import json
import os
from pathlib import Path
import sys
from time import time
import warnings

import numpy as np
from scipy import stats
from scipy.ndimage import binary_dilation

from dipy.core.gradients import gradient_table
from dipy.io import read_bvals_bvecs
from dipy.io.image import load_nifti, save_nifti
from dipy.io.peaks import load_pam
from dipy.io.streamline import load_tractogram
from dipy.io.utils import split_filename_extension
from dipy.reconst.dti import TensorModel
from dipy.segment.bundles import bundle_shape_similarity
from dipy.segment.mask import bounding_box, segment_from_cfa
from dipy.stats.analysis import (
    SpectraGrid,
    anatomical_measures,
    assignment_map,
    buan_profile,
    estimate_length_scales,
    fdr_correction,
    fit_grid_lmm,
    hierarchical_fdr,
    peak_values,
)
from dipy.tracking.streamline import (
    Streamlines,
    set_number_of_points,
    transform_streamlines,
)
from dipy.utils.deprecator import warning_for_keywords
from dipy.utils.logging import logger
from dipy.utils.optpkg import optional_package
from dipy.workflows.multi_io import _resolve_path_or_pattern
from dipy.workflows.workflow import Workflow

pd, have_pd, _ = optional_package("pandas")
smf, have_smf, _ = optional_package("statsmodels.formula.api")
matplt, have_matplotlib, _ = optional_package("matplotlib")
plt, have_plt, _ = optional_package("matplotlib.pyplot")


class SNRinCCFlow(Workflow):
    @classmethod
    def get_short_name(cls):
        return "snrincc"

    def run(
        self,
        data_files,
        bvals_files,
        bvecs_files,
        mask_file,
        bbox_threshold=(0.6, 1, 0, 0.1, 0, 0.1),
        out_dir="",
        out_file="product.json",
        out_mask_cc="cc.nii.gz",
        out_mask_noise="mask_noise.nii.gz",
    ):
        """Compute the signal-to-noise ratio in the corpus callosum.

        Parameters
        ----------
        data_files : string or Path
            Path to the dwi.nii.gz file. This path may contain wildcards to
            process multiple inputs at once.
        bvals_files : string or Path
            Path of bvals.
        bvecs_files : string or Path
            Path of bvecs.
        mask_file : string or Path
            Path of a brain mask file.
        bbox_threshold : variable float, optional
            Threshold for bounding box, values separated with commas for ex.
            [0.6,1,0,0.1,0,0.1].
        out_dir : string or Path, optional
            Where the resulting file will be saved.
        out_file : string, optional
            Name of the result file to be saved.
        out_mask_cc : string, optional
            Name of the CC mask volume to be saved.
        out_mask_noise : string, optional
            Name of the mask noise volume to be saved.

        """
        io_it = self.get_io_iterator()

        for (
            dwi_path,
            bvals_path,
            bvecs_path,
            mask_path,
            out_path,
            cc_mask_path,
            mask_noise_path,
        ) in io_it:
            data, affine = load_nifti(dwi_path)
            bvals, bvecs = read_bvals_bvecs(bvals_path, bvecs_path)
            gtab = gradient_table(bvals=bvals, bvecs=bvecs)

            mask, affine = load_nifti(mask_path)

            logger.info("Computing tensors...")
            tenmodel = TensorModel(gtab)
            tensorfit = tenmodel.fit(data, mask=mask)

            logger.info("Computing worst-case/best-case SNR using the CC...")

            if np.ndim(data) == 4:
                CC_box = np.zeros_like(data[..., 0])
            elif np.ndim(data) == 3:
                CC_box = np.zeros_like(data)
            else:
                raise OSError("DWI data has invalid dimensions")

            mins, maxs = bounding_box(mask)
            mins = np.array(mins)
            maxs = np.array(maxs)
            diff = (maxs - mins) // 4
            bounds_min = mins + diff
            bounds_max = maxs - diff

            CC_box[
                bounds_min[0] : bounds_max[0],
                bounds_min[1] : bounds_max[1],
                bounds_min[2] : bounds_max[2],
            ] = 1

            if len(bbox_threshold) != 6:
                raise OSError("bbox_threshold should have 6 float values")

            mask_cc_part, cfa = segment_from_cfa(
                tensorfit, CC_box, bbox_threshold, return_cfa=True
            )

            if not np.count_nonzero(mask_cc_part.astype(np.uint8)):
                logger.warning(
                    "Empty mask: corpus callosum not found."
                    " Update your data or your threshold"
                )

            save_nifti(cc_mask_path, mask_cc_part.astype(np.uint8), affine)
            logger.info(f"CC mask saved as {cc_mask_path}")

            masked_data = data[mask_cc_part]
            mean_signal = 0
            if masked_data.size:
                mean_signal = np.mean(masked_data, axis=0)
            mask_noise = binary_dilation(mask, iterations=10)
            mask_noise[..., : mask_noise.shape[-1] // 2] = 1
            mask_noise = ~mask_noise

            save_nifti(mask_noise_path, mask_noise.astype(np.uint8), affine)
            logger.info(f"Mask noise saved as {mask_noise_path}")

            noise_std = 0
            if np.count_nonzero(mask_noise.astype(np.uint8)):
                noise_std = np.std(data[mask_noise, :])

            logger.info(f"Noise standard deviation sigma= {noise_std}")

            idx = np.sum(gtab.bvecs, axis=-1) == 0
            gtab.bvecs[idx] = np.inf
            axis_X = np.argmin(np.sum((gtab.bvecs - np.array([1, 0, 0])) ** 2, axis=-1))
            axis_Y = np.argmin(np.sum((gtab.bvecs - np.array([0, 1, 0])) ** 2, axis=-1))
            axis_Z = np.argmin(np.sum((gtab.bvecs - np.array([0, 0, 1])) ** 2, axis=-1))

            SNR_output = []
            SNR_directions = []
            for direction in ["b0", axis_X, axis_Y, axis_Z]:
                if direction == "b0":
                    SNR = mean_signal[0] / noise_std if noise_std else 0
                    logger.info(f"SNR for the b=0 image is : {SNR}")
                else:
                    logger.info(
                        f"SNR for direction {direction} {gtab.bvecs[direction]} is: "
                        f"{SNR}"
                    )
                    SNR_directions.append(direction)
                    SNR = mean_signal[direction] / noise_std if noise_std else 0
                SNR_output.append(SNR)

            snr_str = f"{SNR_output[0]} {SNR_output[1]} {SNR_output[2]} {SNR_output[3]}"
            dir_str = f"b0 {SNR_directions[0]} {SNR_directions[1]} {SNR_directions[2]}"
            data = [{"data": snr_str, "directions": dir_str}]

            with open(Path(out_dir) / out_path, "w") as myfile:
                json.dump(data, myfile)


@warning_for_keywords()
def buan_bundle_profiles(
    model_bundle_folder,
    bundle_folder,
    orig_bundle_folder,
    metric_folder,
    group_id,
    subject,
    *,
    no_disks=100,
    out_dir="",
):
    """
    Applies statistical analysis on bundles and saves the results
    in a directory specified by ``out_dir``.

    See :footcite:p:`Chandio2020a` for further details about the method.

    Parameters
    ----------
    model_bundle_folder : string or Path
        Path to the input model bundle files. This path may contain
        wildcards to process multiple inputs at once.
    bundle_folder : string or Path
        Path to the input bundle files in common space. This path may
        contain wildcards to process multiple inputs at once.
    orig_bundle_folder : string or Path
        Path to the input bundle files in native space. This path may
        contain wildcards to process multiple inputs at once.
    metric_folder : string or Path
        Path to the input dti metric or/and peak files. It will be used as
        metric for statistical analysis of bundles.
    group_id : integer
        what group subject belongs to either 0 for control or 1 for patient.
    subject : string
        subject id e.g. 10001.
    no_disks : integer, optional
        Number of disks used for dividing bundle into disks.
    out_dir : string or Path, optional
        Output directory.

    Notes
    -----
    This function uses :func:`~dipy.stats.analysis.anatomical_measures` to
    store per-point metric values in HDF5 files, which are suitable for
    downstream linear mixed model (LMM) analysis across a group of subjects.
    This is intentionally different from :func:`~dipy.stats.analysis.buan_profile`,
    which computes inverse-distance-weighted along-tract profiles for a single
    subject and returns a ``numpy`` array. Use ``buan_bundle_profiles`` for
    group studies and ``buan_profile`` for single-subject analysis.

    References
    ----------
    .. footbibliography::

    """

    t = time()

    _mb_folder = Path(model_bundle_folder)
    mb = sorted(list(_mb_folder.glob("*.trk")) + list(_mb_folder.glob("*.trx")))
    logger.info(mb)

    _bd_folder = Path(bundle_folder)
    bd = sorted(list(_bd_folder.glob("*.trk")) + list(_bd_folder.glob("*.trx")))
    logger.info(bd)
    _org_folder = Path(orig_bundle_folder)
    org_bd = sorted(list(_org_folder.glob("*.trk")) + list(_org_folder.glob("*.trx")))
    logger.info(org_bd)
    n = len(mb)

    for io in range(n):
        mbundles = load_tractogram(
            mb[io], reference="same", bbox_valid_check=False
        ).streamlines
        bundles = load_tractogram(
            bd[io], reference="same", bbox_valid_check=False
        ).streamlines
        orig_bundles = load_tractogram(
            org_bd[io], reference="same", bbox_valid_check=False
        ).streamlines

        if len(orig_bundles) > 5:
            _, indx = assignment_map(bundles, mbundles, no_disks)
            ind = np.array(indx)

            metric_files_names_dti = list(Path(metric_folder).glob("*.nii.gz"))

            metric_files_names_csa = list(Path(metric_folder).glob("*.pam5"))

            _, affine = load_nifti(metric_files_names_dti[0])

            affine_r = np.linalg.inv(affine)
            transformed_orig_bundles = transform_streamlines(orig_bundles, affine_r)

            for mn in range(len(metric_files_names_dti)):
                metric_name = Path(metric_files_names_dti[mn]).name

                fm = metric_name[:-7]
                bm = Path(mb[io]).name[:-4]

                logger.info(f"bm = {bm}")

                dt = {}

                logger.info(f"metric = {metric_files_names_dti[mn]}")

                metric, _ = load_nifti(metric_files_names_dti[mn])

                anatomical_measures(
                    transformed_orig_bundles,
                    metric,
                    dt,
                    fm,
                    bm,
                    subject,
                    group_id,
                    ind,
                    out_dir,
                )

            for mn in range(len(metric_files_names_csa)):
                metric_name = Path(metric_files_names_csa[mn]).name

                fm = metric_name[:-5]
                bm = Path(mb[io]).name[:-4]

                logger.info(f"bm = {bm}")
                logger.info(f"metric = {metric_files_names_csa[mn]}")
                dt = {}
                metric = load_pam(metric_files_names_csa[mn])

                peak_values(
                    transformed_orig_bundles,
                    metric,
                    dt,
                    fm,
                    bm,
                    subject,
                    group_id,
                    ind,
                    out_dir,
                )

    logger.info(f"total time taken in minutes = {(-t + time()) / 60}")


class BundleAnalysisTractometryFlow(Workflow):
    @classmethod
    def get_short_name(cls):
        return "ba"

    @warning_for_keywords()
    def run(
        self,
        model_bundle_folder,
        subject_folder,
        *,
        bundle_folder=None,
        orig_bundle_folder=None,
        metric_folder=None,
        no_disks=100,
        out_dir="",
    ):
        """Workflow of bundle analytics.

        Applies statistical analysis on bundles of subjects and saves the
        results in a directory specified by ``out_dir``.

        Supports two modes auto-detected from the directory structure:

        - **Group mode**: ``subject_folder`` contains ``patient/`` and
          ``control/`` subdirectories, each with per-subject subdirectories
          containing ``rec_bundles/``, ``org_bundles/``, and
          ``anatomical_measures/``. Outputs HDF5 files suitable for linear
          mixed model analysis.
        - **Single-subject mode**: ``subject_folder`` directly contains
          ``rec_bundles/``, ``org_bundles/``, and ``anatomical_measures/``
          (or the paths are overridden via ``bundle_folder``,
          ``orig_bundle_folder``, and ``metric_folder``). Outputs ``.npy``
          profile arrays (one per bundle/metric pair).

        See :footcite:p:`Chandio2020a` for further details about the method.

        Parameters
        ----------
        model_bundle_folder : string or Path
            Path to the input model bundle files. This path may
            contain wildcards to process multiple inputs at once.
        subject_folder : string or Path
            Path to the subject folder. Either a group-level directory
            (containing ``patient/`` and ``control/`` subdirs) or a
            single-subject directory (directly containing ``rec_bundles/``,
            ``org_bundles/``, and ``anatomical_measures/``).
        bundle_folder : string or Path, optional
            Override path for the registered bundles in common space
            (replaces ``<subject_folder>/rec_bundles``). Only used in
            single-subject mode.
        orig_bundle_folder : string or Path, optional
            Override path for the bundles in native space (replaces
            ``<subject_folder>/org_bundles``). Only used in single-subject
            mode.
        metric_folder : string or Path, optional
            Override path for the metric files (replaces
            ``<subject_folder>/anatomical_measures``). Only used in
            single-subject mode.
        no_disks : integer, optional
            Number of disks used for dividing bundle into disks.
        out_dir : string or Path, optional
            Output directory.

        References
        ----------
        .. footbibliography::

        """
        subject_path = Path(subject_folder)
        if not subject_path.is_dir():
            raise ValueError("Invalid path to subjects")

        has_overrides = any(
            p is not None for p in (bundle_folder, orig_bundle_folder, metric_folder)
        )
        if has_overrides or (subject_path / "rec_bundles").is_dir():
            self._run_single_subject(
                model_bundle_folder,
                subject_path,
                no_disks,
                out_dir,
                bundle_folder=bundle_folder,
                orig_bundle_folder=orig_bundle_folder,
                metric_folder=metric_folder,
            )
        else:
            self._run_group(model_bundle_folder, subject_path, no_disks, out_dir)

    def _run_group(self, model_bundle_folder, subject_path, no_disks, out_dir):
        """Run group-mode tractometry (patient/control subdirectory structure).

        Parameters
        ----------
        model_bundle_folder : string or Path
            Path to the input model bundle files.
        subject_path : Path
            Path object pointing to the group-level subject folder.
        no_disks : integer
            Number of disks used for dividing bundle into disks.
        out_dir : string or Path
            Output directory.

        """
        groups = [p.name for p in subject_path.iterdir()]
        groups.sort()
        for group in groups:
            group_dirname = subject_path / group
            if group_dirname.is_dir():
                logger.info(f"group = {group}")
                all_subjects = os.listdir(group_dirname)
                all_subjects.sort()
                logger.info(all_subjects)
            if group.lower() == "patient":
                group_id = 1  # 1 means patient
            elif group.lower() == "control":
                group_id = 0  # 0 means control
            else:
                logger.info(group)
                raise ValueError("Invalid group. Neither patient nor control")

            for sub in all_subjects:
                logger.info(sub)
                pre = group_dirname / sub
                logger.info(pre)
                b = Path(pre) / "rec_bundles"
                c = Path(pre) / "org_bundles"
                d = Path(pre) / "anatomical_measures"
                buan_bundle_profiles(
                    model_bundle_folder,
                    b,
                    c,
                    d,
                    group_id,
                    sub,
                    no_disks=no_disks,
                    out_dir=out_dir,
                )

    def _run_single_subject(
        self,
        model_bundle_folder,
        subject_path,
        no_disks,
        out_dir,
        *,
        bundle_folder=None,
        orig_bundle_folder=None,
        metric_folder=None,
    ):
        """Run single-subject tractometry saving ``.npy`` profile arrays.

        Parameters
        ----------
        model_bundle_folder : string or Path
            Path to the input model bundle files.
        subject_path : Path
            Path object pointing to the single-subject directory.
        no_disks : integer
            Number of disks used for dividing bundle into disks.
        out_dir : string or Path
            Output directory.
        bundle_folder : string or Path, optional
            Path to the registered bundles in common space. Defaults to
            ``<subject_path>/rec_bundles``.
        orig_bundle_folder : string or Path, optional
            Path to the bundles in native space. Defaults to
            ``<subject_path>/org_bundles``.
        metric_folder : string or Path, optional
            Path to the metric files. Defaults to
            ``<subject_path>/anatomical_measures``.

        """
        bundle_folder = (
            Path(bundle_folder)
            if bundle_folder is not None
            else subject_path / "rec_bundles"
        )
        orig_bundle_folder = (
            Path(orig_bundle_folder)
            if orig_bundle_folder is not None
            else subject_path / "org_bundles"
        )
        metric_folder = (
            Path(metric_folder)
            if metric_folder is not None
            else subject_path / "anatomical_measures"
        )

        _mb_folder = Path(model_bundle_folder)
        mb_list = sorted(
            list(_mb_folder.glob("*.trk")) + list(_mb_folder.glob("*.trx"))
        )
        if not mb_list:
            logger.info("No model bundle files found in the specified folder")
            sys.exit(1)

        bd_list = sorted(
            list(bundle_folder.glob("*.trk")) + list(bundle_folder.glob("*.trx"))
        )
        if not bd_list:
            logger.info("No registered bundle files found in the specified folder")
            sys.exit(1)
        org_list = sorted(
            list(orig_bundle_folder.glob("*.trk"))
            + list(orig_bundle_folder.glob("*.trx"))
        )
        if not org_list:
            logger.info("No original bundle files found in the specified folder")
            sys.exit(1)
        metric_files = sorted(metric_folder.glob("*.nii.gz"))

        Path(out_dir).mkdir(parents=True, exist_ok=True)

        for mb_file, bd_file, org_file in zip(mb_list, bd_list, org_list):
            mbundles = load_tractogram(
                mb_file, reference="same", bbox_valid_check=False
            ).streamlines
            bundles = load_tractogram(
                bd_file, reference="same", bbox_valid_check=False
            ).streamlines
            orig_bundles = load_tractogram(
                org_file, reference="same", bbox_valid_check=False
            ).streamlines

            if len(orig_bundles) <= 5:
                continue

            bname = Path(mb_file).stem
            for metric_file in metric_files:
                metric, affine = load_nifti(metric_file)
                fname = Path(metric_file).stem.replace(".nii", "")
                logger.info(f"Applying metric {metric_file} on bundle {bname}")
                if metric.ndim > 3:
                    logger.info(
                        f"Skipping metric {metric_file} with 4D shape {metric.shape}"
                    )
                    continue
                profile = buan_profile(
                    mbundles,
                    bundles,
                    orig_bundles,
                    metric,
                    affine,
                    no_disks=no_disks,
                )
                np.save(Path(out_dir) / f"{bname}_{fname}_profile.npy", profile)


class LinearMixedModelsFlow(Workflow):
    @classmethod
    def get_short_name(cls):
        return "lmm"

    def get_metric_name(self, path):
        """Splits the path string and returns name of anatomical measure
        (eg: fa), bundle name eg(AF_L) and bundle name with metric name
        (eg: AF_L_fa)

        Parameters
        ----------
        path : string or Path
            Path to the input metric files. This path may
            contain wildcards to process multiple inputs at once.
        """

        name = Path(path).name
        count = 0
        i = len(name) - 1
        while i > 0:
            if name[i] == ".":
                count = i
                break
            i = i - 1

        for j in range(len(name)):
            if name[j] == "_":
                if name[j + 1] != "L" and name[j + 1] != "R" and name[j + 1] != "F":
                    return name[j + 1 : count], name[:j], name[:count]

        return " ", " ", " "

    def save_lmm_plot(self, plot_file, title, bundle_name, x, y):
        """Saves LMM plot with segment/disk number on x-axis and
        -log10(pvalues) on y-axis in out_dir folder.

        Parameters
        ----------
        plot_file : string
            Path to the plot file. This path may
            contain wildcards to process multiple inputs at once.
        title : string
            Title for the plot.
        bundle_name : string
            Bundle name.
        x : list
            list containing segment/disk number for x-axis.
        y : list
            list containing -log10(pvalues) per segment/disk number for y-axis.

        """

        n = len(x)
        dotted = np.ones(n)
        dotted[:] = 2
        c1 = np.random.rand(1, 3)

        y_pos = np.arange(n)

        (l1,) = plt.plot(
            y_pos,
            dotted,
            color="red",
            marker=".",
            linestyle="solid",
            linewidth=0.6,
            markersize=0.7,
            label="p-value < 0.01",
        )

        (l2,) = plt.plot(
            y_pos,
            dotted + 1,
            color="black",
            marker=".",
            linestyle="solid",
            linewidth=0.4,
            markersize=0.4,
            label="p-value < 0.001",
        )

        first_legend = plt.legend(handles=[l1, l2], loc="upper right")

        axes = plt.gca()
        axes.add_artist(first_legend)
        axes.set_ylim([0, 6])

        l3 = plt.bar(y_pos, y, color=c1, alpha=0.5, label=bundle_name)
        plt.legend(handles=[l3], loc="upper left")
        plt.title(title.upper())
        plt.xlabel("Segment Number")
        plt.ylabel("-log10(Pvalues)")
        plt.savefig(plot_file)
        plt.clf()

    @warning_for_keywords()
    def run(self, h5_files, *, no_disks=100, out_dir=""):
        """Workflow of linear Mixed Models.

        Applies linear Mixed Models on bundles of subjects and saves the
        results in a directory specified by ``out_dir``.

        Parameters
        ----------
        h5_files : string or Path
            Path to the input metric files. This path may
            contain wildcards to process multiple inputs at once.
        no_disks : integer, optional
            Number of disks used for dividing bundle into disks.
        out_dir : string or Path, optional
            Output directory.

        """

        io_it = self.get_io_iterator()

        for file_path in io_it:
            logger.info(f"Applying metric {file_path}")

            file_name, bundle_name, save_name = self.get_metric_name(file_path)
            logger.info(f" file name = {file_name}")
            logger.info(f"file path = {file_path}")

            pvalues = np.zeros(no_disks)
            warnings.filterwarnings("ignore")
            # run mixed linear model for every disk
            for i in range(no_disks):
                disk_count = i + 1
                df = pd.read_hdf(file_path, where="disk=disk_count")

                logger.info(f"read the dataframe for disk number {disk_count}")
                # check if data has significant data to perform LMM
                if len(df) < 10:
                    raise ValueError("Dataset for Linear Mixed Model is too small")

                criteria = file_name + " ~ group"
                md = smf.mixedlm(criteria, df, groups=df["subject"])

                mdf = md.fit()

                pvalues[i] = mdf.pvalues["group"]

            x = list(range(1, len(pvalues) + 1))
            y = -1 * np.log10(pvalues)

            save_file = Path(out_dir) / (save_name + "_pvalues.npy")
            np.save(save_file, pvalues)

            save_file = Path(out_dir) / (save_name + "_pvalues_log.npy")
            np.save(save_file, y)

            save_file = Path(out_dir) / (save_name + ".png")
            self.save_lmm_plot(save_file, file_name, bundle_name, x, y)


class BundleShapeAnalysis(Workflow):
    @classmethod
    def get_short_name(cls):
        return "BS"

    @warning_for_keywords()
    def run(self, subject_folder, *, clust_thr=(5, 3, 1.5), threshold=6, out_dir=""):
        """Workflow of bundle analytics.

        Applies bundle shape similarity analysis on bundles of subjects and
        saves the results in a directory specified by ``out_dir``.

        See :footcite:p:`Chandio2020a` for further details about the method.

        Parameters
        ----------
        subject_folder : string or Path
            Path to the input subject folder. This path may contain
            wildcards to process multiple inputs at once.
        clust_thr : variable float, optional
            list of bundle clustering thresholds used in QuickBundlesX.
        threshold : float, optional
            Bundle shape similarity threshold.
        out_dir : string or Path, optional
            Output directory.

        References
        ----------
        .. footbibliography::

        """
        rng = np.random.default_rng()
        all_subjects = []
        if Path(subject_folder).is_dir():
            groups = sorted([p.name for p in Path(subject_folder).iterdir()])
        else:
            raise ValueError("Not a directory")

        for group in groups:
            group_dirname = Path(subject_folder) / group
            if group_dirname.is_dir():
                subjects = sorted([p.name for p in Path(group_dirname).iterdir()])
                logger.info(
                    "first "
                    + str(len(subjects))
                    + " subjects in matrix belong to "
                    + group
                    + " group"
                )

                for sub in subjects:
                    dpath = group_dirname / sub
                    if dpath.is_dir():
                        all_subjects.append(dpath)

        N = len(all_subjects)

        bundles = [p.name for p in (Path(all_subjects[0]) / "rec_bundles").iterdir()]
        for bun in bundles:
            # bundle shape similarity matrix
            ba_matrix = np.zeros((N, N))
            logger.info(bun)
            for i, sub in enumerate(all_subjects):
                bundle1 = load_tractogram(
                    Path(sub) / "rec_bundles" / bun,
                    reference="same",
                    bbox_valid_check=False,
                ).streamlines

                for j, subi in enumerate(all_subjects):
                    logger.info(subi)

                    bundle2 = load_tractogram(
                        Path(subi) / "rec_bundles" / bun,
                        reference="same",
                        bbox_valid_check=False,
                    ).streamlines

                    ba_value = bundle_shape_similarity(
                        bundle1, bundle2, rng, clust_thr=clust_thr, threshold=threshold
                    )

                    ba_matrix[i][j] = ba_value
            logger.info("saving BA score matrix")
            np.save(Path(out_dir) / (bun[:-4] + ".npy"), ba_matrix)

            cmap = matplt.colormaps["Blues"]
            plt.title(bun[:-4])
            plt.imshow(ba_matrix, cmap=cmap)
            plt.colorbar()
            plt.clim(0, 1)
            plt.savefig(Path(out_dir) / f"SM_{bun[:-4]}")
            plt.clf()


def _tractogram_files(path):
    """Sorted tractogram files of a file, folder or pattern with wildcards."""
    files = _resolve_path_or_pattern(path)
    return [Path(f) for f in files if str(f).endswith((".trk", ".trx"))]


def _file_name(path):
    """Name of a file without its directory and extension(s)."""
    name, _ = split_filename_extension(Path(path).name)
    return name


def _parse_group(value):
    """Group label given on the command line, as an int if it is a number."""
    if value is None:
        return None
    value = str(value)
    return int(value) if value.isdecimal() else value


def _spectra_grids(model_bundle_files, segment_length, radial_length, robust_method):
    """SPECTRA grid of each model bundle, keyed by the bundle name."""
    if robust_method not in ("linear", "spline", "none"):
        raise ValueError("robust_method must be 'linear', 'spline' or 'none'")

    grids = {}
    for model_file in _tractogram_files(model_bundle_files):
        model_bundle = load_tractogram(
            model_file, reference="same", bbox_valid_check=False
        ).streamlines
        grids[_file_name(model_file)] = SpectraGrid(
            model_bundle,
            segment_length=segment_length,
            radial_length=radial_length,
            use_robust_centroid=robust_method != "none",
            robust_method=robust_method,
        )
    if not grids:
        raise ValueError("No model bundle found")
    return grids


def _alpha_name(alpha):
    """Column suffix of a Stage 2 level, e.g. ac15 for 0.15, as in SPECTRA."""
    return f"ac{int(alpha * 100):02d}"


def _save_spectra_results(
    bundles,
    *,
    stats_mode,
    alpha,
    adaptive,
    alpha_block,
    alpha_cell,
    combination_method,
    out_dir,
    out_stats,
    out_blocks,
    out_summary,
):
    """Correct the statistics of all the bundles and save them as CSV files.

    ``bundles`` maps each bundle name to a dictionary with ``"grid"``, ``"t"``
    and ``"p"`` (the statistics to correct), ``"l_s"`` and ``"l_r"`` (the
    length scales), ``"mask"`` (the tested cells), ``"columns"`` (the arrays
    saved for each cell) and ``"n_subjects"``.
    """
    alpha_cell = list(alpha_cell)
    block_of_cell = {}
    blocks = []
    if stats_mode == "fdr":
        # Benjamini-Hochberg over the cells of all the bundles
        p_all = np.concatenate([b["p"].ravel() for b in bundles.values()])
        p_corr = fdr_correction(p_all, adaptive=adaptive)
        start = 0
        for bundle in bundles.values():
            size = bundle["p"].size
            bundle["p_corr"] = p_corr[start : start + size].reshape(bundle["p"].shape)
            start += size
        levels = {"significant": alpha}
        p_name = "p_fdr"
    else:
        p_hfdr, blocks = hierarchical_fdr(
            bundles,
            alpha_block=alpha_block,
            use_resel_blocks=stats_mode == "hfdr_resel",
            combination_method=combination_method,
        )
        for name, bundle in bundles.items():
            bundle["p_corr"] = p_hfdr[name]
            bundle["n_blocks"] = 0
            bundle["n_blocks_significant"] = 0
        for block in blocks:
            bundle = bundles[block["bundle"]]
            for k, (s, r) in enumerate(block["cells"]):
                p_cond = np.nan if block["p_cond"] is None else block["p_cond"][k]
                block_of_cell[(block["bundle"], s, r)] = (
                    bundle["n_blocks"],
                    block["significant"],
                    p_cond,
                )
            block["block"] = bundle["n_blocks"]
            bundle["n_blocks"] += 1
            bundle["n_blocks_significant"] += int(block["significant"])
        levels = {f"significant_{_alpha_name(a)}": a for a in alpha_cell}
        p_name = "p_hfdr"

    cell_rows = []
    summary_rows = []
    for name, bundle in bundles.items():
        for s, r in zip(*np.nonzero(bundle["mask"])):
            row = {"bundle": name, "segment": s, "radial": r}
            for column, values in bundle["columns"].items():
                row[column] = values[s, r]
            if blocks:
                block, block_significant, p_cond = block_of_cell.get(
                    (name, s, r), (np.nan, False, np.nan)
                )
                row["block"] = block
                row["block_significant"] = block_significant
                row["p_cond"] = p_cond
            row[p_name] = bundle["p_corr"][s, r]
            for level_name, level in levels.items():
                row[level_name] = bundle["p_corr"][s, r] < level
            cell_rows.append(row)

        summary = {
            "bundle": name,
            "n_subjects": bundle["n_subjects"],
            "n_tested": int(bundle["mask"].sum()),
        }
        for level_name, level in levels.items():
            n_significant = int(np.sum(bundle["p_corr"] < level))
            summary[f"n_{level_name}"] = n_significant
            logger.info(f"{name}: {n_significant} cells {level_name}")
        if blocks:
            summary["n_blocks"] = bundle["n_blocks"]
            summary["n_blocks_significant"] = bundle["n_blocks_significant"]
        summary["l_s"] = bundle["l_s"]
        summary["l_r"] = bundle["l_r"]
        summary_rows.append(summary)

    Path(out_dir).mkdir(parents=True, exist_ok=True)
    pd.DataFrame(cell_rows).to_csv(Path(out_dir) / out_stats, index=False)
    pd.DataFrame(summary_rows).to_csv(Path(out_dir) / out_summary, index=False)
    if blocks:
        block_rows = [
            {
                "bundle": block["bundle"],
                "block": block["block"],
                "n_cells": len(block["cells"]),
                "z": block["z"],
                "p": block["p"],
                "p_adj": block["p_adj"],
                "significant": block["significant"],
            }
            for block in blocks
        ]
        pd.DataFrame(block_rows).to_csv(Path(out_dir) / out_blocks, index=False)
    logger.info(f"SPECTRA statistics saved in {Path(out_dir).resolve()}")


class SpectraProfileFlow(Workflow):
    @classmethod
    def get_short_name(cls):
        return "spectra_profile"

    def run(
        self,
        model_bundle_files,
        bundle_files,
        orig_bundle_files,
        metric_files,
        *,
        segment_length=5.0,
        radial_length=5.0,
        robust_method="linear",
        min_streamlines=20,
        out_dir="",
    ):
        """Workflow of SPECTRA bundle profiles.

        Computes the SPECTRA profiles of the bundles of one subject for each
        metric: the two-dimensional profile along the tract and across the
        radial bins (see :meth:`dipy.stats.analysis.SpectraGrid.profile`) and
        the one-dimensional BUAN profile along the tract, saved as
        ``profile_1d`` (see
        :meth:`dipy.stats.analysis.SpectraGrid.profile_1d_adaptive_buan`). The grid of
        each bundle is built from its model bundle.

        The profiles are saved in ``<out_dir>/<bundle>/<metric>.npz``, where
        ``<bundle>`` is the name of the model bundle file and ``<metric>`` the
        name of the metric file. Use ``<profile_folder>/<subject>`` as
        ``out_dir`` to prepare the profiles of a study for
        ``dipy_spectra_lmm``.

        The model, registered and native bundles are matched in the sorted
        order of their file names.

        See :footcite:p:`Feng2026SPECTRA` for further details about the method.

        Parameters
        ----------
        model_bundle_files : string or Path
            Path to the model (atlas) bundles. It can be a file, a folder or a
            path with wildcards.
        bundle_files : string or Path
            Path to the bundles of the subject in the space of the model
            bundles, e.g. the recognized bundles of RecoBundles.
        orig_bundle_files : string or Path
            Path to the same bundles in the native space of the subject.
        metric_files : string or Path
            Path to the metric volumes of the subject (e.g. FA, MD), in its
            native space.
        segment_length : float, optional
            Desired along-tract segment length in millimeters.
        radial_length : float, optional
            Desired radial bin width in millimeters.
        robust_method : string, optional
            Extension of the centroid to the bundle endpoints: "linear",
            "spline", or "none" to use the QuickBundles centroid.
        min_streamlines : int, optional
            Bundles with fewer streamlines are skipped.
        out_dir : string or Path, optional
            Output directory.

        References
        ----------
        .. footbibliography::

        """
        grids = _spectra_grids(
            model_bundle_files, segment_length, radial_length, robust_method
        )
        bundle_list = _tractogram_files(bundle_files)
        orig_list = _tractogram_files(orig_bundle_files)
        if not len(grids) == len(bundle_list) == len(orig_list):
            raise ValueError(
                "Give the same number of model, registered and native bundles"
            )
        metric_list = [
            f
            for f in _resolve_path_or_pattern(metric_files)
            if str(f).endswith((".nii", ".nii.gz"))
        ]
        if not metric_list:
            raise ValueError("No metric file found")

        for (name, grid), bundle_file, orig_file in zip(
            grids.items(), bundle_list, orig_list
        ):
            bundle = load_tractogram(
                bundle_file, reference="same", bbox_valid_check=False
            ).streamlines
            orig_bundle = load_tractogram(
                orig_file, reference="same", bbox_valid_check=False
            ).streamlines

            if len(bundle) < min_streamlines:
                logger.warning(
                    f"Skipping {name}: {len(bundle)} streamlines, at least "
                    f"{min_streamlines} required"
                )
                continue
            if len(bundle) != len(orig_bundle):
                raise ValueError(
                    f"{bundle_file} and {orig_file} must have the same streamlines"
                )

            # Resample both bundles to the same evenly spaced points, so that
            # they correspond point for point
            n_points = [len(s) for s in orig_bundle]
            orig_bundle = Streamlines(
                set_number_of_points(s, nb_points=n)
                for s, n in zip(orig_bundle, n_points)
            )
            bundle = Streamlines(
                set_number_of_points(s, nb_points=n) for s, n in zip(bundle, n_points)
            )
            logger.info(
                f"{name}: grid of {grid.n_segments} segments and "
                f"{grid.n_radial} radial bins"
            )

            bundle_dir = Path(out_dir) / name
            bundle_dir.mkdir(parents=True, exist_ok=True)
            for metric_file in metric_list:
                metric, affine = load_nifti(metric_file)
                if metric.ndim != 3:
                    logger.warning(f"Skipping {metric_file}: not a 3D volume")
                    continue

                out_file = bundle_dir / f"{_file_name(metric_file)}.npz"
                np.savez(
                    out_file,
                    profile_1d=grid.profile_1d_adaptive_buan(
                        bundle, orig_bundle, metric, affine
                    ),
                    profile_2d=grid.profile(bundle, orig_bundle, metric, affine),
                    n_segments=grid.n_segments,
                    n_radial=grid.n_radial,
                    segment_length=grid.segment_length_actual,
                    radial_length=grid.radial_length_actual,
                )
                logger.info(f"SPECTRA profiles saved as {out_file}")


class SpectraLMMFlow(Workflow):
    @classmethod
    def get_short_name(cls):
        return "spectra_lmm"

    def run(
        self,
        metadata_file,
        profile_folder,
        model_bundle_files,
        *,
        metric="fa",
        mode="1d",
        subject_col="subject",
        target_col="dx",
        target_case=None,
        target_ref=None,
        group_col="site",
        covariates=("age", "sex"),
        continuous=("age",),
        min_prop=0.3,
        min_subjects=20,
        edge_mask=None,
        segment_length=5.0,
        radial_length=5.0,
        robust_method="linear",
        stats_mode="hfdr_resel",
        out_dir="",
    ):
        """Workflow of SPECTRA linear mixed models.

        First stage of the SPECTRA statistics: fits a linear mixed model in
        each cell of the SPECTRA profiles of a group of subjects (see
        :func:`dipy.stats.analysis.fit_grid_lmm`), and saves the results of
        each bundle in ``<out_dir>/<bundle>.npz``. The bundles can be fitted
        in separate runs, e.g. in parallel jobs. The second stage,
        ``dipy_spectra_fdr``, corrects the results of all the bundles for
        multiple comparisons.

        The profiles are read from
        ``<profile_folder>/<subject>/<bundle>/<metric>.npz``, as written by
        ``dipy_spectra_profile``. The bundles are the names of the model
        bundle files, and their grids are built with the same options as in
        ``dipy_spectra_profile``. Subjects without a profile for a bundle are
        left out of that bundle.

        See :footcite:p:`Feng2026SPECTRA` for further details about the method.

        Parameters
        ----------
        metadata_file : string or Path
            CSV file with one row per subject and the columns of the model.
        profile_folder : string or Path
            Folder with one subfolder of profiles per subject.
        model_bundle_files : string or Path
            Path to the model (atlas) bundles to fit. It can be a file, a
            folder or a path with wildcards.
        metric : string, optional
            Name of the metric to analyze, e.g. fa.
        mode : string, optional
            Profiles to analyze: "1d" (along the tract) or "2d" (along the
            tract and across the radial bins).
        subject_col : string, optional
            Column of the metadata with the subject names.
        target_col : string, optional
            Column of the metadata with the variable of interest.
        target_case : string, optional
            Case group of a categorical target, e.g. AD.
        target_ref : string, optional
            Reference group of a categorical target, e.g. CN. Give both
            ``target_case`` and ``target_ref`` for a categorical target,
            otherwise the target is used as a number.
        group_col : string, optional
            Column of the metadata used for the random intercept, e.g. the
            acquisition site. Use None for no random effect.
        covariates : variable string, optional
            Columns of the metadata used as covariates.
        continuous : variable string, optional
            Columns among the target and covariates that are standardized.
        min_prop : float, optional
            Minimum proportion of subjects with data in a cell to fit it.
        min_subjects : int, optional
            Minimum number of subjects with data in a cell to fit it.
        edge_mask : variable int, optional
            Number of cells left out at each end of the tract, and optionally
            at each side of the radial bins, e.g. 2 or 2 1.
        segment_length : float, optional
            Desired along-tract segment length in millimeters, as in
            ``dipy_spectra_profile``.
        radial_length : float, optional
            Desired radial bin width in millimeters, as in
            ``dipy_spectra_profile``.
        robust_method : string, optional
            Extension of the centroid, as in ``dipy_spectra_profile``.
        stats_mode : string, optional
            Correction that will be used in ``dipy_spectra_fdr``: "fdr",
            "hfdr_resel" or "hfdr_bundle". The hierarchical FDR modes also fit
            the model without the target, to estimate the spatial correlation.
        out_dir : string or Path, optional
            Output directory.

        References
        ----------
        .. footbibliography::

        """
        if not have_pd:
            raise ImportError("dipy_spectra_lmm requires pandas")
        if mode not in ("1d", "2d"):
            raise ValueError("mode must be '1d' or '2d'")
        if stats_mode not in ("fdr", "hfdr_resel", "hfdr_bundle"):
            raise ValueError("stats_mode must be 'fdr', 'hfdr_resel' or 'hfdr_bundle'")

        metadata = pd.read_csv(metadata_file)
        fit_reduced = stats_mode != "fdr"
        grids = _spectra_grids(
            model_bundle_files, segment_length, radial_length, robust_method
        )
        Path(out_dir).mkdir(parents=True, exist_ok=True)

        for name, grid in grids.items():
            files = [
                Path(profile_folder) / str(subject) / name / f"{metric}.npz"
                for subject in metadata[subject_col]
            ]
            has_profile = np.array([f.exists() for f in files])
            if not has_profile.any():
                logger.warning(f"Skipping {name}: no {metric} profile found")
                continue

            profiles = np.array(
                [
                    np.load(f)[f"profile_{mode}"]
                    for f, found in zip(files, has_profile)
                    if found
                ]
            )
            if profiles.ndim == 2:
                profiles = profiles[:, :, np.newaxis]
            n_radial = 1 if mode == "1d" else grid.n_radial
            if profiles.shape[1:] != (grid.n_segments, n_radial):
                raise ValueError(
                    f"The {name} profiles do not have the shape of its grid. Use "
                    "the grid options of dipy_spectra_profile."
                )
            data = metadata[has_profile].reset_index(drop=True)

            # Cells left out at the ends of the tract and the sides of the bins
            mask = np.ones(profiles.shape[1:], dtype=bool)
            edges = list(edge_mask or [])
            for axis, edge in enumerate(edges[:2]):
                size = mask.shape[axis]
                if edge <= 0 or size == 1:
                    continue
                if 2 * edge >= size:
                    logger.warning(f"{name}: edge_mask {edge} too large, not used")
                    continue
                index = [slice(None), slice(None)]
                index[axis] = slice(0, edge)
                mask[tuple(index)] = False
                index[axis] = slice(size - edge, size)
                mask[tuple(index)] = False

            logger.info(f"{name}: fitting {len(data)} subjects")
            fit = fit_grid_lmm(
                profiles,
                data,
                target_col,
                covariates=list(covariates),
                groups=group_col or None,
                continuous=list(continuous),
                case_group=_parse_group(target_case),
                ref_group=_parse_group(target_ref),
                mask=mask,
                min_prop=min_prop,
                min_subjects=min_subjects,
                fit_reduced=fit_reduced,
            )
            if not fit["mask"].any():
                logger.warning(f"Skipping {name}: no cell could be tested")
                continue

            l_s = l_r = None
            if fit_reduced:
                try:
                    l_s, l_r = estimate_length_scales(fit["residuals_reduced"], grid)
                except ValueError as error:
                    logger.warning(f"{name}: length scales not estimated ({error})")

            results = {
                key: value
                for key, value in fit.items()
                if not key.startswith("residuals")
            }
            out_file = Path(out_dir) / f"{name}.npz"
            np.savez(
                out_file,
                **results,
                l_s=np.nan if l_s is None else l_s,
                l_r=np.nan if l_r is None else l_r,
                n_subjects_total=len(data),
            )
            logger.info(
                f"{name}: {int(fit['mask'].sum())} cells tested, saved as {out_file}"
            )


class SpectraFDRFlow(Workflow):
    @classmethod
    def get_short_name(cls):
        return "spectra_fdr"

    def run(
        self,
        fit_files,
        model_bundle_files,
        *,
        segment_length=5.0,
        radial_length=5.0,
        robust_method="linear",
        stats_mode="hfdr_resel",
        alpha=0.05,
        adaptive=False,
        alpha_block=0.05,
        alpha_cell=(0.05, 0.15, 0.25),
        combination_method="simes",
        out_dir="",
        out_stats="spectra_stats.csv",
        out_blocks="spectra_blocks.csv",
        out_summary="spectra_summary.csv",
    ):
        """Workflow of SPECTRA corrections for multiple comparisons.

        Second stage of the SPECTRA statistics: corrects the results of
        ``dipy_spectra_lmm`` for all the bundles together, with the
        Benjamini-Hochberg procedure or the hierarchical FDR procedure of
        SPECTRA (see :func:`dipy.stats.analysis.hierarchical_fdr`).

        The results are saved in three CSV files: the statistics of each
        tested cell (``out_stats``), the Stage 1 blocks of the hierarchical
        FDR (``out_blocks``) and a summary of each bundle (``out_summary``).

        See :footcite:p:`Feng2026SPECTRA` for further details about the method.

        Parameters
        ----------
        fit_files : string or Path
            Path to the results of ``dipy_spectra_lmm`` (``<bundle>.npz``). It
            can be a file, a folder or a path with wildcards.
        model_bundle_files : string or Path
            Path to the model (atlas) bundles of the results, used for the
            grids of the hierarchical FDR.
        segment_length : float, optional
            Desired along-tract segment length in millimeters, as in
            ``dipy_spectra_profile``.
        radial_length : float, optional
            Desired radial bin width in millimeters, as in
            ``dipy_spectra_profile``.
        robust_method : string, optional
            Extension of the centroid, as in ``dipy_spectra_profile``.
        stats_mode : string, optional
            Correction for multiple comparisons: "fdr" (Benjamini-Hochberg
            over all the cells), "hfdr_resel" (hierarchical FDR with blocks
            of about one RESEL) or "hfdr_bundle" (hierarchical FDR with one
            block per bundle).
        alpha : float, optional
            FDR level of the "fdr" mode.
        adaptive : bool, optional
            Use the adaptive Benjamini-Hochberg procedure in the "fdr" mode.
        alpha_block : float, optional
            FDR level of the Stage 1 blocks of the hierarchical FDR.
        alpha_cell : variable float, optional
            FDR level(s) of the Stage 2 cells of the hierarchical FDR. Each
            level has its own column of significant cells.
        combination_method : string, optional
            Combination of the cells of a block: "simes" or "stouffer".
        out_dir : string or Path, optional
            Output directory.
        out_stats : string, optional
            Name of the CSV file with the statistics of the tested cells.
        out_blocks : string, optional
            Name of the CSV file with the blocks of the hierarchical FDR.
        out_summary : string, optional
            Name of the CSV file with the summary of each bundle.

        References
        ----------
        .. footbibliography::

        """
        if not have_pd:
            raise ImportError("dipy_spectra_fdr requires pandas")
        if stats_mode not in ("fdr", "hfdr_resel", "hfdr_bundle"):
            raise ValueError("stats_mode must be 'fdr', 'hfdr_resel' or 'hfdr_bundle'")

        files = [
            Path(f)
            for f in _resolve_path_or_pattern(fit_files)
            if str(f).endswith(".npz")
        ]
        if not files:
            raise ValueError("No result of dipy_spectra_lmm found")
        grids = {}
        if stats_mode != "fdr":
            grids = _spectra_grids(
                model_bundle_files, segment_length, radial_length, robust_method
            )

        bundles = {}
        for fit_file in files:
            name = _file_name(fit_file)
            fit = dict(np.load(fit_file))
            l_s = None if np.isnan(fit["l_s"]) else float(fit["l_s"])
            l_r = None if np.isnan(fit["l_r"]) else float(fit["l_r"])
            if stats_mode != "fdr":
                if name not in grids:
                    raise ValueError(f"No model bundle found for {name}")
                if l_s is None:
                    raise ValueError(
                        f"{name} has no length scales. Run dipy_spectra_lmm with "
                        f"--stats_mode {stats_mode}."
                    )
            columns = ["n_subjects", "t", "p", "beta", "se", "dof"]
            columns += ["ci_lower", "ci_upper", "r2_partial", "d_cohen"]
            bundles[name] = {
                "grid": grids.get(name),
                "t": fit["t"],
                "p": fit["p"],
                "l_s": l_s,
                "l_r": l_r,
                "mask": fit["mask"],
                "columns": {c: fit[c] for c in columns if c in fit},
                "n_subjects": int(fit["n_subjects_total"]),
            }

        _save_spectra_results(
            bundles,
            stats_mode=stats_mode,
            alpha=alpha,
            adaptive=adaptive,
            alpha_block=alpha_block,
            alpha_cell=alpha_cell,
            combination_method=combination_method,
            out_dir=out_dir,
            out_stats=out_stats,
            out_blocks=out_blocks,
            out_summary=out_summary,
        )


class SpectraHFDRExternalFlow(Workflow):
    @classmethod
    def get_short_name(cls):
        return "spectra_hfdr_external"

    def run(
        self,
        stats_file,
        model_bundle_files,
        *,
        residuals_folder=None,
        segment_length=5.0,
        radial_length=5.0,
        robust_method="linear",
        no_resel_blocks=False,
        alpha_block=0.05,
        alpha_cell=(0.05, 0.15, 0.25),
        combination_method="simes",
        out_dir="",
        out_stats="spectra_stats.csv",
        out_blocks="spectra_blocks.csv",
        out_summary="spectra_summary.csv",
    ):
        """Workflow of SPECTRA hierarchical FDR for external statistics.

        Applies the hierarchical FDR procedure of SPECTRA (see
        :func:`dipy.stats.analysis.hierarchical_fdr`) to statistics fitted in
        each cell of SPECTRA grids by another software.

        The statistics are read from a CSV file with the columns
        ``grid_name`` (the bundle), ``s_idx`` (the segment), ``r_idx`` (the
        radial bin), ``t_stat`` and ``p_value``, and optionally ``beta``,
        ``se``, ``dof`` and ``d_cohen``. Cells absent from the file are not
        tested. The grids are built from the model bundles, whose names must
        be the grid names.

        The blocks of about one RESEL use the length scales of the residuals
        of the model without the target, read from
        ``<residuals_folder>/<grid_name>_residuals.csv`` with the columns
        ``s_idx``, ``r_idx``, ``subject_id`` and ``residual``. A grid without
        residuals is a single block.

        See :footcite:p:`Feng2026SPECTRA` for further details about the method.

        Parameters
        ----------
        stats_file : string or Path
            CSV file with the statistics of each cell.
        model_bundle_files : string or Path
            Path to the model (atlas) bundles. It can be a file, a folder or a
            path with wildcards.
        residuals_folder : string or Path, optional
            Folder with the residuals of each grid.
        segment_length : float, optional
            Desired along-tract segment length in millimeters, as in
            ``dipy_spectra_profile``.
        radial_length : float, optional
            Desired radial bin width in millimeters, as in
            ``dipy_spectra_profile``.
        robust_method : string, optional
            Extension of the centroid, as in ``dipy_spectra_profile``.
        no_resel_blocks : bool, optional
            Use one Stage 1 block per grid instead of blocks of about one
            RESEL. No residuals are needed.
        alpha_block : float, optional
            FDR level of the Stage 1 blocks.
        alpha_cell : variable float, optional
            FDR level(s) of the Stage 2 cells. Each level has its own column
            of significant cells.
        combination_method : string, optional
            Combination of the cells of a block: "simes" or "stouffer".
        out_dir : string or Path, optional
            Output directory.
        out_stats : string, optional
            Name of the CSV file with the statistics of the tested cells.
        out_blocks : string, optional
            Name of the CSV file with the Stage 1 blocks.
        out_summary : string, optional
            Name of the CSV file with the summary of each grid.

        References
        ----------
        .. footbibliography::

        """
        if not have_pd:
            raise ImportError("dipy_spectra_hfdr_external requires pandas")

        table = pd.read_csv(stats_file)
        missing = {"grid_name", "s_idx", "r_idx", "t_stat", "p_value"} - set(
            table.columns
        )
        if missing:
            raise ValueError(f"Columns not found in {stats_file}: {sorted(missing)}")
        grids = _spectra_grids(
            model_bundle_files, segment_length, radial_length, robust_method
        )
        unknown = sorted(set(table["grid_name"].astype(str)) - set(grids))
        if unknown:
            raise ValueError(f"No model bundle found for {unknown}")

        bundles = {}
        for name, rows in table.groupby(table["grid_name"].astype(str)):
            grid = grids[name]
            s_idx = rows["s_idx"].to_numpy(dtype=int)
            r_idx = rows["r_idx"].to_numpy(dtype=int)
            if s_idx.max() >= grid.n_segments or r_idx.max() >= grid.n_radial:
                raise ValueError(f"The cells of {name} are outside of its grid")

            shape = (grid.n_segments, 1 if r_idx.max() == 0 else grid.n_radial)
            columns = {}
            for column in ("t_stat", "p_value", "beta", "se", "dof", "d_cohen"):
                if column in rows:
                    columns[column] = np.full(shape, np.nan)
                    columns[column][s_idx, r_idx] = rows[column].to_numpy(float)
            mask = ~np.isnan(columns["p_value"])
            p = columns["p_value"]
            if np.any((p[mask] <= 0) | (p[mask] > 1)):
                logger.warning(f"{name}: some p-values are outside of (0, 1]")
            # As in SPECTRA, the z-scores use p-values bounded to stay finite
            p_bounded = np.clip(p, 1e-15, 1 - 1e-15)
            z = np.sign(columns["t_stat"]) * stats.norm.ppf(1 - p_bounded / 2)

            l_s = l_r = None
            residual_file = Path(residuals_folder or "") / f"{name}_residuals.csv"
            if not no_resel_blocks and residuals_folder and residual_file.exists():
                residuals = pd.read_csv(residual_file)
                subjects = sorted(residuals["subject_id"].unique())
                subject_index = residuals["subject_id"].map(
                    {subject: i for i, subject in enumerate(subjects)}
                )
                resid = np.full(shape + (len(subjects),), np.nan)
                resid[
                    residuals["s_idx"].to_numpy(int),
                    residuals["r_idx"].to_numpy(int),
                    subject_index.to_numpy(int),
                ] = residuals["residual"].to_numpy(float)
                try:
                    l_s, l_r = estimate_length_scales(resid, grid)
                except ValueError as error:
                    logger.warning(f"{name}: length scales not estimated ({error})")
            elif not no_resel_blocks:
                logger.warning(f"{name}: no residuals, using a single block")

            bundles[name] = {
                "grid": grid,
                "t": columns["t_stat"],
                "p": p,
                "z": np.clip(z, -10, 10),
                "l_s": l_s,
                "l_r": l_r,
                "mask": mask,
                "columns": columns,
                "n_subjects": None,
            }

        _save_spectra_results(
            bundles,
            stats_mode="hfdr_bundle" if no_resel_blocks else "hfdr_resel",
            alpha=None,
            adaptive=False,
            alpha_block=alpha_block,
            alpha_cell=alpha_cell,
            combination_method=combination_method,
            out_dir=out_dir,
            out_stats=out_stats,
            out_blocks=out_blocks,
            out_summary=out_summary,
        )
