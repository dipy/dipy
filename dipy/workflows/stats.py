import json
import os
from pathlib import Path
from time import time
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.ndimage import binary_dilation
from scipy.stats import norm

from dipy.core.gradients import gradient_table
from dipy.io import read_bvals_bvecs
from dipy.io.image import load_nifti, save_nifti
from dipy.io.peaks import load_peaks
from dipy.io.streamline import load_tractogram
from dipy.reconst.dti import TensorModel
from dipy.segment.bundles import bundle_shape_similarity
from dipy.segment.mask import bounding_box, segment_from_cfa
from dipy.stats.analysis import anatomical_measures, assignment_map, peak_values
from dipy.stats.fosr import fosr, get_covariates
from dipy.testing.decorators import warning_for_keywords
from dipy.tracking.streamline import transform_streamlines
from dipy.utils.logging import logger
from dipy.utils.optpkg import optional_package
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

    References
    ----------
    .. footbibliography::

    """

    t = time()

    mb = list(Path(model_bundle_folder).glob("*.trk"))
    logger.info(mb)

    mb.sort()

    bd = list(Path(bundle_folder).glob("*.trk"))

    bd.sort()
    logger.info(bd)
    org_bd = list(Path(orig_bundle_folder).glob("*.trk"))
    org_bd.sort()
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
            indx = assignment_map(bundles, mbundles, no_disks)
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
                metric = load_peaks(metric_files_names_csa[mn])

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
    def run(self, model_bundle_folder, subject_folder, *, no_disks=100, out_dir=""):
        """Workflow of bundle analytics.

        Applies statistical analysis on bundles of subjects and saves the
        results in a directory specified by ``out_dir``.

        See :footcite:p:`Chandio2020a` for further details about the method.

        Parameters
        ----------
        model_bundle_folder : string or Path
            Path to the input model bundle files. This path may
            contain wildcards to process multiple inputs at once.
        subject_folder : string or Path
            Path to the input subject folder. This path may contain
            wildcards to process multiple inputs at once.
        no_disks : integer, optional
            Number of disks used for dividing bundle into disks.
        out_dir : string or Path, optional
            Output directory.

        References
        ----------
        .. footbibliography::

        """

        if not Path(subject_folder).is_dir():
            raise ValueError("Invalid path to subjects")

        groups = [p.name for p in Path(subject_folder).iterdir()]
        groups.sort()
        for group in groups:
            group_dirname = Path(subject_folder) / group
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

                pvalues[i] = mdf.pvalues[1]

            x = list(range(1, len(pvalues) + 1))
            y = -1 * np.log10(pvalues)

            save_file = Path(out_dir) / (save_name + "_pvalues.npy")
            np.save(save_file, pvalues)

            save_file = Path(out_dir) / (save_name + "_pvalues_log.npy")
            np.save(save_file, y)

            save_file = Path(out_dir) / (save_name + ".png")
            self.save_lmm_plot(save_file, file_name, bundle_name, x, y)


class FOSRFlow(Workflow):
    @classmethod
    def get_short_name(cls):
        return "fosr"

    @warning_for_keywords()
    def run(
        self, hd5_dir, *, no_disks=100, out_dir="", no_streamlines=12000, YMetric="fa"
    ):
        """Workflow of Functional on Scalar Regression models.

        Applies functional on Scalar Regression models on bundles of subjects.

        Parameters
        ----------
        hd5_dir : string
            Path to the input metric files directory containing HDF5 files.
        no_disks : integer, optional
            Number of disks used for dividing bundle into disks.
        out_dir : string, optional
            Output directory. Give full path.
        no_streamlines : integer, optional
            Number of streamlines to use for analysis.
        YMetric : string, optional
            Metric to use for Y variable (e.g., "fa" for fractional anisotropy).
        """
        h5_list = os.listdir(hd5_dir)
        std_error_list = []
        save_dir_beta = os.path.join(out_dir, "beta_values")
        os.makedirs(save_dir_beta, exist_ok=True)
        save_dir_p_value = os.path.join(out_dir, "p_values")
        os.makedirs(save_dir_p_value, exist_ok=True)
        for hd_file in h5_list:
            print("Running fosr for ", hd_file)
            df = pd.read_hdf(os.path.join(hd5_dir, hd_file))
            X, Y = get_covariates(df, no_streamlines, YMetric)

            print("Getting the shape of X", X.shape)
            print("Getting the shape of Y", Y.shape)

            fosr_output = fosr(Y=Y, X=X)

            beta_1 = fosr_output["est.func"][:, 0]
            std_error = fosr_output["se.func"][:, 0]
            std_error_list.append(np.array(std_error))

            beta_1_lower = beta_1 - 1.96 * std_error
            beta_1_upper = beta_1 + 1.96 * std_error

            z_scores = beta_1 / std_error

            p_values = norm.sf(abs(z_scores)) * 2
            # Plot beta values
            plt.figure()
            plt.plot(beta_1_lower, label="lower bound")
            plt.plot(beta_1, label="avg value")
            plt.plot(beta_1_upper, label="upper value")
            plt.axhline(y=0, color="r", linestyle="--")
            plt.legend()
            plt.title(f"{hd_file} - Beta Values")
            plt.savefig(os.path.join(save_dir_beta, f"{hd_file}_beta.png"))

            # Plot p-values
            plt.figure()
            plt.plot(-1 * np.log10(p_values))
            plt.axhline(y=0.01, color="r", linestyle="--")
            plt.axhline(y=0.001, color="b", linestyle="--")
            plt.title(f"{hd_file} -log10(p-values)")
            plt.savefig(os.path.join(save_dir_p_value, f"{hd_file}_p_value.png"))


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
            i = 0
            logger.info(bun)
            for sub in all_subjects:
                j = 0

                bundle1 = load_tractogram(
                    Path(sub) / "rec_bundles" / bun,
                    reference="same",
                    bbox_valid_check=False,
                ).streamlines

                for subi in all_subjects:
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

                    j += 1
                i += 1
            logger.info("saving BA score matrix")
            np.save(Path(out_dir) / (bun[:-4] + ".npy"), ba_matrix)

            cmap = matplt.colormaps["Blues"]
            plt.title(bun[:-4])
            plt.imshow(ba_matrix, cmap=cmap)
            plt.colorbar()
            plt.clim(0, 1)
            plt.savefig(Path(out_dir) / f"SM_{bun[:-4]}")
            plt.clf()
