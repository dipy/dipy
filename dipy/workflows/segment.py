import csv
from pathlib import Path
import sys
from time import time

import numpy as np

from dipy.io.gradients import read_bvals_bvecs
from dipy.io.image import load_nifti, save_nifti
from dipy.io.stateful_tractogram import Space, StatefulTractogram
from dipy.io.streamline import load_tractogram, save_tractogram
from dipy.nn.synthseg import SynthSeg
from dipy.segment.bundles import RecoBundles
from dipy.segment.mask import median_otsu
from dipy.segment.tissue import TissueClassifierHMRF, dam_classifier
from dipy.tracking import Streamlines
from dipy.utils.deprecator import deprecated_params
from dipy.utils.logging import logger
from dipy.workflows.utils import handle_vol_idx
from dipy.workflows.workflow import Workflow


class MedianOtsuFlow(Workflow):
    @classmethod
    def get_short_name(cls):
        return "medotsu"

    @deprecated_params(["autocrop"], since="1.11.0", until="1.13.0")
    def run(
        self,
        input_files,
        bvalues_files=None,
        b0_threshold=50,
        save_masked=False,
        median_radius=2,
        numpass=5,
        autocrop=False,
        vol_idx=None,
        dilate=None,
        finalize_mask=False,
        out_dir="",
        out_mask="brain_mask.nii.gz",
        out_masked="dwi_masked.nii.gz",
    ):
        """Workflow wrapping the median_otsu segmentation method.

        Applies median_otsu segmentation on each file found by 'globing'
        ``input_files`` and saves the results in a directory specified by
        ``out_dir``.

        Parameters
        ----------
        input_files : string or Path
            Path to the input volumes. This path may contain wildcards to
            process multiple inputs at once.
        bvalues_files : variable string or Path, optional
            Path to the b-values files. This path may contain wildcards to
            process multiple inputs at once.
        b0_threshold : float, optional
            Threshold to consider a volume as b0. Used only if bvalues_files
            is provided.
        save_masked : bool, optional
            Save mask.
        median_radius : int, optional
            Radius (in voxels) of the applied median filter.
        numpass : int, optional
            Number of pass of the median filter.
        autocrop : bool, optional
            .. deprecated:: 1.11.0
               This parameter is deprecated and will be removed in 1.13.0.

            If True, the mask is cropped to the bounding box of the
            non-zero elements. This is useful for large 3D volumes
            (e.g. 4D volumes) where the mask is not the same size as
            the input volume.
        vol_idx : str, optional
            1D array representing indices of ``axis=-1`` of a 4D
            `input_volume`. From the command line use something like
            '1,2,3-5,7'. This input is required for 4D volumes if bval files
            are not provided. If bval files are provided, vol_idx is
            ignored and b0 volumes are used for mask computation.
        dilate : int, optional
            number of iterations for binary dilation.
        finalize_mask : bool, optional
            Whether to remove potential holes or islands.
            Useful for solving minor errors.
        out_dir : string or Path, optional
            Output directory.
        out_mask : string, optional
            Name of the mask volume to be saved.
        out_masked : string, optional
            Name of the masked volume to be saved.
        """
        io_it = self.get_io_iterator()

        if vol_idx is not None and bvalues_files is not None and io_it:
            logger.warning(
                "'vol_idx' parameter is ignored when 'bvalues_files' is provided."
            )

        if bvalues_files is not None and not isinstance(bvalues_files, list):
            bvalues_files = [bvalues_files]

        if len(bvalues_files or []) > 0 and io_it:
            if len(bvalues_files) != len(io_it.inputs):
                logger.error(
                    "Number of b-values files must match the number of "
                    "input volumes."
                )
                sys.exit(1)

        vol_idx = handle_vol_idx(vol_idx)

        bvals_counter = 0
        for fpath, mask_out_path, masked_out_path in io_it:
            logger.info(f"Applying median_otsu segmentation on {fpath}")

            data, affine, img = load_nifti(fpath, return_img=True)
            vol_idx_used = vol_idx
            if bvalues_files is not None:
                bvals, _ = read_bvals_bvecs(bvalues_files[bvals_counter], None)
                bvals_counter += 1
                b0_indices = np.where(bvals <= b0_threshold)[0]
                vol_idx_used = b0_indices

            logger.debug(f"vol_idx_used: {vol_idx_used}")
            extra_args = {} if not autocrop else {"autocrop": autocrop}
            masked_volume, mask_volume = median_otsu(
                data,
                vol_idx=vol_idx_used,
                median_radius=median_radius,
                numpass=numpass,
                dilate=dilate,
                finalize_mask=finalize_mask,
                **extra_args,
            )

            save_nifti(mask_out_path, mask_volume.astype(np.float64), affine)

            logger.info(f"Mask saved as {mask_out_path}")

            if save_masked:
                save_nifti(masked_out_path, masked_volume, affine, hdr=img.header)

                logger.info(f"Masked volume saved as {masked_out_path}")

        return io_it


class RecoBundlesFlow(Workflow):
    @classmethod
    def get_short_name(cls):
        return "recobundles"

    def run(
        self,
        streamline_files,
        model_bundle_files,
        greater_than=50,
        less_than=1000000,
        no_slr=False,
        clust_thr=15.0,
        reduction_thr=15.0,
        reduction_distance="mdf",
        model_clust_thr=2.5,
        pruning_thr=8.0,
        pruning_distance="mdf",
        slr_metric="symmetric",
        slr_transform="similarity",
        slr_matrix="small",
        refine=False,
        r_reduction_thr=12.0,
        r_pruning_thr=6.0,
        no_r_slr=False,
        out_dir="",
        out_recognized_transf="recognized.trx",
        out_recognized_labels="labels.npy",
    ):
        """Recognize bundles

        See :footcite:p:`Garyfallidis2018` and :footcite:p:`Chandio2020a` for
        further details about the method.

        Parameters
        ----------
        streamline_files : string or Path
            The path of streamline files where you want to recognize bundles.
        model_bundle_files : string or Path
            The path of model bundle files.
        greater_than : int, optional
            Keep streamlines that have length greater than
            this value in mm.
        less_than : int, optional
            Keep streamlines have length less than this value
            in mm.
        no_slr : bool, optional
            Don't enable local Streamline-based Linear
            Registration.
        clust_thr : float, optional
            MDF distance threshold for all streamlines.
        reduction_thr : float, optional
            Reduce search space by (mm).
        reduction_distance : string, optional
            Reduction distance type can be mdf or mam.
        model_clust_thr : float, optional
            MDF distance threshold for the model bundles.
        pruning_thr : float, optional
            Pruning after matching.
        pruning_distance : string, optional
            Pruning distance type can be mdf or mam.
        slr_metric : string, optional
            Options are None, symmetric, asymmetric or diagonal.
        slr_transform : string, optional
            Transformation allowed. translation, rigid, similarity or scaling.
        slr_matrix : string, optional
            Options are 'nano', 'tiny', 'small', 'medium', 'large', 'huge'.
        refine : bool, optional
            Enable refine recognized bundle.
        r_reduction_thr : float, optional
            Refine reduce search space by (mm).
        r_pruning_thr : float, optional
            Refine pruning after matching.
        no_r_slr : bool, optional
            Don't enable Refine local Streamline-based Linear
            Registration.
        out_dir : string or Path, optional
            Output directory.
        out_recognized_transf : string, optional
            Recognized bundle in the space of the model bundle.
        out_recognized_labels : string, optional
            Indices of recognized bundle in the original tractogram.

        References
        ----------
        .. footbibliography::

        """
        slr = not no_slr
        r_slr = not no_r_slr

        bounds = [
            (-30, 30),
            (-30, 30),
            (-30, 30),
            (-45, 45),
            (-45, 45),
            (-45, 45),
            (0.8, 1.2),
            (0.8, 1.2),
            (0.8, 1.2),
        ]

        slr_matrix = slr_matrix.lower()
        if slr_matrix == "nano":
            slr_select = (100, 100)
        if slr_matrix == "tiny":
            slr_select = (250, 250)
        if slr_matrix == "small":
            slr_select = (400, 400)
        if slr_matrix == "medium":
            slr_select = (600, 600)
        if slr_matrix == "large":
            slr_select = (800, 800)
        if slr_matrix == "huge":
            slr_select = (1200, 1200)

        slr_transform = slr_transform.lower()
        if slr_transform == "translation":
            bounds = bounds[:3]
        if slr_transform == "rigid":
            bounds = bounds[:6]
        if slr_transform == "similarity":
            bounds = bounds[:7]
        if slr_transform == "scaling":
            bounds = bounds[:9]

        logger.info("### RecoBundles ###")

        io_it = self.get_io_iterator()

        t = time()
        logger.info(streamline_files)
        input_obj = load_tractogram(streamline_files, "same", bbox_valid_check=False)
        streamlines = input_obj.streamlines

        logger.info(f" Loading time {time() - t:0.3f} sec")

        rb = RecoBundles(streamlines, greater_than=greater_than, less_than=less_than)

        for _, mb, out_rec, out_labels in io_it:
            t = time()
            logger.info(mb)
            model_bundle = load_tractogram(
                mb, "same", bbox_valid_check=False
            ).streamlines
            logger.info(f" Loading time {time() - t:0.3f} sec")
            logger.info("model file = ")
            logger.info(mb)

            recognized_bundle, labels = rb.recognize(
                model_bundle,
                model_clust_thr=model_clust_thr,
                reduction_thr=reduction_thr,
                reduction_distance=reduction_distance,
                pruning_thr=pruning_thr,
                pruning_distance=pruning_distance,
                slr=slr,
                slr_metric=slr_metric,
                slr_x0=slr_transform,
                slr_bounds=bounds,
                slr_select=slr_select,
                slr_method="L-BFGS-B",
            )

            if refine:
                if len(recognized_bundle) > 1:
                    # affine
                    x0 = np.array([0, 0, 0, 0, 0, 0, 1.0, 1.0, 1, 0, 0, 0])
                    affine_bounds = [
                        (-30, 30),
                        (-30, 30),
                        (-30, 30),
                        (-45, 45),
                        (-45, 45),
                        (-45, 45),
                        (0.8, 1.2),
                        (0.8, 1.2),
                        (0.8, 1.2),
                        (-10, 10),
                        (-10, 10),
                        (-10, 10),
                    ]

                    recognized_bundle, labels = rb.refine(
                        model_bundle,
                        recognized_bundle,
                        model_clust_thr=model_clust_thr,
                        reduction_thr=r_reduction_thr,
                        reduction_distance=reduction_distance,
                        pruning_thr=r_pruning_thr,
                        pruning_distance=pruning_distance,
                        slr=r_slr,
                        slr_metric=slr_metric,
                        slr_x0=x0,
                        slr_bounds=affine_bounds,
                        slr_select=slr_select,
                        slr_method="L-BFGS-B",
                    )

            if len(labels) > 0:
                ba, bmd = rb.evaluate_results(
                    model_bundle, recognized_bundle, slr_select
                )

                logger.info(f"Bundle adjacency Metric {ba}")
                logger.info(f"Bundle Min Distance Metric {bmd}")

            new_tractogram = StatefulTractogram(
                recognized_bundle, streamline_files, Space.RASMM
            )
            save_tractogram(new_tractogram, out_rec, bbox_valid_check=False)
            logger.info("Saving output files ...")
            np.save(out_labels, np.array(labels))
            logger.info(out_rec)
            logger.info(out_labels)


class LabelsBundlesFlow(Workflow):
    @classmethod
    def get_short_name(cls):
        return "labelsbundles"

    def run(
        self,
        streamline_files,
        labels_files,
        out_dir="",
        out_bundle="recognized_orig.trx",
    ):
        """Extract bundles using existing indices (labels)

        See :footcite:p:`Garyfallidis2018` for further details about the method.

        Parameters
        ----------
        streamline_files : string or Path
            The path of streamline files where you want to recognize bundles.
        labels_files : string or Path
            The path of model bundle files.
        out_dir : string or Path, optional
            Output directory.
        out_bundle : string, optional
            Recognized bundle in the space of the model bundle.

        References
        ----------
        .. footbibliography::

        """
        logger.info("### Labels to Bundles ###")

        io_it = self.get_io_iterator()
        for f_steamlines, f_labels, out_bundle in io_it:
            logger.info(f_steamlines)
            sft = load_tractogram(f_steamlines, "same", bbox_valid_check=False)
            streamlines = sft.streamlines

            logger.info(f_labels)
            location = np.load(f_labels)
            if len(location) < 1:
                bundle = Streamlines([])
            else:
                bundle = streamlines[location]

            logger.info("Saving output files ...")
            new_sft = StatefulTractogram(bundle, sft, Space.RASMM)
            save_tractogram(new_sft, out_bundle, bbox_valid_check=False)
            logger.info(out_bundle)


class ClassifyTissueFlow(Workflow):
    @classmethod
    def get_short_name(cls):
        return "extracttissue"

    def run(
        self,
        input_files,
        bvals_file=None,
        method=None,
        wm_threshold=0.5,
        b0_threshold=50,
        low_signal_threshold=50,
        nclass=None,
        beta=0.1,
        tolerance=1e-05,
        max_iter=100,
        out_dir="",
        out_tissue="tissue_classified.nii.gz",
        out_pve="tissue_classified_pve.nii.gz",
        out_csv="tissue_classified.csv",
    ):
        """Extract tissue from a volume.

        Parameters
        ----------
        input_files : string or Path
            Path to the input volumes. This path may contain wildcards to
            process multiple inputs at once.
        bvals_file : string or Path, optional
            Path to the b-values file. Required for 'dam' method.
        method : string, optional
            Method to use for tissue extraction. Options are:

            - 'hmrf': Markov Random Fields modeling approach.
            - 'dam': Directional Average Maps, proposed by
              :footcite:p:`Cheng2020`.
            - 'synthseg': SynthSeg method based on Freesurfer's implementation,
              proposed by :footcite:p:`Billot2023`.

            'hmrf' method is recommended for T1w images, while 'dam' method is
            recommended for DWI Multishell images (single shell are not recommended).
            'synthseg' works for all modalities, but requires more memory.
            No pve maps are generated for 'synthseg' method.
        wm_threshold : float, optional
            The threshold below which a voxel is considered white matter. For data like
            HCP, threshold of 0.5 proves to be a good choice. For data like cfin, higher
            threshold values like 0.7 or 0.8 are more suitable. Used for 'dam' method.
        b0_threshold : float, optional
            The intensity threshold for a b=0 image. used only for 'dam' method.
        low_signal_threshold : float, optional
            The threshold below which a voxel is considered to have low signal.
            Used only for 'dam' method.
        nclass : int, optional
            Number of desired classes. Used only for 'hmrf' method.
        beta : float, optional
            Smoothing parameter, the higher this number the smoother the
            output will be. Used only for 'hmrf' method.
        tolerance : float, optional
            Value that defines the percentage of change tolerated to
            prevent the ICM loop to stop. Default is 1e-05.
            If you want tolerance check to be disabled put 'tolerance = 0'.
            Used only for 'hmrf' method.
        max_iter : int, optional
            Fixed number of desired iterations. Default is 100.
            This parameter defines the maximum number of iterations the
            algorithm will perform. The loop may terminate early if the
            change in energy sum between iterations falls below the
            threshold defined by `tolerance`. However, if `tolerance` is
            explicitly set to 0, this early stopping mechanism is disabled,
            and the algorithm will run for the specified number of
            iterations unless another stopping criterion is met.
            Used only for 'hmrf' method.
        out_dir : string or Path, optional
            Output directory.
        out_tissue : string, optional
            Name of the tissue volume to be saved.
        out_pve : string, optional
            Name of the pve volume to be saved.
        out_csv : string, optional
            Name of the csv file containing label names to be saved.

        REFERENCES
        ----------
        .. footbibliography::

        """
        csv_fieldnames = ["Label", "Name"]
        io_it = self.get_io_iterator()

        if not method or method.lower() not in ["hmrf", "dam", "synthseg"]:
            logger.error(
                f"Unknown method '{method}' for tissue extraction. "
                "Choose '--method hmrf' (for T1w) or '--method dam' (for DWI)"
                " or '--method synthseg' (for all modalities)."
            )
            sys.exit(1)

        for fpath, tissue_out_path, opve, ocsv in io_it:
            logger.info(f"Extracting tissue from {fpath}")

            data, affine = load_nifti(fpath)
            class_list = []

            if method.lower() == "hmrf":
                if nclass is None:
                    logger.error(
                        "Number of classes is required for 'hmrf' method. "
                        "For example, Use '--nclass 4' to specify the number of "
                        "classes."
                    )
                    sys.exit(1)
                classifier = TissueClassifierHMRF()
                _, segmentation_final, PVE = classifier.classify(
                    data, nclass, beta, tolerance=tolerance, max_iter=max_iter
                )

                save_nifti(tissue_out_path, segmentation_final, affine)
                save_nifti(opve, PVE, affine)
                class_list.append(["0", "Background"])
                for i in range(1, nclass + 1):
                    class_list.append([f"{i}", f"Tissue_{i}"])
                class_list.append(
                    [
                        "# Due to the nature of the HMRF algorithm",
                        "tissues are not labeled specifically.",
                    ]
                )

            elif method.lower() == "dam":
                if bvals_file is None or not Path(bvals_file).is_file():
                    logger.error("'--bvals filename' is required for 'dam' method")
                    sys.exit(1)

                bvals, _ = read_bvals_bvecs(bvals_file, None)
                wm_mask, gm_mask = dam_classifier(
                    data,
                    bvals,
                    wm_threshold=wm_threshold,
                    b0_threshold=b0_threshold,
                    low_signal_threshold=low_signal_threshold,
                )
                result = np.zeros(wm_mask.shape, dtype=np.int32)
                result[wm_mask] = 1
                result[gm_mask] = 2
                save_nifti(tissue_out_path, result, affine)
                save_nifti(
                    opve, np.stack([wm_mask, gm_mask], axis=-1).astype(np.int32), affine
                )
                class_list.append(["0", "Background"])
                class_list.append(["1", "White_Matter"])
                class_list.append(["2", "Gray_Matter"])

            elif method.lower() == "synthseg":
                synthseg = SynthSeg()
                segmentation_final, label_dict, _ = synthseg.predict(data, affine)
                for label, name in label_dict.items():
                    class_list.append([f"{label}", name])

                save_nifti(tissue_out_path, segmentation_final.astype(np.int32), affine)

            with open(ocsv, "w", newline="") as csv_object:
                csv_writer = csv.writer(csv_object)
                csv_writer.writerow(csv_fieldnames)
                csv_writer.writerows(class_list)

            if not method.lower() == "synthseg":
                logger.info(f"Tissue saved as {tissue_out_path} and PVE as {opve}")
            else:
                logger.info(f"Tissue saved as {tissue_out_path}")
            logger.info(f"Label names saved as {ocsv}")

        return io_it
