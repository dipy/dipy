
import logging
from dipy.workflows.workflow import Workflow
from dipy.io.image import save_nifti, load_nifti
import nibabel as nib
import numpy as np
from time import time
from dipy.segment.mask import median_otsu
from dipy.segment.bundles import RecoBundles


class MedianOtsuFlow(Workflow):
    @classmethod
    def get_short_name(cls):
        return 'medotsu'

    def run(self, input_files, save_masked=False, median_radius=2, numpass=5,
            autocrop=False, vol_idx=None, dilate=None, out_dir='',
            out_mask='brain_mask.nii.gz', out_masked='dwi_masked.nii.gz'):
        """Workflow wrapping the median_otsu segmentation method.

        Applies median_otsu segmentation on each file found by 'globing'
        ``input_files`` and saves the results in a directory specified by
        ``out_dir``.

        Parameters
        ----------
        input_files : string
            Path to the input volumes. This path may contain wildcards to
            process multiple inputs at once.
        save_masked : bool, optional
            Save mask
        median_radius : int, optional
            Radius (in voxels) of the applied median filter (default 2)
        numpass : int, optional
            Number of pass of the median filter (default 5)
        autocrop : bool, optional
            If True, the masked input_volumes will also be cropped using the
            bounding box defined by the masked data. For example, if diffusion
            images are of 1x1x1 (mm^3) or higher resolution auto-cropping could
            reduce their size in memory and speed up some of the analysis.
            (default False)
        vol_idx : variable int, optional
            1D array representing indices of ``axis=-1`` of a 4D
            `input_volume`. From the command line use something like
            `3 4 5 6`. From script use something like `[3, 4, 5, 6]`. This
            input is required for 4D volumes.
        dilate : int, optional
            number of iterations for binary dilation (default 'None')
        out_dir : string, optional
            Output directory (default input file directory)
        out_mask : string, optional
            Name of the mask volume to be saved (default 'brain_mask.nii.gz')
        out_masked : string, optional
            Name of the masked volume to be saved (default 'dwi_masked.nii.gz')
        """
        io_it = self.get_io_iterator()
        if vol_idx is not None:
            vol_idx = map(int, vol_idx)

        for fpath, mask_out_path, masked_out_path in io_it:
            logging.info('Applying median_otsu segmentation on {0}'.
                         format(fpath))

            data, affine, img = load_nifti(fpath, return_img=True)

            masked_volume, mask_volume = median_otsu(
                data,
                vol_idx=vol_idx,
                median_radius=median_radius,
                numpass=numpass,
                autocrop=autocrop, dilate=dilate)

            save_nifti(mask_out_path, mask_volume.astype(np.float64), affine)

            logging.info('Mask saved as {0}'.format(mask_out_path))

            if save_masked:
                save_nifti(masked_out_path, masked_volume, affine,
                           img.header)

                logging.info('Masked volume saved as {0}'.
                             format(masked_out_path))

        return io_it


class RecoBundlesFlow(Workflow):
    @classmethod
    def get_short_name(cls):
        return 'recobundles'

    def run(self, streamline_files, model_bundle_files,
            greater_than=50, less_than=1000000,
            no_slr=False, clust_thr=15.,
            reduction_thr=15.,
            reduction_distance='mdf',
            model_clust_thr=2.5,
            pruning_thr=8.,
            pruning_distance='mdf',
            slr_metric='symmetric',
            slr_transform='similarity',
            slr_matrix='small',
            refine=False, r_reduction_thr=12.,
            r_pruning_thr=6., no_r_slr=False,
            out_dir='',
            out_recognized_transf='recognized.trk',
            out_recognized_labels='labels.npy'):
        """ Recognize bundles

        Parameters
        ----------
        streamline_files : string
            The path of streamline files where you want to recognize bundles
        model_bundle_files : string
            The path of model bundle files
        greater_than : int, optional
            Keep streamlines that have length greater than
            this value (default 50) in mm.
        less_than : int, optional
            Keep streamlines have length less than this value
            (default 1000000) in mm.
        no_slr : bool, optional
            Don't enable local Streamline-based Linear
            Registration (default False).
        clust_thr : float, optional
            MDF distance threshold for all streamlines (default 15)
        reduction_thr : float, optional
            Reduce search space by (mm) (default 15)
        reduction_distance : string, optional
            Reduction distance type can be mdf or mam (default mdf)
        model_clust_thr : float, optional
            MDF distance threshold for the model bundles (default 2.5)
        pruning_thr : float, optional
            Pruning after matching (default 8).
        pruning_distance : string, optional
            Pruning distance type can be mdf or mam (default mdf)
        slr_metric : string, optional
            Options are None, symmetric, asymmetric or diagonal
            (default symmetric).
        slr_transform : string, optional
            Transformation allowed. translation, rigid, similarity or scaling
            (Default 'similarity').
        slr_matrix : string, optional
            Options are 'nano', 'tiny', 'small', 'medium', 'large', 'huge'
            (default 'small')
        refine : bool, optional
            Enable refine recognized bunle (default False)
        r_reduction_thr : float, optional
            Refine reduce search space by (mm) (default 12)
        r_pruning_thr : float, optional
            Refine pruning after matching (default 6).
        no_r_slr : bool, optional
            Don't enable Refine local Streamline-based Linear
            Registration (default False).
        out_dir : string, optional
            Output directory (default input file directory)
        out_recognized_transf : string, optional
            Recognized bundle in the space of the model bundle
            (default 'recognized.trk')
        out_recognized_labels : string, optional
            Indices of recognized bundle in the original tractogram
            (default 'labels.npy')

        References
        ----------
        .. [Garyfallidis17] Garyfallidis et al. Recognition of white matter
         bundles using local and global streamline-based registration and
         clustering, Neuroimage, 2017.
        """
        slr = not no_slr
        r_slr = not no_r_slr

        bounds = [(-30, 30), (-30, 30), (-30, 30),
                  (-45, 45), (-45, 45), (-45, 45),
                  (0.8, 1.2), (0.8, 1.2), (0.8, 1.2)]

        slr_matrix = slr_matrix.lower()
        if slr_matrix == 'nano':
            slr_select = (100, 100)
        if slr_matrix == 'tiny':
            slr_select = (250, 250)
        if slr_matrix == 'small':
            slr_select = (400, 400)
        if slr_matrix == 'medium':
            slr_select = (600, 600)
        if slr_matrix == 'large':
            slr_select = (800, 800)
        if slr_matrix == 'huge':
            slr_select = (1200, 1200)

        slr_transform = slr_transform.lower()
        if slr_transform == 'translation':
            bounds = bounds[:3]
        if slr_transform == 'rigid':
            bounds = bounds[:6]
        if slr_transform == 'similarity':
            bounds = bounds[:7]
        if slr_transform == 'scaling':
            bounds = bounds[:9]

        logging.info('### RecoBundles ###')

        io_it = self.get_io_iterator()

        t = time()
        logging.info(streamline_files)
        input_obj = nib.streamlines.load(streamline_files)
        streamlines = input_obj.streamlines

        logging.info(' Loading time %0.3f sec' % (time() - t,))

        rb = RecoBundles(streamlines, greater_than=greater_than,
                         less_than=less_than)

        for _, mb, out_rec, out_labels in io_it:
            t = time()
            logging.info(mb)
            model_bundle = nib.streamlines.load(mb).streamlines
            logging.info(' Loading time %0.3f sec' % (time() - t,))
            logging.info("model file = ")
            logging.info(mb)

            recognized_bundle, labels = \
                rb.recognize(
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
                    slr_method='L-BFGS-B')

            if refine:

                if len(recognized_bundle) > 1:

                    # affine
                    x0 = np.array([0, 0, 0, 0, 0, 0, 1., 1., 1, 0, 0, 0])
                    affine_bounds = [(-30, 30), (-30, 30), (-30, 30),
                                     (-45, 45), (-45, 45), (-45, 45),
                                     (0.8, 1.2), (0.8, 1.2), (0.8, 1.2),
                                     (-10, 10), (-10, 10), (-10, 10)]

                    recognized_bundle, labels = \
                        rb.refine(
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
                            slr_method='L-BFGS-B')

            if len(labels) > 0:
                ba, bmd = rb.evaluate_results(
                    model_bundle, recognized_bundle,
                    slr_select)

                logging.info("Bundle adjacency Metric {0}".format(ba))
                logging.info("Bundle Min Distance Metric {0}".format(bmd))

            new_tractogram = nib.streamlines.Tractogram(recognized_bundle,
                                                        affine_to_rasmm=np.eye(4))
            nib.streamlines.save(new_tractogram, out_rec,
                                 header=input_obj.header)
            logging.info('Saving output files ...')
            np.save(out_labels, np.array(labels))
            logging.info(out_rec)
            logging.info(out_labels)


class LabelsBundlesFlow(Workflow):
    @classmethod
    def get_short_name(cls):
        return 'labelsbundles'

    def run(self, streamline_files, labels_files,
            out_dir='',
            out_bundle='recognized_orig.trk'):
        """ Extract bundles using existing indices (labels)

        Parameters
        ----------
        streamline_files : string
            The path of streamline files where you want to recognize bundles
        labels_files : string
            The path of model bundle files
        out_dir : string, optional
            Output directory (default input file directory)
        out_bundle : string, optional
            Recognized bundle in the space of the model bundle
            (default 'recognized_orig.trk')

        References
        ----------
        .. [Garyfallidis17] Garyfallidis et al. Recognition of white matter
         bundles using local and global streamline-based registration and
         clustering, Neuroimage, 2017.

        """
        logging.info('### Labels to Bundles ###')

        io_it = self.get_io_iterator()
        for sf, lb, out_bundle in io_it:

            logging.info(sf)
            tractogram_obj = nib.streamlines.load(sf)
            streamlines = tractogram_obj.streamlines

            logging.info(lb)
            location = np.load(lb)
            logging.info('Saving output files ...')
            new_tractogram = nib.streamlines.Tractogram(streamlines[location],
                                                        affine_to_rasmm=np.eye(4))
            nib.streamlines.save(new_tractogram, out_bundle,
                                 header=tractogram_obj.header)
            logging.info(out_bundle)
