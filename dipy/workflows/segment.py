from __future__ import division, print_function, absolute_import

import logging

from dipy.workflows.workflow import Workflow
from dipy.io.image import save_nifti, load_nifti
import numpy as np
from time import time
from dipy.segment.mask import median_otsu
from dipy.workflows.align import load_trk, save_trk
from dipy.segment.bundles import RecoBundles
from dipy.tracking.streamline import transform_streamlines
from dipy.io.pickles import save_pickle, load_pickle

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
        save_masked : bool
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
            1D array representing indices of ``axis=3`` of a 4D `input_volume`
            'None' (the default) corresponds to ``(0,)`` (assumes first volume
            in 4D array). From cmd line use 3 4 5 6. From script use
            [3, 4, 5, 6].
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

            masked_volume, mask_volume = median_otsu(data, median_radius,
                                                     numpass, autocrop,
                                                     vol_idx, dilate)

            save_nifti(mask_out_path, mask_volume.astype(np.float32), affine)

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
            no_slr=False, clust_thr=15.,
            reduction_thr=10., reduction_distance='mdf',
            model_clust_thr=5.,
            pruning_thr=5., pruning_distance='mdf',
            slr_metric='symmetric',
            slr_transform='similarity',
            slr_matrix='small',
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
        no_slr : boolean, optional
            Enable local Streamline-based Linear Registration (default False).
        clust_thr : float, optional
            MDF distance threshold for all streamlines (default 15)
        reduction_thr : float, optional
            Reduce search space by (mm) (default 10)
        reduction_distance : string, optional
            Reduction distance type can be mdf or mam (default mdf)
        model_clust_thr : float, optional
            MDF distance threshold for the model bundles (default 5)
        pruning_thr : float, optional
            Pruning after matching (default 5).
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

        for sf, mb, out_rec, out_labels in io_it:

            t = time()
            logging.info(sf)
            streamlines, header = load_trk(sf)
            #streamlines = trkfile.streamlines
            logging.info(' Loading time %0.3f sec' % (time() - t,))

            rb = RecoBundles(streamlines)

            t = time()
            logging.info(mb)
            model_bundle, _ = load_trk(mb)
            logging.info(' Loading time %0.3f sec' % (time() - t,))

            recognized_bundle, labels, original_recognized_bundle = \
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

            save_trk(out_rec, recognized_bundle, np.eye(4))

            logging.info('Saving output files ...')
            np.save(out_labels, np.array(labels))
            logging.info(out_rec)
            logging.info(out_labels)


class LabelsBundlesFlow(Workflow):
    @classmethod
    def get_short_name(cls):
        return 'labbundles'

    def run(self, streamline_files, labels_files,
            out_dir='',
            out_bundle='recognized_orig.trk'):
        """ Recognize bundles

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

        io_it = self.get_io_iterator()
        for sf, lb, out_rfile in io_it:

            streamlines, header = load_trk(sf)
            location = np.load(lb)

            save_trk(out_bundle, streamlines[location], np.eye(4))
