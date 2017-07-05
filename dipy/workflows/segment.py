from __future__ import division, print_function, absolute_import

<<<<<<< HEAD
import logging

import numpy as np

from dipy.segment.mask import median_otsu
from dipy.workflows.workflow import Workflow
from dipy.io.image import save_nifti, load_nifti


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
        vol_idx : string, optional
            1D array representing indices of ``axis=3`` of a 4D `input_volume`
            'None' (the default) corresponds to ``(0,)`` (assumes first volume
            in 4D array)
        dilate : string, optional
            number of iterations for binary dilation (default 'None')
        out_dir : string, optional
            Output directory (default input file directory)
        out_mask : string, optional
            Name of the mask volume to be saved (default 'brain_mask.nii.gz')
        out_masked : string, optional
            Name of the masked volume to be saved (default 'dwi_masked.nii.gz')
        """
        io_it = self.get_io_iterator()

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
=======
from glob import glob
from os.path import join, basename, splitext

import nibabel as nib
import numpy as np
from time import time
from dipy.workflows.utils import choose_create_out_dir
from dipy.segment.mask import median_otsu
from dipy.workflows.align import load_trk, save_trk
import os
import numpy as np
from dipy.utils.six import string_types
from dipy.segment.bundles import RecoBundles, KDTreeBundles
from dipy.tracking.streamline import transform_streamlines
from dipy.io.pickles import save_pickle, load_pickle


#def median_otsu_flow(input_files, out_dir='', save_masked=False,
#                     median_radius=4, numpass=4, autocrop=False,
#                     vol_idx=None, dilate=None):
#    """ Workflow wrapping the median_otsu segmentation method.
#
#    It applies median_otsu segmentation on each file found by 'globing'
#    ``input_files`` and saves the results in a directory specified by
#    ``out_dir``.
#
#    Parameters
#    ----------
#    input_files : string
#        Path to the input volumes. This path may contain wildcards to process
#        multiple inputs at once.
#    out_dir : string, optional
#        Output directory (default input file directory)
#    save_masked : bool, optional
#        Save mask
#    median_radius : int, optional
#        Radius (in voxels) of the applied median filter(default 4)
#    numpass : int, optional
#        Number of pass of the median filter (default 4)
#    autocrop : bool, optional
#        If True, the masked input_volumes will also be cropped using the
#        bounding box defined by the masked data. Should be on if DWI is
#        upsampled to 1x1x1 resolution. (default False)
#    vol_idx : string, optional
#        1D array representing indices of ``axis=3`` of a 4D `input_volume`
#        'None' (the default) corresponds to ``(0,)`` (assumes first volume in
#        4D array)
#    dilate : string, optional
#        number of iterations for binary dilation (default 'None')
#    """
#    for fpath in glob(input_files):
#        print('')
#        print('Applying median_otsu segmentation on {0}'.format(fpath))
#        img = nib.load(fpath)
#        volume = img.get_data()
#
#        masked, mask = median_otsu(volume, median_radius,
#                                   numpass, autocrop,
#                                   vol_idx, dilate)
#
#        fname, ext = splitext(basename(fpath))
#        if(fname.endswith('.nii')):
#            fname, _ = splitext(fname)
#            ext = '.nii.gz'
#
#        mask_fname = fname + '_mask' + ext
#
#        out_dir_path = choose_create_out_dir(out_dir, fpath)
#
#        mask_img = nib.Nifti1Image(mask.astype(np.float32), img.get_affine())
#        mask_out_path = join(out_dir_path, mask_fname)
#        mask_img.to_filename(mask_out_path)
#        print('Mask saved as {0}'.format(mask_out_path))
#
#        if bool_param(save_masked):
#            masked_fname = fname + '_bet' + ext
#            masked_img = nib.Nifti1Image(masked,
#                                         img.get_affine(), img.get_header())
#            masked_out_path = join(out_dir_path, masked_fname)
#            masked_img.to_filename(masked_out_path)
#            print('Masked volume saved as {0}'.format(masked_out_path))


def recognize_bundles_flow(streamline_files, model_bundle_files,
                           out_dir=None, clust_thr=15.,
                           reduction_thr=10., reduction_distance='mdf',
                           model_clust_thr=5.,
                           pruning_thr=5., pruning_distance='mdf',
                           slr=True, slr_metric=None,
                           slr_transform='similarity', slr_progressive=True,
                           slr_matrix='small', verbose=True, debug=False):
    """ Recognize bundles

    Parameters
    ----------
    streamline_files : string
        The path of streamline files where you want to recognize bundles
    model_bundle_files : string
        The path of model bundle files
    out_dir : string, optional
        Directory to output the different files
    clust_thr : float, optional
        MDF distance threshold for all streamlines
    reduction_thr : float, optional
        Reduce search space by (mm) (default 20)
    reduction_distance : string, optional
        Reduction distance type can be mdf or mam (default mdf)
    model_clust_thr : float, optional
        MDF distance threshold for the model bundles (default 5)
    pruning_thr : float, optional
        Pruning after matching (default 5).
    pruning_distance : string, optional
        Pruning distance type can be mdf or mam (default mdf)
    slr : bool, optional
        Enable local Streamline-based Linear Registration (default True).
    slr_metric : string, optional
        Options are None, symmetric, asymmetric or diagonal (default None).
    slr_transform : string, optional
        Transformation allowed. translation, rigid, similarity or scaling
        (Default 'similarity').
    slr_progressive : bool, optional
        If for example you selected `rigid` in slr_transform then you will
        do first translation and then rigid (default True).
    slr_matrix : string, optional
        Options are 'nano', 'tiny', 'small', 'medium', 'large', 'huge' (default
        'small')
    verbose : bool, optional
        Enable standard output (defaut True).
    debug : bool, optional
        Write out intremediate results (default False)
    """

    if isinstance(streamline_files, string_types):
        sfiles = glob(streamline_files)
    else:
        raise ValueError('Streamline_files not a string')

    if isinstance(model_bundle_files, string_types):
        mbfiles = glob(model_bundle_files)
        if mbfiles == []:
            raise ValueError('Model_bundle_files is not correct')

    if out_dir is None:
        print('Results will be given in the same folder as input streamlines')

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

    print('### RecoBundles ###')

    # Streamline files where the recognition will take place
    for sf in sfiles:
        print('# Streamline file')
        print(sf)

        if not os.path.exists(sf):
            print('File {} does not exist'.format(sf))
            return

        base_sf = os.path.splitext(os.path.basename(sf))[0]

        t = time()
        trkfile = nib.streamlines.load(sf)
        streamlines = trkfile.streamlines
        print(' Loading time %0.3f sec' % (time() - t,))

        sf_clusters = os.path.join(
            os.path.dirname(sf),
            os.path.splitext(os.path.basename(sf))[0] + '_clusters.pkl')

        if os.path.exists(sf_clusters):
            clusters = load_pickle(sf_clusters)
            print(' Using pre-existing clustering file.')
            print(' To ignore file delete it at {} and rerun'
                  .format(sf_clusters))
        else:
            clusters = None

        rb = RecoBundles(streamlines, clusters,
                         clust_thr=clust_thr)

        if clusters is None:
            print('Clusters of streamlines saved in \n {} '
                  .format(sf_clusters))

            rb.cluster_map.refdata = None
            save_pickle(sf_clusters, rb.cluster_map)
            rb.cluster_map.refdata = rb.streamlines

        # Model bundle
        for mb in mbfiles:
            print('# Model_bundle file')
            print(mb)

            if not os.path.exists(mb):
                print('File {} does not exist'.format(mb))
                return

            t = time()
            model_trkfile = nib.streamlines.load(mb)
            model_bundle = model_trkfile.streamlines
            print(' Loading time %0.3f sec' % (time() - t,))

            recognized_bundle = rb.recognize(
                model_bundle,
                model_clust_thr=float(model_clust_thr),
                reduction_thr=float(reduction_thr),
                reduction_distance=reduction_distance,
                slr=slr,
                slr_metric=slr_metric,
                slr_x0=slr_transform,
                slr_bounds=bounds,
                slr_select=slr_select,
                slr_method='L-BFGS-B',
                slr_use_centroids=False,
                slr_progressive=slr_progressive,
                pruning_thr=float(pruning_thr),
                pruning_distance=pruning_distance)

    # TODO add option to return recognized bundle in the space that you want
    # Or better return the labels of the bundle which I currently do.
    #            extracted_bundle, mat2 = recognize_bundles(
    #                model_bundle, moved_streamlines,
    #                close_centroids_thr=close_centroids_thr,
    #                clean_thr=clean_thr,
    #                local_slr=local_slr,
    #                expand_thr=expand_thr,
    #                scale_range=scale_range,
    #                verbose=verbose,
    #                return_full=False)
    #
    #            extracted_bundle_initial = transform_streamlines(
    #                extracted_bundle,
    #                np.linalg.inv(np.dot(mat2, mat)))

            if out_dir is None:
                out_dir = ''

            base_mb = os.path.splitext(os.path.basename(mb))[0]

            sf_bundle_file = os.path.join(
                out_dir,
                base_mb + '_of_' + base_sf + '.trk')

            sf_bundle_file_initial = os.path.join(
                out_dir,
                base_mb + '_of_' + base_sf + '_initial.trk')

            sf_bundle_labels = os.path.join(
                out_dir,
                base_mb + '_of_' + base_sf + '_labels.npy')

            # if not os.path.exists(os.path.dirname(sf_bundle_file)):
            #     os.makedirs(os.path.dirname(sf_bundle_file))

            recognized_tractogram = nib.streamlines.Tractogram(
                recognized_bundle)
            recognized_trkfile = nib.streamlines.TrkFile(recognized_tractogram)
            nib.streamlines.save(recognized_trkfile, sf_bundle_file)

            np.save(sf_bundle_labels, np.array(rb.labels))

            print('# Output files')
            print('Recognized bundle saved in \n {} '
                  .format(sf_bundle_file))
            print('Recognized bundle labels saved in \n {} '
                  .format(sf_bundle_labels))
            print('Recognized bundle in initial saved in \n {} '
                  .format(sf_bundle_file_initial))


            if debug:
                sf_bundle_neighb = os.path.join(
                    out_dir,
                    base_mb + '_of_' + base_sf + '_neighb.trk')

                neighb_tractogram = nib.streamlines.Tractogram(
                    rb.neighb_streamlines)
                neighb_trkfile = nib.streamlines.TrkFile(neighb_tractogram)
                nib.streamlines.save(neighb_trkfile, sf_bundle_neighb)

                print('Recognized bundle\'s neighbors saved in \n {} '
                      .format(sf_bundle_neighb))

        if debug:
            sf_centroids = os.path.join(
                os.path.dirname(sf),
                base_sf + '_centroids.trk')

            centroid_tractogram = nib.streamlines.Tractogram(
                rb.centroids)
            centroid_trkfile = nib.streamlines.TrkFile(centroid_tractogram)
            nib.streamlines.save(centroid_trkfile, sf_centroids)

            print('Centroids of streamlines saved in \n {} '
                  .format(sf_centroids))


def assign_bundle_labels_flow(streamline_file, labels_files, verbose=True):
    """ Show recognized bundles in their original space

    Parameters
    ----------
    streamline_file : string
    labels_files : string
    verbose : bool, optional
        Print standard output (default True)
    """
    print(streamline_file)
    print(labels_files)

    if isinstance(labels_files, string_types):
        lfiles = glob(labels_files)

    streamlines_trk = nib.streamlines.load(streamline_file)
    streamlines = streamlines_trk.streamlines

    for lf in lfiles:

        labels = np.load(lf)
        recognized_bundle = streamlines[labels.tolist()]
        recognized_tractogram = nib.streamlines.Tractogram(recognized_bundle)
        recognized_trkfile = nib.streamlines.TrkFile(
            recognized_tractogram,
            header=streamlines_trk.header)
        base = os.path.splitext(os.path.basename(lf))[0].split('_labels')[0]
        fname = os.path.join(
            os.path.dirname(lf),
            base + '_of_' + os.path.basename(streamline_file))
        nib.streamlines.save(recognized_trkfile, fname)
        if verbose:
            print('Bundle saved in \n {} '.format(fname))


def kdtrees_bundles_flow(streamline_file, labels_file,
                         metric='mdf',
                         verbose=True):
    """ Expand reduce bundles using kdtrees

    Parameters
    ----------
    streamline_file : string
    labels_file : string
    metric : string
    verbose : bool, optional
        Print standard output (default True)
    """

    global bundle_actor
    global bundle
    global final_streamlines

    streamlines, hdr = load_trk(streamline_file)
    labels = np.load(labels_file)

    bundle = [streamlines[i] for i in labels]
    final_streamlines = bundle

    kdb = KDTreeBundles(streamlines, labels)
    kdb.build_kdtree(mam_metric=metric)

    from dipy.viz import window, actor, widget
    from dipy.data import fetch_viz_icons, read_viz_icons
    fetch_viz_icons()

    ren = window.Renderer()
    bundle_actor = actor.line(kdb.model_bundle)
    ren.add(bundle_actor)

    show_m = window.ShowManager(ren, size=(1200, 900))
    show_m.initialize()

    def expand_callback(obj, event):
        global bundle_actor
        global bundle
        global final_streamlines

        z = int(np.round(obj.get_value()))
        if z > len(bundle):
            dists, actual_indices, extra_streamlines = kdb.expand(z, True)
            final_streamlines = bundle + extra_streamlines
        if z < len(bundle):
            reduction = len(bundle) - z
            # print(reduction)
            dists, actual_indices, final_streamlines = kdb.reduce(reduction,
                                                                  True)
        if z == len(bundle):
            final_streamlines = bundle
        ren.rm(bundle_actor)
        bundle_actor = actor.line(final_streamlines)
        ren.add(bundle_actor)
        show_m.render()

    slider = widget.slider(show_m.iren, show_m.ren,
                           callback=expand_callback,
                           min_value=1,
                           max_value=len(bundle) + 10 * len(bundle),
                           value=len(bundle),
                           label="Expand/Reduce",
                           right_normalized_pos=(.98, 0.6),
                           size=(120, 0), label_format="%0.lf",
                           color=(1., 1., 1.),
                           selected_color=(0.86, 0.33, 1.))

    def button_save_callback(obj, event):
        global final_streamlines
        ftypes = (("Trackvis file", "*.trk"), ("All Files", "*.*"))
        fname = window.save_file_dialog(initial_file='dipy.trk',
                                        default_ext='.trk',
                                        file_types=ftypes)
        if fname != '':
            print('Saving new trk file ...')
            save_trk(fname, final_streamlines, hdr=hdr)
            print('File saved at ' + fname)

    button_png = read_viz_icons(fname='drive.png')
    button = widget.button(show_m.iren,
                           show_m.ren,
                           button_save_callback,
                           button_png, (.94, .9), (50, 50))

    global size
    size = ren.GetSize()

    def win_callback(obj, event):
        global size
        if size != obj.GetSize():

            slider.place(ren)
            button.place(ren)
            size = obj.GetSize()

    show_m.add_window_callback(win_callback)
    show_m.render()
    show_m.start()
>>>>>>> 673537700ce0828891541d053481f728b7ed5253
