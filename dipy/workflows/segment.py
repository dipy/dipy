from __future__ import division, print_function, absolute_import

from glob import glob
from os.path import join, basename, splitext

import nibabel as nib
import numpy as np

from dipy.workflows.utils import (choose_create_out_dir,
                                  int_param,
                                  bool_param,
                                  int_list_param)
from dipy.segment.mask import median_otsu
import os
import numpy as np
from dipy.utils.six import string_types
from glob import glob
from dipy.align.streamlinear import whole_brain_slr
from dipy.segment.bundles import recognize_bundles
from dipy.tracking.streamline import transform_streamlines
from nibabel import trackvis as tv


def median_otsu_flow(input_files, out_dir='', save_masked=False,
                     median_radius=4, numpass=4, autocrop=False,
                     vol_idx=None, dilate=None):
    """ Workflow wrapping the median_otsu segmentation method.

    It applies median_otsu segmentation on each file found by 'globing'
    ``input_files`` and saves the results in a directory specified by
    ``out_dir``.

    Parameters
    ----------
    input_files : string
        Path to the input volumes. This path may contain wildcards to process
        multiple inputs at once.
    out_dir : string, optional
        Output directory (default input file directory)
    save_masked : bool
        Save mask
    median_radius : int, optional
        Radius (in voxels) of the applied median filter(default 4)
    numpass : int, optional
        Number of pass of the median filter (default 4)
    autocrop : bool, optional
        If True, the masked input_volumes will also be cropped using the
        bounding box defined by the masked data. Should be on if DWI is
        upsampled to 1x1x1 resolution. (default False)
    vol_idx : string, optional
        1D array representing indices of ``axis=3`` of a 4D `input_volume`
        'None' (the default) corresponds to ``(0,)`` (assumes first volume in
        4D array)
    dilate : string, optional
        number of iterations for binary dilation (default 'None')
    """
    for fpath in glob(input_files):
        print('')
        print('Applying median_otsu segmentation on {0}'.format(fpath))
        img = nib.load(fpath)
        volume = img.get_data()

        masked, mask = median_otsu(volume, median_radius,
                                   numpass, autocrop,
                                   vol_idx, dilate)

        fname, ext = splitext(basename(fpath))
        if(fname.endswith('.nii')):
            fname, _ = splitext(fname)
            ext = '.nii.gz'

        mask_fname = fname + '_mask' + ext

        out_dir_path = choose_create_out_dir(out_dir, fpath)

        mask_img = nib.Nifti1Image(mask.astype(np.float32), img.get_affine())
        mask_out_path = join(out_dir_path, mask_fname)
        mask_img.to_filename(mask_out_path)
        print('Mask saved as {0}'.format(mask_out_path))

        if bool_param(save_masked):
            masked_fname = fname + '_bet' + ext
            masked_img = nib.Nifti1Image(masked, img.get_affine(), img.get_header())
            masked_out_path = join(out_dir_path, masked_fname)
            masked_img.to_filename(masked_out_path)
            print('Masked volume saved as {0}'.format(masked_out_path))


def load_trk(fname):
    streams, hdr = tv.read(fname, points_space='rasmm')
    return [i[0] for i in streams], hdr


def save_trk(fname, streamlines, hdr=None):
    streams = ((s, None, None) for s in streamlines)
    if hdr is not None:
        hdr_dict = {key: hdr[key] for key in hdr.dtype.names}
        tv.write(fname, streams, hdr_mapping=hdr_dict, points_space='rasmm')
    else:
        tv.write(fname, streams, points_space='rasmm')


def show_bundles(static, moving, linewidth=1., tubes=False,
                 opacity=1., fname=None):

    from dipy.viz import fvtk
    ren = fvtk.ren()
    ren.SetBackground(1, 1, 1.)

    if tubes:
        static_actor = fvtk.streamtube(static, fvtk.colors.red,
                                       linewidth=linewidth, opacity=opacity)
        moving_actor = fvtk.streamtube(moving, fvtk.colors.green,
                                       linewidth=linewidth, opacity=opacity)

    else:
        static_actor = fvtk.line(static, fvtk.colors.red,
                                 linewidth=linewidth, opacity=opacity)
        moving_actor = fvtk.line(moving, fvtk.colors.green,
                                 linewidth=linewidth, opacity=opacity)

    fvtk.add(ren, static_actor)
    fvtk.add(ren, moving_actor)

    fvtk.add(ren, fvtk.axes(scale=(2, 2, 2)))

    fvtk.show(ren, size=(900, 900))
    if fname is not None:
        fvtk.record(ren, size=(900, 900), out_path=fname)


def recognize_bundles_flow(streamline_files, model_bundle_files,
                           model_streamlines_file, out_dir=None,
                           close_centroids_thr=20,
                           clean_thr=5.,
                           local_slr=True,
                           expand_thr=None,
                           scale_range=(0.8, 1.2),
                           verbose=True,
                           disp=False):

    # scale_range = tuple([float(i) for i in scale_range.split(':')])

    if isinstance(streamline_files, string_types):
        sfiles = glob(streamline_files)
    else:
        raise ValueError('# Streamline_files not a string')

    if isinstance(model_bundle_files, string_types):
        mbfiles = glob(model_bundle_files)

    if out_dir is None:
        print('No out_dir was provided results will be given in the same folder as input streamlines')

    print('### Recognition of bundles ###')
    model_streamlines, hrd_model = load_trk(model_streamlines_file)

    print('# Model\'s whole brain streamlines file')
    print(model_streamlines_file)

    print('# Streamline files')
    for sf in sfiles:
        print(sf)
        streamlines, hdr = load_trk(sf)
        ret = whole_brain_slr(model_streamlines, streamlines,
                              maxiter=150, select_random=50000,
                              verbose=verbose)
        moved_streamlines, mat, centroids1, centroids2 = ret

        print(mat)

        print('# Model_bundle files')
        for mb in mbfiles:
            print(mb)
            model_bundle, hdr_model_bundle = load_trk(mb)
            extracted_bundle, mat2 = recognize_bundles(
                model_bundle, moved_streamlines,
                close_centroids_thr=close_centroids_thr,
                clean_thr=clean_thr,
                local_slr=local_slr,
                expand_thr=expand_thr,
                scale_range=scale_range,
                verbose=verbose,
                return_full=False)

            extracted_bundle_initial = transform_streamlines(
                extracted_bundle,
                np.linalg.inv(np.dot(mat2, mat)))

            if disp:
                show_bundles(model_bundle, extracted_bundle)
                # show_bundles(model_streamlines, moved_streamlines)

            if out_dir is None:
                sf_bundle_file = os.path.join(os.path.dirname(sf),
                                              os.path.basename(mb))
            else:
                sf_bundle_file = os.path.join(
                    out_dir,
                    os.path.basename(os.path.dirname(sf)),
                    os.path.basename(mb))

            if not os.path.exists(os.path.dirname(sf_bundle_file)):
                os.makedirs(os.path.dirname(sf_bundle_file))
            save_trk(sf_bundle_file, extracted_bundle_initial, hdr=hdr)

            print('Recognized bundle saved in %s ' % (sf_bundle_file,))
