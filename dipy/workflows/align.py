import numpy as np
import os.path as path
from dipy.utils.six import string_types
from dipy.align.streamlinear import whole_brain_slr
from dipy.tracking.streamline import transform_streamlines
from glob import glob
from dipy.io.trackvis import load_trk, save_trk


def whole_brain_slr_flow(moving_streamlines_files,
                         static_streamlines_file, out_dir=None,
                         slr_transform='affine', slr_progressive=True,
                         maxiter=150, select_random=50000, verbose=True):
    """ Whole brain Streamline-based Registration

    Parameters
    ----------
    moving_streamlines_files : string
        Paths of streamline files to be registered to the static streamlines.
    static_streamlines_file : string
        Path of static (fixed) streamlines
    out_dir : string, optional
        Output directory (default input file directory)
    slr_transform : string, optional
        Can be 'translation', 'rigid', 'similarity', 'scaling' or 'affine'.
        (Default 'affine').
    slr_progressive : bool, optional
        If for example you selected `rigid` in slr_transform then you will
        do first translation and then rigid (default True). From the command
        line call this using 0 (False) or 1 (True).
    maxiter : int, optional
        Maximum iterations for registation optimization
    select_random : int, optional
        Random streamlines for starting QuickBundles.
    verbose : bool, optional
        Print results as they are being generated. From the command
        line call this using 0 (False) or 1 (True).

    """

    if isinstance(moving_streamlines_files, string_types):
        sfiles = glob(moving_streamlines_files)
    else:
        raise ValueError('# Moving_streamline_files not a string')

    static_streamlines, hdr_static = load_trk(static_streamlines_file)

    if verbose:
        print('# Static streamlines file')
        print(static_streamlines_file)
        print('# Moving streamlines files')

    for sf in sfiles:

        if verbose:
            print(sf)

        moving_streamlines, hdr = load_trk(sf)

        ret = whole_brain_slr(static_streamlines, moving_streamlines,
                              x0=slr_transform,
                              maxiter=maxiter, select_random=select_random,
                              verbose=verbose, progressive=slr_progressive)
        moved_streamlines, mat, static_centroids, moving_centroids = ret

        moved_centroids = transform_streamlines(moving_centroids, mat)

        moving_basename = path.splitext(path.basename(sf))[0]
        static_basename = path.splitext(
            path.basename(static_streamlines_file))[0]
        ext = path.splitext(path.basename(static_streamlines_file))[1]

        if out_dir is None:
            out_dir = path.dirname(sf)

        moved_bundle_file = path.join(out_dir,
                                      moving_basename + '_moved' + ext)
        moving_centroids_file = path.join(
            out_dir,
            moving_basename + '_moving_centroids' + ext)
        static_centroids_file = path.join(
            out_dir,
            static_basename + '_' + moving_basename + '_static_centroids' + ext)
        moved_centroids_file = path.join(
            out_dir, moving_basename + '_moved_centroids' + ext)
        mat_file = path.join(
            out_dir,
            moving_basename + '_to_' + static_basename + '_affine.txt')

        save_trk(moved_bundle_file, moved_streamlines, hdr=hdr_static)
        save_trk(static_centroids_file, static_centroids, hdr=hdr_static)
        save_trk(moving_centroids_file, moving_centroids, hdr=hdr)
        save_trk(moved_centroids_file, moved_centroids, hdr=hdr_static)
        np.savetxt(mat_file, mat)

        if verbose:
            print('\n Saved results at:')
            print(moved_bundle_file)
            print(static_centroids_file)
            print(moving_centroids_file)
            print(moved_centroids_file)
            print(mat_file)