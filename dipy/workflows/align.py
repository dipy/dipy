import numpy as np
import os.path as path
from dipy.utils.six import string_types
from nibabel import trackvis as tv
from dipy.align.streamlinear import whole_brain_slr
from dipy.tracking.streamline import transform_streamlines
from glob import glob


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


def whole_brain_slr_flow(moving_streamlines_files,
                         static_streamlines_file, out_dir=None,
                         maxiter=150, select_random=50000, verbose=True):
    """ Whole brain Streamline-based Registration

    Parameters
    ----------
    moving_streamlines_files : string
        Paths of streamline files to be registered to the static streamlines.
    stratic_streamlines_file : string
        Path of static (fixed) streamlines
    out_dir : string, optional
        Output directory (default input file directory)
    maxiter : int, optional
        Maximum iterations for registation optimization
    select_random : int, optional
        Random streamlines for starting QuickBundles.
    verbose : bool, optional
        Print results as they are being generated.

    """

    if isinstance(moving_streamlines_files, string_types):
        sfiles = glob(moving_streamlines_files)
    else:
        raise ValueError('# Moving_streamline_files not a string')

    static_streamlines, hdr_static = load_trk(static_streamlines_file)

    print('# Static streamlines file')
    print(static_streamlines_file)

    print('# Moving streamlines files')
    for sf in sfiles:
        print(sf)
        moving_streamlines, hdr = load_trk(sf)
        ret = whole_brain_slr(static_streamlines, moving_streamlines,
                              maxiter=maxiter, select_random=select_random,
                              verbose=verbose)
        moved_streamlines, mat, static_centroids, moving_centroids = ret

        moved_centroids = transform_streamlines(moving_centroids, mat)

        # first = path.splitext(path.basename(sf))[0]
        # second = path.splitext(path.basename(static_streamlines_file))[0]
        ext = path.splitext(path.basename(static_streamlines_file))[1]

        if out_dir is None:
            out_dir = path.dirname(sf)

        moved_bundle_file = path.join(out_dir,
                                      'moved' + ext)
        moving_centroids_file = path.join(out_dir,
                                          'moving_centroids' + ext)
        static_centroids_file = path.join(out_dir,
                                          'static_centroids' + ext)
        moved_centroids_file = path.join(
            out_dir,  'moved_centroids' + ext)
        mat_file = path.join(out_dir, 'affine.txt')

        print('Saving results at :')
        print(moved_bundle_file)
        print(static_centroids_file)
        print(moving_centroids_file)
        print(moved_centroids_file)
        print(mat_file)
        save_trk(moved_bundle_file, moved_streamlines, hdr=hdr_static)
        save_trk(static_centroids_file, static_centroids, hdr=hdr_static)
        save_trk(moving_centroids_file, moving_centroids, hdr=hdr)
        save_trk(moved_centroids_file, moved_centroids, hdr=hdr_static)
        np.savetxt(mat_file, mat)
