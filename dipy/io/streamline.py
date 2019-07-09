import os
from functools import partial
import nibabel as nib
from nibabel.streamlines import (Field, TrkFile, TckFile,
                                 Tractogram, LazyTractogram,
                                 detect_format)
from nibabel.orientations import aff2axcodes
from dipy.io.dpy import Dpy, Streamlines


def save_tractogram(fname, streamlines, affine, vox_size=None, shape=None,
                    header=None, reduce_memory_usage=False,
                    tractogram_file=None):
    """ Saves tractogram files (*.trk or *.tck or *.dpy)

    Parameters
    ----------
    fname : str
        output trk filename
    streamlines : list of 2D arrays, generator or ArraySequence
        Each 2D array represents a sequence of 3D points (points, 3).
    affine : array_like (4, 4)
        The mapping from voxel coordinates to streamline points.
    vox_size : array_like (3,), optional
        The sizes of the voxels in the reference image (default: None)
    shape : array, shape (dim,), optional
        The shape of the reference image (default: None)
    header : dict, optional
        Metadata associated to the tractogram file(*.trk). (default: None)
    reduce_memory_usage : {False, True}, optional
        If True, save streamlines in a lazy manner i.e. they will not be kept
        in memory. Otherwise, keep all streamlines in memory until saving.
    tractogram_file : class TractogramFile, optional
        Define tractogram class type (TrkFile vs TckFile)
        Default is None which means auto detect format
    """
    if 'dpy' in os.path.splitext(fname)[1].lower():
        dpw = Dpy(fname, 'w')
        dpw.write_tracks(Streamlines(streamlines))
        dpw.close()
        return

    tractogram_file = tractogram_file or detect_format(fname)
    if tractogram_file is None:
        raise ValueError("Unknown format for 'fname': {}".format(fname))

    if vox_size is not None and shape is not None:
        if not isinstance(header, dict):
            header = {}
        header[Field.VOXEL_TO_RASMM] = affine.copy()
        header[Field.VOXEL_SIZES] = vox_size
        header[Field.DIMENSIONS] = shape
        header[Field.VOXEL_ORDER] = "".join(aff2axcodes(affine))

    if reduce_memory_usage and not callable(streamlines):
        sg = lambda: (s for s in streamlines)
    else:
        sg = streamlines

    tractogram_loader = LazyTractogram if reduce_memory_usage else Tractogram
    tractogram = tractogram_loader(sg)
    tractogram.affine_to_rasmm = affine
    track_file = tractogram_file(tractogram, header=header)
    nib.streamlines.save(track_file, fname)


def load_tractogram(filename, lazy_load=False):
    """ Loads tractogram files (*.trk or *.tck or *.dpy)

    Parameters
    ----------
    filename : str
        input trk filename
    lazy_load : {False, True}, optional
        If True, load streamlines in a lazy manner i.e. they will not be kept
        in memory and only be loaded when needed.
        Otherwise, load all streamlines in memory.

    Returns
    -------
    streamlines : list of 2D arrays
        Each 2D array represents a sequence of 3D points (points, 3).
    hdr : dict
        header from a trk file
    """
    if 'dpy' in os.path.splitext(filename)[1].lower():
        dpw = Dpy(filename, 'r')
        streamlines = dpw.read_tracks()
        dpw.close()
        return streamlines, {}

    trk_file = nib.streamlines.load(filename, lazy_load)
    return trk_file.streamlines, trk_file.header


load_tck = load_tractogram
load_tck.__doc__ = load_tractogram.__doc__.replace("(*.trk or *.tck or *.dpy)",
                                                   "(*.tck)")


load_trk = load_tractogram
load_trk.__doc__ = load_tractogram.__doc__.replace("(*.trk or *.tck or *.dpy)",
                                                   "(*.trk)")

load_dpy = load_tractogram
load_dpy.__doc__ = load_tractogram.__doc__.replace("(*.trk or *.tck or *.dpy)",
                                                   "(*.dpy)")

save_tck = partial(save_tractogram, tractogram_file=TckFile)
save_tck.__doc__ = save_tractogram.__doc__.replace("(*.trk or *.tck or *.dpy)",
                                                   "(*.tck)")


save_trk = partial(save_tractogram, tractogram_file=TrkFile)
save_trk.__doc__ = save_tractogram.__doc__.replace("(*.trk or *.tck or *.dpy)",
                                                   "(*.trk)")

save_dpy = partial(save_tractogram, affine=None)
save_dpy.__doc__ = save_tractogram.__doc__.replace("(*.trk or *.tck or *.dpy)",
                                                   "(*.dpy)")
