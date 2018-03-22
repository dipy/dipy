import nibabel as nib
from nibabel.streamlines import Field
from nibabel.orientations import aff2axcodes


def save_trk(fname, streamlines, affine, vox_size=None, shape=None, header=None):
    """ Saves tractogram files (*.trk)

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
    """
    if vox_size is not None and shape is not None:
        if not isinstance(header, dict):
            header = {}
        header[Field.VOXEL_TO_RASMM] = affine.copy()
        header[Field.VOXEL_SIZES] = vox_size
        header[Field.DIMENSIONS] = shape
        header[Field.VOXEL_ORDER] = "".join(aff2axcodes(affine))

    tractogram = nib.streamlines.Tractogram(streamlines)
    tractogram.affine_to_rasmm = affine
    trk_file = nib.streamlines.TrkFile(tractogram, header=header)
    nib.streamlines.save(trk_file, fname)


def load_trk(filename):
    """ Loads tractogram files(*.trk)

    Parameters
    ----------
    filename : str
        input trk filename

    Returns
    -------
    streamlines : list of 2D arrays
        Each 2D array represents a sequence of 3D points (points, 3).
    hdr : dict
        header from a trk file
    """
    trk_file = nib.streamlines.load(filename)
    return trk_file.streamlines, trk_file.header
