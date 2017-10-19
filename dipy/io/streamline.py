import nibabel as nib
from nibabel.streamlines import Field
from nibabel.orientations import aff2axcodes


def save_trk(fname, streamlines, affine, vox_size=None, shape=None, header=None):
    """ function Helper for saving trk files.

    Parameters
    ----------
    fname : str
        output trk filename
    streamlines : list of 2D arrays
        Each 2D array represents a sequence of 3D points (points, 3).
    affine : array_like (4, 4)
        The mapping from voxel coordinates to streamline points.
    vox_size : array_like (3,)
        The sizes of the voxels in the reference image.
    shape : array, shape (dim,)
        The shape of the reference image.
    header : dict
        header from a trk file

    """
    if vox_size and shape:
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
    """ function Helper for Loading trk files.

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
