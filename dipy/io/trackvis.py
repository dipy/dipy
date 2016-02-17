import nibabel as nib
import numpy as np

from dipy.tracking import utils


def save_trk(filename, points, vox_to_ras, shape):
    """A temporary helper function for saving trk files.

    This function will soon be replaced by better trk file support in nibabel.
    """
    voxel_order = nib.orientations.aff2axcodes(vox_to_ras)
    voxel_order = "".join(voxel_order)

    # Compute the vox_to_ras of "trackvis space"
    zooms = np.sqrt((vox_to_ras * vox_to_ras).sum(0))
    vox_to_trk = np.diag(zooms)
    vox_to_trk[3, 3] = 1
    vox_to_trk[:3, 3] = zooms[:3] / 2.

    points = utils.move_streamlines(points,
                                    input_space=vox_to_ras,
                                    output_space=vox_to_trk)

    data = ((p, None, None) for p in points)

    hdr = nib.trackvis.empty_header()
    hdr['dim'] = shape
    hdr['voxel_order'] = voxel_order
    hdr['voxel_size'] = zooms[:3]

    nib.trackvis.write(filename, data, hdr)
