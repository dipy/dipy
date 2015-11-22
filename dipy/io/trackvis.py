import nibabel as nib
import numpy as np

from dipy.tracking import utils


def load_trk(fname):
#    import nibabel.trackvis as tv
#    streams, hdr = tv.read(fname, points_space='rasmm')
#    return [i[0] for i in streams], hdr
    trkfile = nib.streamlines.load(fname)
    return trkfile.streamlines, trkfile.header


def save_trk(fname, streamlines, hdr=None):
#    import nibabel.trackvis as tv
#    streams = ((s, None, None) for s in streamlines)
#    if hdr is not None:
#        hdr_dict = {key: hdr[key] for key in hdr.dtype.names}
#        tv.write(fname, streams, hdr_mapping=hdr_dict, points_space='rasmm')
#    else:
#        tv.write(fname, streams, points_space='rasmm')

    tractogram = nib.streamlines.Tractogram(streamlines)
    trkfile = nib.streamlines.TrkFile(tractogram, header=hdr)
    nib.streamlines.save(trkfile, fname)

#
#def save_trk(filename, points, vox_to_ras, shape):
#    """A temporary helper function for saving trk files.
#
#    This function will soon be replaced by better trk file support in nibabel.
#    """
#    voxel_order = nib.orientations.aff2axcodes(vox_to_ras)
#    voxel_order = "".join(voxel_order)
#
#    # Compute the vox_to_ras of "trackvis space"
#    zooms = np.sqrt((vox_to_ras * vox_to_ras).sum(0))
#    vox_to_trk = np.diag(zooms)
#    vox_to_trk[3, 3] = 1
#    vox_to_trk[:3, 3] = zooms[:3] / 2.
#
#    points = utils.move_streamlines(points,
#                                    input_space=vox_to_ras,
#                                    output_space=vox_to_trk)
#
#    data = ((p, None, None) for p in points)
#
#    hdr = nib.trackvis.empty_header()
#    hdr['dim'] = shape
#    hdr['voxel_order'] = voxel_order
#    hdr['voxel_size'] = zooms[:3]
#
#    nib.trackvis.write(filename, data, hdr)

