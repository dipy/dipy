import nibabel as nib


def load_trk(fname):
    trkfile = nib.streamlines.load(fname)
    return trkfile.streamlines, trkfile.header


def save_trk(fname, streamlines, hdr=None, affine_to_rasmm=None):
    tractogram = nib.streamlines.Tractogram(streamlines,
                                            affine_to_rasmm=affine_to_rasmm)

    trkfile = nib.streamlines.TrkFile(tractogram, header=hdr)
    nib.streamlines.save(trkfile, fname)
