import numpy as np
from os.path import splitext


def read_bvals_bvecs(fbvals, fbvecs):
    """
    Read b-values and b-vectors from the disk

    Parameters
    ----------
    fbvals: str
            path of file with b-values
    fbvecs: str
            path of file with b-vectors

    Returns
    -------
    bvals: array, (N,)
    bvecs: array, (N, 3)

    Notes
    -----
    Files can be either '.bvals'/'.bvecs' or '.txt' or '.npy' (containing arrays
    stored with the appropriate values).

    """

    # Loop over the provided inputs, reading each one in turn and adding them
    # to this list:
    vals = []
    for this_fname in [fbvals, fbvecs]:
        if isinstance(this_fname, basestring):
            base, ext = splitext(this_fname)
            if ext in ['.bvals', '.bval', '.bvecs', '.bvec', '.txt', '']:
                vals.append(np.squeeze(np.loadtxt(this_fname)))
            elif ext == '.npy':
                vals.append(np.squeeze(np.load(this_fname)))
            else:
                e_s = "File type %s is not recognized"%ext
                raise ValueError(e_s)
        else:
            raise ValueError('String with full path to file is required')

    # Once out of the loop, unpack them:
    bvals, bvecs = vals[0], vals[1]

    if bvecs.shape[1] > bvecs.shape[0]:
        bvecs = bvecs.T
    if min(bvecs.shape) != 3:
        raise IOError('bvec file should have three rows')
    if bvecs.ndim != 2:
        raise IOError('bvec file should be saved as a two dimensional array')
    if max(bvals.shape) != max(bvecs.shape):
            raise IOError('b-values and b-vectors shapes do not correspond')

    return bvals, bvecs


def read_btable(fbtab):
    """ Read b-table from the disk

    B-table is a 2D array which contains b-values in the
    first column and b-vectors in the last column.

    Parameters
    ----------
    fbtab: str,
        path of file with b-table

    Returns
    -------
    bvals: array, (N,)
    bvecs: array, (N, 3)

    """
    if isinstance(fbtab, basestring):
        base, ext = splitext(fbvals)
        if ext in ['.txt', '']:
            btable=np.loadtxt(fbtab)
        if ext == '.npy':
            btable = np.load(fbvecs)
        if bvecs.shape[1] > bvecs.shape[0]:
            btable = btable.T
        if min(bvecs.shape) != 4:
            raise IOError('btable file should have 4 rows')
        bvals = np.squeeze(btable[:, 0])
        bvecs = btable[:, 1:]
        if bvals.ndim != 1 and bvecs.ndim != 2:
            raise IOError('b-table was not loaded correctly')
        if max(bvals.shape) != max(bvecs.shape):
            raise IOError('b-values and b-vectors shapes do not correspond')
    else:
        raise ValueError('A string with the full filepath is required')
    return bvals, bvecs
