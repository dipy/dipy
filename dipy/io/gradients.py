import numpy as np
from os.path import splitext


def read_bvals_bvecs(fbvals, fbvecs):
    """ Read b-values and b-vectors from the disk

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

    """
    if isinstance(fbvals, basestring) and isinstance(fbvecs, basestring):
        base, ext = splitext(fbvals)
        if ext in ['.bvals', '.bval', '.txt', '']:
            bvals = np.squeeze(np.loadtxt(fbvals))
            bvecs = np.loadtxt(fbvecs)
        if ext == '.npy':
            bvals = np.squeeze(np.load(fbvals))
            bvecs = np.load(fbvecs)
        if bvecs.shape[1] > bvecs.shape[0]:
            bvecs = bvecs.T
        if min(bvecs.shape) != 3:
            raise IOError('bvec file should have three rows')
        if bvecs.ndim != 2:
            raise IOError('bvec file should be saved as a two dimensional array')
        if max(bvals.shape) != max(bvecs.shape):
            raise IOError('b-values and b-vectors shapes do not correspond')
    else:
        raise ValueError('Two strings with the full filepaths are required')
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
