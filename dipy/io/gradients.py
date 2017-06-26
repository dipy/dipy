from __future__ import division, print_function, absolute_import

from os.path import splitext
from dipy.utils.six import string_types
import numpy as np


def read_bvals_bvecs(fbvals, fbvecs):
    """
    Read b-values and b-vectors from disk

    Parameters
    ----------
    fbvals : str
       Full path to file with b-values. None to not read bvals.
    fbvecs : str
       Full path of file with b-vectors. None to not read bvecs.

    Returns
    -------
    bvals : array, (N,) or None
    bvecs : array, (N, 3) or None

    Notes
    -----
    Files can be either '.bvals'/'.bvecs' or '.txt' or '.npy' (containing
    arrays stored with the appropriate values).
    """

    # Loop over the provided inputs, reading each one in turn and adding them
    # to this list:
    vals = []
    for this_fname in [fbvals, fbvecs]:
        # If the input was None, we don't read anything and move on:
        if this_fname is None:
            vals.append(None)
        else:
            if isinstance(this_fname, string_types):
                base, ext = splitext(this_fname)
                if ext in ['.bvals', '.bval', '.bvecs', '.bvec', '.txt', '']:
                    vals.append(np.squeeze(np.loadtxt(this_fname)))
                elif ext == '.npy':
                    vals.append(np.squeeze(np.load(this_fname)))
                else:
                    e_s = "File type %s is not recognized" % ext
                    raise ValueError(e_s)
            else:
                raise ValueError('String with full path to file is required')

    # Once out of the loop, unpack them:
    bvals, bvecs = vals[0], vals[1]

    # If bvecs is None, you can just return now w/o making more checks:
    if bvecs is None:
        return bvals, bvecs

    if bvecs.shape[1] > bvecs.shape[0]:
        bvecs = bvecs.T
    if min(bvecs.shape) != 3:
        raise IOError('bvec file should have three rows')
    if bvecs.ndim != 2:
        raise IOError('bvec file should be saved as a two dimensional array')

    # If bvals is None, you don't need to check that they have the same shape:
    if bvals is None:
        return bvals, bvecs

    if len(bvals.shape) > 1:
        raise IOError('bval file should have one row')

    if max(bvals.shape) != max(bvecs.shape):
            raise IOError('b-values and b-vectors shapes do not correspond')

    return bvals, bvecs
