from os.path import splitext
import re
import io
import tempfile
import warnings
import numpy as np


def read_bvals_bvecs(fbvals, fbvecs):
    """Read b-values and b-vectors from disk.

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
        # If the input was None or empty string, we don't read anything and
        # move on:
        if this_fname is None or not this_fname:
            vals.append(None)
            continue

        if not isinstance(this_fname, str):
            raise ValueError('String with full path to file is required')

        base, ext = splitext(this_fname)
        if ext in ['.bvals', '.bval', '.bvecs', '.bvec', '.txt',
                   '.eddy_rotated_bvecs', '']:
            with open(this_fname, 'r') as f:
                content = f.read()

            munged_content = io.StringIO(re.sub(r'(\t|,)', ' ', content))
            vals.append(np.squeeze(np.loadtxt(munged_content)))
        elif ext == '.npy':
            vals.append(np.squeeze(np.load(this_fname)))
        else:
            e_s = "File type %s is not recognized" % ext
            raise ValueError(e_s)

    # Once out of the loop, unpack them:
    bvals, bvecs = vals[0], vals[1]

    # If bvecs is None, you can just return now w/o making more checks:
    if bvecs is None:
        return bvals, bvecs

    if 3 not in bvecs.shape:
        raise OSError('bvec file should have three rows')
    if bvecs.ndim != 2:
        bvecs = bvecs[None, ...]
        bvals = bvals[None, ...]
        msg = "Detected only 1 direction on your bvec file. For diffusion "
        msg += "dataset, it is recommended to have at least 3 directions."
        msg += "You may have problems during the reconstruction step."
        warnings.warn(msg)
    if bvecs.shape[1] != 3:
        bvecs = bvecs.T

    # If bvals is None, you don't need to check that they have the same shape:
    if bvals is None:
        return bvals, bvecs

    if len(bvals.shape) > 1:
        raise OSError('bval file should have one row')

    if bvals.shape[0] != bvecs.shape[0]:
        raise OSError('b-values and b-vectors shapes do not correspond')

    return bvals, bvecs
