''' FSL IO '''

import os
from os.path import join as pjoin

import numpy as np


_VAL_FMT = '   %e'


def write_bvals_bvecs(bvals, bvecs, outpath=None, prefix=''):
    ''' Write FSL FDT bvals and bvecs files

    Parameters
    ----------
    bvals : (N,) sequence
       Vector with diffusion gradient strength (one per diffusion
       acquisition, N=no of acquisitions)
    bvecs : (N, 3) array-like
       diffusion gradient directions
    outpath : None or str
       path to write FDT bvals, bvecs text files
       None results in current working directory.
    prefix : str
       prefix for bvals, bvecs files in directory.  Defaults to ''
    '''
    if outpath is None:
        outpath = os.getcwd()
    bvals = tuple(bvals)
    bvecs = np.asarray(bvecs)
    bvecs[np.isnan(bvecs)] = 0
    N = len(bvals)
    fname = pjoin(outpath, prefix + 'bvals')
    fmt = _VAL_FMT * N + '\n'
    open(fname, 'wt').write(fmt % bvals)
    fname = pjoin(outpath, prefix + 'bvecs')
    bvf = open(fname, 'wt')
    for dim_vals in bvecs.T:
        bvf.write(fmt % tuple(dim_vals))
