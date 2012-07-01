import numpy as np
from os.path import splitext

class DiffusionGradients(object):
    def __init__(self, bvals, bvecs=None, b0_thr=20, atol=1e-2):
        """ A general class for handling diffusion MRI gradients' info.

        This class is especially useful as an input for signal reconstruction methods.
        It reads, loads and prepares scanner parameters like the b-values and b-vectors
        so that they can be useful during the reconstruction process.

        Parameters
        ----------

        bvals: can be any of the four options
            1. an array of shape (N,) or (1, N) or (N, 1) with the b-values.
            2. a path for the file which contains an array like the above (1).
            3. an array of shape (N, 4) or (4, N). Then this parameter is considered
            to be a b-table which contains both bvals and bvecs. In this case
            the next parameter is skipped.
            4. a path for the file which contains an array like the one at (3).

        bvecs: can be any of two options
            1. an array of shape (N, 3) or (3, N) with the b-vectors.
            2. a path for the file wich contains an array like the previous.

        b0_thr: float
            all b-values with values lower than this threshold
            are considered as b0s i.e. without diffusion weighting
            (See notes).
        atol: float
            all b-vectors need to be unit vectors up to a tolerance.

        Attributes
        ----------
        bvals: array, shape (N,)
                b-values
        bvecs: array, shape (N,3)
                b-vectors
        info: short information string

        Methods
        -------
        expose_b0s: show which b-values are smaller or equal
                    than a low diffusion signal threshold (default 20)

        expose_nonb0s: show which b-values are smaller
                    than a low diffusion signal threshold (default 20)

        transform: align the b-vectors to a different coordinate system

        Examples
        --------
        >>> from dipy.core.gradients import DiffusionGradients
        >>> bvals=1500*np.ones(7)
        >>> bvals[0]=0
        >>> sq2=np.sqrt(2)/2
        >>> bvecs=np.array([[0, 0, 0],
                            [1, 0, 0],
                            [0, 1, 0],
                            [0, 0, 1],
                            [sq2, sq2, 0],
                            [sq2, 0, sq2],
                            [0, sq2, sq2]])
        >>> gt = DiffusionGradients((bvals, bvecs))
        >>> gt.bvecs.shape == bvecs.shape
        True
        >>> gt = DiffusionGradients((bvals, bvecs.T))
        >>> gt.bvecs.shape == bvecs.T.shape
        False

        Notes
        -----
        1. Often b0s (b-values which correspond to images without diffusion
        weighting) have 0 values however in some cases the scanner cannot
        provide b0s of an exact 0 value and it gives a bit higher values
        e.g. 6 or 12. This is the purpose of the b0_thr in the __init__.

        2. We assume that the minimum number of b-values is 7.

        3. B-vectors should be unit vectors.

        See Also
        --------
        dipy.reconst.dti.Tensor

        """
        self.b0_thr=b0_thr
        if isinstance(bvals,str) and isinstance(bvecs,str):
            self.bvals, self.bvecs=read_bvals_bvecs_files(bvals, bvecs)
        if isinstance(bvals,str) and bvecs==None:
            self.bvals, self.bvecs=read_btable_file(bvals)
        if hasattr(bvals,'shape') and hasattr(bvals,'shape'):
            self.bvals, self.bvecs=bvals, bvecs
        if hasattr(bvals,'shape') and bvecs==None:
            if bvals.shape[-1]==4:
                self.bvals=np.squeeze(bvals[:, 0])
                self.bvecs=bvals[:, 1:]
            if bvals.shape[0]==4:
                self.bvals=np.squeeze(bvals[0, :])
                self.bvecs=bvals[1:, :].T
        if max(self.bvals.shape)!=max(self.bvecs.shape):
            raise ValueError('b-values and b-vectors shapes do not correspond')
        if self.bvecs.shape[1]>self.bvecs.shape[0]:
            self.bvecs=self.bvecs.T
        #check if bvecs are unit vectors
        sqrtb=np.sqrt(self.bvecs[:,0]**2+self.bvecs[:,1]**2+self.bvecs[:,2]**2)
        sqrtb=sqrtb[self.expose_nonb0s()]
        if not np.allclose(sqrtb,np.ones(len(sqrtb)),atol=atol):
            raise ValueError('B-vectors should be unit vectors')


    def norm_q_vectors(self, b0_thr=None):
        ''' normalized q_vectors from bvals and bvecs
        '''
        if b0_thr==None:
            b0_thr=self.b0_thr
        bv=self.bvals[self.bvals>b0_thr]
        bmin=np.sort(bv)[1]
        bv=np.sqrt(bv/bmin)
        return np.vstack((bv, bv, bv)).T*self.bvecs

    def expose_b0s(self, b0_thr=None):
        """ find where b0s are held
        """
        if b0_thr==None:
            b0_thr=self.b0_thr
        return np.where(self.bvals<=b0_thr)[0]

    def expose_nonb0s(self, b0_thr=None):
        """ find where b0s are held
        """
        if b0_thr==None:
            b0_thr=self.b0_thr
        return np.where(self.bvals>b0_thr)[0]

    def transform(self):
        """ reorient gradients to a different coordinate system
        """
        raise NotImplementedError('Transformation of Gradients not implemented yet')

    @property
    def info(self):
        print('B-values shape (%d,)' % self.bvals.shape)
        print('         min %f ' % self.bvals.min())
        print('         max %f ' % self.bvals.max())
        print('B-vectors shape (%d, %d)' % self.bvecs.shape)
        print('         min %f ' % self.bvecs.min())
        print('         max %f ' % self.bvecs.max())


def read_bvals_bvecs_files(fbvals,fbvecs):
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
    if isinstance(fbvals,str) and isinstance(fbvecs,str):
        base, ext = splitext(fbvals)
        if ext in ['.bvals', '.bval', '.txt', '']:
            bvals=np.squeeze(np.loadtxt(fbvals))
            bvecs=np.loadtxt(fbvecs)
        if ext=='.npy':
            bvals=np.squeeze(np.load(fbvals))
            bvecs=np.load(fbvecs)
        if bvecs.shape[1]>bvecs.shape[0]:
            bvecs=bvecs.T
        if min(bvecs.shape) != 3:
            raise IOError('bvec file should have three rows')
        if bvecs.ndim != 2:
            raise IOError('bvec file should be saved as a two dimensional array')
        if max(bvals.shape)!=max(bvecs.shape):
            raise IOError('b-values and b-vectors shapes do not correspond')
    else:
        raise ValueError('Two strings with the full filepaths are required')

    return bvals, bvecs

def read_btable_file(fbtab):
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
    if isinstance(fbtab,str):
        base, ext = splitext(fbvals)
        if ext in ['.txt', '']:
            btable=np.loadtxt(fbtab)
        if ext=='.npy':
            btable=np.load(fbvecs)
        if bvecs.shape[1]>bvecs.shape[0]:
            btable=btable.T
        if min(bvecs.shape) != 4:
            raise IOError('btable file should have 4 rows')
        bvals=np.squeeze(btable[:,0])
        bvecs=btable[:,1:]
        if bvals.ndim!=1 and bvecs.ndim != 2:
            raise IOError('b-table was not loaded correctly')
        if max(bvals.shape)!=max(bvecs.shape):
            raise IOError('b-values and b-vectors shapes do not correspond')
    else:
        raise ValueError('A string with the full filepath is required')
