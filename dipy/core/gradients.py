import numpy as np


class DiffusionGradients(object):
    def __init__(self, gtab, b0_thr=20):
        """ A general class for handling diffusion MRI gradients

        Parameters
        ----------
        gtab: can be any of the following options
            list or tuple with (b-values, b-vectors) where
                b-values has shape (1, N) or (N, 1) or (N,) and
                b-vectors has shape (N, 3) or (3, N)
            or (N, 4) or (4, N) array with b-values in the first
                column and b-vectors in the last 3 columns
            or (path_b-values, path_b-vectors)
                filenames of text files with b-values and b-vectors
            or (fileobj_b-values, fileobj_b-vectors)
                fileobjects of the above.

        b0_thr: float
            all b-values with values lower than this threshold
            are considered as b0s i.e. without diffusion weighting
            (See notes).

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

        """
        self.b0_thr=b0_thr
        if isinstance(gtab, (tuple, list)):
            if len(gtab)==2:
                self.bvals=gtab[0]
                self.bvecs=gtab[1]
                if max(self.bvals.shape)!=max(self.bvecs.shape):
                    raise ValueError('b-values and b-vectors shapes do not correspond')
                self.gtab=np.zeros((len(self.bvals.ravel()), 4))
                self.gtab[:, 0]=self.bvals
                if self.bvecs.shape[1]>self.bvecs.shape[0]:
                    self.gtab[:, 1:]=self.bvecs.T
                else:
                    self.gtab[:, 1:]=self.bvecs
        if hasattr(gtab, 'shape'):
            if gtab.ndim==2:
                if gtab.shape[-1]==4:
                    self.gtab=gtab
                    self.bvals=gtab[:, 0]
                    self.bvecs=gtab[:, 1:]
                if gtab.shape[0]==4:
                    self.gtab=gtab.T
                    self.bvals=self.gtab[:, 0]
                    self.bvecs=self.gtab[:, 1:]
        #check if bvecs are unit vectors
        sqrtb=np.sqrt(self.bvecs[:,0]**2+self.bvecs[:,1]**2+self.bvecs[:,2]**2)
        sqrtb=sqrtb[self.expose_nonb0s()]
        if not np.allclose(sqrtb,np.ones(len(sqrtb)),atol=1e-2):
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
        print('B-values shape (%d,)',self.bvals.shape)
        print('         min %f ', self.bvals.min())
        print('         max %f ', self.bvals.max())
        print('B-vectors shape (%d, %d)',self.bvecs.shape)
        print('         min %f ', self.bvecs.min())
        print('         max %f ', self.bvecs.max())
