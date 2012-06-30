import numpy as np


class DiffusionGradients(object):
    def __init__(self, gtab, b0_thr=20):
        """ A general class for handling diffusion MRI gradients

        Parameters
        ----------
        gtab: list or tuple with (b-values, b-vectors) where
            b-values has shape (1, N) or (N, 1) or (N,) and
            b-vectors has shape (N, 3) or (3, N)
              or an (N, 4) or (4, N) array with b-values in the first
            column and b-vectors in the last 3 columns.

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
        expose_b0s: show which b-values are smaller
                    than a low threshold (default 20)

        transform: align the b-vectors to a different coordinate system

        Examples
        --------
        >>> from dipy.core.gradients import DiffusionGradients
        >>> bvals=1500*np.ones(4)
        >>> bvals[0]=0
        >>> bvecs=np.array([[0,0,0],[1,0,0],[0,1,0],[0,0,1]])
        >>> gt = DiffusionGradients((bvals, bvecs))
        >>> gt.bvecs.shape == bvecs.shape
        True
        >>> gt = DiffusionGradients((bvals, bvecs.T))
        >>> gt.bvecs.shape == bvecs.T.shape
        False

        Notes
        -----
        Often b0s (b-values which correspond to images without diffusion
        weighting) have 0 values however in some cases the scanner cannot
        provide b0s of an exact 0 value and it gives a bit higher values
        e.g. 6 or 12. This is the purpose of the b0_thr in the __init__.

        """
        self.b0_thr=b0_thr
        if isinstance(gtab, (tuple, list)):
            self.bvals=gtab[0]
            self.bvecs=gtab[1]
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
        sqrtb=sqrtb[self.expose_non_b0s()]
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

    def expose_non_b0s(self, b0_thr=None):
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
        print('B-values shape (%d, %d)',self.bvals.shapem)
        print('         min %f ', self.bvals.min())
        print('         max %f ', self.bvals.max())
        print('B-vectors shape (%d, %d)',self.bvecs.shapem)
        print('         min %f ', self.bvecs.min())
        print('         max %f ', self.bvecs.max())
