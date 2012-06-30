import numpy as np

class GradientTable(object):
    def __init__(self, gtab):
        """ A general class for handling diffusion gradient tables

        Parameters
        ----------
        gtab: list or tuple with (b-values, b-vectors) where
            b-values has shape (1, N) or (N, 1) or (N,) and
            b-vectors has shape (N, 3) or (3, N)
              or an (N, 4) or (4, N) array with b-values in the first
            column and b-vectors in the last 3 columns.

        Attributes
        ----------
        bvals: array, shape (N,)
                b-values
        bvecs: array, shape (N,3)
                b-vectors

        Methods
        -------
        expose_b0s: show which bvalues are smaller
                    than a low threshold

        reorient: align the bvecs to a different coordinate system

        Examples
        --------
        >>> from dipy.core.gtable import GradientTable
        >>> from dipy.data import get_data
        >>> fimg,fbvals,fbvecs = get_data('small_64D')
        >>> bvals=np.load(fbvals)
        >>> bvecs=np.load(fbvecs)
        >>> gt = GradientTable((bvals, bvecs))
        >>> gt.bvecs.shape == bvecs.shape
        True
        >>> gt = GradientTable((bvals, bvecs.T))
        >>> gt.bvecs.shape == bvecs.T.shape
        False

        """

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

    def qvecs_from_bvecs(self, thr=20):
        bv=self.bvals[self.bvals>thr]
        bmin=np.sort(bv)[1]
        bv=np.sqrt(bv/bmin)
        qtable=np.vstack((bv, bv, bv)).T*self.bvecs
        qtable=np.floor(qtable+.5)
        self.qvecs=qtable
        return self.qvecs

    def expose_b0s(self, thr=20):
        """ find where b0s are held
        """
        return np.where(self.bvals<=thr)[0]

    def reorient(self):
        pass

    def info(self):
        pass
