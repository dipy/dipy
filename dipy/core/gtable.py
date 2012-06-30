import numpy as np


class GradientTable(object):
    def __init__(self, gtab):
        """ A generic class for handling diffusion gradients

        Parameters
        ----------
        gtab: list or tuple with (bvals, bvecs) 
            or an Nx4 table with bvals in the first column 
            and bvecs at the last 3 columns.             
            bvals can be arrays of shape N or (1, N) or (N, 1).
            bvecs could be Nx3 or 3xN arrays of floats.             
            btable can be insterted as Nx4 or 4xN.
            qvectors can be inserted as Nx3 arrays.

        Attributes
        ----------
        bvals: array, shape (N,)
                b-values
        bvecs: array, shape (N,3)
                b-vectors
        qvecs: array, shape (N,3)
                a normalized combination of bvals and bvecs

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
        >>> gt = GradientTable((bvals,bvecs))
        >>> gt.bvecs.shape == bvecs.shape
        True

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
                if gtab.shape[-1]==3:
                    self.qvecs=self.gtab
                    self.bvals=None
                    self.bvecs=None
                if gtab.shape[0]==3:
                    self.qvecs=self.gtab.T
                    self.bvals=None
                    self.bvecs=None

    def qvecs_from_bvecs(self, thr=20):
        bv=self.bvals[self.bvals>thr]
        bmin=np.sort(bv)[1]
        bv=np.sqrt(bv/bmin)
        qtable=np.vstack((bv, bv, bv)).T*self.bvecs
        qtable=np.floor(qtable+.5)
        self.qvecs=qtable
        return self.qvecs

    def qvecs_in_qspace(self, origin, volume_shape, thr=20):
        """
        """
        self.qvecs=self.qvecs_from_bvecs(thr=thr)
        self.Q=self.qtable+origin
        self.Q=self.q.astype('i8')
        return self.Q, np.zeros(volume_shape)

    def expose_b0s(self, thr=20):
        """ find where b0s are held
        """
        return np.where(self.bvals<=thr)[0]

    def reorient(self):
        pass

    def info(self):
        pass
