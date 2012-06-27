import numpy as np

class GradientTable(object):
    def __init__(self, gtab):
        """ A generic class for diffusion gradients

        Parameters
        ----------
        gtab: list or tuple with (bvals, bvecs)
            or Nx4 table with bvals in the first column and bvecs at the last 3 columns
            bvecs could be Nx3 or 3xN. Diffusion gradient table can be insterted as Nx4 or 4xN.


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

        if isinstance(gtab,(tuple,list)):
            self.bvals= gtab[0]
            self.bvecs= gtab[1]
            self.gtab = np.zeros((len(self.bvals.ravel()),4))
            self.gtab[:,0]=self.bvals
            if self.bvecs.shape[1]>self.bvecs.shape[0]:
                self.gtab[:,1:]=self.bvecs.T
            else:
                self.gtab[:,1:]=self.bvecs

        if hasattr(gtab,'shape'):
            if gtab.ndim==2:
                if gtab.shape[-1]==4:
                    self.gtab= gtab
                    self.bvals= gtab[:,0]
                    self.bvecs= gtab[:,1:]
                if gtab.shape[0]==4:
                    self.gtab= gtab.T
                    self.bvals= self.gtab[:,0]
                    self.bvecs= self.gtab[:,1:]
                if gtab.shape[-1]==3:
                    print('We are not handling yet Nx3 gradient tables. Please use an Nx4 gradient table or a tuple with bvals and bvecs.')


    def expose_b0s(self,thr=20):
        """ find where b0s are held
        """
        return np.where(self.bvals<=thr)[0]

    def reorient(self):
        pass

    def info(self):
        pass
