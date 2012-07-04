import numpy as np
from dipy.io.bvalues import read_bvals_bvecs_files, read_btable_file

class GradientTable(object):
    def __init__(self, bvals, bvecs=None, big_delta=None, small_delta=None, b0_threshold=20, atol=1e-2):
        """ A general class for handling information about diffusion MR gradients .

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

        big_delta: float
            acquisition timing duration (default None)

        small_delta: float
            acquisition timing duration (default None)           
        
        b0_threshold: float 
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

        b0s_mask: show which b-values are smaller or equal
                    than a low diffusion signal value threshold (default 20)

        Methods
        -------
        transform: align the b-vectors to a different coordinate system

        Examples
        --------
        >>> from dipy.core.gradients import GradientTable
        >>> bvals=1500*np.ones(7)
        >>> bvals[0]=0
        >>> sq2=np.sqrt(2)/2
        >>> bvecs=np.array([[0, 0, 0],\
                            [1, 0, 0],\
                            [0, 1, 0],\
                            [0, 0, 1],\
                            [sq2, sq2, 0],\
                            [sq2, 0, sq2],\
                            [0, sq2, sq2]])
        >>> gt = GradientTable(bvals, bvecs)
        >>> gt.bvecs.shape == bvecs.shape
        True
        >>> gt = GradientTable(bvals, bvecs.T)
        >>> gt.bvecs.shape == bvecs.T.shape
        False

        Notes
        -----
        1. Often b0s (b-values which correspond to images without diffusion
        weighting) have 0 values however in some cases the scanner cannot 
        provide b0s of an exact 0 value and it gives a bit higher values 
        e.g. 6 or 12. This is the purpose of the b0_threshold in the __init__.

        2. We assume that the minimum number of b-values is 7.

        3. B-vectors should be unit vectors.

        See Also
        --------
        dipy.reconst.dti.Tensor

        """
        self.b0_threshold=b0_threshold
        if hasattr(bvals,'shape') and hasattr(bvals,'shape'):
            self.bvals, self.bvecs=bvals, bvecs 
        if hasattr(bvals,'shape') and bvecs is None:
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
        sqrtb=sqrtb[~self.b0s_mask]
        if not np.allclose(sqrtb,np.ones(len(sqrtb)),atol=atol):
            raise ValueError('B-vectors should be unit vectors')
        self.big_delta=None
        self.small_delta=None

    @property
    def b0s_mask(self):
        """ find where b0s are held
        """
        return self.bvals<=self.b0_threshold

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



           

