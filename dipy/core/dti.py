#!/usr/bin/python
# 5/17/2010

#import modules
import time
import sys, os, traceback, optparse
import numpy as np
import scipy as sp
from copy import copy, deepcopy

#dipy modules
from dipy.core.maskedview import MaskedView

def _makearray(a):
    new = np.asarray(a)
    wrap = getattr(a, "__array_wrap__", new.__array_wrap__)
    return new, wrap

def _filled(a):
    if hasattr(a, 'filled'):
        return a.filled()
    else:
        return a

class Tensor(object):
    """
    Fits a diffusion tensor given diffusion-weighted signals and gradient info

    Tensor object that when initialized calculates single self diffusion
    tensor[1]_ in each voxel using selected fitting algorithm
    (DEFAULT: weighted least squares[1]_)
    Requires a given gradient table, b value for each diffusion-weighted
    gradient vector, and image data given all as ndarrays.

    Parameters
    ----------
    data : ndarray ([X, Y, Z, ...], g)
        Diffusion-weighted signals. The dimension corresponding to the
        diffusion weighting must be the last dimenssion
    bval : ndarray (g,)
        Diffusion weighting factor b for each vector in gtab.
    gtab : ndarray (g, 3)
        Diffusion gradient table found in DICOM header as a ndarray.
    mask : ndarray, optional
        The tensor will only be fit where mask is True. Mask must must
        broadcast to the shape of data and must have fewer dimensions than data
    thresh : float, default = None
        The tensor will not be fit where data[bval == 0] < thresh. If multiple
        b0 volumes are given, the minimum b0 signal is used.
    min_signal : float
        All diffusion weighted signals below min_signal are replaced with
        min_signal. min_signal must be > 0.

    Attributes
    ----------
    D : ndarray (..., 3, 3)
        Self diffusion tensor calculated from cached eigenvalues and 
        eigenvectors.
    mask : ndarray
        True in voxels where a tensor was fit, false if the voxel was skipped
    B : ndarray (g, 7)
        Design matrix or B matrix constructed from given gradient table and
        b-value vector.
    evals : ndarray (..., 3) 
        Cached eigenvalues of self diffusion tensor for given index. 
        (eval1, eval2, eval3)
    evecs : ndarray (..., 3, 3)
        Cached associated eigenvectors of self diffusion tensor for given 
        index. Note: evals[..., j] is associated with evecs[..., :, j]


    Methods
    -------
    fa : ndarray
        Calculates fractional anisotropy [2]_.
    md : ndarray
        Calculates the mean diffusivity [2]_. 
        Note: [units ADC] ~ [units b value]*10**-1
    
    See Also
    --------
    dipy.io.bvectxt.read_bvec_file, WLS_tensor, design_matrix, 
    dipy.core.qball.ODF
    
    Notes
    -----
    Due to the fact that diffusion MRI entails large volumes (e.g. [256,256,
    50,64]), memory can be an issue. Therefore, only the following parameters 
    of the self diffusion tensor are cached for each voxel:

        -All three eigenvalues
        -Primary and secondary eigenvectors

    From these cached parameters, one can presumably construct any desired
    parameter.

    References
    ----------
    ..  [1] Basser, P.J., Mattiello, J., LeBihan, D., 1994. Estimation of 
        the effective self-diffusion tensor from the NMR spin echo. J Magn 
        Reson B 103, 247-254.
    ..  [2] Basser, P., Pierpaoli, C., 1996. Microstructural and physiological
        features of tissues elucidated by quantitative diffusion-tensor MRI. 
        Journal of Magnetic Resonance 111, 209-219.
    
    Examples
    --------
    >>> data = np.ones((5, 6, 7, 56)) * 100.
    >>> data[..., 0] *= 10
    >>> x = 1
    >>> y = 1
    >>> z = 1
    >>> tensor = dti.Tensor(data, gtab, bvals)

    To get the tensor for a particular voxel
    
    >>> tensor[x, y, z].D.shape
    (3, 3)

    To get the tensors of all the voxels in a slice

    >>> tensor[:, :, 2].D.shape
    (5, 6, 3, 3)
    
    """
    ### Shape Property ###
    def _getshape(self):
        """
        Gives the shape of the tensor array

        """

        return self._evals.shape[:-1]

    def _setshape(self, shape):
        """
        Sets the shape of the tensor array

        """
        self._evals.shape = shape + (3,)
        self._evecs.shape = shape + (3,3)

    shape = property(_getshape, _setshape, doc = "Shape of tensor array")

    ### Ndim Property ###
    @property
    def ndim(self):
        return self._evals.ndim - 1

    @property
    def mask(self):
        if hasattr(self._evals, 'mask'):
            return self._evals.mask
        else:
            return np.ones(self.shape, 'bool')

    ### Getitem Property ###
    def __getitem__(self, index):
        """
        Returns part of the tensor array

        """
        if type(index) is not tuple:
            index = (index,)
        if len(index) > self.ndim:
            raise IndexError('invalid index')
        for ii, slc in enumerate(index):
            if slc is Ellipsis:
                n_ellipsis = len(self.shape) - len(index) + 1
                index = index[:ii] + n_ellipsis*(slice(None),) + index[ii+1:]
                break

        new_tensor = copy(self)
        new_tensor._evals = self._evals[index]
        new_tensor._evecs = self._evecs[index]
        return new_tensor

    ### Eigenvalues Property ###
    @property
    def evals(self):
        """
        Returns the eigenvalues of the tensor as an ndarray

        """

        return _filled(self._evals)

    ### Eigenvectors Property ###
    @property
    def evecs(self):
        """
        Returns the eigenvectors of teh tensor as an ndarray

        """

        return _filled(self._evecs)

    def __init__(self, data, b_values, grad_table, mask=True, thresh=None,
                 min_signal=1, verbose=False):
        """
        Fits a tensors to diffusion weighted data.

        """

        if min_signal <= 0:
            raise ValueError('min_signal must be > 0')

        #64 bit design matrix makes for faster pinv
        B = design_matrix(grad_table.T, b_values)
        self.B = B

        mask = np.atleast_1d(mask)
        if mask is not None:
            #Define total mask from thresh and mask
            mask = mask & (np.min(data[..., b_values == 0], -1) > thresh)

        #if mask is all False
        if not mask.any():
            raise ValueError('between mask and thresh, there is no data to '+
            'fit')

        #and the mask is not all True
        if not mask.all():
            #leave only data[mask is True]
            data = data[mask]
            data = MaskedView(mask, data)

        #Perform WLS fit on masked data
        evals, evecs = wls_fit_tensor(B, data, min_signal=min_signal)
        self._evals = evals
        self._evecs = evecs

    ### Self Diffusion Tensor Property ###
    def _getD(self):
        evals_flat = self.evals.reshape((-1, 3))
        evecs_flat = self.evecs.reshape((-1, 3, 3))
        D_flat = np.empty(evecs_flat.shape)
        for ii, eval in enumerate(evals_flat):
            L = eval
            Q = evecs_flat[ii]
            D_flat[ii] = np.dot(Q*L, Q.T) #timeit = 11.5us
        return D_flat.reshape(self.evecs.shape)
    
    D = property(_getD, doc = "Self diffusion tensor")

    @property
    def FA(self):
        return self.fa()

    def fa(self):
        """
        Fractional anisotropy (FA) calculated from cached eigenvalues. 
        
        Returns
        -------
        fa : ndarray (V, 1)
            Calculated FA. Note: range is 0 <= FA <= 1.

        Notes
        -----
        FA is calculated with the following equation:

        .. math::

        FA = \sqrt{\frac{1}{2}\frac{(\lambda_1-\lambda_2)^2+(\lambda_1-
                    \lambda_3)^2+(\lambda_2-lambda_3)^2}{\lambda_1^2+
                    \lambda_2^2+\lambda_3^2} }
        """
        evals, wrap = _makearray(self._evals)
        ev1 = evals[..., 0]
        ev2 = evals[..., 1]
        ev3 = evals[..., 2]

        fa = np.sqrt(0.5 * ((ev1 - ev2)**2 + (ev2 - ev3)**2 + (ev3 - ev1)**2)
                      / (ev1**2 + ev2**2 + ev3**2))
        fa = wrap(np.asarray(fa))
        return _filled(fa)

    @property
    def MD(self):
        return self.md()
    
    def md(self):
        """
        Mean diffusitivity (MD) calculated from cached eigenvalues. 
        
        Returns
        -------
        md : ndarray (V, 1)
            Calculated MD.

        Notes
        -----
        MD is calculated with the following equation:

        .. math:: ADC = \frac{\lambda_1+\lambda_2+\lambda_3}{3}
        """
        #adc/md = (ev1+ev2+ev3)/3
        return self.evals.mean(-1)

    @property
    def IN(self):
        ''' Quantizes eigenvectors with maximum eigenvalues  on an
        evenly distributed sphere so that the can be used for tractography.

        Returns
        -------
        IN: array, shape(x,y,z) integer indices for the points of the
        evenly distributed sphere representing tensor  eigenvectors of
        maximum eigenvalue
    
        '''
        return quantize_evecs(self.evecs,odf_vertices=None)

def wls_fit_tensor(design_matrix, data, min_signal=1):
    """
    Computes weighted least squares (WLS) fit to calculate self-diffusion 
    tensor using a linear regression model [1]_.
    
    Parameters
    ----------
    design_matrix : ndarray (g, 7)
        Design matrix holding the covariants used to solve for the regression
        coefficients.
    data : ndarray ([X, Y, Z, ...], g)
        Data or response variables holding the data. Note that the last 
        dimension should contain the data. It makes no copies of data.

    Returns
    -------
    eigvals : ndarray (..., 3)
        Eigenvalues from eigen decomposition of the tensor.
    eigvecs : ndarray (..., 3, 3)
        Associated eigenvectors from eigen decomposition of the tensor.
        Eigenvectors are columnar (e.g. eigvecs[:,j] is associated with 
        eigvals[j])


    See Also
    --------
    decompose_tensor

    Notes
    -----
    In Chung, et al. 2006, the regression of the WLS fit needed an unbiased
    preliminary estimate of the weights and therefore the ordinary least 
    squares (OLS) estimates were used. A "two pass" method was implemented:
    
        (1) calculate OLS estimates of the data
        (2) apply the OLS estimates as weights to the WLS fit of the data 
    
    This ensured heteroscadasticity could be properly modeled for various 
    types of bootstrap resampling (namely residual bootstrap).
    
    .. math::

    y = \mathrm{data} \\
    X = \mathrm{design matrix} \\
    \hat{\beta}_WLS = \mathrm{desired regression coefficients (e.g. tensor)}\\
    \\
    \hat{\beta}_WLS = (X^T W X)^-1 X^T W y \\
    \\
    W = \mathrm{diag}((X \hat{\beta}_OLS)^2),
    \mathrm{where} \hat{\beta}_OLS = (X^T X)^-1 X^T y

    References
    ----------
    ..  [1] Chung, SW., Lu, Y., Henry, R.G., 2006. Comparison of bootstrap
        approaches for estimation of uncertainties of DTI parameters.
        NeuroImage 33, 531-541.
    """

    data, wrap = _makearray(data)
    data_flat = data.reshape((-1, data.shape[-1]))
    evals = np.empty((len(data_flat), 3))
    evecs = np.empty((len(data_flat), 3, 3))
    
    #obtain OLS fitting matrix
    #U,S,V = np.linalg.svd(design_matrix, False)
    #math: beta_ols = inv(X.T*X)*X.T*y
    #math: ols_fit = X*beta_ols*inv(y)
    #ols_fit = np.dot(U, U.T)
    ols_fit = _ols_fit_matrix(design_matrix)
    
    for ii, sig in enumerate(data_flat):
        evals[ii], evecs[ii] = _wls_iter(ols_fit, design_matrix, 
                                             sig, min_signal=min_signal)
    evals.shape = data.shape[:-1]+(3,)
    evecs.shape = data.shape[:-1]+(3,3)
    evals = wrap(evals)
    evecs = wrap(evecs)
    return evals, evecs


def _wls_iter(ols_fit, design_matrix, sig, min_signal=1):
    ''' 
    Function used by wls_fit_tensor for later optimization.
    '''
    sig = np.maximum(sig, min_signal) #throw out zero signals
    log_s = np.log(sig)
    w = np.exp(np.dot(ols_fit, log_s))
    D = np.dot(np.linalg.pinv(design_matrix*w[:,None]), w*log_s)
    tensor = _full_tensor(D)
    return decompose_tensor(tensor)

def ols_fit_tensor(design_matrix, data):
    """
    Computes ordinary least squares (OLS) fit to calculate self-diffusion 
    tensor using a linear regression model [1]_.
    
    Parameters
    ----------
    design_matrix : ndarray (g, 7)
        Design matrix holding the covariants used to solve for the regression
        coefficients. Use design_matrix to build a valid design matrix from 
        bvalues and a gradient table.
    data : ndarray ([X, Y, Z, ...], g)
        Data or response variables holding the data. Note that the last 
        dimension should contain the data. It makes no copies of data.

    Returns
    -------
    eigvals : ndarray (..., 3)
        Eigenvalues from eigen decomposition of the tensor.
    eigvecs : ndarray (..., 3, 3)
        Associated eigenvectors from eigen decomposition of the tensor.
        Eigenvectors are columnar (e.g. eigvecs[:,j] is associated with 
        eigvals[j])


    See Also
    --------
    WLS_fit_tensor, decompose_tensor, design_matrix

    Notes
    -----
    This function is offered mainly as a quick comparison to WLS.

    .. math::
    y = \mathrm{data} \\
    X = \mathrm{design matrix} \\
    
    \hat{\beta}_OLS = (X^T X)^-1 X^T y

    References
    ----------
    ..  [1] Chung, SW., Lu, Y., Henry, R.G., 2006. Comparison of bootstrap
        approaches for estimation of uncertainties of DTI parameters.
        NeuroImage 33, 531-541.
    """

    data = _makearray(data)
    data_flat = data.reshape((-1, data.shape[-1]))
    evals = np.empty((len(data_flat), 3))
    evecs = np.empty((len(data_flat), 3, 3))
    
    #obtain OLS fitting matrix
    #U,S,V = np.linalg.svd(design_matrix, False)
    #math: beta_ols = inv(X.T*X)*X.T*y
    #math: ols_fit = X*beta_ols*inv(y)
    #ols_fit =  np.dot(U, U.T)
    
    Ds = np.dot(data_flat,np.linalg.pinv(design_matrix.T))

    for ii, sig in enumerate(data_flat):
        tensor = _full_tensor(Ds[ii, :])
        evals[ii, :], evecs[ii, :, :] = decompose_tensor(tensor)

    evals.shape = data.shape[:-1]+(3,)
    evecs.shape = data.shape[:-1]+(3,3)
    evals = wrap(evals)
    evecs = wrap(evecs)
    return evals, evecs

def _ols_fit_matrix(design_matrix):
    """
    Helper function to calculate the ordinary least squares (OLS)
    fit as a matrix multiplication. Mainly used to calculate WLS weights. Can
    be used to calculate regression coefficients in OLS but not recommended.

    See Also:
    ---------
    wls_fit_tensor, ols_fit_tensor

    Example:
    --------
    ols_fit = _ols_fit_matrix(design_mat) 
    ols_data = np.dot(ols_fit, data)
    """

    U,S,V = np.linalg.svd(design_matrix, False)
    return np.dot(U, U.T)

def _full_tensor(D):
    """
    Returns a tensor given the six unique tensor elements

    Given the six unique tensor elments (in the order: Dxx, Dyy, Dzz, Dxy, Dxz,
    Dyz) returns a 3 by 3 tensor. All elements after the sixth are ignored.

    """

    tensor = np.empty((3,3),dtype=D.dtype)

    tensor[0, 0] = D[0]  #Dxx
    tensor[1, 1] = D[1]  #Dyy
    tensor[2, 2] = D[2]  #Dzz
    tensor[1, 0] = tensor[0, 1] = D[3]  #Dxy
    tensor[2, 0] = tensor[0, 2] = D[4]  #Dxz
    tensor[2, 1] = tensor[1, 2] = D[5]  #Dyz

    return tensor

def decompose_tensor(tensor):
    """
    Returns eigenvalues and eigenvectors given a diffusion tensor

    Computes tensor eigen decomposition to calculate eigenvalues and
    eigenvectors of self-diffusion tensor. (Basser et al., 1994a)

    Parameters
    ----------
    D : ndarray (3,3)
        array holding a tensor. Assumes D has units on order of
        ~ 10^-4 mm^2/s

    Results
    -------
    eigvals : ndarray (3,)
        Eigenvalues from eigen decomposition of the tensor. Negative
        eigenvalues are replaced by zero. Sorted from largest to smallest.
    eigvecs : ndarray (3,3)
        Associated eigenvectors from eigen decomposition of the tensor.
        Eigenvectors are columnar (e.g. eigvecs[:,j] is associated with
        eigvals[j])

    See Also
    --------
    numpy.linalg.eig
    """

    #outputs multiplicity as well so need to unique
    eigenvals, eigenvecs = np.linalg.eig(tensor)

    #need to sort the eigenvalues and associated eigenvectors
    order = eigenvals.argsort()[::-1]
    eigenvecs = eigenvecs[:, order]
    eigenvals = eigenvals[order]

    #Forcing negative eigenvalues to 0
    eigenvals = np.maximum(eigenvals, 0)
    # b ~ 10^3 s/mm^2 and D ~ 10^-4 mm^2/s
    # eigenvecs: each vector is columnar

    return eigenvals, eigenvecs

def design_matrix(gtab, bval, dtype=None):
    """
    Constructs design matrix for DTI weighted least squares or least squares 
    fitting. (Basser et al., 1994a)

    Parameters
    ----------
    gtab : ndarray with shape (3,g)
        Diffusion gradient table found in DICOM header as a numpy ndarray.
    bval : ndarray with shape (g,)
        Diffusion weighting factor b for each vector in gtab.
    dtype : string
        Parameter to control the dtype of returned designed matrix

	Return
	------
	design_matrix : ndarray (g,7)
		Design matrix or B matrix assuming Gaussian distributed tensor model.
		Note: design_matrix[j,:] = (Bxx,Byy,Bzz,Bxy,Bxz,Byz,dummy)

    """
    G = gtab
    B = np.zeros((bval.size, 7), dtype = G.dtype)
    if gtab.shape[1] != bval.shape[0]:
        raise ValueError('The number of b values and gradient directions must'
                          +' be the same')
    B[:, 0] = G[0, :] * G[0, :] * 1. * bval   #Bxx
    B[:, 1] = G[1, :] * G[1, :] * 1. * bval   #Byy
    B[:, 2] = G[2, :] * G[2, :] * 1. * bval   #Bzz
    B[:, 3] = G[0, :] * G[1, :] * 2. * bval   #Bxy
    B[:, 4] = G[0, :] * G[2, :] * 2. * bval   #Bxz
    B[:, 5] = G[1, :] * G[2, :] * 2. * bval   #Byz
    B[:, 6] = np.ones(bval.size)
    return -B




def quantize_evecs(evecs,odf_vertices=None):

    ''' Find the closest orientation of an evenly distributed sphere
    '''
    
    max_evecs=evecs[...,:,0]

    if odf_vertices==None:
        
        eds=np.load(os.path.join(os.path.dirname(__file__),'matrices','evenly_distributed_sphere_362.npz'))
        odf_vertices=eds['vertices']

    tup=max_evecs.shape[:-1]
    mec=max_evecs.reshape(np.prod(np.array(tup)),3)
    IN=np.array([np.argmin(np.dot(odf_vertices,m)) for m in mec])
    IN=IN.reshape(tup)

    return IN



##############################################################################
#The following are/will be DEPRECATED
##############################################################################


def __WLS_fit (data,gtab,bval,verbose=False):    
    """
    Computes weighted least squares (WLS) fit to calculate self-diffusion tensor. 
    (Basser et al., 1994a)

    NB: This function is the deprecated version of WLS_fit ... it works fine but
    is not as robust as WLS_fit.

    Parameters
    ----------
    data : ndarray (X,Y,Z,g)
        The image data will be masked if ndarray is given with threshold = 25.
    gtab : ndarray (3,g)
        Diffusion gradient table found in DICOM header as a numpy ndarray.
    bval : ndarray (g,1)
        Diffusion weighting factor b for each vector in gtab.
    verbose : boolean
        Boolean to indicate verbose output such as timing.

    Returns
    -------
    eig_decomp : ndarray (X,Y,Z,12)
        Eigenvalues and eigenvectors from eigen decomposition of the tensor
    design_mat : ndarray (g,7)
        DTI design matrix to reconstruct fitted data if desired

    """
    #timer for verbose
    start_time = time.time()
    
    #hold original shape
    dims = data.shape
        
    ### Prepare data for analysis    
    if isinstance(data,np.ndarray):#len(dims) == 4:
        #Create conservative mask; also makes sure log behaves
        mask = data[:,:,:,0] > 25

        #turn data into maskedview
        data = MaskedView(mask,data[mask],fill_value=0)
    elif not isinstance(data,MaskedView):
        raise ValueError('input must be of type ndarray or MaskedView')
    
    #reshape data to be (X*Y*Z,1)
    data.shape = (-1,1)

    ###Create log of signal and reshape it to be x:y:z by grad
    data = np.maximum(data,1)
    log_s = np.log(data) #type(log_s) == MaskedView
    
    ###Construct design matrix
    #For DTI this is the so called B matrix
    # X matrix from Chris' paper
    B = design_matrix(gtab,bval) # [g by 7]
	
    ###Weighted Least Squares (WLS) to solve "linear" regression
    # Y hat OLS from Chris' paper
    #  ( [x*y*z by g] [g by 7] [7 by g ] ) = [x*y*z by g]
    log_s_ols = np.dot(log_s, np.dot(B, np.linalg.pinv(B))) #type(log_s_ols) = ndarray
    del log_s #freeing up memory

    #Setting these arrays later to allow the previous step to have all memory
    eig_decomp = np.zeros((log_s_ols.shape[0],12),dtype='float32')#'int16')
    
    time_diff = list((0,0))
    time_iter = time.time()
    # This step is because we cannot vectorize diagonal vector and tensor fit
    for ii,data_ii in enumerate(data): 
        #Split up weighting vector into little w to perform pinv
        w = np.exp(log_s_ols[ii,:])[:,np.newaxis]

        #pointwise broadcasting to avoid diagonal matrix multiply!
        D = np.dot(np.linalg.pinv(B*w), w.ravel()*np.log(data_ii)) #log_s[i,:]
        
        ###Obtain eigenvalues and eigenvectors
        eig_decomp[ii,:] = decompose_tensor(D[0:6])
        
        #Check every check_percent%
        ch_percent=0.05
        if verbose and ii % np.round(ch_percent*log_s_ols.shape[0]) == 0:
            percent = 100.*ii/log_s_ols.shape[0]
            time_diff.append(time.time()-time_iter)
            min = np.mean(time_diff[2:len(time_diff)])/60.0/ch_percent
            sec = np.round((min - np.fix(min)) * 60.0)
            min = np.fix(min)
            print str(np.round(percent)) + '% ... time left: ' + str(min) + ' MIN ' \
                + str(sec) + ' SEC ... memory: ' + str(np.round(memory()/1024.)) + 'MB'
            time_iter=time.time()

    #clear variables not needed to save memory
    del log_s_ols

    # Reshape the output
    data.shape = dims
    eig_decomp = MaskedView(data.mask,eig_decomp,fill_value=0)
    
    #Report how long it took to make the fit  
    if verbose:
        min = (time.time() - start_time) / 60.0
        sec = (min - np.fix(min)) * 60.0
        print 'TOTAL TIME: ' + str(np.fix(min)) + ' MIN ' + str(np.round(sec)) + ' SEC'

    return(eig_decomp.filled(), B)

def __save_scalar_maps(scalar_maps, img=None, coordmap=None):
    """
    Deprecated version of writing scalar maps out to disk. Do not use.
    """

    #for io of writing and reading nifti images
    from nipy import load_image, save_image
    from nipy.core.api import fromarray #data --> image
    
    #For writing out with save_image with appropriate affine matrix
    if img != None:
        coordmap = get_coord_4D_to_3D(img.affine)
        header = img.header.copy()

    ###Save scalar maps if requested
    print ''
    print 'Saving t2di map ... '+out_root+'_t2di.nii.gz'
        
    #fyi the dtype flag for save image does not appear to work right now...
    t2di_img = fromarray(data[:,:,:,0],'ijk','xyz',coordmap=coordmap)
    if img != []: 
        t2di_img.header = header
    save_image(t2di_img,out_root+'_t2di.nii.gz',dtype=np.int16)

        
    scalar_fnames = ('ev1','ev2','ev3','adc','fa','ev1p','ev1f','ev1s')
    for i in range(np.size(scalar_maps,axis=3)):
        #need to send in 4 x 4 affine matrix for 3D image not 5 x 5 from original 4D image
        print 'Saving '+ scalar_fnames[i] + ' map ... '+out_root+'_'+scalar_fnames[i]+'.nii.gz'
        scalar_img = fromarray(np.int16(scalar_maps[:,:,:,i]),'ijk' ,'xyz',coordmap=coordmap)
        if img != []:
            scalar_img.header = header
        save_image(scalar_img,out_root+'_'+scalar_fnames[i]+'.nii.gz',dtype=np.int16)

    print ''
    print 'Saving D = [Dxx,Dyy,Dzz,Dxy,Dxz,Dyz] map ... '+out_root+'_self_diffusion.nii.gz'
    #Saving 4D matrix holding diffusion coefficients
    if img != [] :
        coordmap = img.coordmap
        header = img.header.copy()
    tensor_img = fromarray(tensor_data,'ijkl','xyzt',coordmap=coordmap)
    tensor_img.header = header
    save_image(tensor_img,out_root+'_self_diffusion.nii.gz',dtype=np.int16)

    print

    return

