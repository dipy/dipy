#!/usr/bin/python
""" Classes and functions for fitting tensors """
# 5/17/2010

import numpy as np

from dipy.reconst.maskedview import MaskedView, _makearray, _filled
from dipy.reconst.modelarray import ModelArray
from dipy.data import get_sphere


class TensorModel(object):
    """

    """
    def __init__(self, bval, bvec, fit_method="WLS", *args, **kwargs):
        """

        """
        if not callable(fit_method):
            try:
                self.fit_method = common_fit_methods[fit_method]
            except KeyError:
                raise ValueError('"'+str(fit_method)+'" is not a known fit '
                                 'method, the fit method should either be a '
                                 'function or one of the common fit methods')
        self.bvec = bvec
        self.bval = bval
        #64 bit design matrix makes for faster pinv
        self.design_matrix = design_matrix(bvec.T, bval)


    def fit(self, data):
        """
        """
        dti_params = self.fit_method(self.design_matrix, data, *args, **kargs)
        return TensorFit(self, dti_params)


class TensorFit(object):
    """ Fits a diffusion tensor given diffusion-weighted signals and gradient
    info

    Tensor object that when initialized calculates single self diffusion
    tensor [1]_ in each voxel using selected fitting algorithm
    (DEFAULT: weighted least squares [2]_)
    Requires a given gradient table, b value for each diffusion-weighted
    gradient vector, and image data given all as arrays.

    Parameters
    ----------
    data : array ([X, Y, Z, ...], g)
        Diffusion-weighted signals. The dimension corresponding to the
        diffusion weighting must be the last dimenssion
    bval : array (g,)
        Diffusion weighting factor b for each vector in gtab.
    gtab : array (g, 3)
        Diffusion gradient table found in DICOM header as a array.
    mask : array, optional
        The tensor will only be fit where mask is True. Mask must must
        broadcast to the shape of data and must have fewer dimensions than data
    thresh : float, default = None
        The tensor will not be fit where data[bval == 0] < thresh. If multiple
        b0 volumes are given, the minimum b0 signal is used.
    fit_method : funciton or string, default = 'WLS'
        The method to be used to fit the given data to a tensor. Any function
        that takes the B matrix and the data and returns eigen values and eigen
        vectors can be passed as the fit method. Any of the common fit methods
        can be passed as a string.
    *args, **kargs :
        Any other arguments or keywards will be passed to fit_method.

    common fit methods:
        'WLS' : weighted least squares
            dti.wls_fit_tensor
        'LS' : ordinary least squares
            dti.ols_fit_tensor

    Attributes
    ----------
    D : array (..., 3, 3)
        Self diffusion tensor calculated from cached eigenvalues and 
        eigenvectors.
    mask : array
        True in voxels where a tensor was fit, false if the voxel was skipped
    B : array (g, 7)
        Design matrix or B matrix constructed from given gradient table and
        b-value vector.
    evals : array (..., 3) 
        Cached eigenvalues of self diffusion tensor for given index. 
        (eval1, eval2, eval3)
    evecs : array (..., 3, 3)
        Cached associated eigenvectors of self diffusion tensor for given 
        index. Note: evals[..., j] is associated with evecs[..., :, j]


    Methods
    -------
    fa : array
        Calculates fractional anisotropy [2]_.
    md : array
        Calculates the mean diffusivity [2]_.
        Note: [units ADC] ~ [units b value]*10**-1

    See Also
    --------
    dipy.io.bvectxt.read_bvec_file, dipy.core.qball.ODF

    References
    ----------
    .. [1] Basser, P.J., Mattiello, J., LeBihan, D., 1994. Estimation of
       the effective self-diffusion tensor from the NMR spin echo. J Magn
       Reson B 103, 247-254.
    .. [2] Basser, P., Pierpaoli, C., 1996. Microstructural and physiological
       features of tissues elucidated by quantitative diffusion-tensor MRI.
       Journal of Magnetic Resonance 111, 209-219.

    Examples
    ----------
    For a complete example have a look at the main dipy/examples folder
    """
    def __init__(self, model, model_params):
        """
        Initialize a TensorFit class instance.
        """
        self.model_params = model_params


    @property
    def directions(self):
        """
        For tracking - return the PDD in each voxel
        """
        return self.evecs[..., np.newaxis, :, 0]


    @property
    def evals(self):
        """
        Returns the eigenvalues of the tensor as an array
        """
        return _filled(self.model_params[..., :3])

    @property
    def evecs(self):
        """
        Returns the eigenvectors of teh tensor as an array

        """
        evecs = _filled(self.model_params[..., 3:])
        return evecs.reshape(self.shape + (3, 3))

    ### Self Diffusion Tensor Property ###
    @property
    def quadratic_form(self):
        """Calculates the 3x3 diffusion tensor for each voxel"""
        params, wrap = _makearray(self.model_params)
        evals = params[..., :3]
        evecs = params[..., 3:]
        evals_flat = evals.reshape((-1, 3))
        evecs_flat = evecs.reshape((-1, 3, 3))
        D_flat = np.empty(evecs_flat.shape)
        for ii in xrange(len(D_flat)):
            Q = evecs_flat[ii]
            L = evals_flat[ii]
            D_flat[ii] = np.dot(Q*L, Q.T)
        D = _filled(wrap(D_flat))
        D.shape = self.shape + (3, 3)
        return D

    def lower_triangular(self, b0=None):
        return lower_triangular(self.quadratic_form, b0)

    def fa(self, fill_value=0, nonans=True):
        r"""
        Fractional anisotropy (FA) calculated from cached eigenvalues.

        Parameters
        ----------
        fill_value : float
            value of fa where self.mask == True.

        nonans : Bool
            When True, fa is 0 when all eigenvalues are 0, otherwise fa is nan

        Returns
        ---------
        fa : array (V, 1)
            Calculated FA. Note: range is 0 <= FA <= 1.

        Notes
        --------
        FA is calculated with the following equation:

        .. math::

            FA = \sqrt{\frac{1}{2}\frac{(\lambda_1-\lambda_2)^2+(\lambda_1-
                        \lambda_3)^2+(\lambda_2-lambda_3)^2}{\lambda_1^2+
                        \lambda_2^2+\lambda_3^2} }

        """
        evals, wrap = _makearray(self.model_params[..., :3])
        ev1 = evals[..., 0]
        ev2 = evals[..., 1]
        ev3 = evals[..., 2]

        if nonans:
            all_zero = (ev1 == 0) & (ev2 == 0) & (ev3 == 0)
        else:
            all_zero = 0.

        fa = np.sqrt(0.5 * ((ev1 - ev2)**2 + (ev2 - ev3)**2 + (ev3 - ev1)**2)
                      / (ev1*ev1 + ev2*ev2 + ev3*ev3 + all_zero))

        fa = wrap(np.asarray(fa))
        return _filled(fa, fill_value)


    def md(self):
        r"""
        Mean diffusitivity (MD) calculated from cached eigenvalues.

        Returns
        ---------
        md : array (V, 1)
            Calculated MD.

        Notes
        --------
        MD is calculated with the following equation:

        .. math::

            MD = \frac{\lambda_1+\lambda_2+\lambda_3}{3}
        """
        return self.evals.mean(-1)
    

def wls_fit_tensor(design_matrix, data, min_signal=1):
    r"""
    Computes weighted least squares (WLS) fit to calculate self-diffusion 
    tensor using a linear regression model [1]_.
    
    Parameters
    ----------
    design_matrix : array (g, 7)
        Design matrix holding the covariants used to solve for the regression
        coefficients.
    data : array ([X, Y, Z, ...], g)
        Data or response variables holding the data. Note that the last 
        dimension should contain the data. It makes no copies of data.
    min_signal : default = 1
        All values below min_signal are repalced with min_signal. This is done
        in order to avaid taking log(0) durring the tensor fitting.

    Returns
    -------
    eigvals : array (..., 3)
        Eigenvalues from eigen decomposition of the tensor.
    eigvecs : array (..., 3, 3)
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
    
        1. calculate OLS estimates of the data
        2. apply the OLS estimates as weights to the WLS fit of the data 
    
    This ensured heteroscadasticity could be properly modeled for various 
    types of bootstrap resampling (namely residual bootstrap).
    
    .. math::

        y = \mathrm{data} \\
        X = \mathrm{design matrix} \\
        \hat{\beta}_\mathrm{WLS} = \mathrm{desired regression coefficients (e.g. tensor)}\\
        \\
        \hat{\beta}_\mathrm{WLS} = (X^T W X)^{-1} X^T W y \\
        \\
        W = \mathrm{diag}((X \hat{\beta}_\mathrm{OLS})^2),
        \mathrm{where} \hat{\beta}_\mathrm{OLS} = (X^T X)^{-1} X^T y

    References
    ----------
    ..  _[1] Chung, SW., Lu, Y., Henry, R.G., 2006. Comparison of bootstrap
        approaches for estimation of uncertainties of DTI parameters.
        NeuroImage 33, 531-541.
    """
    if min_signal <= 0:
        raise ValueError('min_signal must be > 0')

    data, wrap = _makearray(data)
    data_flat = data.reshape((-1, data.shape[-1]))
    dti_params = np.empty((len(data_flat), 4, 3))

    #obtain OLS fitting matrix
    #U,S,V = np.linalg.svd(design_matrix, False)
    #math: beta_ols = inv(X.T*X)*X.T*y
    #math: ols_fit = X*beta_ols*inv(y)
    #ols_fit = np.dot(U, U.T)
    ols_fit = _ols_fit_matrix(design_matrix)

    for param, sig in zip(dti_params, data_flat):
        param[0], param[1:] = _wls_iter(ols_fit, design_matrix, sig,
                                        min_signal=min_signal)
    dti_params.shape = data.shape[:-1]+(12,)
    dti_params = wrap(dti_params)
    return dti_params

def _wls_iter(ols_fit, design_matrix, sig, min_signal=1):
    '''
    Function used by wls_fit_tensor for later optimization.
    '''
    sig = np.maximum(sig, min_signal) #throw out zero signals
    log_s = np.log(sig)
    w = np.exp(np.dot(ols_fit, log_s))
    D = np.dot(np.linalg.pinv(design_matrix*w[:,None]), w*log_s)
    tensor = from_lower_triangular(D)
    return decompose_tensor(tensor)

def _ols_iter(inv_design, sig, min_signal=1):
    '''
    Function used by ols_fit_tensor for later optimization.
    '''
    sig = np.maximum(sig, min_signal) #throw out zero signals
    log_s = np.log(sig)
    D = np.dot(inv_design, log_s)
    tensor = from_lower_triangular(D)
    return decompose_tensor(tensor)


def ols_fit_tensor(design_matrix, data, min_signal=1):
    r"""
    Computes ordinary least squares (OLS) fit to calculate self-diffusion 
    tensor using a linear regression model [1]_.
    
    Parameters
    ----------
    design_matrix : array (g, 7)
        Design matrix holding the covariants used to solve for the regression
        coefficients. Use design_matrix to build a valid design matrix from 
        bvalues and a gradient table.
    data : array ([X, Y, Z, ...], g)
        Data or response variables holding the data. Note that the last 
        dimension should contain the data. It makes no copies of data.
    min_signal : default = 1
        All values below min_signal are repalced with min_signal. This is done
        in order to avaid taking log(0) durring the tensor fitting.

    Returns
    -------
    eigvals : array (..., 3)
        Eigenvalues from eigen decomposition of the tensor.
    eigvecs : array (..., 3, 3)
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
    
        \hat{\beta}_\mathrm{OLS} = (X^T X)^{-1} X^T y

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
    dti_params = np.empty((len(data_flat), 4, 3))

    #obtain OLS fitting matrix
    #U,S,V = np.linalg.svd(design_matrix, False)
    #math: beta_ols = inv(X.T*X)*X.T*y
    #math: ols_fit = X*beta_ols*inv(y)
    #ols_fit =  np.dot(U, U.T)

    inv_design = np.linalg.pinv(design_matrix)

    for param, sig in zip(dti_params, data_flat):
        param[0], param[1:] = _ols_iter(inv_design, sig, min_signal)

    dti_params.shape = data.shape[:-1]+(12,)
    dti_params = wrap(dti_params)
    return dti_params

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

_lt_indices = np.array([[0, 1, 3],
                        [1, 2, 4],
                        [3, 4, 5]])
def from_lower_triangular(D):
    """
    Returns a tensor given the six unique tensor elements

    Given the six unique tensor elments (in the order: Dxx, Dxy, Dyy, Dxz, Dyz,
    Dzz) returns a 3 by 3 tensor. All elements after the sixth are ignored.

    Parameters:
    -----------
    D : array_like, (..., >6)
        Unique elements of the tensors

    Returns:
    --------
    tensor : ndarray (..., 3, 3)
        3 by 3 tensors

    """
    return D[..., _lt_indices]

_lt_rows = np.array([0, 1, 1, 2, 2, 2])
_lt_cols = np.array([0, 0, 1, 0, 1, 2])
def lower_triangular(tensor, b0=None):
    """
    Returns the six lower triangular values of the tensor and a dummy variable
    if b0 is not None

    Parameters:
    ----------
    tensor - array_like (..., 3, 3)
        a collection of 3, 3 diffusion tensors
    b0 - float
        if b0 is not none log(b0) is returned as the dummy variable

    Returns:
    -------
    D - ndarray
        If b0 is none, then the shape will be (..., 6) otherwise (..., 7)

    """
    if tensor.shape[-2:] != (3, 3):
        raise ValueError("Diffusion tensors should be (..., 3, 3)")
    if b0 is None:
        return tensor[..., _lt_rows, _lt_cols]
    else:
        D = np.empty(tensor.shape[:-2] + (7,), dtype=tensor.dtype)
        D[..., 6] = -np.log(b0)
        D[..., :6] = tensor[..., _lt_rows, _lt_cols]
        return D

def tensor_eig_from_lo_tri(B, data):
    """Calculates parameters for creating a Tensor instance

    Calculates tensor parameters from the six unique tensor elements. This
    function can be passed to the Tensor class as a fit_method for creating a
    Tensor instance from tensors stored in a nifti file.

    Parameters:
    -----------
    B :
        not currently used
    data : array_like (..., 6)
        diffusion tensors elements stored in lower triangular order

    Returns
    -------
    dti_params
        Eigen values and vectors, used by the Tensor class to create an
        instance
    """
    data, wrap = _makearray(data)
    data_flat = data.reshape((-1, data.shape[-1]))
    dti_params = np.empty((len(data_flat), 4, 3))

    for ii in xrange(len(data_flat)):
        tensor = from_lower_triangular(data_flat[ii])
        eigvals, eigvecs = decompose_tensor(tensor)
        dti_params[ii, 0] = eigvals
        dti_params[ii, 1:] = eigvecs

    dti_params.shape = data.shape[:-1]+(12,)
    dti_params = wrap(dti_params)
    return dti_params

def decompose_tensor(tensor):
    """
    Returns eigenvalues and eigenvectors given a diffusion tensor

    Computes tensor eigen decomposition to calculate eigenvalues and
    eigenvectors of self-diffusion tensor. (Basser et al., 1994a)

    Parameters
    ----------
    D : array (3,3)
        array holding a tensor. Assumes D has units on order of
        ~ 10^-4 mm^2/s

    Returns
    -------
    eigvals : array (3,)
        Eigenvalues from eigen decomposition of the tensor. Negative
        eigenvalues are replaced by zero. Sorted from largest to smallest.
    eigvecs : array (3,3)
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
    gtab : array with shape (3,g)
        Diffusion gradient table found in DICOM header as a numpy array.
    bval : array with shape (g,)
        Diffusion weighting factor b for each vector in gtab.
    dtype : string
        Parameter to control the dtype of returned designed matrix

	Returns
	-------
	design_matrix : array (g,7)
		Design matrix or B matrix assuming Gaussian distributed tensor model.
		Note: design_matrix[j,:] = (Bxx,Byy,Bzz,Bxy,Bxz,Byz,dummy)
    """
    G = gtab
    B = np.zeros((bval.size, 7), dtype = G.dtype)
    if gtab.shape[1] != bval.shape[0]:
        raise ValueError('The number of b values and gradient directions must'
                          +' be the same')
    B[:, 0] = G[0, :] * G[0, :] * 1. * bval   #Bxx
    B[:, 1] = G[0, :] * G[1, :] * 2. * bval   #Bxy
    B[:, 2] = G[1, :] * G[1, :] * 1. * bval   #Byy
    B[:, 3] = G[0, :] * G[2, :] * 2. * bval   #Bxz
    B[:, 4] = G[1, :] * G[2, :] * 2. * bval   #Byz
    B[:, 5] = G[2, :] * G[2, :] * 1. * bval   #Bzz
    B[:, 6] = np.ones(bval.size)
    return -B

def quantize_evecs(evecs, odf_vertices=None):
    ''' Find the closest orientation of an evenly distributed sphere

    Parameters
    ----------
    evecs : ndarray
    odf_vertices : None or ndarray
        If None, then set vertices from symmetric362 sphere.  Otherwise use
        passed ndarray as vertices

    Returns
    -------
    IN : ndarray
    '''
    max_evecs=evecs[...,:,0]
    if odf_vertices==None:
        odf_vertices = get_sphere('symmetric362').vertices
    tup=max_evecs.shape[:-1]
    mec=max_evecs.reshape(np.prod(np.array(tup)),3)
    IN=np.array([np.argmin(np.dot(odf_vertices,m)) for m in mec])
    IN=IN.reshape(tup)
    return IN

class TensorStepper(object):
    """Used for tracking diffusion tensors, has a next_step method"""
    def _get_angel_limit(self):
        return np.arccos(self.dot_limit)*180/pi
    def _set_angel_limit(self, angle):
        if angle >= 0 and angle <= 90:
            self.dot_limit = cos(angle*pi/180)
        else:
            raise ValueError("angle should be between 0 and 180")
    angel_limit = property(_get_angel_limit, _set_angel_limit)

    def _get_fa_limit(self):
        return self._fa_limit
    def _set_fa_limit(self, arg):
        self._fa_limit = arg
        mask = self.fa_vol > arg
        self._interp_inst = _interpolator(self.evec1_vol, self.voxe_size, mask)
    fa_limit = property(_get_fa_limit, _set_fa_limit)

    def __init__(self, fa_vol, evec1_vol, voxel_size, interpolator,
                 fa_limit=None, angle_limit=None):
        self.voxel_size = voxel_size
        self.angle_limit = angle_limit
        if fa_vol.shape != evec1_vol.shape[:-1]:
            msg = "the fa and eigen vector volumes are not the same shape"
            raise ValueError(msg)
        if evec1_vol.shape[-1] != 3:
            msg = "eigen vector volume should have vecetors of length 3 " + \
                  "along the last dimmension"
            raise ValueError(msg)
        self.evec1_vol = evec1_vol
        self.fa_vol = fa_vol
        self._interpolator = interpolator
        #self._interp_inst is created when fa_limit is set
        self.fa_limit = fa_limit

    def next_step(location, prev_step):
        """Returns the nearest neighbor tensor for location"""
        step = self._interp_inst[location]
        angle_dot = dot(step, prev_step)
        if np.abs(angle_dot) < self.dot_limit:
            raise StopIteration
        if angle_dot > 0:
            return step
        else:
            return -step

def stepper_from_tensor(tensor, *args, **kargs):
    """stepper_from_tensor(tensor, fa_vol, evec1_vol, voxel_size, interpolator)
    """
    fa_vol = tensor.fa()
    evec1_vol = tensor.evec[..., 0]
    stepper = TensorStepper(fa_vol, evec1_vol, *args, **kargs)
    return stepper

common_fit_methods = {'WLS': wls_fit_tensor,
                      'LS': ols_fit_tensor,
                      'from lower triangular': tensor_eig_from_lo_tri,
                     }


# For backwards compatibility:
class Tensor(TensorFit, ModelArray):
    """
    For backwards compatibility, we continue to support this form of the Tensor
    fitting.
    """
    def __init__(self, data, b_values, grad_table, mask=True, thresh=None,
                 fit_method='WLS', verbose=False, *args, **kargs):
        """
        Fits a tensors to diffusion weighted data.

        """

        if not callable(fit_method):
            try:
                fit_method = common_fit_methods[fit_method]
            except KeyError:
                raise ValueError('"'+str(fit_method)+'" is not a known fit '+
                                 'method, the fit method should either be a '+
                                 'function or one of the common fit methods')

        #64 bit design matrix makes for faster pinv
        B = design_matrix(grad_table.T, b_values)
        self.B = B

        mask = np.atleast_1d(mask)
        if thresh is not None:
            #Define total mask from thresh and mask
            #mask = mask & (np.min(data[..., b_values == 0], -1) >
            #thresh)
            #the assumption that the lowest b_value is always 0 is
            #incorrect the lowest b_value could also be higher than 0
            #this is common with grid q-spaces
            min_b0_sig = np.min(data[..., b_values == b_values.min()], -1)
            mask = mask & (min_b0_sig > thresh)

        #if mask is all False
        if not mask.any():
            raise ValueError('between mask and thresh, there is no data to '+
            'fit')

        #and the mask is not all True
        if not mask.all():
            #leave only data[mask is True]
            data = data[mask]
            data = MaskedView(mask, data, fill_value=0)

        #Perform WLS fit on masked data
        dti_params = fit_method(B, data, *args, **kargs)
        self.model_params = dti_params

    # For backwards compatibility:
    D = TensorFit.quadratic_form

    def ind(self):
        """
        Quantizes eigenvectors with maximum eigenvalues  on an
        evenly distributed sphere so that the can be used for tractography.

        Returns
        ---------
        IN : array, shape(x,y,z) integer indices for the points of the
        evenly distributed sphere representing tensor  eigenvectors of
        maximum eigenvalue

        """
        return quantize_evecs(self.evecs, odf_vertices=None)
