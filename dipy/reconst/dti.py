#!/usr/bin/python
import warnings
import numpy as np
from .maskedview import MaskedView, _makearray, _filled
from .modelarray import ModelArray
from ..data import get_sphere
from ..core.geometry import vector_norm
from .vec_val_sum import vec_val_vect
from ..core.onetime import auto_attr


def fractional_anisotropy(evals, axis=-1):
    r"""
    Fractional anisotropy (FA) of a diffusion tensor.

    Parameters
    ----------
    evals : array-like
        Eigenvalues of a diffusion tensor.
    axis : int
        Axis of `evals` which contains 3 eigenvalues.

    Returns
    -------
    fa : array
        Calculated FA. Range is 0 <= FA <= 1.

    Notes
    --------
    FA is calculated using the following equation:

    .. math::

        FA = \sqrt{\frac{1}{2}\frac{(\lambda_1-\lambda_2)^2+(\lambda_1-
                    \lambda_3)^2+(\lambda_2-\lambda_3)^2}{\lambda_1^2+
                    \lambda_2^2+\lambda_3^2}}

    """
    evals = np.rollaxis(evals, axis)
    if evals.shape[0] != 3:
        msg = "Expecting 3 eigenvalues, got {}".format(evals.shape[0])
        raise ValueError(msg)

    # Make sure not to get nans
    all_zero = (evals == 0).all(axis=0)
    ev1, ev2, ev3 = evals
    fa = np.sqrt(0.5 * ((ev1 - ev2)**2 + (ev2 - ev3)**2 + (ev3 - ev1)**2)
                  / ((evals*evals).sum(0) + all_zero))

    return fa


def mean_diffusivity(evals, axis=-1):
    r"""
    Mean Diffusivity (MD) of a diffusion tensor. Also, called
    Apparent diffusion coefficient (ADC)

    Parameters
    ----------
    evals : array-like
        Eigenvalues of a diffusion tensor.
    axis : int
        Axis of `evals` which contains 3 eigenvalues.

    Returns
    -------
    md : array
        Calculated MD. 

    Notes
    --------
    MD is calculated with the following equation:

    .. math::

        MD = \frac{\lambda_1 + \lambda_2 + \lambda_3}{3}

    """
    evals = np.rollaxis(evals, axis)
    if evals.shape[0] != 3:
        msg = "Expecting 3 eigenvalues, got {}".format(evals.shape[0])
        raise ValueError(msg)

    ev1, ev2, ev3 = evals
    return (ev1 + ev2 + ev3) / 3



def axial_diffusivity(evals, axis=-1):
    r"""
    Axial Diffusivity (AD) of a diffusion tensor.
    Also called parallel diffusivity. 
    
    Parameters
    ----------
    evals : array-like
        Eigenvalues of a diffusion tensor.
    axis : int
        Axis of `evals` which contains 3 eigenvalues.

    Returns
    -------
    ad : array
        Calculated AD. 

    Notes
    --------
    AD is calculated with the following equation:

    .. math::

        AD = \lambda_1

    """
    evals = np.rollaxis(evals, axis)
    if evals.shape[0] != 3:
        msg = "Expecting 3 eigenvalues, got {}".format(evals.shape[0])
        raise ValueError(msg)

    ev1, ev2, ev3 = evals
    return ev1


def radial_diffusivity(evals, axis=-1):
    r"""
    Radial Diffusivity (RD) of a diffusion tensor.
    Also called perpendicular diffusivity.
    
    Parameters
    ----------
    evals : array-like
        Eigenvalues of a diffusion tensor.
    axis : int
        Axis of `evals` which contains 3 eigenvalues.

        Returns
    -------
    rd : array
        Calculated RD. 

    Notes
    --------
    RD is calculated with the following equation:

    .. math::

        RD = \frac{\lambda_2 + \lambda_3}{2}

    """
    evals = np.rollaxis(evals, axis)
    if evals.shape[0] != 3:
        msg = "Expecting 3 eigenvalues, got {}".format(evals.shape[0])
        raise ValueError(msg)

    ev1, ev2, ev3 = evals
    return (ev2 + ev3) / 2


def trace(evals, axis=-1):
    r"""
    Trace of a diffusion tensor.

    Parameters
    ----------
    evals : array-like
        Eigenvalues of a diffusion tensor.
    axis : int
        Axis of `evals` which contains 3 eigenvalues.

    Returns
    -------
    trace : array
        Calculated trace of the diffusion tensor. 

    Notes
    --------
    Trace is calculated with the following equation:

    .. math::

        MD = \lambda_1 + \lambda_2 + \lambda_3

    """
    evals = np.rollaxis(evals, axis)
    if evals.shape[0] != 3:
        msg = "Expecting 3 eigenvalues, got {}".format(evals.shape[0])
        raise ValueError(msg)

    ev1, ev2, ev3 = evals
    return (ev1 + ev2 + ev3)


def color_fa(fa, evecs):
    r""" Color fractional anisotropy of diffusion tensor

    Parameters
    ----------
    fa : array-like
        Array of the fractional anisotropy (can be 1D, 2D or 3D)

    evecs : array-like
        eigen vectors from the tensor model

    Returns
    -------
    rgb : Array with 3 channels for each color as the last dimension.
        Colormap of the FA with red for the x value, y for the green
        value and z for the blue value.

    Notes
    -----

    it is computed from the clipped FA between 0 and 1 using the following
    formula

    .. math::

        rgb = abs(max(eigen_vector)) \times fa
    """
    if (fa.shape != evecs[..., 0, 0].shape) or ((3, 3) != evecs.shape[-2:]):
        raise ValueError("Wrong number of dimensions for evecs")

    fa = np.clip(fa, 0, 1)
    rgb = np.abs(evecs[..., 0]) * fa[..., None]
    return rgb


class TensorModel(object):
    """ Diffusion Tensor
    """
    def __init__(self, gtab, fit_method="WLS", *args, **kwargs):
        """ A Diffusion Tensor Model [1]_, [2]_.

        Parameters
        ----------
        gtab : GradientTable
        fit_method : str or callable
            str can be one of the following:
            'WLS' for weighted least squares
                dti.wls_fit_tensor
            'LS' for ordinary least squares
                dti.ols_fit_tensor

            callable has to have the signature:
              fit_method(design_matrix, data, *args, **kwargs)

        args, kwargs : arguments and key-word arguments passed to the
           fit_method. See dti.wls_fit_tensor, dti.ols_fit_tensor for details

        References
        ----------
        .. [1] Basser, P.J., Mattiello, J., LeBihan, D., 1994. Estimation of
           the effective self-diffusion tensor from the NMR spin echo. J Magn
           Reson B 103, 247-254.
        .. [2] Basser, P., Pierpaoli, C., 1996. Microstructural and
           physiological features of tissues elucidated by quantitative
           diffusion-tensor MRI.  Journal of Magnetic Resonance 111, 209-219.

        """
        if not callable(fit_method):
            try:
                self.fit_method = common_fit_methods[fit_method]
            except KeyError:
                raise ValueError('"'+str(fit_method)+'" is not a known fit '
                                 'method, the fit method should either be a '
                                 'function or one of the common fit methods')
        self.bvec = gtab.bvecs
        self.bval = gtab.bvals
        self.design_matrix = design_matrix(self.bvec.T, self.bval)
        self.args = args
        self.kwargs = kwargs


    def fit(self, data, mask=None):
        """ Fit method of the DTI model class

        Parameters
        ----------
        data : array
            The measured signal from one voxel.

        mask : array
            A boolean array used to mark the coordinates in the data that
            should be analyzed that has the shape data.shape[-1]
        """
        # If a mask is provided, we will use it to access the data
        if mask is not None:
            # Make sure it's boolean, so that it can be used to mask
            mask = np.array(mask, dtype=bool, copy=False)
            data_in_mask = data[mask]
        else:
            data_in_mask = data

        params_in_mask = self.fit_method(self.design_matrix, data_in_mask,
                                         *self.args, **self.kwargs)

        dti_params = np.zeros(data.shape[:-1] + (12,))

        dti_params[mask,:] = params_in_mask

        return TensorFit(self, dti_params)


class TensorFit(object):
    def __init__(self, model, model_params):
        """ Initialize a TensorFit class instance.
        """
        self.model = model
        self.model_params = model_params


    def __getitem__(self, index):
        model_params = self.model_params
        N = model_params.ndim
        if type(index) is not tuple:
            index = (index,)
        elif len(index) >= model_params.ndim:
            raise IndexError("IndexError: invalid index")
        index = index + (slice(None),) * (N - len(index))
        return type(self)(self.model, model_params[index])


    @property
    def shape(self):
        return self.model_params.shape[:-1]


    @property
    def directions(self):
        """
        For tracking - return the primary direction in each voxel
        """
        return self.evecs[..., None, :, 0]


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


    @property
    def quadratic_form(self):
        """Calculates the 3x3 diffusion tensor for each voxel"""
        # do `evecs * evals * evecs.T` where * is matrix multiply
        # einsum does this with:
        # np.einsum('...ij,...j,...kj->...ik', evecs, evals, evecs)
        return vec_val_vect(self.evecs, self.evals)


    def lower_triangular(self, b0=None):
        return lower_triangular(self.quadratic_form, b0)


    @auto_attr
    def fa(self):
        """Fractional anisotropy (FA) calculated from cached eigenvalues."""
        return fractional_anisotropy(self.evals)


    @auto_attr
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
        return self.trace/3.0
    
    @auto_attr
    def rd(self):
        r"""
        Radial diffusitivity (RD) calculated from cached eigenvalues.

        Returns
        ---------
        rd : array (V, 1)
            Calculated RD.

        Notes
        --------
        RD is calculated with the following equation:

        .. math::

          RD = \frac{\lambda_2 + \lambda_3}{2}


        """
        return radial_diffusivity(self.evals)


    @auto_attr
    def ad(self):
        r"""
        Radial diffusitivity (RD) calculated from cached eigenvalues.

        Returns
        ---------
        ad : array (V, 1)
            Calculated AD.

        Notes
        --------
        RD is calculated with the following equation:

        .. math::

          AD = \lambda_1


        """
        return axial_diffusivity(self.evals)
    
    @auto_attr
    def trace(self):
        r"""
        Trace of the tensor calculated from cached eigenvalues.

        Returns
        ---------
        trace : array (V, 1)
            Calculated trace.

        Notes
        --------
        The trace is calculated with the following equation:

        .. math::

          trace = \lambda_1 + \lambda_2 + \lambda_3
        """
        return trace(self.evals)

    def odf(self, sphere):
        lower = 4 * np.pi * np.sqrt(np.prod(self.evals, -1))
        projection = np.dot(sphere.vertices, self.evecs)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            projection /=  np.sqrt(self.evals)
            odf = (vector_norm(projection) ** -3) / lower
        # Zero evals are non-physical, we replace nans with zeros
        any_zero = (self.evals == 0).any(-1)
        odf = np.where(any_zero, 0, odf)
        # Move odf to be on the last dimension
        odf = np.rollaxis(odf, 0, odf.ndim)
        return odf


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
    .. [1] Chung, SW., Lu, Y., Henry, R.G., 2006. Comparison of bootstrap
       approaches for estimation of uncertainties of DTI parameters.
       NeuroImage 33, 531-541.
    """
    tol = 1e-6
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
    min_diffusivity = tol / -design_matrix.min()

    for param, sig in zip(dti_params, data_flat):
        param[0], param[1:] = _wls_iter(ols_fit, design_matrix, sig,
                                        min_signal, min_diffusivity)
    dti_params.shape = data.shape[:-1]+(12,)
    dti_params = wrap(dti_params)
    return dti_params


def _wls_iter(ols_fit, design_matrix, sig, min_signal, min_diffusivity):
    ''' Function used by wls_fit_tensor for later optimization.
    '''
    sig = np.maximum(sig, min_signal) #throw out zero signals
    log_s = np.log(sig)
    w = np.exp(np.dot(ols_fit, log_s))
    D = np.dot(np.linalg.pinv(design_matrix * w[:,None]), w*log_s)
    # D, _, _, _ = np.linalg.lstsq(design_matrix * w[:, None], log_s)
    tensor = from_lower_triangular(D)
    return decompose_tensor(tensor, min_diffusivity=min_diffusivity)


def _ols_iter(inv_design, sig, min_signal, min_diffusivity):
    ''' Function used by ols_fit_tensor for later optimization.
    '''
    sig = np.maximum(sig, min_signal) #throw out zero signals
    log_s = np.log(sig)
    D = np.dot(inv_design, log_s)
    tensor = from_lower_triangular(D)
    return decompose_tensor(tensor, min_diffusivity=min_diffusivity)


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
    tol = 1e-6

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

    min_diffusivity = tol / -design_matrix.min()
    inv_design = np.linalg.pinv(design_matrix)

    for param, sig in zip(dti_params, data_flat):
        param[0], param[1:] = _ols_iter(inv_design, sig, min_signal, min_diffusivity)

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
    """ Returns a tensor given the six unique tensor elements

    Given the six unique tensor elments (in the order: Dxx, Dxy, Dyy, Dxz, Dyz,
    Dzz) returns a 3 by 3 tensor. All elements after the sixth are ignored.

    Parameters
    -----------
    D : array_like, (..., >6)
        Unique elements of the tensors

    Returns
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

    Parameters
    ----------
    tensor : array_like (..., 3, 3)
        a collection of 3, 3 diffusion tensors
    b0 : float
        if b0 is not none log(b0) is returned as the dummy variable

    Returns
    -------
    D : ndarray
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


def tensor_eig_from_lo_tri(data):
    """Calculates parameters for creating a Tensor instance

    Calculates tensor parameters from the six unique tensor elements. This
    function can be passed to the Tensor class as a fit_method for creating a
    Tensor instance from tensors stored in a nifti file.

    Parameters
    ----------
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


def decompose_tensor(tensor, min_diffusivity=0):
    """ Returns eigenvalues and eigenvectors given a diffusion tensor

    Computes tensor eigen decomposition to calculate eigenvalues and
    eigenvectors (Basser et al., 1994a).

    Parameters
    ----------
    tensor : array (3, 3)
        Hermitian matrix representing a diffusion tensor.
    min_diffusivity : float
        Because negative eigenvalues are not physical and small eigenvalues,
        much smaller than the diffusion weighting, cause quite a lot of noise
        in metrics such as fa, diffusivity values smaller than
        `min_diffusivity` are replaced with `min_diffusivity`.

    Returns
    -------
    eigvals : array (3,)
        Eigenvalues from eigen decomposition of the tensor. Negative
        eigenvalues are replaced by zero. Sorted from largest to smallest.
    eigvecs : array (3, 3)
        Associated eigenvectors from eigen decomposition of the tensor.
        Eigenvectors are columnar (e.g. eigvecs[:,j] is associated with
        eigvals[j])

    """
    #outputs multiplicity as well so need to unique
    eigenvals, eigenvecs = np.linalg.eigh(tensor)

    #need to sort the eigenvalues and associated eigenvectors
    order = eigenvals.argsort()[::-1]
    eigenvecs = eigenvecs[:, order]
    eigenvals = eigenvals[order]

    eigenvals = eigenvals.clip(min=min_diffusivity)
    # eigenvecs: each vector is columnar

    return eigenvals, eigenvecs


def design_matrix(gtab, bval, dtype=None):
    """  Constructs design matrix for DTI weighted least squares or
    least squares fitting. (Basser et al., 1994a)

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
		Design matrix or B matrix assuming Gaussian distributed tensor model
		design_matrix[j,:] = (Bxx,Byy,Bzz,Bxy,Bxz,Byz,dummy)
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
    """ Find the closest orientation of an evenly distributed sphere

    Parameters
    ----------
    evecs : ndarray
    odf_vertices : None or ndarray
        If None, then set vertices from symmetric362 sphere.  Otherwise use
        passed ndarray as vertices

    Returns
    -------
    IN : ndarray
    """
    max_evecs=evecs[...,:,0]
    if odf_vertices==None:
        odf_vertices = get_sphere('symmetric362').vertices
    tup=max_evecs.shape[:-1]
    mec=max_evecs.reshape(np.prod(np.array(tup)),3)
    IN=np.array([np.argmin(np.dot(odf_vertices,m)) for m in mec])
    IN=IN.reshape(tup)
    return IN

common_fit_methods = {'WLS': wls_fit_tensor,
                      'LS': ols_fit_tensor,
                      'OLS': ols_fit_tensor,
                     }


# For backwards compatibility:
class Tensor(ModelArray, TensorFit):
    """
    For backwards compatibility, we continue to support this form of the Tensor
    fitting.

    """
    def __init__(self, data, b_values, b_vectors, mask=True, thresh=None,
                 fit_method='WLS', verbose=False, *args, **kargs):
        """ Fits tensors to diffusion weighted data.

        Fits a diffusion tensor given diffusion-weighted signals and gradient
        info. Tensor object that when initialized calculates single self
        diffusion tensor [1]_ in each voxel using selected fitting algorithm
        (DEFAULT: weighted least squares [2]_) Requires a given b-vector table,
        b value for each diffusion-weighted gradient vector, and image data
        given all as arrays.

        Parameters
        ----------
        data : array ([X, Y, Z, ...], g)
            Diffusion-weighted signals. The dimension corresponding to the
            diffusion weighting must be the last dimension

        bval : array (g,)
            Diffusion weighting factor b for each vector in gtab.

        bvec : array (g, 3)
            Diffusion gradient table found in DICOM header as a array.

        mask : array, optional
            The tensor will only be fit where mask is True. Mask must
            broadcast to the shape of data and must have fewer dimensions than
            data

        thresh : float, default = None
            The tensor will not be fit where data[bval == 0] < thresh. If
            multiple b0 volumes are given, the minimum b0 signal is used.

        fit_method : funciton or string, default = 'WLS'
            The method to be used to fit the given data to a tensor. Any
            function that takes the B matrix and the data and returns eigen
            values and eigen vectors can be passed as the fit method. Any of
            the common fit methods can be passed as a string.

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

        Examples
        ----------
        For a complete example have a look at the main dipy/examples folder

        """
        warnings.warn("This implementation of DTI will be deprecated in a future release, consider using TensorModel", DeprecationWarning)
        if not callable(fit_method):
            try:
                fit_method = common_fit_methods[fit_method]
            except KeyError:
                raise ValueError('"'+str(fit_method)+'" is not a known fit '+
                                 'method, the fit method should either be a '+
                                 'function or one of the common fit methods')

        #64 bit design matrix makes for faster pinv
        B = design_matrix(b_vectors.T, b_values)
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
            Calculated FA. Range is 0 <= FA <= 1.

        Notes
        --------
        FA is calculated with the following equation:

        .. math::

            FA = \sqrt{\frac{1}{2}\frac{(\lambda_1-\lambda_2)^2+(\lambda_1-
                        \lambda_3)^2+(\lambda_2-\lambda_3)^2}{\lambda_1^2+
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
