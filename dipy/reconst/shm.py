""" Tools for using spherical homonic models to fit diffussion data
"""
"""
Note about the Transpose:
In the literature the matrix representation of these methods is often written
as Y = Bx where B is some design matrix and Y and x are column vectors. In our
case the input data, a dwi stored as a nifti file for example, is stored as row
vectors (ndarrays) of the form (x, y, z, n), where n is the number of diffusion
directions. We could transpose and reshape the data to be (n, x*y*z), so that
we could directly plug it into the above equation. However, I have chosen to
keep the data as is and implement the relevant equations rewritten in the
following form: Y.T = x.T B.T, or in python syntax data = np.dot(sh_coef, B.T)
where data is Y.T and sh_coef is x.T.
"""
from numpy import arange, arccos, arctan2, array, asarray, atleast_1d, \
                  broadcast_arrays, concatenate, cos, diag, dot, empty, \
                  eye, log, minimum, maximum, pi, repeat, sqrt, eye
from numpy.linalg import pinv, svd
from numpy.random import randint
from scipy.special import sph_harm, lpn
from dipy.core.geometry import cart2sphere
from .recspeed import peak_finding_onedge, _robust_peaks

def real_sph_harm(m, n, theta, phi):
    """
    Compute real spherical harmonics, where the real harmonic $Y^m_n$ is
    defined to be:
        Real($Y^m_n$) * sqrt(2) if m > 0
        $Y^m_n$                 if m == 0
        Imag($Y^m_n$) * sqrt(2) if m < 0

    This may take scalar or array arguments. The inputs will be broadcasted
    against each other.

    Parameters
    -----------
      - `m` : int |m| <= n
        The order of the harmonic.
      - `n` : int >= 0
        The degree of the harmonic.
      - `theta` : float [0, 2*pi]
        The azimuthal (longitudinal) coordinate.
      - `phi` : float [0, pi]
        The polar (colatitudinal) coordinate.

    Returns
    --------
      - `y_mn` : real float
        The real harmonic $Y^m_n$ sampled at `theta` and `phi`.

    :See also:
        scipy.special.sph_harm
    """
    m = atleast_1d(m)
    # find where m is =,< or > 0 and broadcasts to the size of the output
    m_eq0,junk,junk,junk = broadcast_arrays(m == 0, n, theta, phi)
    m_gt0,junk,junk,junk = broadcast_arrays(m > 0, n, theta, phi)
    m_lt0,junk,junk,junk = broadcast_arrays(m < 0, n, theta, phi)

    sh = sph_harm(m, n, theta, phi)
    real_sh = empty(sh.shape, 'double')
    real_sh[m_eq0] = sh[m_eq0].real
    real_sh[m_gt0] = sh[m_gt0].real * sqrt(2)
    real_sh[m_lt0] = sh[m_lt0].imag * sqrt(2)
    return real_sh

def sph_harm_ind_list(sh_order):
    """
    Returns the degree (n) and order (m) of all the symmetric spherical
    harmonics of degree less then or equal it sh_order. The results, m_list
    and n_list are kx1 arrays, where k depends on sh_order. They can be
    passed to real_sph_harm.

    Parameters
    ----------
    sh_order : int
        even int > 0, max degree to return

    Returns
    -------
    m_list : array
        orders of even spherical harmonics
    n_list : array
        degrees of even spherical hormonics

    See also
    --------
    real_sph_harm
    """
    if sh_order % 2 != 0:
        raise ValueError('sh_order must be an even integer >= 0')

    n_range = arange(0, sh_order+1, 2, dtype='int')
    n_list = repeat(n_range, n_range*2+1)

    ncoef = (sh_order + 2)*(sh_order + 1)/2
    offset = 0
    m_list = empty(ncoef, 'int')
    for ii in n_range:
        m_list[offset:offset+2*ii+1] = arange(-ii, ii+1)
        offset = offset + 2*ii + 1

    # makes the arrays ncoef by 1, allows for easy broadcasting later in code
    return (m_list, n_list)

def smooth_pinv(B, L):
    """Regularized psudo-inverse

    Computes a regularized least square inverse of B

    Parameters
    ----------
    B : array_like (n, m)
        Matrix to be inverted
    L : array_like (n,)

    Returns
    -------
    inv : ndarray (m, n)
        regularized least square inverse of B

    Notes
    -----
    In the literature this inverse is often written $(B^{T}B+L^{2})^{-1}B^{T}$.
    However here this inverse is implemented using the psudo-inverse because it
    is more numerically stable than the direct implementation of the matrix
    product.

    """
    L = diag(L)
    inv = pinv(concatenate((B, L)))
    return inv[:, :len(B)]

class SphHarmModel(object):
    @property
    def sampling_points(self):
        return self._sampling_points

    @property
    def sampling_edges(self):
        return self._sampling_edges

    def __init__(self, sh_order, bval, bvec, smooth=0, sampling_points=None,
                 sampling_edges=None):
        """Creates a model that can be used to fit or sample diffusion data

        Arguments
        ---------
        sh_order : even int >= 0
            the spherical harmonic order of the model
        bval : ndarray (n,)
            the b values for the data, where n is the number of volumes in data
        bvec : ndarray (3, n)
            the diffusing weighting gradient directions for the data, n is the
            number of volumes in the data
        smoothness : float between 0 and 1
            The regulization peramater of the model
        sampling_points : ndarray (3, m), optional
            points for sampling the model, these points are used when the
            sample method is called
        sampling_edges : ndarray (e, 2), dtype=int, optional
            Indices to sampling_points so that every unique pair of neighbors
            in sampling_points is one of the m edges.

        """
        bvec = bvec[:, bval > 0]
        m, n = sph_harm_ind_list(sh_order)
        x, y, z = bvec
        r, pol, azi = cart2sphere(x, y, z)
        B = real_sph_harm(m, n, azi[:, None], pol[:, None])
        L = -n*(n+1)
        legendre0 = lpn(sh_order, 0)[0]
        F = legendre0[n]
        self.B = B
        self._m = m
        self._n = n
        self._set_fit_matrix(B, L, F, smooth)
        if sampling_points is not None:
            self.set_sampling_points(sampling_points, sampling_edges)

    def set_sampling_points(self, sampling_points, sampling_edges=None):
        """Sets the sampling points

        The sampling points are the points at which the modle is sampled when
        the sample method is called.

        Parameters
        ----------
        sampling_points : ndarray (n, 3), dtype=float
            The x, y, z coordinates of n points on a unit sphere.
        sampling_edges : ndarray (m, 2), dtype=int, optional
            Indices to sampling_points so that every unique pair of neighbors
            in sampling_points is one of the m edges.

        """
        x, y, z = sampling_points.T
        r, pol, azi = cart2sphere(x, y, z)
        S = real_sph_harm(self._m, self._n, azi[:, None], pol[:, None])

        self._sampling_matrix = dot(S, self._fit_matrix)
        self._sampling_points = sampling_points
        self._sampling_edges = sampling_edges

    def _set_fit_matrix(self, *args):
        """Should be set in a sublcass and is called by __init__"""
        msg = "User must implement this method in a subclass"
        raise NotImplementedError(msg)

class MonoExpOpdfModel(SphHarmModel):
    """Implementaion of Solid Angle method with mono-exponential assumtion

    References
    ----------
    Aganj, I., et. al. 2009. ODF Reconstruction in Q-Ball Imaging With Solid
    Angle Consideration.
    Tristan-Vega, A., et. al. 2010. A new methodology for estimation of fiber
    populations in white matter of the brain with Funk-Radon transform.
    Decoteaux, M., et. al. 2007. Regularized, fast, and robust analytical
    Q-ball imaging.
    """

    def _set_fit_matrix(self, B, L, F, smooth):
        """The fit matrix, is used by fit_data to return the coefficients of
        the model"""
        invB = smooth_pinv(B, sqrt(smooth)*L)
        L = L[:, None]
        F = F[:, None]
        self._fit_matrix = F*L*invB

    def fit_data(self, data):
        """Fits the model to diffusion data and returns the coefficients
        """
        d = log(-log(data))
        return dot(d, self._fit_matrix.T)

    def evaluate(self, data):
        """Fits the model to diffusion data and evaluates the model at
        sampling_points

        The points used to sample the model can be set using the
        set_sampling_points method

        Parameters
        ----------
        data : ndarray (..., n)
            Diffusion data to be fit using the model. The data should be
            normilzed before it is fit.

        """
        d = log(-log(data))
        return dot(d, self._sampling_matrix.T)

class SlowAdcOpdfModel(SphHarmModel):
    """Implementaion of Tristen-Vega 2009 method with slow varying ADC
    assumption

    References
    ----------
    Aganj, I., et. al. 2009. ODF Reconstruction in Q-Ball Imaging With Solid
    Angle Consideration.
    Tristan-Vega, A., et. al. 2010. A new methodology for estimation of fiber
    populations in white matter of the brain with Funk-Radon transform.
    Decoteaux, M., et. al. 2007. Regularized, fast, and robust analytical
    Q-ball imaging.

    """
    def _set_fit_matrix(self, B, L, F, smooth):
        invB = smooth_pinv(B, sqrt(smooth)*L)
        L = L[:, None]
        F = F[:, None]
        delta_b = F*L*invB
        delta_q = 4*F*invB
        self._fit_matrix = delta_b, delta_q

    def set_sampling_points(self, sampling_points, sampling_edges=None):
        """Sets the sampling points

        The sampling points are the points at which the modle is sampled when
        the sample method is called.

        Parameters
        ----------
        sampling_points : ndarray (n, 3), dtype=float
            The x, y, z coordinates of n points on a unit sphere.
        sampling_edges : ndarray (m, 2), dtype=int, optional
            Indices to sampling_points so that every unique pair of neighbors
            in sampling_points is one of the m edges.

        """
        x, y, z = sampling_points.T
        r, pol, azi = cart2sphere(x, y, z)
        S = real_sph_harm(self._m, self._n, azi[:, None], pol[:, None])

        delta_b, delta_q = self._fit_matrix
        delta_b = dot(S, delta_b)
        delta_q = dot(S, delta_q)
        self._sampling_matrix = delta_b, delta_q
        self._sampling_points = sampling_points
        self._sampling_edges = sampling_edges

    def fit_data(self, data):
        """The fit matrix, is used by fit_data to return the coefficients of
        the model"""
        delta_b, delta_q = self._fit_matrix
        return _slowadc_formula(data, delta_b, delta_q)

    def evaluate(self, data):
        """Fits the model to diffusion data and evaluates the model at
        sampling_points

        The points used to sample the model can be set using the
        set_sampling_points method

        Parameters
        ----------
        data : ndarray (..., n)
            Diffusion data to be fit using the model. The data should be
            normilzed before it is fit.

        """
        delta_b, delta_q = self._sampling_matrix
        return _slowadc_formula(data, delta_b, delta_q)

def _slowadc_formula(data, delta_b, delta_q):
    """formula used in SlowAdcOpdfModel"""
    logd = -log(data)
    return dot(logd*(1.5-logd)*data, delta_q.T) - dot(data, delta_b.T)

class QballOdfModel(SphHarmModel):
    """Implementaion Qball Odf Model
    """

    def _set_fit_matrix(self, B, L, F, smooth):
        invB = smooth_pinv(B, sqrt(smooth)*L)
        F = F[:, None]
        self._fit_matrix = F*invB

    def fit_data(self, data):
        """Fits the model to diffusion data and returns the coefficients
        """
        return dot(data, self._fit_matrix.T)

    def evaluate(self, data):
        """Fits the model to diffusion data and returns samples

        The points used to sample the model can be set using the
        set_sampling_points method

        Parameters
        ----------
        data : array_like, shape (..., n)
            Diffusion data to be fit using the model. The data should have n
            diffusion weighed signals along the last dimenssion

        """
        return dot(data, self._sampling_matrix.T)

def normalize_data(data, bval, min_signal=1e-5, mask=None):
    """Normalizes the data with respect to the mean b0
    """
    if min_signal <= 0:
        raise ValueError("min_signal must be > 0")
    where_b0 = bval == 0
    if len(where_b0) != data.shape[-1]:
        message = "number of bvalues does not match number of input signals"
        raise ValueError(message)
    if not where_b0.any():
        raise ValueError("data must contain at least one image set with no "+
                         "diffusion weighting")
    elif where_b0.all():
        raise ValueError("data must contain at least one dwi set")

    dwi = data[..., ~where_b0]
    dwi = asarray(dwi, 'float')
    b0 = data[..., where_b0].mean(-1)
    b0 = b0[..., None]
    b0 = asarray(b0, 'float')

    maximum(dwi, min_signal, dwi)
    maximum(b0, min_signal, b0)
    dwi /= b0
    minimum(dwi, 1, dwi)
    return dwi

def gfa(samples):
    """gfa of some function from a set of samples os that function"""
    diff = samples - samples.mean(-1)[..., None]
    n = samples.shape[-1]
    numer = n*(diff*diff).sum(-1)
    denom = (n-1)*(samples*samples).sum(-1)
    return sqrt(numer/denom)

def _closest_peak(peak_points, prev_step, dot_limit):
    """Returns peak form peak_points closest to prev_step

    Returns either the closest peak or None if dot(prev, closets) < dot_limit
    """
    peak_dots = dot(peak_points, prev_step)
    closest_peak = abs(peak_dots).argmax()
    dot_closest_peak = peak_dots[closest_peak]
    if abs(dot_closest_peak) < dot_limit:
        raise StopIteration("angle between peaks too large")
    if dot_closest_peak > 0:
        return peak_points[closest_peak]
    else:
        return -peak_points[closest_peak]

def hat(B):
    """Returns the hat matrix for the design matrix B
    """

    U, S, V = svd(B, False)
    H = dot(U, U.T)
    return H

def lcr_matrix(H):
    """Returns a matrix for computing leveraged, centered residuals from data

    if r = (d-Hd), the leveraged centered residuals are lcr = (r/l)-mean(r/l)
    ruturns the matrix R, such lcr = Rd

    """
    if H.ndim != 2 or H.shape[0] != H.shape[1]:
        raise ValueError('H should be a square matrix')

    leverages = sqrt(1-H.diagonal())
    leverages = leverages[:, None]
    R = (eye(len(H)) - H) / leverages
    return R - R.mean(0)

def bootstrap_data_array(data, H, R, permute=None):
    """Applies the Residual Bootstraps to the data given H and R

    data must be normalized, ie 0 < data <= 1

    This function, and the bootstrap_data_voxel function, calculat
    residual-bootsrap samples given a Hat matrix and a Residual matrix. These
    samples can be used for non-parametric statistics or for bootstrap
    probabilistic tractography:

    References:
    -----------
    J. I. Berman, et al., "Probabilistic streamline q-ball tractography using
        the residual bootstrap" 2008
    HA Haroon, et al., "Using the model-based residual bootstrap to quantify
        uncertainty in fiber orientations from Q-ball analysis" 2009
    B. Jeurissen, et al., "Probabilistic Fiber Tracking Using the Residual
        Bootstrap with Constrained Spherical Deconvolution" 2011
    """

    if permute is None:
        permute = randint(data.shape[-1], size=data.shape[-1])
    assert R.shape == H.shape
    assert len(permute) == R.shape[-1]
    R = R[permute]
    data = dot(data, (H+R).T)
    return data

def bootstrap_data_voxel(data, H, R, permute=None):
    """Like bootstrap_data_array but faster when for a single voxel

    data must be 1d and normalized
    """
    if permute is None:
        permute = randint(data.shape[-1], size=data.shape[-1])
    r = dot(data, R.T)
    r = r[permute]
    d = dot(data, H.T) + r
    return d

class ResidualBootstrapWrapper(object):
    """Returns a residual bootstrap sample of the signal_object when indexed

    Wraps a signal_object, this signal object can be an interpolator. When
    indexed, the the wrapper indexes the signal_object to get the signal.
    There wrapper than samples the residual boostrap distribution of signal and
    returns that sample.
    """
    def __init__(self, signal_object, B, min_signal=0):
        """Builds a ResidualBootstrapWapper

        Given some linear model described by B, the design matrix, and a
        signal_object, returns an object which can sample the residual
        bootstrap distribution of the signal. We assume that the signals are
        normalized so we clip the bootsrap samples to be between min_signal and
        1.

        Parameters
        ----------
        signal_object : some object that can be indexed
            This object should return diffusion weighted signals when indexed.
        B : ndarray, ndim=2
            The design matrix of spherical hormonic model usded to fit the
            data. This is the model that will be used to compute the residuals
            and sample the residual bootstrap distribution
        min_signal : float
            The lowest allowable signal.
        """
        self._signal_object = signal_object
        self._H = hat(B)
        self._R = lcr_matrix(self._H)
        self._min_signal = min_signal

    def __getitem__(self, index):
        """Indexes self._singal_object and bootsraps the result"""
        d = self._signal_object[index]
        d = bootstrap_data_voxel(d, self._H, self._R)
        d.clip(self._min_signal, 1., d)
        return d

class ClosestPeakSelector(object):
    """Step selector with next_step method to be used for fiber tracking

    Attributes:
    -----------
    angle_limit : float, 0 <= angle_limit <= 90
        angle_limit is used when next_step is called, see also dot_limit.
    dot_limit : float, 0 <= dot_limit <= 1
        Same as cos(angle_limit), chaning dot_limit will change angle_limit
        and vice versa
    min_relative_peak : float, 0 <= min_relative_peak < 1
        Peaks smaller than min_relative_peak of the largest peak are assumed to
        be artifacts and ignored
    peak_spacing : float, 0 <= peak_spacing <= 1
        The minimum spacing between neighboring peaks, spacing_angle is
        arccos(peak_spacing). If two peaks are less than spacing_angle appart
        it is assumed to be an artifact and only the greater of the two peaks
        is treated as a true peak.

    """
    def _get_angle_limit(self):
        return 180/pi * arccos(self.dot_limit)

    def _set_angle_limit(self, angle_limit):
        if angle_limit < 0 or angle_limit > 90:
            raise ValueError("angle_limit must be between 0 and 90")
        self.dot_limit = cos(angle_limit*pi/180)

    angle_limit = property(_get_angle_limit, _set_angle_limit)

    def __init__(self, model, interpolator, angle_limit=None, dot_limit=0, \
                 min_relative_peak=.5, peak_spacing=.75):
        """Creates a peakfinder which can be used to get next_step

        Parameters:
        -----------
        model : must have evaluate method
            A model used to fit data
        interpolator : must be indexable
            An object which returns diffusion weighted data when indexed
        angle_limit : float
            see angle_limit attribute
        dot_limit : float
            see dot_limit attribute
        min_relative_peak : float
            see min_relative_peak attritube
        peak_spacing : float
            see peak_spacing attribute
        """

        self._interpolator = interpolator
        self._model = model
        self.min_relative_peak = min_relative_peak
        self.peak_spacing = peak_spacing
        if angle_limit is not None:
            self.angle_limit = angle_limit
        else:
            self.dot_limit = dot_limit

    def next_step(self, location, prev_step):
        """Returns the peak closest to prev_step at location

        Fits the data from location using model and evaluates that model on the
        surface of a sphere. Then the point on the sphere which is both a
        local maxima and closest to prev_step is returned.

        Parameters
        ----------
        location : point in space
            location is passed to the interpolator in order to get data
        prev_step: array_like (3,)
            the direction of the previous tracking step

        """
        vox_data = self._interpolator[location]

        sampling_points = self._model.sampling_points
        sampling_edges = self._model.sampling_edges
        samples = self._model.evaluate(vox_data)

        peak_values, peak_inds = peak_finding_onedge(samples, sampling_edges)
        peak_points = sampling_points[peak_inds]
        peak_points = _robust_peaks(peak_points, peak_values,
                                    self.min_relative_peak, self.peak_spacing)
        step = _closest_peak(peak_points, prev_step, self.dot_limit)
        return step
