""" Tools for using spherical homonic models to fit diffussion data

Note about the Transpose:
In the literature the matrix represenation of these methods is often writen as
Y = Bd where B is some design matrix and Y and d are column vectors. In our
case the incomming data, for example, is stored as row vectors (ndarrays) of
the form (x, y, z, n), where n is the number of diffusion directions. In this
case we can implement the method by doing something like Y' = dot(data, B.T) or
equivelently writen Y' = d.T B.T, where Y' is simply Y.T. Please forgive all
B.T and R.T, but I thought that would be easier to read than a lot of
data.reshape(...) and parmas.reshape(...).
"""

from numpy import arange, arccos, arctan2, array, atleast_1d, \
                  broadcast_arrays, c_, cos, diag, dot, empty, eye, log, \
                  maximum, pi, r_, repeat, sqrt, eye
from numpy.linalg import inv, pinv, svd
from numpy.random import randint
from scipy.special import sph_harm, lpn
from .recspeed import peak_finding_edges

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

def cartesian2polar(x=0, y=0, z=0):
    """Converts cartesian coordinates to polar coordinates

    converts a list of cartesian coordinates (x, y, z) to polar coordinates
    (R, theta, phi).

    """
    R = sqrt(x*x+y*y+z*z)
    theta = arctan2(y, x)
    phi = arccos(z)

    R, theta, phi = broadcast_arrays(R, theta, phi)

    return R, theta, phi

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
    inv = pinv(r_[B, diag(L)])
    return inv[:, :len(B)]

class OpdfModel(object):
    """A model for fitting diffussion data
    """
    def __init__(self, sh_order, bval, bvec, smooth=0, sampling_points=None,
                 sampling_edges=None):
        """Creates a model that can be used to fit and sample diffusion data

        Arguments
        ---------
        sh_order : even int
            the spherical harmonic order of the model
        bval : array_like (n,)
            the b values for the data, where n is the number of volumes in data
        bvec : array_like (3, n)
            the diffusing weighting gradient directions for the data, n is the
            number of volumes in the data
        smoothness : float between 0 and 1
            The regulization peramater of the model
        sampling_points : array_like (3, m)
            points for sampling the model, these points are used when the
            sample method is called

        """
        m, n = sph_harm_ind_list(sh_order)
        L = n*(n+1)
        legendre0 = lpn(sh_order, 0)[0]
        F = legendre0[n]
        bvec = bvec[:, bval > 0]
        x, y, z = bvec
        r, theta, phi = cartesian2polar(x, y, z)
        B = real_sph_harm(m, n, theta[:, None], phi[:, None])
        invB = smooth_pinv(B, sqrt(smooth)*L)
        L = L[:, None]
        F = F[:, None]
        delta_b = F*L*invB
        delta_q = 4*F*invB
        self._fit_matrix = delta_b, delta_q
        self._m = m
        self._n = n
        if sampling_points is not None:
            self.set_sampling_points(sampling_points, sampling_edges)

    def fit_data(self, data):
        """Fits the model to diffusion data and returns the coefficients
        """
        delta_b, delta_q = self._fit_matrix
        logd = log(data)
        return dot(data, delta_b.T) - dot(logd*(1.5-logd)*data, delta_q.T)

    def set_sampling_points(self, sampling_points, sampling_edges=None):
        """Sets the sampling points

        The sampling points are the points at which the modle is sampled when
        the sample method is called.

        Parameters
        ----------
        sampling_points : ndarray (n, 3), dtype=float
            The x, y, z coordinates of n points on a unit sphere.
        sampling_edges : ndarray (m, 2), dtype=int
            Indices to sampling_points so that every unique pair of neighbors
            in sampling_points is one of the m edges.

        """
        x, y, z = sampling_points.T
        r, theta, phi = cartesian2polar(x, y, z)
        theta = theta[:, None]
        phi = phi[:, None]
        S = real_sph_harm(self._m, self._n, theta, phi)

        delta_b, delta_q = self._fit_matrix
        delta_b = dot(S, delta_b)
        delta_q = dot(S, delta_q)
        self._sampling_matrix = delta_b, delta_q
        self.sampling_points = sampling_points
        self.sampling_edges = sampling_edges

    def sample(self, data):
        """Fits the model to diffusion data and returns samples

        The points used to sample the model can be set using the
        set_sampling_points method

        Parameters
        ----------
        data : ndarray (..., n)
            Diffusion data to be fit using the model. The data should be
            normilzed before it is fit.

        """
        delta_b, delta_q = self._sampling_matrix
        logd = log(data)
        return dot(data, delta_b.T) - dot(logd*(1.5-logd)*data, delta_q.T)

def normalize_data(data, bval, min_signal=1e-5):
    """Normalizes the data with respect to the mean b0
    """
    dwi = data[..., bval > 0]
    b0 = data[..., bval == 0].mean(-1)
    b0 = b0[..., None]
    dwi = maximum(dwi, min_signal)
    b0 = maximum(b0, min_signal)
    return dwi/b0

def gfa(samples):
    diff = samples - samples.mean(-1)[..., None]
    n = samples.shape[-1]
    numer = n*(diff*diff).sum(-1)
    denom = (n-1)*(samples*samples).sum(-1)
    return sqrt(numer/denom)

class ClosestPeakSelector(object):
    """Step selector with next_step method to be used for fiber tracking

    """
    def _get_angle_limit(self):
        return 180/pi * arccos(self.dot_limit)

    def _set_angle_limit(self, angle_limit):
        if angle_limit < 0 or angle_limit > 90:
            raise ValueError("angle_limit must be between 0 and 90")
        self.dot_limit = cos(angle_limit*pi/180)

    angle_limit = property(_get_angle_limit, _set_angle_limit)

    def __init__(self, model, data, gfa_limit=0, dot_limit=0,
                 angle_limit=None, min_relative_peak=.5, peak_spacing=.75):
        self._samples = model.sample(data)
        self._gfa = gfa(self._samples)
        self.gfa_limit = gfa_limit
        self.min_relative_peak = min_relative_peak
        self.peak_spacing = peak_spacing
        if angle_limit is not None:
            self.angle_limit = angle_limit
        else:
            self.dot_limit = dot_limit
        self.sampling_points = model.sampling_points
        self.sampling_edges = model.sampling_edges

    def next_step(self, vox_loc, prev_step):
        if self._gfa[vox_loc] < self.gfa_limit:
            return False
        vox_samples = self._samples[vox_loc]
        peak_values, peak_inds = peak_finding_edges(vox_samples,
                                                    self.sampling_edges)
        peak_points = self.sampling_points[peak_inds]
        peak_points = _robust_peaks(peak_points, peak_values,
                                    self.min_relative_peak, self.peak_spacing)
        return _closest_peak(peak_points, prev_step, self.dot_limit)

def _robust_peaks(peak_points, peak_values, min_relative_value,
                        closest_neighbor):
    """Removes peaks that are too small and child peaks too close to a parent
    peak
    """
    if peak_points.ndim == 1:
        return peak_points
    min_value = peak_values[0] * min_relative_value
    good_peaks = [peak_points[0]]
    for ii in xrange(1, len(peak_values)):
        if peak_values[ii] < min_value:
            break
        inst = peak_points[ii]
        dist = dot(good_peaks, inst)
        if abs(dist).max() < closest_neighbor:
            good_peaks.append(inst)

    return array(good_peaks)

def _closest_peak(peak_points, prev_step, dot_limit):
    """Returns peak form peak_points closest to prev_step

    Returns either the closest peak or False if dot(prev, closets) < dot_limit
    """
    peak_dots = dot(peak_points, prev_step)
    closest_peak = abs(peak_dots).argmax()
    dot_closest_peak = peak_dots[closest_peak]
    if abs(dot_closest_peak) < dot_limit:
        return False
    if dot_closest_peak > 0:
        return peak_points[closest_peak]
    else:
        return -peak_points[closest_peak]

def hat(B, bvec):
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
    """Returns a bootsrapped sample of data given H and R

    Calculated the bootsrapped
    """

    if permute is None:
        permute = randint(data.shape[-1], size=data.shape[-1])
    R = R[:, permute]
    return dot(data, (H+R).T)

def bootstrap_data_voxel(data, H, R, permute=None):
    if permute is None:
        permute = randint(data.shape[-1], size=data.shape[-1])
    r = dot(data, R.T)
    r = r[permute]
    return dot(data, H.T) + r

