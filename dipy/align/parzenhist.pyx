#!python
#cython: boundscheck=False
#cython: wraparound=False
#cython: cdivision=True

import numpy as np
cimport numpy as cnp
cimport cython
import numpy.random as random
from dipy.align.fused_types cimport floating
from dipy.align import vector_fields as vf

from dipy.align.vector_fields cimport(_apply_affine_3d_x0,
                                      _apply_affine_3d_x1,
                                      _apply_affine_3d_x2,
                                      _apply_affine_2d_x0,
                                      _apply_affine_2d_x1)

from dipy.align.transforms cimport (Transform)

cdef extern from "dpy_math.h" nogil:
    double cos(double)
    double sin(double)
    double log(double)

class ParzenJointHistogram(object):
    def __init__(self, nbins):
        r""" Computes joint histogram and derivatives with Parzen windows

        Base class to compute joint and marginal probability density
        functions and their derivatives with respect to a transform's
        parameters. The smooth histograms are computed by using Parzen
        windows [Parzen62] with a cubic spline kernel, as proposed by
        Mattes et al. [Mattes03]. This implementation is not tied to any
        optimization (registration) method, the idea is that
        information-theoretic matching functionals (such as Mutual
        Information) can inherit from this class to perform the low-level
        computations of the joint intensity distributions and its gradient
        w.r.t. the transform parameters. The derived class can then compute
        the similarity/dissimilarity measure and gradient, and finally
        communicate the results to the appropriate optimizer.

        Parameters
        ----------
        nbins : int
            the number of bins of the joint and marginal probability density
            functions (the actual number of bins of the joint PDF is nbins**2)

        References
        ----------
        [Parzen62] E. Parzen. On the estimation of a probability density
                   function and the mode. Annals of Mathematical Statistics,
                   33(3), 1065-1076, 1962.
        [Mattes03] Mattes, D., Haynor, D. R., Vesselle, H., Lewellen, T. K.,
                   & Eubank, W. PET-CT image registration in the chest using
                   free-form deformations. IEEE Transactions on Medical
                   Imaging, 22(1), 120-8, 2003.

        Notes
        -----
        We need this class in cython to allow _joint_pdf_gradient_dense_2d and
        _joint_pdf_gradient_dense_3d to use a nogil Jacobian function (obtained
        from an instance of the Transform class), which allows us to evaluate
        Jacobians at all the sampling points (maybe the full grid) inside a
        nogil loop.

        The reason we need a class is to encapsulate all the parameters related
        to the joint and marginal distributions.
        """
        self.nbins = nbins
        # Since the kernel used to compute the Parzen histogram covers more
        # than one bin, we need to add extra bins to both sides of the
        # histogram to account for the contributions of the minimum and maximum
        # intensities. Padding is the number of extra bins used at each side
        # of the histogram (a total of [2 * padding] extra bins). Since the
        # support of the cubic spline is 5 bins (the center plus 2 bins at each
        # side) we need a padding of 2, in the case of cubic splines.
        self.padding = 2
        self.setup_called = False

    def setup(self, static, moving, smask=None, mmask=None):
        r""" Compute histogram settings to store the PDF of input images

        Parameters
        ----------
        static : array
            static image
        moving : array
            moving image
        smask : array
            mask of static object being registered (a binary array with 1's
            inside the object of interest and 0's along the background).
            If None, the behaviour is equivalent to smask=ones_like(static)
        mmask : array
            mask of moving object being registered (a binary array with 1's
            inside the object of interest and 0's along the background).
            If None, the behaviour is equivalent to mmask=ones_like(static)
        """

        if smask is None:
            smask = np.ones_like(static)
        if mmask is None:
            mmask = np.ones_like(moving)

        self.smin = np.min(static[smask != 0])
        self.smax = np.max(static[smask != 0])
        self.mmin = np.min(moving[mmask != 0])
        self.mmax = np.max(moving[mmask != 0])

        self.sdelta = (self.smax - self.smin) / (self.nbins - 2 * self.padding)
        self.mdelta = (self.mmax - self.mmin) / (self.nbins - 2 * self.padding)
        self.smin = self.smin / self.sdelta - self.padding
        self.mmin = self.mmin / self.mdelta - self.padding

        self.joint_grad = None
        self.metric_grad = None
        self.metric_val = 0
        self.joint = np.zeros(shape=(self.nbins, self.nbins))
        self.smarginal = np.zeros(shape=(self.nbins,), dtype=np.float64)
        self.mmarginal = np.zeros(shape=(self.nbins,), dtype=np.float64)

        self.setup_called = True

    def bin_normalize_static(self, x):
        r""" Maps intensity x to the range covered by the static histogram

        If the input intensity is in [self.smin, self.smax] then the normalized
        intensity will be in [self.padding, self.nbins - self.padding]

        Parameters
        ----------
        x : float
            the intensity to be normalized

        Returns
        -------
        xnorm : float
            normalized intensity to the range covered by the static histogram
        """
        return _bin_normalize(x, self.smin, self.sdelta)

    def bin_normalize_moving(self, x):
        r""" Maps intensity x to the range covered by the moving histogram

        If the input intensity is in [self.mmin, self.mmax] then the normalized
        intensity will be in [self.padding, self.nbins - self.padding]

        Parameters
        ----------
        x : float
            the intensity to be normalized

        Returns
        -------
        xnorm : float
            normalized intensity to the range covered by the moving histogram
        """
        return _bin_normalize(x, self.mmin, self.mdelta)

    def bin_index(self, xnorm):
        r""" Bin index associated with the given normalized intensity

        The return value is an integer in [padding, nbins - 1 - padding]

        Parameters
        ----------
        xnorm : float
            intensity value normalized to the range covered by the histogram

        Returns
        -------
        bin : int
            the bin index associated with the given normalized intensity
        """
        return _bin_index(xnorm, self.nbins, self.padding)

    def update_pdfs_dense(self, static, moving, smask=None, mmask=None):
        r""" Computes the Probability Density Functions of two images

        The joint PDF is stored in self.joint. The marginal distributions
        corresponding to the static and moving images are computed and
        stored in self.smarginal and self.mmarginal, respectively.

        Parameters
        ----------
        static : array, shape (S, R, C)
            static image
        moving : array, shape (S, R, C)
            moving image
        smask : array, shape (S, R, C)
            mask of static object being registered (a binary array with 1's
            inside the object of interest and 0's along the background).
            If None, ones_like(static) is used as mask.
        mmask : array, shape (S, R, C)
            mask of moving object being registered (a binary array with 1's
            inside the object of interest and 0's along the background).
            If None, ones_like(moving) is used as mask.
        """
        if static.shape != moving.shape:
            raise ValueError("Images must have the same shape")
        dim = len(static.shape)
        if not dim in [2, 3]:
            msg = 'Only dimensions 2 and 3 are supported. ' +\
                    str(dim) + ' received'
            raise ValueError(msg)
        if not self.setup_called:
            self.setup(static, moving, smask=None, mmask=None)

        if dim == 2:
            _compute_pdfs_dense_2d(static, moving, smask, mmask, self.smin,
                                   self.sdelta, self.mmin, self.mdelta,
                                   self.nbins, self.padding, self.joint,
                                   self.smarginal, self.mmarginal)
        elif dim == 3:
            _compute_pdfs_dense_3d(static, moving, smask, mmask, self.smin,
                                   self.sdelta, self.mmin, self.mdelta,
                                   self.nbins, self.padding, self.joint,
                                   self.smarginal, self.mmarginal)

    def update_pdfs_sparse(self, sval, mval):
        r""" Computes the Probability Density Functions from a set of samples

        The list of intensities `sval` and `mval` are assumed to be sampled
        from the static and moving images, respectively, at the same
        physical points. Of course, the images may not be perfectly aligned
        at the moment the sampling was performed. The resulting  distributions
        corresponds to the paired intensities according to the alignment at the
        moment the images were sampled.

        The joint PDF is stored in self.joint. The marginal distributions
        corresponding to the static and moving images are computed and
        stored in self.smarginal and self.mmarginal, respectively.

        Parameters
        ----------
        sval : array, shape (n,)
            sampled intensities from the static image at sampled_points
        mval : array, shape (n,)
            sampled intensities from the moving image at sampled_points
        """
        if not self.setup_called:
            self.setup(sval, mval)

        energy = _compute_pdfs_sparse(sval, mval, self.smin, self.sdelta,
                                      self.mmin, self.mdelta, self.nbins,
                                      self.padding, self.joint,
                                      self.smarginal, self.mmarginal)

    def update_gradient_dense(self, theta, transform, static, moving,
                              grid2world, mgradient, smask=None, mmask=None):
        r""" Computes the Gradient of the joint PDF w.r.t. transform parameters

        Computes the vector of partial derivatives of the joint histogram
        w.r.t. each transformation parameter.

        The gradient is stored in self.joint_grad.

        Parameters
        ----------
        theta : array, shape (n,)
            parameters of the transformation to compute the gradient from
        transform : instance of Transform
            the transformation with respect to whose parameters the gradient
            must be computed
        static : array, shape (S, R, C)
            static image
        moving : array, shape (S, R, C)
            moving image
        grid2world : array, shape (4, 4)
            we assume that both images have already been sampled at a common
            grid. This transform must map voxel coordinates of this common grid
            to physical coordinates of its corresponding voxel in the moving
            image. For example, if the moving image was sampled on the static
            image's grid (this is the typical setting) using an aligning
            matrix A, then

            (1) grid2world = A.dot(static_affine)

            where static_affine is the transformation mapping static image's
            grid coordinates to physical space.

        mgradient : array, shape (S, R, C, 3)
            the gradient of the moving image
        smask : array, shape (S, R, C), optional
            mask of static object being registered (a binary array with 1's
            inside the object of interest and 0's along the background).
            The default is None, indicating all voxels are considered.
        mmask : array, shape (S, R, C), optional
            mask of moving object being registered (a binary array with 1's
            inside the object of interest and 0's along the background).
            The default is None, indicating all voxels are considered.
        """
        if static.shape != moving.shape:
            raise ValueError("Images must have the same shape")
        dim = len(static.shape)
        if not dim in [2, 3]:
            msg = 'Only dimensions 2 and 3 are supported. ' +\
                str(dim) + ' received'
            raise ValueError(msg)

        if mgradient.shape != moving.shape + (dim,):
            raise ValueError('Invalid gradient field dimensions.')

        if not self.setup_called:
            self.setup(static, moving, smask, mmask)

        n = theta.shape[0]
        nbins = self.nbins

        if (self.joint_grad is None) or (self.joint_grad.shape[2] != n):
            self.joint_grad = np.zeros((nbins, nbins, n))
        if dim == 2:
            if mgradient.dtype == np.float64:
                _joint_pdf_gradient_dense_2d[cython.double](theta, transform,
                    static, moving, grid2world, mgradient, smask, mmask,
                    self.smin, self.sdelta, self.mmin, self.mdelta,
                    self.nbins, self.padding, self.joint_grad)
            elif mgradient.dtype == np.float32:
                _joint_pdf_gradient_dense_2d[cython.float](theta, transform,
                    static, moving, grid2world, mgradient, smask, mmask,
                    self.smin, self.sdelta, self.mmin, self.mdelta,
                    self.nbins, self.padding, self.joint_grad)
            else:
                raise ValueError('Grad. field dtype must be floating point')

        elif dim == 3:
            if mgradient.dtype == np.float64:
                _joint_pdf_gradient_dense_3d[cython.double](theta, transform,
                    static, moving, grid2world, mgradient, smask, mmask,
                    self.smin, self.sdelta, self.mmin, self.mdelta,
                    self.nbins, self.padding, self.joint_grad)
            elif mgradient.dtype == np.float32:
                _joint_pdf_gradient_dense_3d[cython.float](theta, transform,
                    static, moving, grid2world, mgradient, smask, mmask,
                    self.smin, self.sdelta, self.mmin, self.mdelta,
                    self.nbins, self.padding, self.joint_grad)
            else:
                raise ValueError('Grad. field dtype must be floating point')

    def update_gradient_sparse(self, theta, transform, sval, mval,
                               sample_points, mgradient):
        r""" Computes the Gradient of the joint PDF w.r.t. transform parameters

        Computes the vector of partial derivatives of the joint histogram
        w.r.t. each transformation parameter.

        The list of intensities `sval` and `mval` are assumed to be sampled
        from the static and moving images, respectively, at the same
        physical points. Of course, the images may not be perfectly aligned
        at the moment the sampling was performed. The resulting  gradient
        corresponds to the paired intensities according to the alignment at the
        moment the images were sampled.

        The gradient is stored in self.joint_grad.

        Parameters
        ----------
        theta : array, shape (n,)
            parameters to compute the gradient at
        transform : instance of Transform
            the transformation with respect to whose parameters the gradient
            must be computed
        sval : array, shape (m,)
            sampled intensities from the static image at sampled_points
        mval : array, shape (m,)
            sampled intensities from the moving image at sampled_points
        sample_points : array, shape (m, 3)
            coordinates (in physical space) of the points the images were
            sampled at
        mgradient : array, shape (m, 3)
            the gradient of the moving image at the sample points
        """
        dim = sample_points.shape[1]
        if mgradient.shape[1] != dim:
            raise ValueError('Dimensions of gradients and points are different')

        nsamples = sval.shape[0]
        if ((mgradient.shape[0] != nsamples) or (mval.shape[0] != nsamples)
            or sample_points.shape[0] != nsamples):
            raise ValueError('Number of points and gradients are different.')

        if not mgradient.dtype in [np.float32, np.float64]:
            raise ValueError('Gradients dtype must be floating point')

        n = theta.shape[0]
        nbins = self.nbins

        if (self.joint_grad is None) or (self.joint_grad.shape[2] != n):
            self.joint_grad = np.zeros(shape=(nbins, nbins, n))

        if dim == 2:
            if mgradient.dtype == np.float64:
                _joint_pdf_gradient_sparse_2d[cython.double](theta, transform,
                    sval, mval, sample_points, mgradient, self.smin,
                    self.sdelta, self.mmin, self.mdelta, self.nbins,
                    self.padding, self.joint_grad)
            elif mgradient.dtype == np.float32:
                _joint_pdf_gradient_sparse_2d[cython.float](theta, transform,
                    sval, mval, sample_points, mgradient, self.smin,
                    self.sdelta, self.mmin, self.mdelta, self.nbins,
                    self.padding, self.joint_grad)
            else:
                raise ValueError('Gradients dtype must be floating point')

        elif dim == 3:
            if mgradient.dtype == np.float64:
                _joint_pdf_gradient_sparse_3d[cython.double](theta, transform,
                    sval, mval, sample_points, mgradient, self.smin,
                    self.sdelta, self.mmin, self.mdelta, self.nbins,
                    self.padding, self.joint_grad)
            elif mgradient.dtype == np.float32:
                _joint_pdf_gradient_sparse_3d[cython.float](theta, transform,
                    sval, mval, sample_points, mgradient, self.smin,
                    self.sdelta, self.mmin, self.mdelta, self.nbins,
                    self.padding, self.joint_grad)
            else:
                raise ValueError('Gradients dtype must be floating point')
        else:
            msg = 'Only dimensions 2 and 3 are supported. ' + str(dim) +\
                ' received'
            raise ValueError(msg)


cdef inline double _bin_normalize(double x, double mval, double delta) nogil:
    r""" Normalizes intensity x to the range covered by the Parzen histogram
    We assume that mval was computed as:

    (1) mval = xmin / delta - padding

    where xmin is the minimum observed image intensity and delta is the
    bin size, computed as:

    (2) delta = (xmax - xmin)/(nbins - 2 * padding)

    If the minimum and maximum intensities were assigned to the first and last
    bins (with no padding), it could be possible that samples at the first and
    last bins contribute to "non-existing" bins beyond the boundary (because
    the support of the Parzen window may be larger than one bin). The padding
    bins are used to collect such contributions (i.e. the probability of
    observing a value beyond the minimum and maximum observed intensities may
    correctly be assigned a positive value).

    The normalized intensity is (from eq(1) ):

    (3) nx = (x - xmin) / delta + padding = x / delta - mval

    This means that normalized intensity nx must lie in the closed interval
    [padding, nbins-padding], which contains bins with indices
    padding, padding+1, ..., nbins - 1 - padding (i.e., nbins - 2*padding bins)

    """
    return x / delta - mval


cdef inline cnp.npy_intp _bin_index(double normalized, int nbins,
                                    int padding) nogil:
    r""" Index of the bin in which the normalized intensity `normalized` lies.

    The intensity is assumed to have been normalized to the range of
    intensities covered by the histogram: the bin index is the integer part of
    `normalized`, which must be within the interval
    [padding, nbins - 1 - padding].

    Parameters
    ----------
    normalized : float
        normalized intensity
    nbins : int
        number of histogram bins
    padding : int
        number of bins used as padding (the total bins used for padding at
        both sides of the histogram is actually 2*padding)

    Returns
    -------
    bin : int
        index of the bin in which the normalized intensity 'normalized' lies
    """
    cdef:
        cnp.npy_intp bin

    bin = <cnp.npy_intp>(normalized)
    if bin < padding:
        return padding
    if bin > nbins - 1 - padding:
        return nbins - 1 - padding
    return bin


def cubic_spline(double[:] x):
    r""" Evaluates the cubic spline at a set of values

    Parameters
    ----------
    x : array, shape (n)
        input values
    """
    cdef:
        cnp.npy_intp i
        cnp.npy_intp n = x.shape[0]
        double[:] sx = np.zeros(n, dtype=np.float64)
    with nogil:
        for i in range(n):
            sx[i] = _cubic_spline(x[i])
    return np.asarray(sx)


cdef inline double _cubic_spline(double x) nogil:
    r""" Cubic B-Spline evaluated at x
    See eq. (3) of [Matttes03].

    References
    ----------
    [Mattes03] Mattes, D., Haynor, D. R., Vesselle, H., Lewellen, T. K.,
               & Eubank, W. PET-CT image registration in the chest using
               free-form deformations. IEEE Transactions on Medical Imaging,
               22(1), 120-8, 2003.
    """
    cdef:
        double absx = -x if x < 0.0 else x
        double sqrx = x * x

    if absx < 1.0:
        return (4.0 - 6.0 * sqrx + 3.0 * sqrx * absx) / 6.0
    elif absx < 2.0:
        return (8.0 - 12 * absx + 6.0 * sqrx - sqrx * absx) / 6.0
    return 0.0


def cubic_spline_derivative(double[:] x):
    r""" Evaluates the cubic spline derivative at a set of values

    Parameters
    ----------
    x : array, shape (n)
        input values
    """
    cdef:
        cnp.npy_intp i
        cnp.npy_intp n = x.shape[0]
        double[:] sx = np.zeros(n, dtype=np.float64)
    with nogil:
        for i in range(n):
            sx[i] = _cubic_spline_derivative(x[i])
    return np.asarray(sx)


cdef inline double _cubic_spline_derivative(double x) nogil:
    r""" Derivative of cubic B-Spline evaluated at x
    See eq. (3) of [Mattes03].

    References
    ----------
    [Mattes03] Mattes, D., Haynor, D. R., Vesselle, H., Lewellen, T. K.,
               & Eubank, W. PET-CT image registration in the chest using
               free-form deformations. IEEE Transactions on Medical Imaging,
               22(1), 120-8, 2003.
    """
    cdef:
        double absx = -x if x < 0.0 else x
    if absx < 1.0:
        if x >= 0.0:
            return -2.0 * x + 1.5 * x * x
        else:
            return -2.0 * x - 1.5 * x * x
    elif absx < 2.0:
        if x >= 0:
            return -2.0 + 2.0 * x - 0.5 * x * x
        else:
            return 2.0 + 2.0 * x + 0.5 * x * x
    return 0.0


cdef _compute_pdfs_dense_2d(double[:, :] static, double[:, :] moving,
                            int[:, :] smask, int[:, :] mmask,
                            double smin, double sdelta,
                            double mmin, double mdelta,
                            int nbins, int padding, double[:, :] joint,
                            double[:] smarginal, double[:] mmarginal):
    r""" Joint Probability Density Function of intensities of two 2D images

    Parameters
    ----------
    static : array, shape (R, C)
        static image
    moving : array, shape (R, C)
        moving image
    smask : array, shape (R, C)
        mask of static object being registered (a binary array with 1's inside
        the object of interest and 0's along the background)
    mmask : array, shape (R, C)
        mask of moving object being registered (a binary array with 1's inside
        the object of interest and 0's along the background)
    smin : float
        the minimum observed intensity associated with the static image, which
        was used to define the joint PDF
    sdelta : float
        bin size associated with the intensities of the static image
    mmin : float
        the minimum observed intensity associated with the moving image, which
        was used to define the joint PDF
    mdelta : float
        bin size associated with the intensities of the moving image
    nbins : int
        number of histogram bins
    padding : int
        number of bins used as padding (the total bins used for padding at both
        sides of the histogram is actually 2*padding)
    joint : array, shape (nbins, nbins)
        the array to write the joint PDF
    smarginal : array, shape (nbins,)
        the array to write the marginal PDF associated with the static image
    mmarginal : array, shape (nbins,)
        the array to write the marginal PDF associated with the moving image
    """
    cdef:
        cnp.npy_intp nrows = static.shape[0]
        cnp.npy_intp ncols = static.shape[1]
        cnp.npy_intp offset, valid_points
        cnp.npy_intp i, j, r, c
        double rn, cn
        double val, spline_arg, sum

    joint[...] = 0
    sum = 0
    valid_points = 0
    with nogil:
        smarginal[:] = 0
        for i in range(nrows):
            for j in range(ncols):
                if smask is not None and smask[i, j] == 0:
                    continue
                if mmask is not None and mmask[i, j] == 0:
                    continue
                valid_points += 1
                rn = _bin_normalize(static[i, j], smin, sdelta)
                r = _bin_index(rn, nbins, padding)
                cn = _bin_normalize(moving[i, j], mmin, mdelta)
                c = _bin_index(cn, nbins, padding)
                spline_arg = (c - 2) - cn

                smarginal[r] += 1
                for offset in range(-2, 3):
                    val = _cubic_spline(spline_arg)
                    joint[r, c + offset] += val
                    sum += val
                    spline_arg += 1.0
        if sum > 0:
            for i in range(nbins):
                for j in range(nbins):
                    joint[i, j] /= valid_points

            for i in range(nbins):
                smarginal[i] /= valid_points

            for j in range(nbins):
                mmarginal[j] = 0
                for i in range(nbins):
                    mmarginal[j] += joint[i, j]


cdef _compute_pdfs_dense_3d(double[:, :, :] static, double[:, :, :] moving,
                            int[:, :, :] smask, int[:, :, :] mmask,
                            double smin, double sdelta,
                            double mmin, double mdelta,
                            int nbins, int padding, double[:, :] joint,
                            double[:] smarginal, double[:] mmarginal):
    r""" Joint Probability Density Function of intensities of two 3D images

    Parameters
    ----------
    static : array, shape (S, R, C)
        static image
    moving : array, shape (S, R, C)
        moving image
    smask : array, shape (S, R, C)
        mask of static object being registered (a binary array with 1's inside
        the object of interest and 0's along the background)
    mmask : array, shape (S, R, C)
        mask of moving object being registered (a binary array with 1's inside
        the object of interest and 0's along the background)
    smin : float
        the minimum observed intensity associated with the static image, which
        was used to define the joint PDF
    sdelta : float
        bin size associated with the intensities of the static image
    mmin : float
        the minimum observed intensity associated with the moving image, which
        was used to define the joint PDF
    mdelta : float
        bin size associated with the intensities of the moving image
    nbins : int
        number of histogram bins
    padding : int
        number of bins used as padding (the total bins used for padding at both
        sides of the histogram is actually 2*padding)
    joint : array, shape (nbins, nbins)
        the array to write the joint PDF to
    smarginal : array, shape (nbins,)
        the array to write the marginal PDF associated with the static image
    mmarginal : array, shape (nbins,)
        the array to write the marginal PDF associated with the moving image
    """
    cdef:
        cnp.npy_intp nslices = static.shape[0]
        cnp.npy_intp nrows = static.shape[1]
        cnp.npy_intp ncols = static.shape[2]
        cnp.npy_intp offset, valid_points
        cnp.npy_intp k, i, j, r, c
        double rn, cn
        double val, spline_arg, sum

    joint[...] = 0
    sum = 0
    with nogil:
        valid_points = 0
        smarginal[:] = 0
        for k in range(nslices):
            for i in range(nrows):
                for j in range(ncols):
                    if smask is not None and smask[k, i, j] == 0:
                        continue
                    if mmask is not None and mmask[k, i, j] == 0:
                        continue
                    valid_points += 1
                    rn = _bin_normalize(static[k, i, j], smin, sdelta)
                    r = _bin_index(rn, nbins, padding)
                    cn = _bin_normalize(moving[k, i, j], mmin, mdelta)
                    c = _bin_index(cn, nbins, padding)
                    spline_arg = (c - 2) - cn

                    smarginal[r] += 1
                    for offset in range(-2, 3):
                        val = _cubic_spline(spline_arg)
                        joint[r, c + offset] += val
                        sum += val
                        spline_arg += 1.0

        if sum > 0:
            for i in range(nbins):
                for j in range(nbins):
                    joint[i, j] /= sum

            for i in range(nbins):
                smarginal[i] /= valid_points

            for j in range(nbins):
                mmarginal[j] = 0
                for i in range(nbins):
                    mmarginal[j] += joint[i, j]


cdef _compute_pdfs_sparse(double[:] sval, double[:] mval, double smin,
                          double sdelta, double mmin, double mdelta,
                          int nbins, int padding, double[:, :] joint,
                          double[:] smarginal, double[:] mmarginal):
    r""" Probability Density Functions of paired intensities

    Parameters
    ----------
    sval : array, shape (n,)
        sampled intensities from the static image at sampled_points
    mval : array, shape (n,)
        sampled intensities from the moving image at sampled_points
    smin : float
        the minimum observed intensity associated with the static image, which
        was used to define the joint PDF
    sdelta : float
        bin size associated with the intensities of the static image
    mmin : float
        the minimum observed intensity associated with the moving image, which
        was used to define the joint PDF
    mdelta : float
        bin size associated with the intensities of the moving image
    nbins : int
        number of histogram bins
    padding : int
        number of bins used as padding (the total bins used for padding at both
        sides of the histogram is actually 2*padding)
    joint : array, shape (nbins, nbins)
        the array to write the joint PDF to
    smarginal : array, shape (nbins,)
        the array to write the marginal PDF associated with the static image
    mmarginal : array, shape (nbins,)
        the array to write the marginal PDF associated with the moving image
    """
    cdef:
        cnp.npy_intp n = sval.shape[0]
        cnp.npy_intp offset, valid_points
        cnp.npy_intp i, r, c
        double rn, cn
        double val, spline_arg, sum

    joint[...] = 0
    sum = 0

    with nogil:
        valid_points = 0
        smarginal[:] = 0
        for i in range(n):
            valid_points += 1
            rn = _bin_normalize(sval[i], smin, sdelta)
            r = _bin_index(rn, nbins, padding)
            cn = _bin_normalize(mval[i], mmin, mdelta)
            c = _bin_index(cn, nbins, padding)
            spline_arg = (c - 2) - cn

            smarginal[r] += 1
            for offset in range(-2, 3):
                val = _cubic_spline(spline_arg)
                joint[r, c + offset] += val
                sum += val
                spline_arg += 1.0

        if sum > 0:
            for i in range(nbins):
                for j in range(nbins):
                    joint[i, j] /= sum

            for i in range(nbins):
                smarginal[i] /= valid_points

            for j in range(nbins):
                mmarginal[j] = 0
                for i in range(nbins):
                    mmarginal[j] += joint[i, j]


cdef _joint_pdf_gradient_dense_2d(double[:] theta, Transform transform,
                                  double[:, :] static, double[:, :] moving,
                                  double[:, :] grid2world,
                                  floating[:, :, :] mgradient, int[:, :] smask,
                                  int[:, :] mmask, double smin, double sdelta,
                                  double mmin, double mdelta, int nbins,
                                  int padding, double[:, :, :] grad_pdf):
    r""" Gradient of the joint PDF w.r.t. transform parameters theta

    Computes the vector of partial derivatives of the joint histogram w.r.t.
    each transformation parameter. The transformation itself is not necessary
    to compute the gradient, but only its Jacobian.

    Parameters
    ----------
    theta : array, shape (n,)
        parameters of the transformation to compute the gradient from
    transform : instance of Transform
        the transformation with respect to whose parameters the gradient
        must be computed
    static : array, shape (R, C)
        static image
    moving : array, shape (R, C)
        moving image
    grid2world : array, shape (3, 3)
        the grid-to-space transform associated with images static and moving
        (we assume that both images have already been sampled at a common grid)
    mgradient : array, shape (R, C, 2)
        the gradient of the moving image
    smask : array, shape (R, C)
        mask of static object being registered (a binary array with 1's inside
        the object of interest and 0's along the background)
    mmask : array, shape (R, C)
        mask of moving object being registered (a binary array with 1's inside
        the object of interest and 0's along the background)
    smin : float
        the minimum observed intensity associated with the static image, which
        was used to define the joint PDF
    sdelta : float
        bin size associated with the intensities of the static image
    mmin : float
        the minimum observed intensity associated with the moving image, which
        was used to define the joint PDF
    mdelta : float
        bin size associated with the intensities of the moving image
    nbins : int
        number of histogram bins
    padding : int
        number of bins used as padding (the total bins used for padding at both
        sides of the histogram is actually 2*padding)
    grad_pdf : array, shape (nbins, nbins, len(theta))
        the array to write the gradient to
    """
    cdef:
        cnp.npy_intp nrows = static.shape[0]
        cnp.npy_intp ncols = static.shape[1]
        cnp.npy_intp n = theta.shape[0]
        cnp.npy_intp offset, valid_points
        int constant_jacobian = 0
        cnp.npy_intp k, i, j, r, c
        double rn, cn
        double val, spline_arg, norm_factor
        double[:, :] J = np.empty(shape=(2, n), dtype=np.float64)
        double[:] prod = np.empty(shape=(n,), dtype=np.float64)
        double[:] x = np.empty(shape=(2,), dtype=np.float64)

    grad_pdf[...] = 0
    with nogil:
        valid_points = 0
        for i in range(nrows):
            for j in range(ncols):
                if smask is not None and smask[i, j] == 0:
                    continue
                if mmask is not None and mmask[i, j] == 0:
                    continue

                valid_points += 1
                x[0] = _apply_affine_2d_x0(i, j, 1, grid2world)
                x[1] = _apply_affine_2d_x1(i, j, 1, grid2world)

                if constant_jacobian == 0:
                    constant_jacobian = transform._jacobian(theta, x, J)

                for k in range(n):
                    prod[k] = (J[0, k] * mgradient[i, j, 0] +
                               J[1, k] * mgradient[i, j, 1])

                rn = _bin_normalize(static[i, j], smin, sdelta)
                r = _bin_index(rn, nbins, padding)
                cn = _bin_normalize(moving[i, j], mmin, mdelta)
                c = _bin_index(cn, nbins, padding)
                spline_arg = (c - 2) - cn

                for offset in range(-2, 3):
                    val = _cubic_spline_derivative(spline_arg)
                    for k in range(n):
                        grad_pdf[r, c + offset, k] -= val * prod[k]
                    spline_arg += 1.0

        norm_factor = valid_points * mdelta
        if norm_factor > 0:
            for i in range(nbins):
                for j in range(nbins):
                    for k in range(n):
                        grad_pdf[i, j, k] /= norm_factor


cdef _joint_pdf_gradient_dense_3d(double[:] theta, Transform transform,
                                  double[:, :, :] static,
                                  double[:, :, :] moving,
                                  double[:, :] grid2world,
                                  floating[:, :, :, :] mgradient,
                                  int[:, :, :] smask,
                                  int[:, :, :] mmask, double smin,
                                  double sdelta, double mmin, double mdelta,
                                  int nbins, int padding,
                                  double[:, :, :] grad_pdf):
    r""" Gradient of the joint PDF w.r.t. transform parameters theta

    Computes the vector of partial derivatives of the joint histogram w.r.t.
    each transformation parameter. The transformation itself is not necessary
    to compute the gradient, but only its Jacobian.

    Parameters
    ----------
    theta : array, shape (n,)
        parameters of the transformation to compute the gradient from
    transform : instance of Transform
        the transformation with respect to whose parameters the gradient
        must be computed
    static : array, shape (S, R, C)
        static image
    moving : array, shape (S, R, C)
        moving image
    grid2world : array, shape (4, 4)
        the grid-to-space transform associated with images static and moving
        (we assume that both images have already been sampled at a common grid)
    mgradient : array, shape (S, R, C, 3)
        the gradient of the moving image
    smask : array, shape (S, R, C)
        mask of static object being registered (a binary array with 1's inside
        the object of interest and 0's along the background)
    mmask : array, shape (S, R, C)
        mask of moving object being registered (a binary array with 1's inside
        the object of interest and 0's along the background)
    smin : float
        the minimum observed intensity associated with the static image, which
        was used to define the joint PDF
    sdelta : float
        bin size associated with the intensities of the static image
    mmin : float
        the minimum observed intensity associated with the moving image, which
        was used to define the joint PDF
    mdelta : float
        bin size associated with the intensities of the moving image
    nbins : int
        number of histogram bins
    padding : int
        number of bins used as padding (the total bins used for padding at both
        sides of the histogram is actually 2*padding)
    grad_pdf : array, shape (nbins, nbins, len(theta))
        the array to write the gradient to
    """
    cdef:
        cnp.npy_intp nslices = static.shape[0]
        cnp.npy_intp nrows = static.shape[1]
        cnp.npy_intp ncols = static.shape[2]
        cnp.npy_intp n = theta.shape[0]
        cnp.npy_intp offset, valid_points
        int constant_jacobian = 0
        cnp.npy_intp l, k, i, j, r, c
        double rn, cn
        double val, spline_arg, norm_factor
        double[:, :] J = np.empty(shape=(3, n), dtype=np.float64)
        double[:] prod = np.empty(shape=(n,), dtype=np.float64)
        double[:] x = np.empty(shape=(3,), dtype=np.float64)

    grad_pdf[...] = 0
    with nogil:
        valid_points = 0
        for k in range(nslices):
            for i in range(nrows):
                for j in range(ncols):
                    if smask is not None and smask[k, i, j] == 0:
                        continue
                    if mmask is not None and mmask[k, i, j] == 0:
                        continue
                    valid_points += 1
                    x[0] = _apply_affine_3d_x0(k, i, j, 1, grid2world)
                    x[1] = _apply_affine_3d_x1(k, i, j, 1, grid2world)
                    x[2] = _apply_affine_3d_x2(k, i, j, 1, grid2world)

                    if constant_jacobian == 0:
                        constant_jacobian = transform._jacobian(theta, x, J)

                    for l in range(n):
                        prod[l] = (J[0, l] * mgradient[k, i, j, 0] +
                                   J[1, l] * mgradient[k, i, j, 1] +
                                   J[2, l] * mgradient[k, i, j, 2])

                    rn = _bin_normalize(static[k, i, j], smin, sdelta)
                    r = _bin_index(rn, nbins, padding)
                    cn = _bin_normalize(moving[k, i, j], mmin, mdelta)
                    c = _bin_index(cn, nbins, padding)
                    spline_arg = (c - 2) - cn

                    for offset in range(-2, 3):
                        val = _cubic_spline_derivative(spline_arg)
                        for l in range(n):
                            grad_pdf[r, c + offset, l] -= val * prod[l]
                        spline_arg += 1.0

        norm_factor = valid_points * mdelta
        if norm_factor > 0:
            for i in range(nbins):
                for j in range(nbins):
                    for k in range(n):
                        grad_pdf[i, j, k] /= norm_factor


cdef _joint_pdf_gradient_sparse_2d(double[:] theta, Transform transform,
                                   double[:] sval, double[:] mval,
                                   double[:, :] sample_points,
                                   floating[:, :] mgradient, double smin,
                                   double sdelta, double mmin,
                                   double mdelta, int nbins, int padding,
                                   double[:, :, :] grad_pdf):
    r""" Gradient of the joint PDF w.r.t. transform parameters theta

    Computes the vector of partial derivatives of the joint histogram w.r.t.
    each transformation parameter. The transformation itself is not necessary
    to compute the gradient, but only its Jacobian.

    Parameters
    ----------
    theta : array, shape (n,)
        parameters to compute the gradient at
    transform : instance of Transform
        the transformation with respect to whose parameters the gradient
        must be computed
    sval : array, shape (m,)
        sampled intensities from the static image at sampled_points
    mval : array, shape (m,)
        sampled intensities from the moving image at sampled_points
    sample_points : array, shape (m, 2)
        positions (in physical space) of the points the images were sampled at
    mgradient : array, shape (m, 2)
        the gradient of the moving image at the sample points
    smin : float
        the minimum observed intensity associated with the static image, which
        was used to define the joint PDF
    sdelta : float
        bin size associated with the intensities of the static image
    mmin : float
        the minimum observed intensity associated with the moving image, which
        was used to define the joint PDF
    mdelta : float
        bin size associated with the intensities of the moving image
    nbins : int
        number of histogram bins
    padding : int
        number of bins used as padding (the total bins used for padding at both
        sides of the histogram is actually 2*padding)
    grad_pdf : array, shape (nbins, nbins, len(theta))
        the array to write the gradient to
    """
    cdef:
        cnp.npy_intp n = theta.shape[0]
        cnp.npy_intp m = sval.shape[0]
        cnp.npy_intp offset
        int constant_jacobian = 0
        cnp.npy_intp i, j, r, c, valid_points
        double rn, cn
        double val, spline_arg, norm_factor
        double[:, :] J = np.empty(shape=(2, n), dtype=np.float64)
        double[:] prod = np.empty(shape=(n,), dtype=np.float64)

    grad_pdf[...] = 0
    with nogil:
        valid_points = 0
        for i in range(m):
            valid_points += 1
            if constant_jacobian == 0:
                constant_jacobian = transform._jacobian(theta,
                                                        sample_points[i], J)

            for j in range(n):
                prod[j] = (J[0, j] * mgradient[i, 0] +
                           J[1, j] * mgradient[i, 1])

            rn = _bin_normalize(sval[i], smin, sdelta)
            r = _bin_index(rn, nbins, padding)
            cn = _bin_normalize(mval[i], mmin, mdelta)
            c = _bin_index(cn, nbins, padding)
            spline_arg = (c - 2) - cn

            for offset in range(-2, 3):
                val = _cubic_spline_derivative(spline_arg)
                for j in range(n):
                    grad_pdf[r, c + offset, j] -= val * prod[j]
                spline_arg += 1.0

        norm_factor = valid_points * mdelta
        if norm_factor > 0:
            for i in range(nbins):
                for j in range(nbins):
                    for k in range(n):
                        grad_pdf[i, j, k] /= norm_factor


cdef _joint_pdf_gradient_sparse_3d(double[:] theta, Transform transform,
                                   double[:] sval, double[:] mval,
                                   double[:, :] sample_points,
                                   floating[:, :] mgradient, double smin,
                                   double sdelta, double mmin,
                                   double mdelta, int nbins, int padding,
                                   double[:, :, :] grad_pdf):
    r""" Gradient of the joint PDF w.r.t. transform parameters theta

    Computes the vector of partial derivatives of the joint histogram w.r.t.
    each transformation parameter. The transformation itself is not necessary
    to compute the gradient, but only its Jacobian.

    Parameters
    ----------
    theta : array, shape (n,)
        parameters to compute the gradient at
    transform : instance of Transform
        the transformation with respect to whose parameters the gradient
        must be computed
    sval : array, shape (m,)
        sampled intensities from the static image at sampled_points
    mval : array, shape (m,)
        sampled intensities from the moving image at sampled_points
    sample_points : array, shape (m, 3)
        positions (in physical space) of the points the images were sampled at
    mgradient : array, shape (m, 3)
        the gradient of the moving image at the sample points
    smin : float
        the minimum observed intensity associated with the static image, which
        was used to define the joint PDF
    sdelta : float
        bin size associated with the intensities of the static image
    mmin : float
        the minimum observed intensity associated with the moving image, which
        was used to define the joint PDF
    mdelta : float
        bin size associated with the intensities of the moving image
    nbins : int
        number of histogram bins
    padding : int
        number of bins used as padding (the total bins used for padding at both
        sides of the histogram is actually 2*padding)
    grad_pdf : array, shape (nbins, nbins, len(theta))
        the array to write the gradient to
    """
    cdef:
        cnp.npy_intp n = theta.shape[0]
        cnp.npy_intp m = sval.shape[0]
        cnp.npy_intp offset, valid_points
        int constant_jacobian = 0
        cnp.npy_intp i, j, r, c
        double rn, cn
        double val, spline_arg, norm_factor
        double[:, :] J = np.empty(shape=(3, n), dtype=np.float64)
        double[:] prod = np.empty(shape=(n,), dtype=np.float64)

    grad_pdf[...] = 0
    with nogil:
        valid_points = 0
        for i in range(m):
            valid_points += 1

            if constant_jacobian == 0:
                constant_jacobian = transform._jacobian(theta,
                                                        sample_points[i], J)

            for j in range(n):
                prod[j] = (J[0, j] * mgradient[i, 0] +
                           J[1, j] * mgradient[i, 1] +
                           J[2, j] * mgradient[i, 2])

            rn = _bin_normalize(sval[i], smin, sdelta)
            r = _bin_index(rn, nbins, padding)
            cn = _bin_normalize(mval[i], mmin, mdelta)
            c = _bin_index(cn, nbins, padding)
            spline_arg = (c - 2) - cn

            for offset in range(-2, 3):
                val = _cubic_spline_derivative(spline_arg)
                for j in range(n):
                    grad_pdf[r, c + offset, j] -= val * prod[j]
                spline_arg += 1.0

        norm_factor = valid_points * mdelta
        if norm_factor > 0:
            for i in range(nbins):
                for j in range(nbins):
                    for k in range(n):
                        grad_pdf[i, j, k] /= norm_factor


def compute_parzen_mi(double[:, :] joint,
                      double[:, :, :] joint_gradient,
                      double[:] smarginal, double[:] mmarginal,
                      double[:] mi_gradient):
    r""" Computes the mutual information and its gradient (if requested)

    Parameters
    ----------
    joint : array, shape (nbins, nbins)
        the joint intensity distribution
    joint_gradient : array, shape (nbins, nbins, n)
        the gradient of the joint distribution w.r.t. the transformation
        parameters
    smarginal : array, shape (nbins,)
        the marginal intensity distribution of the static image
    mmarginal : array, shape (nbins,)
        the marginal intensity distribution of the moving image
    mi_gradient : array, shape (n,)
        the buffer in which to write the gradient of the mutual information.
        If None, the gradient is not computed
    """
    cdef:
        double epsilon = 2.2204460492503131e-016
        double metric_value
        cnp.npy_intp nrows = joint.shape[0]
        cnp.npy_intp ncols = joint.shape[1]
        cnp.npy_intp n = joint_gradient.shape[2]
    with nogil:
        mi_gradient[:] = 0
        metric_value = 0
        for i in range(nrows):
            for j in range(ncols):
                if joint[i, j] < epsilon or mmarginal[j] < epsilon:
                    continue

                factor = log(joint[i, j] / mmarginal[j])

                if mi_gradient is not None:
                    for k in range(n):
                        mi_gradient[k] += joint_gradient[i, j, k] * factor

                if smarginal[i] > epsilon:
                    metric_value += joint[i, j] * (factor - log(smarginal[i]))

    return metric_value


def sample_domain_regular(int k, int[:] shape, double[:, :] grid2world,
                          double sigma=0.25, int seed=1234):
    r""" Take floor(total_voxels/k) samples from a (2D or 3D) grid

    The sampling is made by taking all pixels whose index (in lexicographical
    order) is a multiple of k. Each selected point is slightly perturbed by
    adding a realization of a normally distributed random variable and then
    mapped to physical space by the given grid-to-space transform.

    The lexicographical order of a pixels in a grid of shape (a, b, c) is
    defined by assigning to each voxel position (i, j, k) the integer index

    F((i, j, k)) = i * (b * c) + j * (c) + k

    and sorting increasingly by this index.

    Parameters
    ----------
    k : int
        the sampling rate, as described before
    shape : array, shape (dim,)
        the shape of the grid to be sampled
    grid2world : array, shape (dim+1, dim+1)
        the grid-to-space transform
    sigma : float
        the standard deviation of the Normal random distortion to be applied
        to the sampled points

    Returns
    -------
    samples : array, shape (total_pixels//k, dim)
        the matrix whose rows are the sampled points

    Examples
    --------
    >>> from dipy.align.parzenhist import sample_domain_regular
    >>> import dipy.align.vector_fields as vf
    >>> shape = np.array((10, 10), dtype=np.int32)
    >>> sigma = 0
    >>> dim = len(shape)
    >>> grid2world = np.eye(dim+1)
    >>> n = shape[0]*shape[1]
    >>> k = 2
    >>> samples = sample_domain_regular(k, shape, grid2world, sigma)
    >>> (samples.shape[0], samples.shape[1]) == (n//k, dim)
    True
    >>> isamples = np.array(samples, dtype=np.int32)
    >>> indices = (isamples[:, 0] * shape[1] + isamples[:, 1])
    >>> len(set(indices)) == len(indices)
    True
    >>> (indices%k).sum()
    0
    """
    cdef:
        cnp.npy_intp i, dim, n, m, slice_size
        double s, r, c
        double[:, :] samples
    dim = len(shape)
    if not vf.is_valid_affine(grid2world, dim):
        raise ValueError("Invalid grid-to-space matrix")

    random.seed(seed)
    if dim == 2:
        n = shape[0] * shape[1]
        m = n // k
        samples = random.randn(m, dim) * sigma
        with nogil:
            for i in range(m):
                r = ((i * k) // shape[1]) + samples[i, 0]
                c = ((i * k) % shape[1]) + samples[i, 1]
                samples[i, 0] = _apply_affine_2d_x0(r, c, 1, grid2world)
                samples[i, 1] = _apply_affine_2d_x1(r, c, 1, grid2world)
    else:
        slice_size = shape[1] * shape[2]
        n = shape[0] * slice_size
        m = n // k
        samples = random.randn(m, dim) * sigma
        with nogil:
            for i in range(m):
                s = ((i * k) // slice_size) + samples[i, 0]
                r = (((i * k) % slice_size) // shape[2]) + samples[i, 1]
                c = (((i * k) % slice_size) % shape[2]) + samples[i, 2]
                samples[i, 0] = _apply_affine_3d_x0(s, r, c, 1, grid2world)
                samples[i, 1] = _apply_affine_3d_x1(s, r, c, 1, grid2world)
                samples[i, 2] = _apply_affine_3d_x2(s, r, c, 1, grid2world)
    return np.asarray(samples)
