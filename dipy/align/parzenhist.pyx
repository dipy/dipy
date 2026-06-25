#!python
#cython: boundscheck=False
#cython: wraparound=False
#cython: cdivision=True

import numpy as np
cimport numpy as cnp
cimport cython
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

class ParzenJointHistogram:
    def __init__(self, nbins):
        r""" Computes joint histogram and derivatives with Parzen windows
        :footcite:p:`Parzen1962`.

        Base class to compute joint and marginal probability density
        functions and their derivatives with respect to a transform's
        parameters. The smooth histograms are computed by using Parzen
        windows :footcite:p:`Parzen1962` with a cubic spline kernel, as proposed
        by :footcite:t:`Mattes2003`. This implementation is not tied to any
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
        .. footbibliography::

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

        numerator = self.smax - self.smin
        denominator = self.nbins - 2 * self.padding
        self.sdelta = np.divide(numerator, denominator,
                                out=np.zeros_like(numerator, dtype=np.float64),
                                where=denominator!=0)
        numerator = self.mmax - self.mmin
        self.mdelta = np.divide(numerator, denominator,
                                out=np.zeros_like(numerator, dtype=np.float64),
                                where=denominator!=0)

        self.smin = np.divide(self.smin, self.sdelta,
                              out=np.zeros_like(self.smin, dtype=np.float64),
                              where=self.sdelta!=0) - self.padding
        self.mmin = np.divide(self.mmin, self.mdelta,
                              out=np.zeros_like(self.mmin, dtype=np.float64),
                              where=self.mdelta!=0) - self.padding

        self.joint_grad = None
        self.mi_weights = None
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

        if static.dtype != moving.dtype:
            raise ValueError("Static and moving images must have the same dtype")

        if static.dtype == np.float64:
            if dim == 2:
                _compute_pdfs_dense_2d[cython.double](
                    static, moving, smask, mmask,
                    self.smin, self.sdelta, self.mmin, self.mdelta,
                    self.nbins, self.padding,
                    self.joint, self.smarginal, self.mmarginal
                )
            elif dim == 3:
                _compute_pdfs_dense_3d[cython.double](
                    static, moving, smask, mmask,
                    self.smin, self.sdelta, self.mmin, self.mdelta,
                    self.nbins, self.padding,
                    self.joint, self.smarginal, self.mmarginal
                )
        elif static.dtype == np.float32:
            if dim == 2:
                _compute_pdfs_dense_2d[cython.float](
                    static, moving, smask, mmask,
                    self.smin, self.sdelta, self.mmin, self.mdelta,
                    self.nbins, self.padding,
                    self.joint, self.smarginal, self.mmarginal
                )
            elif dim == 3:
                _compute_pdfs_dense_3d[cython.float](
                    static, moving, smask, mmask,
                    self.smin, self.sdelta, self.mmin, self.mdelta,
                    self.nbins, self.padding,
                    self.joint, self.smarginal, self.mmarginal
                )
        else:
            raise ValueError("Images must be of dtype float32 or float64")

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

    def compute_mi(self, local_support=False):
        """Computes mutual information using the current joint PDF.

        This method supports two derivative-reduction paths. With
        ``local_support=False``, it follows the global-parameter path used by
        affine registration: the joint PDF derivative stored in
        ``self.joint_grad`` is reduced into ``metric_grad``. With
        ``local_support=True``, it follows the dense local-support path used by
        displacement-field registration: no global ``joint_grad`` tensor is
        reduced, and the per-bin MI derivative weights are stored in
        ``self.mi_weights`` for a later voxelwise update computation.

        Parameters
        ----------
        local_support : bool, optional
            If True, compute MI bin weights instead of a global parameter
            gradient.
        """
        if local_support:
            self.metric_val = compute_parzen_mi_weights(
                self.joint, self.smarginal, self.mmarginal, self.mi_weights)
            return self.metric_val

        self.metric_val = compute_parzen_mi(
            self.joint, self.joint_grad, self.smarginal, self.mmarginal,
            self.metric_grad)
        return self.metric_val

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
            self.setup(static, moving, smask=smask, mmask=mmask)

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

    def compute_dense_mi_update(self, static, moving, mgradient, update_field,
                                smask=None, mmask=None):
        r""" Computes the dense local-support MI update field

        Computes the dense vector field of partial derivatives of the mutual
        information w.r.t. local displacement parameters. The update is stored
        in `update_field`.

        Parameters
        ----------
        static : array, shape (S, R, C)
            static image
        moving : array, shape (S, R, C)
            moving image
        mgradient : array, shape (S, R, C, 3)
            gradient of the moving image
        update_field : array, shape (S, R, C, 3)
            the buffer in which to write the dense update field
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
        if dim not in [2, 3]:
            msg = "Only dimensions 2 and 3 are supported. " + str(dim) + " received"
            raise ValueError(msg)

        if mgradient.shape != moving.shape + (dim,):
            raise ValueError("Invalid gradient field dimensions.")

        if update_field is None:
            raise ValueError("An update field buffer is required.")

        if update_field.shape != moving.shape + (dim,):
            raise ValueError("Invalid update field dimensions.")

        if static.dtype != moving.dtype:
            raise ValueError("Static and moving images must have the same dtype.")

        if static.dtype != mgradient.dtype:
            raise ValueError("Images and gradient field must have the same dtype.")

        if update_field.dtype != mgradient.dtype:
            raise ValueError("Gradient and update field dtypes must match.")

        if static.dtype not in [np.float32, np.float64]:
            raise ValueError("Images must be of dtype float32 or float64.")

        if not self.setup_called:
            self.setup(static, moving, smask=smask, mmask=mmask)

        n_offsets = 2 * self.padding + 1
        local_derivative_by_parzen_bin = np.zeros(
            (n_offsets,) + moving.shape + (dim,), dtype=mgradient.dtype)
        joint_pdf_index = np.zeros(moving.shape, dtype=np.intp)

        if self.mi_weights is None or self.mi_weights.shape != self.joint.shape:
            self.mi_weights = np.zeros_like(self.joint)

        if dim == 2:
            if mgradient.dtype == np.float64:
                _compute_pdfs_dense_and_local_derivatives_2d[cython.double](
                    static, moving, mgradient, smask, mmask,
                    self.smin, self.sdelta, self.mmin, self.mdelta,
                    self.nbins, self.padding, self.joint, self.smarginal,
                    self.mmarginal, local_derivative_by_parzen_bin,
                    joint_pdf_index)

                self.compute_mi(local_support=True)

                _apply_mi_weights_to_cached_local_derivatives_2d[cython.double](
                    local_derivative_by_parzen_bin, joint_pdf_index,
                    smask, mmask, self.mi_weights, self.mdelta,
                    self.nbins, self.padding, update_field)

            elif mgradient.dtype == np.float32:
                _compute_pdfs_dense_and_local_derivatives_2d[cython.float](
                    static, moving, mgradient, smask, mmask,
                    self.smin, self.sdelta, self.mmin, self.mdelta,
                    self.nbins, self.padding, self.joint, self.smarginal,
                    self.mmarginal, local_derivative_by_parzen_bin,
                    joint_pdf_index)

                self.compute_mi(local_support=True)

                _apply_mi_weights_to_cached_local_derivatives_2d[cython.float](
                    local_derivative_by_parzen_bin, joint_pdf_index,
                    smask, mmask, self.mi_weights, self.mdelta,
                    self.nbins, self.padding, update_field)

            else:
                raise ValueError("Grad. field dtype must be floating point")

        elif dim == 3:
            if mgradient.dtype == np.float64:
                _compute_pdfs_dense_and_local_derivatives_3d[cython.double](
                    static, moving, mgradient, smask, mmask,
                    self.smin, self.sdelta, self.mmin, self.mdelta,
                    self.nbins, self.padding, self.joint, self.smarginal,
                    self.mmarginal, local_derivative_by_parzen_bin,
                    joint_pdf_index)

                self.compute_mi(local_support=True)

                _apply_mi_weights_to_cached_local_derivatives_3d[cython.double](
                    local_derivative_by_parzen_bin, joint_pdf_index,
                    smask, mmask, self.mi_weights, self.mdelta,
                    self.nbins, self.padding, update_field)

            elif mgradient.dtype == np.float32:
                _compute_pdfs_dense_and_local_derivatives_3d[cython.float](
                    static, moving, mgradient, smask, mmask,
                    self.smin, self.sdelta, self.mmin, self.mdelta,
                    self.nbins, self.padding, self.joint, self.smarginal,
                    self.mmarginal, local_derivative_by_parzen_bin,
                    joint_pdf_index)

                self.compute_mi(local_support=True)

                _apply_mi_weights_to_cached_local_derivatives_3d[cython.float](
                    local_derivative_by_parzen_bin, joint_pdf_index,
                    smask, mmask, self.mi_weights, self.mdelta,
                    self.nbins, self.padding, update_field)

            else:
                raise ValueError("Grad. field dtype must be floating point")

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
    if delta == 0:
        return 0
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
    normalized : double
        normalized intensity
    nbins : int
        number of histogram bins
    padding : int
        number of bins used as padding (the total bins used for padding at
        both sides of the histogram is actually 2*padding)

    Returns
    -------
    bin_id : int
        index of the bin in which the normalized intensity 'normalized' lies
    """
    cdef:
        cnp.npy_intp bin_id

    bin_id = <cnp.npy_intp>normalized
    if bin_id < padding:
        return padding
    if bin_id > nbins - 1 - padding:
        return nbins - 1 - padding
    return bin_id


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

    See eq. (3) of :footcite:t:`Mattes2003`.

    References
    ----------
    .. footbibliography::
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

    See eq. (3) of :footcite:t:`Mattes2003`.

    References
    ----------
    .. footbibliography::
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


cdef _compute_pdfs_dense_2d(floating[:, :] static, floating[:, :] moving,
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
        double val, spline_arg, total_sum

    joint[...] = 0
    total_sum = 0
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
                spline_arg = (c - padding) - cn

                smarginal[r] += 1
                for offset in range(-padding, padding + 1):
                    val = _cubic_spline(spline_arg)
                    joint[r, c + offset] += val
                    total_sum += val
                    spline_arg += 1.0
        if total_sum > 0:
            for i in range(nbins):
                for j in range(nbins):
                    joint[i, j] /= valid_points

            for i in range(nbins):
                smarginal[i] /= valid_points

            for j in range(nbins):
                mmarginal[j] = 0
                for i in range(nbins):
                    mmarginal[j] += joint[i, j]


cdef _compute_pdfs_dense_and_local_derivatives_2d(
        floating[:, :] static, floating[:, :] moving,
        floating[:, :, :] mgradient, int[:, :] smask, int[:, :] mmask,
        double smin, double sdelta, double mmin, double mdelta,
        int nbins, int padding, double[:, :] joint,
        double[:] smarginal, double[:] mmarginal,
        floating[:, :, :, :] local_derivative_by_parzen_bin,
        cnp.npy_intp[:, :] joint_pdf_index):
    r"""Joint Probability Density Function of intensities and cached dense
    local-support derivatives for two 2D images.

    This function has the same PDF-building role as `_compute_pdfs_dense_2d`,
    but it also caches the local derivative contributions required by the dense
    local-support MI update.

    Parameters
    ----------
    static : array, shape (R, C)
        static image
    moving : array, shape (R, C)
        moving image
    mgradient : array, shape (R, C, 2)
        gradient of the moving image
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
    local_derivative_by_parzen_bin : array, shape (2*padding + 1, R, C, 2)
        the array to write the unweighted local derivative contribution of each
        pixel displacement component to each affected moving Parzen bin
    joint_pdf_index : array, shape (R, C)
        the array to write the flattened index of the first joint-PDF bin affected
        by each pixel. For a voxel assigned to static bin r and moving bin c, this
        value is r * nbins + (c - padding)
    """
    cdef:
        cnp.npy_intp nrows = static.shape[0]
        cnp.npy_intp ncols = static.shape[1]
        cnp.npy_intp offset, offset_id, valid_points
        cnp.npy_intp i, j, r, c
        double rn, cn
        double val, dval, spline_arg, total_sum

    joint[...] = 0
    smarginal[:] = 0
    mmarginal[:] = 0
    local_derivative_by_parzen_bin[...] = 0
    joint_pdf_index[...] = 0
    total_sum = 0
    with nogil:
        valid_points = 0
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

                joint_pdf_index[i, j] = r * nbins + (c - padding)

                spline_arg = (c - padding) - cn
                smarginal[r] += 1

                offset_id = 0
                for offset in range(-padding, padding + 1):
                    val = _cubic_spline(spline_arg)
                    dval = _cubic_spline_derivative(spline_arg)

                    joint[r, c + offset] += val
                    total_sum += val

                    local_derivative_by_parzen_bin[offset_id, i, j, 0] = (
                        -dval * mgradient[i, j, 0])
                    local_derivative_by_parzen_bin[offset_id, i, j, 1] = (
                        -dval * mgradient[i, j, 1])

                    spline_arg += 1.0
                    offset_id += 1

        if total_sum > 0:
            for i in range(nbins):
                for j in range(nbins):
                    joint[i, j] /= valid_points

            for i in range(nbins):
                smarginal[i] /= valid_points

            for j in range(nbins):
                mmarginal[j] = 0
                for i in range(nbins):
                    mmarginal[j] += joint[i, j]


cdef _compute_pdfs_dense_3d(floating[:, :, :] static, floating[:, :, :] moving,
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
        double val, spline_arg, total_sum

    joint[...] = 0
    total_sum = 0
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
                    spline_arg = (c - padding) - cn

                    smarginal[r] += 1
                    for offset in range(-padding, padding + 1):
                        val = _cubic_spline(spline_arg)
                        joint[r, c + offset] += val
                        total_sum += val
                        spline_arg += 1.0

        if total_sum > 0:
            for i in range(nbins):
                for j in range(nbins):
                    joint[i, j] /= total_sum

            for i in range(nbins):
                smarginal[i] /= valid_points

            for j in range(nbins):
                mmarginal[j] = 0
                for i in range(nbins):
                    mmarginal[j] += joint[i, j]


cdef _compute_pdfs_dense_and_local_derivatives_3d(
        floating[:, :, :] static, floating[:, :, :] moving,
        floating[:, :, :, :] mgradient, int[:, :, :] smask,
        int[:, :, :] mmask, double smin, double sdelta,
        double mmin, double mdelta, int nbins, int padding,
        double[:, :] joint, double[:] smarginal, double[:] mmarginal,
        floating[:, :, :, :, :] local_derivative_by_parzen_bin,
        cnp.npy_intp[:, :, :] joint_pdf_index):
    r"""Joint Probability Density Function of intensities and cached dense
    local-support derivatives for 3D images.

    This function has the same PDF-building role as `_compute_pdfs_dense_3d`,
    but it also caches the local derivative contributions required by the dense
    local-support MI update.

    Parameters
    ----------
    static : array, shape (S, R, C)
        static image
    moving : array, shape (S, R, C)
        moving image
    mgradient : array, shape (S, R, C, 3)
        gradient of the moving image
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
        the array to write the joint PDF
    smarginal : array, shape (nbins,)
        the array to write the marginal PDF associated with the static image
    mmarginal : array, shape (nbins,)
        the array to write the marginal PDF associated with the moving image
    local_derivative_by_parzen_bin : array, shape (2*padding + 1, S, R, C, 3)
        the array to write the unweighted local derivative contribution of each
        voxel displacement component to each affected moving Parzen bin
    joint_pdf_index : array, shape (S, R, C)
        the array to write the flattened index of the first joint-PDF bin affected
        by each voxel. For a voxel assigned to static bin r and moving bin c, this
        value is r * nbins + (c - padding)
    """
    cdef:
        cnp.npy_intp nslices = static.shape[0]
        cnp.npy_intp nrows = static.shape[1]
        cnp.npy_intp ncols = static.shape[2]
        cnp.npy_intp offset, offset_id, valid_points
        cnp.npy_intp k, i, j, r, c
        double rn, cn
        double val, dval, spline_arg, total_sum

    joint[...] = 0
    smarginal[:] = 0
    mmarginal[:] = 0
    local_derivative_by_parzen_bin[...] = 0
    joint_pdf_index[...] = 0

    total_sum = 0
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
                    rn = _bin_normalize(static[k, i, j], smin, sdelta)
                    r = _bin_index(rn, nbins, padding)
                    cn = _bin_normalize(moving[k, i, j], mmin, mdelta)
                    c = _bin_index(cn, nbins, padding)

                    joint_pdf_index[k, i, j] = r * nbins + (c - padding)

                    spline_arg = (c - padding) - cn
                    smarginal[r] += 1

                    offset_id = 0
                    for offset in range(-padding, padding + 1):
                        val = _cubic_spline(spline_arg)
                        dval = _cubic_spline_derivative(spline_arg)

                        joint[r, c + offset] += val
                        total_sum += val

                        local_derivative_by_parzen_bin[offset_id, k, i, j, 0] = (
                            -dval * mgradient[k, i, j, 0])
                        local_derivative_by_parzen_bin[offset_id, k, i, j, 1] = (
                            -dval * mgradient[k, i, j, 1])
                        local_derivative_by_parzen_bin[offset_id, k, i, j, 2] = (
                            -dval * mgradient[k, i, j, 2])

                        spline_arg += 1.0
                        offset_id += 1

        if total_sum > 0:
            for i in range(nbins):
                for j in range(nbins):
                    joint[i, j] /= total_sum

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
        double val, spline_arg, total_sum

    joint[...] = 0
    total_sum = 0

    with nogil:
        valid_points = 0
        smarginal[:] = 0
        for i in range(n):
            valid_points += 1
            rn = _bin_normalize(sval[i], smin, sdelta)
            r = _bin_index(rn, nbins, padding)
            cn = _bin_normalize(mval[i], mmin, mdelta)
            c = _bin_index(cn, nbins, padding)
            spline_arg = (c - padding) - cn

            smarginal[r] += 1
            for offset in range(-padding, padding + 1):
                val = _cubic_spline(spline_arg)
                joint[r, c + offset] += val
                total_sum += val
                spline_arg += 1.0

        if total_sum > 0:
            for i in range(nbins):
                for j in range(nbins):
                    joint[i, j] /= total_sum

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
                spline_arg = (c - padding) - cn

                for offset in range(-padding, padding + 1):
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
                    spline_arg = (c - padding) - cn

                    for offset in range(-padding, padding + 1):
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
            spline_arg = (c - padding) - cn

            for offset in range(-padding, padding + 1):
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
            spline_arg = (c - padding) - cn

            for offset in range(-padding, padding + 1):
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


cdef _apply_mi_weights_to_cached_local_derivatives_2d(
        floating[:, :, :, :] local_derivative_by_parzen_bin,
        cnp.npy_intp[:, :] joint_pdf_index,
        int[:, :] smask, int[:, :] mmask,
        double[:, :] mi_weights, double mdelta, int nbins, int padding,
        floating[:, :, :] update_field):
    r"""Apply MI bin weights to cached 2D local-support derivatives.

    This function combines the MI weights computed from the joint PDF with the
    cached local derivative contributions produced by
    `_compute_pdfs_dense_and_local_derivatives_2d`. For each valid pixel, the
    flattened joint-PDF index is used to recover the affected joint histogram bins.
    The corresponding MI weights are multiplied by the cached local derivatives and
    accumulated into the dense update field.

    Parameters
    ----------
    local_derivative_by_parzen_bin : array, shape (2*padding + 1, R, C, 2)
        cached unweighted local derivative contribution of each pixel displacement
        component to each affected moving Parzen bin
    joint_pdf_index : array, shape (R, C)
        flattened index of the first joint-PDF bin affected by each pixel
    smask : array, shape (R, C)
        mask of static object being registered. Pixels with value 0 are ignored
        when applying the cached local derivative contributions.
    mmask : array, shape (R, C)
        mask of moving object being registered. Pixels with value 0 are ignored
        when applying the cached local derivative contributions.
    mi_weights : array, shape (nbins, nbins)
        MI derivative weight associated with each joint histogram bin
    mdelta : float
        bin size associated with the intensities of the moving image
    nbins : int
        number of histogram bins
    padding : int
        number of bins used as padding (the total bins used for padding at both
        sides of the histogram is actually 2*padding)
    update_field : array, shape (R, C, 2)
        the array to write the dense local-support MI update field
    """
    cdef:
        cnp.npy_intp nrows = update_field.shape[0]
        cnp.npy_intp ncols = update_field.shape[1]
        cnp.npy_intp i, j, offset_id, idx, r, c, valid_points
        double weight, norm_factor

    update_field[...] = 0

    with nogil:
        valid_points = 0
        for i in range(nrows):
            for j in range(ncols):
                if smask is not None and smask[i, j] == 0:
                    continue
                if mmask is not None and mmask[i, j] == 0:
                    continue

                valid_points += 1
                idx = joint_pdf_index[i, j]

                for offset_id in range(2 * padding + 1):
                    r = idx // nbins
                    c = idx - r * nbins
                    weight = mi_weights[r, c]

                    update_field[i, j, 0] += (
                        weight * local_derivative_by_parzen_bin[
                            offset_id, i, j, 0])
                    update_field[i, j, 1] += (
                        weight * local_derivative_by_parzen_bin[
                            offset_id, i, j, 1])

                    idx += 1

        norm_factor = valid_points * mdelta
        if norm_factor > 0:
            for i in range(nrows):
                for j in range(ncols):
                    update_field[i, j, 0] /= norm_factor
                    update_field[i, j, 1] /= norm_factor


cdef _apply_mi_weights_to_cached_local_derivatives_3d(
        floating[:, :, :, :, :] local_derivative_by_parzen_bin,
        cnp.npy_intp[:, :, :] joint_pdf_index,
        int[:, :, :] smask, int[:, :, :] mmask,
        double[:, :] mi_weights, double mdelta, int nbins, int padding,
        floating[:, :, :, :] update_field):
    r"""Apply MI bin weights to cached 3D local-support derivatives.

    This function combines the MI weights computed from the joint PDF with the
    cached local derivative contributions produced by
    `_compute_pdfs_dense_and_local_derivatives_3d`. For each valid voxel, the
    flattened joint-PDF index is used to recover the affected joint histogram bins.
    The corresponding MI weights are multiplied by the cached local derivatives and
    accumulated into the dense update field.

    Parameters
    ----------
    local_derivative_by_parzen_bin : array, shape (2*padding + 1, S, R, C, 3)
        cached unweighted local derivative contribution of each voxel displacement
        component to each affected moving Parzen bin
    joint_pdf_index : array, shape (S, R, C)
        flattened index of the first joint-PDF bin affected by each voxel
    smask : array, shape (S, R, C)
        mask of static object being registered. Voxels with value 0 are ignored
        when applying the cached local derivative contributions.
    mmask : array, shape (S, R, C)
        mask of moving object being registered. Voxels with value 0 are ignored
        when applying the cached local derivative contributions.
    mi_weights : array, shape (nbins, nbins)
        MI derivative weight associated with each joint histogram bin
    mdelta : float
        bin size associated with the intensities of the moving image
    nbins : int
        number of histogram bins
    padding : int
        number of bins used as padding (the total bins used for padding at both
        sides of the histogram is actually 2*padding)
    update_field : array, shape (S, R, C, 3)
        the array to write the dense local-support MI update field
    """
    cdef:
        cnp.npy_intp nslices = update_field.shape[0]
        cnp.npy_intp nrows = update_field.shape[1]
        cnp.npy_intp ncols = update_field.shape[2]
        cnp.npy_intp k, i, j, offset_id, idx, r, c, valid_points
        double weight, norm_factor

    update_field[...] = 0

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
                    idx = joint_pdf_index[k, i, j]

                    for offset_id in range(2 * padding + 1):
                        r = idx // nbins
                        c = idx - r * nbins
                        weight = mi_weights[r, c]

                        update_field[k, i, j, 0] += (
                            weight * local_derivative_by_parzen_bin[
                                offset_id, k, i, j, 0])
                        update_field[k, i, j, 1] += (
                            weight * local_derivative_by_parzen_bin[
                                offset_id, k, i, j, 1])
                        update_field[k, i, j, 2] += (
                            weight * local_derivative_by_parzen_bin[
                                offset_id, k, i, j, 2])

                        idx += 1

        norm_factor = valid_points * mdelta
        if norm_factor > 0:
            for k in range(nslices):
                for i in range(nrows):
                    for j in range(ncols):
                        update_field[k, i, j, 0] /= norm_factor
                        update_field[k, i, j, 1] /= norm_factor
                        update_field[k, i, j, 2] /= norm_factor


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


def compute_parzen_mi_weights(double[:, :] joint,
                              double[:] smarginal,
                              double[:] mmarginal,
                              double[:, :] mi_weights):
    r""" Computes the mutual information and its bin weights (if requested)

    Parameters
    ----------
    joint : array, shape (nbins, nbins)
        the joint intensity distribution
    smarginal : array, shape (nbins,)
        the marginal intensity distribution of the static image
    mmarginal : array, shape (nbins,)
        the marginal intensity distribution of the moving image
    mi_weights : array, shape (nbins, nbins), optional
        the buffer in which to write the mutual information weight associated
        with each joint bin. If None, the bin weights are not computed
    """
    cdef:
        double epsilon = 2.2204460492503131e-016
        double metric_value
        double factor
        cnp.npy_intp i, j
        cnp.npy_intp nrows = joint.shape[0]
        cnp.npy_intp ncols = joint.shape[1]

    with nogil:
        if mi_weights is not None:
            mi_weights[:, :] = 0
        metric_value = 0
        for i in range(nrows):
            for j in range(ncols):
                if joint[i, j] < epsilon or mmarginal[j] < epsilon:
                    continue

                factor = log(joint[i, j] / mmarginal[j])
                if mi_weights is not None:
                    mi_weights[i, j] = factor

                if smarginal[i] > epsilon:
                    metric_value += joint[i, j] * (factor - log(smarginal[i]))

    return metric_value


def sample_domain_regular(int k, int[:] shape, double[:, :] grid2world,
                          double sigma=0.25, object rng=None):
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

    if rng is None:
        rng = np.random.default_rng(1234)
    if dim == 2:
        n = shape[0] * shape[1]
        m = n // k
        samples = rng.standard_normal((m, dim)) * sigma
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
        samples = rng.standard_normal((m, dim)) * sigma
        with nogil:
            for i in range(m):
                s = ((i * k) // slice_size) + samples[i, 0]
                r = (((i * k) % slice_size) // shape[2]) + samples[i, 1]
                c = (((i * k) % slice_size) % shape[2]) + samples[i, 2]
                samples[i, 0] = _apply_affine_3d_x0(s, r, c, 1, grid2world)
                samples[i, 1] = _apply_affine_3d_x1(s, r, c, 1, grid2world)
                samples[i, 2] = _apply_affine_3d_x2(s, r, c, 1, grid2world)
    return np.asarray(samples)
