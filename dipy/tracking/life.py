"""
This is an implementation of the Linear Fascicle Evaluation (LiFE) algorithm
described in:

Pestilli, F., Yeatman, J, Rokem, A. Kay, K. and Wandell B.A. (2014). Validation
and statistical inference in living connectomes. Nature Methods 11:
1058-1063. doi:10.1038/nmeth.3098
"""
import numpy as np
import scipy.sparse as sps
import scipy.linalg as la

from dipy.reconst.base import ReconstModel, ReconstFit
import dipy.core.sphere as dps
from dipy.tracking.utils import unique_rows, transform_sl
from dipy.tracking.vox2track import _voxel2streamline
import dipy.data as dpd


def gradient(f, *varargs):
    """
    Return the gradient of an N-dimensional array.

    The gradient is computed using central differences in the interior
    and first differences at the boundaries. The returned gradient hence has
    the same shape as the input array.

    Parameters
    ----------
    f : array_like
      An N-dimensional array containing samples of a scalar function.
    `*varargs` : scalars
      0, 1, or N scalars specifying the sample distances in each direction,
      that is: `dx`, `dy`, `dz`, ... The default distance is 1.


    Returns
    -------
    gradient : ndarray
      N arrays of the same shape as `f` giving the derivative of `f` with
      respect to each dimension.

    Examples
    --------
    >>> x = np.array([1, 2, 4, 7, 11, 16], dtype=np.float)
    >>> gradient(x)
    array([ 1. ,  1.5,  2.5,  3.5,  4.5,  5. ])
    >>> gradient(x, 2)
    array([ 0.5 ,  0.75,  1.25,  1.75,  2.25,  2.5 ])

    >>> gradient(np.array([[1, 2, 6], [3, 4, 5]], dtype=np.float))
    [array([[ 2.,  2., -1.],
           [ 2.,  2., -1.]]), array([[ 1. ,  2.5,  4. ],
           [ 1. ,  1. ,  1. ]])]

    Note
    ----
    This is the implementation of gradient that is part of numpy 1.8. In order
    to mitigate the effects of changes added to this implementation in version
    1.9 of numpy, we include this implementation here.

    """
    f = np.asanyarray(f)
    N = len(f.shape)  # number of dimensions
    n = len(varargs)
    if n == 0:
        dx = [1.0]*N
    elif n == 1:
        dx = [varargs[0]]*N
    elif n == N:
        dx = list(varargs)
    else:
        raise SyntaxError(
          "invalid number of arguments")

    # use central differences on interior and first differences on endpoints
    outvals = []

    # create slice objects --- initially all are [:, :, ..., :]
    slice1 = [slice(None)]*N
    slice2 = [slice(None)]*N
    slice3 = [slice(None)]*N

    for axis in range(N):
        # select out appropriate parts for this dimension
        out = np.empty_like(f)
        slice1[axis] = slice(1, -1)
        slice2[axis] = slice(2, None)
        slice3[axis] = slice(None, -2)
        # 1D equivalent -- out[1:-1] = (f[2:] - f[:-2])/2.0
        out[slice1] = (f[slice2] - f[slice3])/2.0
        slice1[axis] = 0
        slice2[axis] = 1
        slice3[axis] = 0
        # 1D equivalent -- out[0] = (f[1] - f[0])
        out[slice1] = (f[slice2] - f[slice3])
        slice1[axis] = -1
        slice2[axis] = -1
        slice3[axis] = -2
        # 1D equivalent -- out[-1] = (f[-1] - f[-2])
        out[slice1] = (f[slice2] - f[slice3])

        # divide by step size
        outvals.append(out / dx[axis])
        # reset the slice object in this dimension to ":"
        slice1[axis] = slice(None)
        slice2[axis] = slice(None)
        slice3[axis] = slice(None)

    if N == 1:
        return outvals[0]
    else:
        return outvals


def spdot(A, B):
    """The same as np.dot(A, B), except it works even if A or B or both
    are sparse matrices.

    Parameters
    ----------
    A, B : arrays of shape (m, n), (n, k)

    Returns
    -------
    The matrix product AB.

    See discussion here:
    http://mail.scipy.org/pipermail/scipy-user/2010-November/027700.html
    """
    if sps.issparse(A) and sps.issparse(B):
        return A * B
    elif sps.issparse(A) and not sps.issparse(B):
        return (A * B).view(type=B.__class__)
    elif not sps.issparse(A) and sps.issparse(B):
        return (B.T * A.T).T.view(type=A.__class__)
    else:
        return np.dot(A, B)


def rsq(ss_residuals, ss_residuals_to_mean):
    """
    Calculate: $R^2 = \frac{1-SSE}{\sigma^2}$

    Parameters
    ----------
    ss_residuals : array
        Model fit errors relative to the data
    ss_residuals_to_mean : array
        Residuals of the data relative to the mean of the data (variance)

    Returns
    -------
    rsq : the variance explained.
    """
    return 100 * (1 - ss_residuals/ss_residuals_to_mean)


def sparse_nnls(y, X,
                momentum=0,
                step_size=0.1,
                non_neg=True,
                prop_bad_checks=0.1,
                check_error_iter=10,
                max_error_checks=10,
                converge_on_r=0.2):
    """

    Solve y=Xh for h, using gradient descent, with X a sparse matrix

    Parameters
    ----------

    y : 1-d array of shape (N)
        The data

    X : ndarray. May be either sparse or dense. Shape (N, M)
       The regressors

    momentum : float, optional (default: 0).
        The persistence of the gradient.

    step_size : float, optional (default: 0.05).
        The increment of parameter update in each iteration

    non_neg : Boolean, optional (default: True)
        Whether to enforce non-negativity of the solution.

    prop_bad_checks : float (default: 0.1)
       If this proportion of error checks so far has not yielded an improvement
       in r squared, we halt the optimization.

    check_error_iter : int (default:10)
        How many rounds to run between error evaluation for
        convergence-checking.

    max_error_checks : int (default: 10)
        Don't check errors more than this number of times if no improvement in
        r-squared is seen.

    converge_on_r : float (default: 1)
      a percentage improvement in rsquared that is required each time to say
      that things are still going well.

    Returns
    -------
    h_best : The best estimate of the parameters.

    """
    num_data = y.shape[0]
    num_regressors = X.shape[1]
    # Initialize the parameters at the origin:
    h = np.zeros(num_regressors)
    # If nothing good happens, we'll return that in the end:
    h_best = np.zeros(num_regressors)
    gradient = np.zeros(num_regressors)
    iteration = 1
    count = 1
    ss_residuals_min = np.inf  # This will keep track of the best solution
    ss_residuals_to_mean = np.sum((y - np.mean(y)) ** 2)  # The variance of y
    rsq_max = -np.inf   # This will keep track of the best r squared so far
    count_bad = 0  # Number of times estimation error has gone up.
    error_checks = 0  # How many error checks have we done so far

    while 1:
        if iteration > 1:
            # The sum of squared error given the current parameter setting:
            sse = np.sum((y - spdot(X, h)) ** 2)
            # The gradient is (Kay 2008 supplemental page 27):
            gradient = spdot(X.T, spdot(X, h) - y)
            gradient += momentum * gradient
            # Normalize to unit-length
            unit_length_gradient = (gradient /
                                    np.sqrt(np.dot(gradient, gradient)))
            # Update the parameters in the direction of the gradient:
            h -= step_size * unit_length_gradient
            if non_neg:
                # Set negative values to 0:
                h[h < 0] = 0

        # Every once in a while check whether it's converged:
        if np.mod(iteration, check_error_iter):
            # This calculates the sum of squared residuals at this point:
            this_ss_resid = (np.sum(np.power(y - spdot(X, h), 2)))
            rsq_est = rsq(this_ss_resid, ss_residuals_to_mean)

            # Did we do better this time around?
            if this_ss_resid < ss_residuals_min:
                # Update your expectations about the minimum error:
                ss_residuals_min = this_ss_resid
                n_iterations = iteration  # This holds the number of iterations
                                          # for the best solution so far.
                h_best = h  # This holds the best params we have so far

                # Are we generally (over iterations) converging on
                # improvement in r-squared?
                if rsq_est > (rsq_max * (1 + converge_on_r / 100)):
                    rsq_max = rsq_est
                    count_bad = 0  # We're doing good. Null this count for now
                else:
                    count_bad += 1
            else:
                count_bad += 1

            if count_bad >= np.max([max_error_checks,
                                    np.round(prop_bad_checks * error_checks)]):
                return h_best
            error_checks += 1
        iteration += 1


def streamline_gradients(streamline):
    """
    Calculate the gradients of the streamline along the spatial dimension

    Parameters
    ----------
    streamline : array-like of shape (n, 3)
        The 3d coordinates of a single streamline

    Returns
    -------
    Array of shape (3, n): Spatial gradients along the length of the
    streamline.

    """
    return np.array(gradient(np.asarray(streamline), 1)[0])


def grad_tensor(grad, evals):
    """
    Calculate the 3 by 3 tensor for a given spatial gradient, given a canonical
    tensor shape (also as a 3 by 3), pointing at [1,0,0]

    Parameters
    ----------
    grad : 1d array of shape (3,)
        The spatial gradient (e.g between two nodes of a streamline).

    evals: 1d array of shape (3,)
        The eigenvalues of a canonical tensor to be used as a response
        function.

    """
    # This is the rotation matrix from [1, 0, 0] to this gradient of the sl:
    R = la.svd(np.matrix(grad), overwrite_a=True)[2]
    # This is the 3 by 3 tensor after rotation:
    T = np.dot(np.dot(R, np.diag(evals)), R.T)
    return T


def streamline_tensors(streamline, evals=[0.001, 0, 0]):
    """
    The tensors generated by this fiber.

    Parameters
    ----------
    streamline : array-like of shape (n, 3)
        The 3d coordinates of a single streamline

    evals : iterable with three entries
        The estimated eigenvalues of a single fiber tensor.
        (default: [0.001, 0, 0]).

    Returns
    -------
    An n_nodes by 3 by 3 array with the tensor for each node in the fiber.

    Note
    ----
    Estimates of the radial/axial diffusivities may rely on
    empirical measurements (for example, the AD in the Corpus Callosum), or
    may be based on a biophysical model of some kind.
    """

    grad = streamline_gradients(streamline)

    # Preallocate:
    tensors = np.empty((grad.shape[0], 3, 3))

    for grad_idx, this_grad in enumerate(grad):
        tensors[grad_idx] = grad_tensor(this_grad, evals)
    return tensors


def streamline_signal(streamline, gtab, evals=[0.001, 0, 0]):
    """
    The signal from a single streamline estimate along each of its nodes.

    Parameters
    ----------
    streamline : a single streamline

    gtab : GradientTable class instance

    evals : list of length 3 (optional. Default: [0.0015, 0.0005, 0.0005])
        The eigenvalues of the canonical tensor used as an estimate of the
        signal generated by each node of the streamline.
    """
    # Gotta have those tensors:
    tensors = streamline_tensors(streamline, evals)
    sig = np.empty((len(streamline), np.sum(~gtab.b0s_mask)))
    # Extract them once:
    bvecs = gtab.bvecs[~gtab.b0s_mask]
    bvals = gtab.bvals[~gtab.b0s_mask]
    for ii, tensor in enumerate(tensors):
        ADC = np.diag(np.dot(np.dot(bvecs, tensor), bvecs.T))
        # Use the Stejskal-Tanner equation with the ADC as input, and S0 = 1:
        sig[ii] = np.exp(-bvals * ADC)
    return sig


class LifeSignalMaker(object):
    """
    A class for generating signals from streamlines in an efficient and speedy
    manner.
    """
    def __init__(self, gtab, evals=[0.0015, 0.0005, 0.0005], sphere=None):
        """
        Initialize a signal maker

        Parameters
        ----------
        gtab : GradientTable class instance
            The gradient table on which the signal is calculated.
        evals : list of 3 items
            The eigenvalues of the canonical tensor to use in calculating the
            signal.
        n_points : `dipy.core.Sphere` class instance
            The discrete sphere to use as an approximation for the continuous
            sphere on which the signal is represented. If integer - we will use
            an instance of one of the symmetric spheres cached in
            `dps.get_sphere`. If a 'dipy.core.Sphere' class instance is
            provided, we will use this object. Default: the :mod:`dipy.data`
            symmetric sphere with 724 vertices
        """
        if sphere is None:
            self.sphere = dpd.get_sphere('symmetric724')
        else:
            self.sphere = sphere

        self.gtab = gtab
        self.evals = evals
        # Initialize an empty dict to fill with signals for each of the sphere
        # vertices:
        self.signal = np.empty((self.sphere.vertices.shape[0],
                                np.sum(~gtab.b0s_mask)))
        # We'll need to keep track of what we've already calculated:
        self._calculated = []

    def find_closest(self, xyz):
        """
        Find the index of the closest point on the discrete sphere to an input
        vector

        Parameters
        ----------
        xyz : 1d array, shape (3,)

        Returns
        -------
        The index into `self.sphere.vertices` to the vertex closest to the
        input vector, taking into account antipodal symmetry.
        """
        ang = np.arccos(np.dot(self.sphere.vertices, xyz))
        ang = np.min(np.vstack([ang, np.pi - ang]), 0)
        return np.argmin(ang)

    def calc_signal(self, xyz):
        idx = self.find_closest(xyz)
        if idx not in self._calculated:
            bvecs = self.gtab.bvecs[~self.gtab.b0s_mask]
            bvals = self.gtab.bvals[~self.gtab.b0s_mask]
            tensor = grad_tensor(self.sphere.vertices[idx], self.evals)
            ADC = np.diag(np.dot(np.dot(bvecs, tensor), bvecs.T))
            self.signal[idx] = np.exp(-bvals * ADC)
            self._calculated.append(idx)

        return self.signal[idx]

    def streamline_signal(self, streamline):
        """
        Approximate the signal for a given streamline
        """
        grad = streamline_gradients(streamline)
        return [self.calc_signal(g) for g in grad]


def voxel2streamline(streamline, transformed=False, affine=None, unique_idx=None):
    """
    Maps voxels to streamlines and streamlines to voxels, for setting up
    the LiFE equations matrix

    Parameters
    ----------
    streamline : list
        A collection of streamlines, each n by 3, with n being the number of
        nodes in the fiber.

    affine : 4 by 4 array (optional)
       Defines the spatial transformation from streamline to data. Default: np.eye(4)

    transformed : bool (optional)
        Whether the streamlines have been already transformed (in which case
        they don't need to be transformed in here).

    unique_idx : array (optional).
       The unique indices in the streamlines

    Returns
    -------
    v2f, v2fn : tuple of arrays

    The first array in the tuple answers the question: Given a voxel (from
    the unique indices in this model), which fibers pass through it? Shape:
    (n_voxels, n_fibers).

    The second answers the question: Given a voxel, for each fiber, which
    nodes are in that voxel? Shape: (n_voxels, max(n_nodes per fiber)).
    """
    if transformed:
        transformed_streamline = streamline
    else:
        transformed_streamline = transform_sl(streamline, affine=affine)

    if unique_idx is None:
        all_coords = np.concatenate(transformed_streamline)
        unique_idx = unique_rows(all_coords.astype(int))
    else:
        unique_idx = unique_idx

    return _voxel2streamline(transformed_streamline, unique_idx)


class FiberModel(ReconstModel):
    """
    A class for representing and solving predictive models based on
    tractography solutions.

    Notes
    -----
    This is an implementation of the LiFE model described in [1]_

    [1] Pestilli, F., Yeatman, J, Rokem, A. Kay, K. and Wandell
        B.A. (2014). Validation and statistical inference in living
        connectomes. Nature Methods.
    """
    def __init__(self, gtab):
        """
        Parameters
        ----------
        gtab : a GradientTable class instance

        """
        # Initialize the super-class:
        ReconstModel.__init__(self, gtab)

    def setup(self, streamline, affine, evals=[0.001, 0, 0], sphere=None):
        """
        Set up the necessary components for the LiFE model: the matrix of
        fiber-contributions to the DWI signal, and the coordinates of voxels
        for which the equations will be solved

        Parameters
        ----------
        streamline : list
            Streamlines, each is an array of shape (n, 3)
        affine : 4 by 4 array
            Mapping from the streamline coordinates to the data
        evals : list (3 items, optional)
            The eigenvalues of the canonical tensor used as a response function

        sphere: `dipy.core.Sphere` instance.
            Whether to approximate (and cache) the signal on a discrete
            sphere. This may confer a significant speed-up in setting up the
            problem, but is not as accurate. If `False`, we use the exact
            gradients along the streamlines to calculate the matrix, instead of
            an approximation. Defaults to use the 724-vertex symmetric sphere
            from :mod:`dipy.data`
        """
        if sphere is not False:
            SignalMaker = LifeSignalMaker(self.gtab,
                                          evals=evals,
                                          sphere=sphere)

        streamline = transform_sl(streamline, affine)
        # Assign some local variables, for shorthand:
        all_coords = np.concatenate(streamline)
        vox_coords = unique_rows(all_coords.astype(int))
        n_vox = vox_coords.shape[0]
        # We only consider the diffusion-weighted signals:
        n_bvecs = self.gtab.bvals[~self.gtab.b0s_mask].shape[0]

        v2f, v2fn = voxel2streamline(streamline, transformed=True, affine=affine,
                                unique_idx=vox_coords)

        # How many fibers in each voxel (this will determine how many
        # components are in the fiber part of the matrix):
        n_unique_f = np.sum(v2f)

        # Preallocate these, which will be used to generate the two sparse
        # matrices:

        # This one will hold the fiber-predicted signal
        f_matrix_sig = np.zeros(n_unique_f * n_bvecs)
        f_matrix_row = np.zeros(n_unique_f * n_bvecs)
        f_matrix_col = np.zeros(n_unique_f * n_bvecs)

        keep_ct = 0
        if sphere is not False:
            fiber_signal = [SignalMaker.streamline_signal(s) for s in streamline]
        else:
            fiber_signal = [streamline_signal(s, self.gtab, evals) for s in streamline]

        # In each voxel:
        for v_idx, vox in enumerate(vox_coords):
            # dbg:
            # print(100*float(v_idx)/n_vox)
            # For each fiber:
            for f_idx in np.where(v2f[v_idx])[0]:
                # Sum the signal from each node of the fiber in that voxel:
                vox_fiber_sig = np.zeros(n_bvecs)
                for node_idx in np.where(v2fn[f_idx] == v_idx)[0]:
                    this_signal = fiber_signal[f_idx][node_idx]
                    vox_fiber_sig += (this_signal - np.mean(this_signal))
                # For each fiber-voxel combination, we now store the row/column
                # indices and the signal in the pre-allocated linear arrays
                f_matrix_row[keep_ct:keep_ct+n_bvecs] =\
                    np.arange(n_bvecs) + v_idx * n_bvecs
                f_matrix_col[keep_ct:keep_ct+n_bvecs] =\
                    np.ones(n_bvecs) * f_idx
                f_matrix_sig[keep_ct:keep_ct+n_bvecs] = vox_fiber_sig
                keep_ct += n_bvecs

        # Allocate the sparse matrix, using the more memory-efficient 'csr'
        # format (converted from the coo format, which we rely on for the
        # initial allocation):
        life_matrix = sps.coo_matrix((f_matrix_sig,
                                      [f_matrix_row, f_matrix_col])).tocsr()

        return life_matrix, vox_coords

    def _signals(self, data, vox_coords):
        """
        Helper function to extract and separate all the signals we need to fit
        and evaluate a fit of this model

        Parameters
        ----------
        data : 4D array

        vox_coords: n by 3 array
            The coordinates into the data array of the fiber nodes.
        """
        # Fitting is done on the S0-normalized-and-demeaned diffusion-weighted
        # signal:
        idx_tuple = (vox_coords[:, 0], vox_coords[:, 1], vox_coords[:, 2])
        # We'll look at a 2D array, extracting the data from the voxels:
        vox_data = data[idx_tuple]
        weighted_signal = vox_data[:, ~self.gtab.b0s_mask]
        b0_signal = np.mean(vox_data[:, self.gtab.b0s_mask], -1)
        relative_signal = (weighted_signal/b0_signal[:, None])

        # The mean of the relative signal across directions in each voxel:
        mean_sig = np.mean(relative_signal, -1)
        to_fit = (relative_signal - mean_sig[:, None]).ravel()

        return (to_fit, weighted_signal, b0_signal, relative_signal, mean_sig,
                vox_data)

    def fit(self, data, streamline, affine=None, evals=[0.001, 0, 0],
            sphere=None):
        """
        Fit the LiFE FiberModel for data and a set of streamlines associated
        with this data

        Parameters
        ----------
        data : 4D array
           Diffusion-weighted data

        streamline : list
           A bunch of streamlines

        affine: 4 by 4 array (optional)
           The affine to go from the streamline coordinates to the data
           coordinates. Defaults to use `np.eye(4)`

        evals : list (optional)
           The eigenvalues of the tensor response function used in constructing
           the model signal. Default: [0.0015, 0.0005, 0.0005]

        sphere: `dipy.core.Sphere` instance, or False
            Whether to approximate (and cache) the signal on a discrete
            sphere. This may confer a significant speed-up in setting up the
            problem, but is not as accurate. If `False`, we use the exact
            gradients along the streamlines to calculate the matrix, instead of
            an approximation.

        Returns
        -------
        FiberFit class instance
        """
        life_matrix, vox_coords = \
            self.setup(streamline, affine, evals=evals, sphere=sphere)

        to_fit, weighted_signal, b0_signal, relative_signal, mean_sig, vox_data=\
            self._signals(data, vox_coords)
        beta = sparse_nnls(to_fit, life_matrix)
        return FiberFit(self, life_matrix, vox_coords, to_fit, beta,
                        weighted_signal, b0_signal, relative_signal, mean_sig,
                        vox_data, streamline, affine, evals)


class FiberFit(ReconstFit):
    """
    A fit of the LiFE model to diffusion data
    """
    def __init__(self, fiber_model, life_matrix, vox_coords, to_fit, beta,
                 weighted_signal, b0_signal, relative_signal, mean_sig,
                 vox_data, streamline, affine, evals):
        """
        Parameters
        ----------
        fiber_model : A FiberModel class instance

        params : the parameters derived from a fit of the model to the data.

        """
        ReconstFit.__init__(self, fiber_model, vox_data)

        self.life_matrix = life_matrix
        self.vox_coords = vox_coords
        self.fit_data = to_fit
        self.beta = beta
        self.weighted_signal = weighted_signal
        self.b0_signal = b0_signal
        self.relative_signal = relative_signal
        self.mean_signal = mean_sig
        self.streamline = streamline
        self.affine = affine
        self.evals = evals

    def predict(self, gtab=None, S0=None):
        """
        Predict the signal

        Parameters
        ----------
        """
        # We generate the prediction and in each voxel, we add the
        # offset, according to the isotropic part of the signal, which was
        # removed prior to fitting:

        if gtab is None:
            _matrix = self.life_matrix
            gtab = self.model.gtab
        else:
            _model = FiberModel(gtab)
            _matrix, _ = self.model.setup(self.streamline,
                                          self.affine,
                                          self.evals)
        pred_weighted = np.reshape(spdot(_matrix, self.beta),
                                   (self.vox_coords.shape[0],
                                    np.sum(~gtab.b0s_mask)))

        pred = np.empty((self.vox_coords.shape[0], gtab.bvals.shape[0]))
        if S0 is None:
            S0 = self.b0_signal

        pred[..., gtab.b0s_mask] = S0[:, None]
        pred[..., ~gtab.b0s_mask] =\
            (pred_weighted + self.mean_signal[:, None]) * S0[:, None]

        return pred
