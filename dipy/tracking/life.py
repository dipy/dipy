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
import scipy.spatial.distance as dist

from dipy.reconst.base import ReconstModel, ReconstFit
from dipy.utils.six.moves import range
from dipy.tracking.utils import unique_rows
from dipy.tracking.streamline import transform_streamlines
import dipy.data as dpd
from dipy.tracking.vox2track import streamline_mapping, track_counts
from dipy.tracking.spdot import spdot, spdot_t, gradient_change

def grad_tensor(grad, evals):
    """
    Calculate the 3 by 3 tensor for a given spatial gradient.

    Parameters
    ----------
    grad : 1d array of shape (3,)
        The orientation of the tensor (for example, the spatial gradient
        between two nodes of a streamline).

    evals: 1d array of shape (3,)
        The eigenvalues of a canonical tensor to be used as a response
        function.

    """
    # This is the rotation matrix from [1, 0, 0] to this gradient of the sl:
    R = la.svd(np.matrix(grad), overwrite_a=True)[2]
    # This is the 3 by 3 tensor after rotation:
    T = np.dot(np.dot(R, np.diag(evals)), R.T)
    return T


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


    def signal_maker(self, evals=[0.001, 0, 0], sphere=None):
        """
        Initialize a signal maker

        Parameters
        ----------
        gtab : GradientTable class instance
            The gradient table on which the signal is calculated.
        evals : list of 3 items
            The eigenvalues of the canonical tensor to use in calculating the
            signal.
        """
        if sphere is None:
            sphere = dpd.get_sphere()

        bvecs = self.gtab.bvecs[~self.gtab.b0s_mask]
        bvals = self.gtab.bvals[~self.gtab.b0s_mask]
        # This will be the output
        signal = np.empty((sphere.vertices.shape[0], bvals.shape[0]))

        # Calculate for every direction on the sphere:
        for idx in range(sphere.vertices.shape[0]):
            tensor = grad_tensor(sphere.vertices[idx], evals)
            ADC = np.diag(np.dot(np.dot(bvecs, tensor), bvecs.T))
            sig = np.exp(-bvals * ADC)
            sig = sig - np.mean(sig)
            signal[idx] = sig

        return signal

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
            sphere=None, check_error_iter=5, converge_on_sse=0.8,
            max_error_checks=5, step_size=0.01):
        """
        Fit the LiFE model.

        Parameters
        ----------
        data : ndarray
        streamline : list
            Streamlines, each is an array of shape (n, 3)
        affine : 4 by 4 array
            Mapping from the streamline coordinates to the data
        evals : list (3 items, optional)
            The eigenvalues of the canonical tensor used as a response
            function. Default:[0.001, 0, 0].
        sphere: `dipy.core.Sphere` instance.
            Whether to approximate (and cache) the signal on a discrete
            sphere. This may confer a significant speed-up in setting up the
            problem, but is not as accurate. If `False`, we use the exact
            gradients along the streamlines to calculate the matrix, instead of
            an approximation. Defaults to use the :mod:`dipy.data`
            `default_sphere`.
        """
        if sphere is None:
            sphere = dpd.get_sphere()

        signal_maker = self.signal_maker(evals=evals,
                                         sphere=sphere)

        if affine is None:
            affine = np.eye(4)
        else:
            streamline = transform_streamlines(streamline, affine)

        sl_as_coords = [np.round(s).astype(np.intp) for s in streamline]
        cat_streamline = np.concatenate(sl_as_coords)
        vox_coords = unique_rows(cat_streamline)
        v2f = streamline_mapping(sl_as_coords, affine=np.eye(4))

        (to_fit, weighted_signal, b0_signal, relative_signal, mean_sig,
         vox_data) = self._signals(data, vox_coords)

        closest = {}
        for sl_idx, ss in enumerate(streamline):
            closest[sl_idx] = []
            for node_idx in range(ss.shape[0]):
                if node_idx == 0:
                    g = ss[1] - ss[0]
                elif node_idx == ss.shape[0]:
                    g = ss[-1] - ss[-2]
                else:
                    g = ss[node_idx] - ss[node_idx-1]
                closest[sl_idx].append(sphere.find_closest(g))

        # We only consider the diffusion-weighted signals in fitting:
        n_bvecs = self.gtab.bvals[~self.gtab.b0s_mask].shape[0]
        beta = np.zeros(len(streamline))
        range_bvecs = np.arange(n_bvecs).astype(np.intp)

        # Optimization-related stuff:
        iteration = 0
        ss_residuals_min = np.inf
        sse_best = np.inf
        error_checks = 0  # How many error checks have we done so far
        y_hat = np.zeros(to_fit.shape)

        # Cache some facts about relationship between voxels and fibers:
        max_s_per_vox = np.max([len(t) for t in v2f.values()])
        s_in_vox = {}
        for v_idx in range(vox_coords.shape[0]):
            s_in_vox[v_idx] = []
            mat_row_idx = (range_bvecs + v_idx * n_bvecs).astype(np.intp)
            this_vox = (vox_coords[v_idx][0], vox_coords[v_idx][1],
                        vox_coords[v_idx][2])
            # For each fiber in that voxel:
            for sl_idx in v2f[this_vox]:
                find_vox = np.logical_and(
                  np.logical_and(
                    sl_as_coords[sl_idx][:, 0] == vox_coords[v_idx][0],
                    sl_as_coords[sl_idx][:, 1] == vox_coords[v_idx][1]),
                  sl_as_coords[sl_idx][:, 2] == vox_coords[v_idx][2])
                nodes_in_vox = np.where(find_vox)[0]
                s_in_vox[v_idx].append((sl_idx, nodes_in_vox))

        row_col_list = np.array([np.zeros(n * n_bvecs, dtype=np.intp)
                                 for n in range(1, max_s_per_vox+1)])
        sigs_list = np.array([np.zeros(n * n_bvecs, dtype=np.float)
                              for n in range(1, max_s_per_vox+1)])
        while 1:
            delta = np.zeros(beta.shape)
            for v_idx in range(vox_coords.shape[0]):
                mat_row_idx = range_bvecs + v_idx * n_bvecs
                f_matrix_row = row_col_list[len(s_in_vox[v_idx])-1].copy()
                f_matrix_col = row_col_list[len(s_in_vox[v_idx])-1].copy()
                f_matrix_sig = sigs_list[len(s_in_vox[v_idx])-1].copy()
                for ii, (sl_idx, nodes_in_vox) in enumerate(s_in_vox[v_idx]):
                    ss = streamline[sl_idx]
                    f_matrix_row[ii*n_bvecs:ii*n_bvecs+n_bvecs] = range_bvecs
                    f_matrix_col[ii*n_bvecs:ii*n_bvecs+n_bvecs] = sl_idx
                    vox_fib_sig = np.zeros(n_bvecs)
                    for node_idx in nodes_in_vox:
                        signal_idx = closest[sl_idx][node_idx]
                        this_signal = signal_maker[signal_idx]
                        # Sum the signal from each node of the fiber in that
                        # voxel:
                        vox_fib_sig += this_signal
                    # And add the summed thing into the corresponding rows:
                    f_matrix_sig[ii*n_bvecs:ii*n_bvecs+n_bvecs] += vox_fib_sig

                if (iteration > 1 and
                   (np.mod(iteration, check_error_iter) == 0)):
                    y_hat[mat_row_idx] = spdot(f_matrix_row,
                                               f_matrix_col,
                                               f_matrix_sig,
                                               beta,
                                               f_matrix_row.shape[0],
                                               mat_row_idx.shape[0])
                else:
                    XtXh = gradient_change(f_matrix_row,
                                           f_matrix_col,
                                           f_matrix_sig, beta,
                                           to_fit[mat_row_idx],
                                           mat_row_idx.shape[0],
                                           delta.shape[0])
                    delta = delta + XtXh

            if iteration > 1 and (np.mod(iteration, check_error_iter) == 0):
                sse = np.sum((to_fit - y_hat) ** 2)
                # Did we do better this time around?
                if sse < ss_residuals_min:
                    # Update your expectations about the minimum error:
                    ss_residuals_min = sse
                    beta_best = beta
                    # Are we generally (over iterations) converging?
                    if sse < sse_best * converge_on_sse:
                        sse_best = sse
                        count_bad = 0
                    else:
                        count_bad += 1
                else:
                    count_bad += 1
                if count_bad >= max_error_checks:
                    return FiberFit(self,
                                    vox_coords,
                                    to_fit,
                                    beta_best,
                                    weighted_signal,
                                    b0_signal,
                                    relative_signal,
                                    mean_sig,
                                    vox_data,
                                    streamline,
                                    affine,
                                    evals,
                                    v2f,
                                    closest,
                                    s_in_vox)
                error_checks += 1
            else:
                beta = beta - step_size * delta
                # Set negative values to 0 (non-negative!)
                beta[beta < 0] = 0
            iteration += 1


class FiberFit(ReconstFit):
    """
    A fit of the LiFE model to diffusion data
    """
    def __init__(self, fiber_model, vox_coords, to_fit, beta,
                 weighted_signal, b0_signal, relative_signal, mean_sig,
                 vox_data, streamline, affine, evals, v2f, closest, s_in_vox):
        """
        Parameters
        ----------
        fiber_model : A FiberModel class instance

        params : the parameters derived from a fit of the model to the data.

        """
        ReconstFit.__init__(self, fiber_model, vox_data)

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
        self.v2f = v2f
        self.closest = closest
        self.s_in_vox = s_in_vox

    def predict(self, gtab=None, S0=None, sphere=None):
        """
        Predict the signal

        Parameters
        ----------
        gtab : GradientTable
            Default: use self.gtab
        S0 : float or array
            The non-diffusion-weighted signal in the voxels for which a
            prediction is made. Default: use self.b0_signal

        Returns
        -------
        prediction : ndarray of shape (voxels, bvecs)
            An array with a prediction of the signal in each voxel/direction
        """
        # We generate the prediction and in each voxel, we add the
        # offset, according to the isotropic part of the signal, which was
        # removed prior to fitting:
        if gtab is None:
            gtab = self.model.gtab

        if sphere is None:
            sphere = dpd.get_sphere()

        signal_maker = self.model.signal_maker(evals=self.evals,
                                               sphere=sphere)

        n_bvecs = gtab.bvals[~gtab.b0s_mask].shape[0]
        f_matrix_shape = (self.fit_data.shape[0], len(self.streamline))
        range_bvecs = np.arange(n_bvecs).astype(int)
        pred_weighted = np.zeros(self.fit_data.shape)

        for v_idx in range(self.vox_coords.shape[0]):
                mat_row_idx = (range_bvecs + v_idx * n_bvecs).astype(np.intp)
                s_in_vox = self.s_in_vox[v_idx]
                f_matrix_row = np.zeros(len(s_in_vox) *
                                        n_bvecs, dtype=np.intp)
                f_matrix_col = np.zeros(len(s_in_vox) *
                                        n_bvecs, dtype=np.intp)
                f_matrix_sig = np.zeros(len(s_in_vox) *
                                        n_bvecs, dtype=np.float)
                for ii, (sl_idx, nodes_in_vox) in enumerate(s_in_vox):
                    ss = self.streamline[sl_idx]
                    f_matrix_row[ii*n_bvecs:ii*n_bvecs+n_bvecs] = range_bvecs
                    f_matrix_col[ii*n_bvecs:ii*n_bvecs+n_bvecs] = sl_idx
                    vox_fib_sig = np.zeros(n_bvecs)
                    for node_idx in nodes_in_vox:
                        signal_idx = self.closest[sl_idx][node_idx]
                        this_signal = signal_maker[signal_idx]
                        # Sum the signal from each node of the fiber in that
                        # voxel:
                        vox_fib_sig += this_signal
                    # And add the summed thing into the corresponding rows:
                    f_matrix_sig[ii*n_bvecs:ii*n_bvecs+n_bvecs] += vox_fib_sig

                pred_weighted[mat_row_idx] = spdot(f_matrix_row,
                                                   f_matrix_col,
                                                   f_matrix_sig,
                                                   self.beta,
                                                   f_matrix_row.shape[0],
                                                   mat_row_idx.shape[0])

        pred = np.empty((self.vox_coords.shape[0], gtab.bvals.shape[0]))
        pred[..., ~gtab.b0s_mask] = pred_weighted.reshape(
                                            pred[..., ~gtab.b0s_mask].shape)
        if S0 is None:
            S0 = self.b0_signal

        pred[..., gtab.b0s_mask] = S0[:, None]
        pred[..., ~gtab.b0s_mask] =\
            (pred[..., ~gtab.b0s_mask] +
             self.mean_signal[:, None]) * S0[:, None]

        return pred
