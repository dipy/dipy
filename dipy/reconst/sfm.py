"""
This is an implementation of the sparse fascicle model described in
[Rokem2014]_.


.. [Rokem2014] Ariel Rokem, Jason D. Yeatman, Franco Pestilli, Kendrick
   N. Kay, Aviv Mezer, Stefan van der Walt, Brian A. Wandell
   (2014). Evaluating the accuracy of diffusion MRI models in white
   matter. http://arxiv.org/abs/1411.0721

"""
import warnings

import numpy as np
from dipy.utils.optpkg import optional_package
import dipy.core.geometry as geo
import dipy.core.gradients as grad
import dipy.core.optimize as opt
import dipy.sims.voxel as sims
import dipy.reconst.dti as dti
import dipy.data as dpd
from dipy.reconst.base import ReconstModel, ReconstFit
from dipy.reconst.cache import Cache
from dipy.core.onetime import auto_attr

lm, has_sklearn, _ = optional_package('sklearn.linear_model')

# If sklearn is unavailable, we can fall back on nnls (but we also warn the
# user that we are about to do that):
if not has_sklearn:
    w = "sklearn is not available, you can use 'nnls' method to fit"
    w += " the SparseFascicleModel"
    warnings.warn(w)


def sfm_design_matrix(gtab, sphere, response, mode='signal'):
    """
    Construct the SFM design matrix

    Parameters
    ----------
    gtab : GradientTable or Sphere
        Sets the rows of the matrix, if the mode is 'signal', this should be a
        GradientTable. If mode is 'odf' this should be a Sphere
    sphere : Sphere
        Sets the columns of the matrix
    response : list of 3 elements
        The eigenvalues of a tensor which will serve as a kernel
        function.
    mode : str {'signal' | 'odf'}
        Choose the (default) 'signal' for a design matrix containing predicted
        signal in the measurements defined by the gradient table for putative
        fascicles oriented along the vertices of the sphere. Otherwise, choose
        'odf' for an odf convolution matrix, with values of the odf calculated
        from a tensor with the provided response eigenvalues, evaluated at the
        b-vectors in the gradient table, for the tensors with prinicipal
        diffusion directions along the vertices of the sphere.

    Returns
    -------
    mat : ndarray
        A design matrix that can be used for one of the following operations:
        when the 'signal' mode is used, each column contains the putative
        signal in each of the bvectors of the `gtab` if a fascicle is oriented
        in the direction encoded by the sphere vertex corresponding to this
        column. This is used for deconvolution with a measured DWI signal. If
        the 'odf' mode is chosen, each column instead contains the values of
        the tensor ODF for a tensor with a principal diffusion direction
        corresponding to this vertex. This is used to generate odfs from the
        fits of the SFM for the purpose of tracking.

    Examples
    --------
    >>> import dipy.data as dpd
    >>> data, gtab = dpd.dsi_voxels()
    >>> sphere = dpd.get_sphere()
    >>> from dipy.reconst.sfm import sfm_design_matrix

    A canonical tensor approximating corpus-callosum voxels [Rokem2014]_:

    >>> tensor_matrix=sfm_design_matrix(gtab, sphere, [0.0015, 0.0005, 0.0005])

    A 'stick' function ([Behrens2007]_):

    >>> stick_matrix = sfm_design_matrix(gtab, sphere, [0.001, 0, 0])

    Notes
    -----
    .. [Rokem2014] Ariel Rokem, Jason D. Yeatman, Franco Pestilli, Kendrick
       N. Kay, Aviv Mezer, Stefan van der Walt, Brian A. Wandell
       (2014). Evaluating the accuracy of diffusion MRI models in white
       matter. http://arxiv.org/abs/1411.0721

    .. [Behrens2007] Behrens TEJ, Berg HJ, Jbabdi S, Rushworth MFS, Woolrich MW
       (2007): Probabilistic diffusion tractography with multiple fibre
       orientations: What can we gain? Neuroimage 34:144-55.
    """
    # Each column of the matrix is the signal in each measurement, as
    # predicted by a "canonical", symmetrical tensor rotated towards this
    # vertex of the sphere:
    canonical_tensor = np.diag(response)

    if mode == 'signal':
        mat_gtab = grad.gradient_table(gtab.bvals[~gtab.b0s_mask],
                                       gtab.bvecs[~gtab.b0s_mask])
        # Preallocate:
        mat = np.empty((np.sum(~gtab.b0s_mask),
                        sphere.vertices.shape[0]))
    elif mode == 'odf':
        mat = np.empty((gtab.x.shape[0], sphere.vertices.shape[0]))

    # Calculate column-wise:
    for ii, this_dir in enumerate(sphere.vertices):
        # Rotate the canonical tensor towards this vertex and calculate the
        # signal you would have gotten in the direction
        rot_matrix = geo.vec2vec_rotmat(np.array([1, 0, 0]), this_dir)
        this_tensor = np.dot(rot_matrix, canonical_tensor)
        evals, evecs = dti.decompose_tensor(this_tensor)
        if mode == 'signal':
            sig = sims.single_tensor(mat_gtab, evals=response, evecs=evecs)
            mat[:, ii] = sig - np.mean(sig)
        elif mode == 'odf':
            # Stick function
            if response[1] == 0 or response[2] == 0:
                jj = sphere.find_closest(evecs[0])
                mat[jj, ii] = 1
            else:
                odf = sims.single_tensor_odf(gtab.vertices,
                                             evals=response, evecs=evecs)
                mat[:, ii] = odf
    return mat


class SparseFascicleModel(ReconstModel, Cache):
    def __init__(self, gtab, sphere=None, response=[0.0015, 0.0005, 0.0005],
                 solver='ElasticNet', l1_ratio=0.5, alpha=0.001):
        """
        Initialize a Sparse Fascicle Model

        Parameters
        ----------
        gtab : GradientTable class instance
        sphere : Sphere class instance
        response : (3,) array-like
            The eigenvalues of a canonical tensor to be used as the response
            function of single-fascicle signals.
            Default:[0.0015, 0.0005, 0.0005]

        solver : string or SKLearnLinearSolver object.
            This will determine the algorithm used to solve the set of linear
            equations underlying this model. If it is a string it needs to be
            one of the following: {'ElasticNet', 'NNLS'}. Otherwise, it can be
            an object that inherits from `dipy.optimize.SKLearnLinearSolver`

        l1_ratio : float
            Sets the balance betwee L1 and L2 regularization in ElasticNet
            [Zou2005]_.
        alpha : float
            Sets the balance between least-squares error and L1/L2
            regularization in ElasticNet [Zou2005]_.

        Notes
        -----
        This is an implementation of the SFM, described in [Rokem2014]_.

        .. [Rokem2014] Ariel Rokem, Jason D. Yeatman, Franco Pestilli, Kendrick
           N. Kay, Aviv Mezer, Stefan van der Walt, Brian A. Wandell
           (2014). Evaluating the accuracy of diffusion MRI models in white
           matter. http://arxiv.org/abs/1411.0721

        .. [Zou2005] Zou H, Hastie T (2005). Regularization and variable
           selection via the elastic net. J R Stat Soc B:301-320
        """
        ReconstModel.__init__(self, gtab)
        if sphere is None:
            sphere = dpd.get_sphere()
        self.sphere = sphere
        self.response = np.asarray(response)

        if solver == 'ElasticNet':
            self.solver = lm.ElasticNet(l1_ratio=l1_ratio, alpha=alpha,
                                        positive=True, warm_start=True)
        elif solver == 'NNLS' or solver == 'nnls':
            self.solver = opt.NonNegativeLeastSquares()
        elif isinstance(solver, opt.SKLearnLinearSolver):
            self.solver = solver
        else:
            e_s = "The `solver` key-word argument needs to be: "
            e_s += "'ElasticNet', 'NNLS', or a "
            e_s += "`dipy.optimize.SKLearnLinearSolver` object"
            raise ValueError(e_s)

    @auto_attr
    def design_matrix(self):
        return sfm_design_matrix(self.gtab, self.sphere, self.response)

    def fit(self, data, mask=None):
        """
        Fit the SparseFascicleModel object to data

        Parameters
        ----------
        data : array
            The measured signal from one voxel.

        mask : array
            A boolean array used to mark the coordinates in the data that
            should be analyzed that has the shape data.shape[-1]

        Returns
        -------
        SparseFascicleFit object

        """
        if mask is None:
            flat_data = np.reshape(data, (-1, data.shape[-1]))
        else:
            mask = np.array(mask, dtype=bool, copy=False)
            flat_data = np.reshape(data[mask], (-1, data.shape[-1]))

        # Fitting is done on the relative signal (S/S0):
        flat_S0 = np.mean(flat_data[..., self.gtab.b0s_mask], -1)
        flat_S = flat_data[..., ~self.gtab.b0s_mask] / flat_S0[..., None]
        flat_mean = np.mean(flat_S, -1)
        flat_params = np.zeros((flat_data.shape[0],
                                self.design_matrix.shape[-1]))

        for vox, vox_data in enumerate(flat_S):
            if np.any(np.isnan(vox_data)):
                flat_params[vox] = (np.zeros(self.design_matrix.shape[-1]))
            else:
                fit_it = vox_data - flat_mean[vox]
                flat_params[vox] = self.solver.fit(self.design_matrix,
                                                   fit_it).coef_
        if mask is None:
            out_shape = data.shape[:-1] + (-1, )
            beta = flat_params.reshape(out_shape)
            mean_out = flat_mean.reshape(out_shape)
            S0 = flat_S0.reshape(out_shape).squeeze()
        else:
            beta = np.zeros(data.shape[:-1] +
                            (self.design_matrix.shape[-1],))
            beta[mask, :] = flat_params
            mean_out = np.zeros(data.shape[:-1])
            mean_out[mask, ...] = flat_mean.squeeze()
            S0 = np.zeros(data.shape[:-1])
            S0[mask] = flat_S0

        return SparseFascicleFit(self, beta, S0, mean_out.squeeze())


class SparseFascicleFit(ReconstFit):
    def __init__(self, model, beta, S0, mean_signal):
        """
        Initalize a SparseFascicleFit class instance

        Parameters
        ----------
        model : a SparseFascicleModel object.
        beta : ndarray
            The parameters of fit to data.
        S0 : ndarray
            The mean non-diffusion-weighted signal.
        mean_signal : ndarray
            The mean of the diffusion-weighted signal
        """
        self.model = model
        self.beta = beta
        self.S0 = S0
        self.mean_signal = mean_signal

    def odf(self, sphere):
        """
        The orientation distribution function of the SFM

        Parameters
        ----------
        sphere : Sphere
            The points in which the ODF is evaluated

        Returns
        -------
        odf :  ndarray of shape (x, y, z, sphere.vertices.shape[0])

        """
        odf_matrix = self.model.cache_get('odf_matrix', key=sphere)
        if odf_matrix is None:
            odf_matrix = sfm_design_matrix(sphere, self.model.sphere,
                                           self.model.response, mode='odf')
            self.model.cache_set('odf_matrix', key=sphere, value=odf_matrix)

        flat_beta = self.beta.reshape(-1, self.beta.shape[-1])
        flat_odf = np.dot(odf_matrix, flat_beta.T)
        return flat_odf.T.reshape(self.beta.shape[:-1] +
                                  (odf_matrix.shape[0], ))

    def predict(self, gtab=None, response=None, S0=None):
        """
        Predict the signal based on the SFM parameters

        Parameters
        ----------
        gtab : GradientTable.
            The bvecs/bvals to predict the signal on. Default: the gtab from
            the model object
        response : list of 3 elements.
            The eigenvalues of a tensor which will serve as a kernel
            function. Default: the response of the model object
        S0 : float or array.
             The non-diffusion-weighted signal. Default: use the S0 of the data

        Returns
        -------
        pred_sig : ndarray
            The signal predicted in each voxel/direction
        """
        if response is None:
            response = self.model.response
        if gtab is None:
            _matrix = self.model.design_matrix
            gtab = self.model.gtab
        # The only thing we can't change at this point is the sphere we use
        # (which sets the width of our design matrix):
        else:
            _matrix = sfm_design_matrix(gtab, self.model.sphere, response)
        # Get them all at once:
        beta_all = self.beta.reshape(-1, self.beta.shape[-1])
        pred_weighted = np.dot(_matrix, beta_all.T).T
        pred_weighted = pred_weighted.reshape(self.beta.shape[:-1] +
                                              (_matrix.shape[0],))
        if S0 is None:
            S0 = self.S0
        if isinstance(S0, np.ndarray):
            S0 = S0[..., None]
        if isinstance(self.mean_signal, np.ndarray):
            mean_signal = self.mean_signal[..., None]
        pre_pred_sig = S0 * (pred_weighted + mean_signal)
        pred_sig = np.zeros(pre_pred_sig.shape[:-1] + (gtab.bvals.shape[0],))
        pred_sig[..., ~gtab.b0s_mask] = pre_pred_sig
        pred_sig[..., gtab.b0s_mask] = S0
        return pred_sig.squeeze()
