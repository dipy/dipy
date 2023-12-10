"""
The Sparse Fascicle Model.

This is an implementation of the sparse fascicle model described in
[Rokem2015]_. The multi b-value version of this model is described in
[Rokem2014]_.


.. [Rokem2015] Ariel Rokem, Jason D. Yeatman, Franco Pestilli, Kendrick
   N. Kay, Aviv Mezer, Stefan van der Walt, Brian A. Wandell
   (2015). Evaluating the accuracy of diffusion MRI models in white
   matter. PLoS ONE 10(4): e0123272. doi:10.1371/journal.pone.0123272

.. [Rokem2014] Ariel Rokem, Kimberly L. Chan, Jason D. Yeatman, Franco
   Pestilli,  Brian A. Wandell (2014). Evaluating the accuracy of diffusion
   models at multiple b-values with cross-validation. ISMRM 2014.
"""
import warnings

import numpy as np
import gc
from collections import OrderedDict

try:
    from numpy import nanmean
except ImportError:
    from scipy.stats import nanmean

from dipy.utils.optpkg import optional_package
from dipy.utils.multiproc import determine_num_processes
import dipy.core.gradients as grad
import dipy.core.optimize as opt
import dipy.sims.voxel as sims
import dipy.data as dpd
from dipy.reconst.base import ReconstModel, ReconstFit
from dipy.reconst.cache import Cache
from dipy.core.onetime import auto_attr

joblib, has_joblib, _ = optional_package('joblib')
sklearn, has_sklearn, _ = optional_package('sklearn')
lm, _, _ = optional_package('sklearn.linear_model')


# Isotropic signal models: these are models of the part of the signal that
# changes with b-value, but does not change with direction. This collection is
# extensible, by inheriting from IsotropicModel/IsotropicFit below:

# First, a helper function to derive the fit signal for these models:
def _to_fit_iso(data, gtab, mask=None):
    if mask is None:
        mask = np.ones(data.shape[:-1], dtype=bool)
    # Turn it into a 2D thing:
    if len(mask.shape) > 0:
        data = data[mask]
    else:
        # This handles the corner case of fitting a single voxel:
        data = data.reshape((-1, data.shape[0]))
    data_no_b0 = data[:, ~gtab.b0s_mask]
    nzb0 = data_no_b0 > 0
    nzb0_idx = np.where(nzb0)
    zb0_idx = np.where(~nzb0)
    if np.sum(gtab.b0s_mask) > 0:
        s0 = np.mean(data[:, gtab.b0s_mask], -1)
        to_fit = np.empty(data_no_b0.shape)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            to_fit[nzb0_idx] = data_no_b0[nzb0_idx] / s0[nzb0_idx[0]]
        to_fit[zb0_idx] = 0
    else:
        to_fit = data_no_b0
    return to_fit


class IsotropicModel(ReconstModel):
    """
    A base-class for the representation of isotropic signals.

    The default behavior, suitable for single b-value data is to calculate the
    mean in each voxel as an estimate of the signal that does not depend on
    direction.
    """
    def __init__(self, gtab):
        """Initialize an IsotropicModel.

        Parameters
        ----------
        gtab : a GradientTable class instance

        """
        ReconstModel.__init__(self, gtab)

    def fit(self, data, mask=None, **kwargs):
        """Fit an IsotropicModel.

        This boils down to finding the mean diffusion-weighted signal in each
        voxel

        Parameters
        ----------
        data : ndarray

        Returns
        -------
        IsotropicFit class instance.

        """
        # This returns as a 2D thing:
        params = np.mean(_to_fit_iso(data, self.gtab, mask=mask), -1)
        if mask is None:
            params = np.reshape(params, data.shape[:-1])
        else:
            out_params = np.zeros(data.shape[:-1])
            out_params[mask] = params
            params = out_params
        return IsotropicFit(self, params)


class IsotropicFit(ReconstFit):
    """
    A fit object for representing the isotropic signal as the mean of the
    diffusion-weighted signal.
    """
    def __init__(self, model, params):
        """Initialize an IsotropicFit object.

        Parameters
        ----------
        model : IsotropicModel class instance
        params : ndarray
            The mean isotropic model parameters (the mean diffusion-weighted
            signal in each voxel).
        n_vox : int
            The number of voxels for which the fit was done.

        """
        super().__init__(self, model)
        self.model = model
        self.params = params

    def predict(self, gtab=None):
        """Predict the isotropic signal.

        Based on a gradient table. In this case, the (naive!) prediction will
        be the mean of the diffusion-weighted signal in the voxels.

        Parameters
        ----------
        gtab : a GradientTable class instance (optional)
            Defaults to use the gtab from the IsotropicModel from which this
            fit was derived.

        """
        if gtab is None:
            gtab = self.model.gtab
        if len(self.params.shape) == 0:
            return self.params[..., np.newaxis] + np.zeros(
                                                        np.sum(~gtab.b0s_mask))
        else:
            return self.params[..., np.newaxis] + np.zeros(
                self.params.shape + (np.sum(~gtab.b0s_mask),))


class ExponentialIsotropicModel(IsotropicModel):
    """
    Representing the isotropic signal as a fit to an exponential decay function
    with b-values
    """
    def fit(self, data, mask=None, **kwargs):
        """

        Parameters
        ----------
        data : ndarray

        mask : array, optional
            A boolean array used to mark the coordinates in the data that
            should be analyzed. Has the shape `data.shape[:-1]`. Default: None,
            which implies that all points should be analyzed.

        Returns
        -------
        ExponentialIsotropicFit class instance.
        """
        to_fit = _to_fit_iso(data, self.gtab, mask=mask)
        # Fitting to the log-transformed relative data is much faster:
        nz_idx = to_fit > 0
        to_fit[nz_idx] = np.log(to_fit[nz_idx])
        to_fit[~nz_idx] = -np.inf
        params = -nanmean(to_fit / self.gtab.bvals[~self.gtab.b0s_mask], -1)
        if mask is None:
            params = np.reshape(params, data.shape[:-1])
        else:
            out_params = np.zeros(data.shape[:-1])
            out_params[mask] = params
            params = out_params
        return ExponentialIsotropicFit(self, params)


class ExponentialIsotropicFit(IsotropicFit):
    """
    A fit to the ExponentialIsotropicModel object, based on data.
    """
    def predict(self, gtab=None):
        """
        Predict the isotropic signal, based on a gradient table. In this case,
        the prediction will be for an exponential decay with the mean
        diffusivity derived from the data that was fit.

        Parameters
        ----------
        gtab : a GradientTable class instance (optional)
            Defaults to use the gtab from the IsotropicModel from which this
            fit was derived.
        """
        if gtab is None:
            gtab = self.model.gtab
        if len(self.params.shape) == 0:
            return np.exp(-gtab.bvals[~gtab.b0s_mask] *
                          (np.zeros(np.sum(~gtab.b0s_mask)) +
                          self.params[..., np.newaxis]))
        else:
            return np.exp(-gtab.bvals[~gtab.b0s_mask] *
                          (np.zeros((self.params.shape[0],
                                     np.sum(~gtab.b0s_mask))) +
                          self.params[..., np.newaxis]))


def sfm_design_matrix(gtab, sphere, response, mode='signal'):
    """
    Construct the SFM design matrix

    Parameters
    ----------
    gtab : GradientTable or Sphere
        Sets the rows of the matrix, if the mode is 'signal', this should be a
        GradientTable. If mode is 'odf' this should be a Sphere.

    sphere : Sphere
        Sets the columns of the matrix

    response : list of 3 elements
        The eigenvalues of a tensor which will serve as a kernel
        function.

    mode : str {'signal' | 'odf'}, optional
        Choose the (default) 'signal' for a design matrix containing predicted
        signal in the measurements defined by the gradient table for putative
        fascicles oriented along the vertices of the sphere. Otherwise, choose
        'odf' for an odf convolution matrix, with values of the odf calculated
        from a tensor with the provided response eigenvalues, evaluated at the
        b-vectors in the gradient table, for the tensors with principal
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

    >>> tensor_matrix = sfm_design_matrix(gtab, sphere,
    ...                                   [0.0015, 0.0005, 0.0005])

    A 'stick' function ([Behrens2007]_):

    >>> stick_matrix = sfm_design_matrix(gtab, sphere, [0.001, 0, 0])

    Notes
    -----
    .. [Rokem2015] Ariel Rokem, Jason D. Yeatman, Franco Pestilli, Kendrick
       N. Kay, Aviv Mezer, Stefan van der Walt, Brian A. Wandell
       (2015). Evaluating the accuracy of diffusion MRI models in white
       matter. PLoS ONE 10(4): e0123272. doi:10.1371/journal.pone.0123272

    .. [Rokem2014] Ariel Rokem, Kimberly L. Chan, Jason D. Yeatman, Franco
       Pestilli,  Brian A. Wandell (2014). Evaluating the accuracy of diffusion
       models at multiple b-values with cross-validation. ISMRM 2014.

    .. [Behrens2007] Behrens TEJ, Berg HJ, Jbabdi S, Rushworth MFS, Woolrich MW
       (2007): Probabilistic diffusion tractography with multiple fibre
       orientations: What can we gain? Neuroimage 34:144-55.
    """
    if mode == 'signal':
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
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
        if mode == 'signal':
            # For regressors based on the single tensor, remove $e^{-bD}$
            mat[:, ii] = sims.single_tensor(
                mat_gtab,
                evals=response,
                evecs=sims.all_tensor_evecs(this_dir)
            ) - np.exp(-mat_gtab.bvals * np.mean(response))

        elif mode == 'odf':
            # Stick function
            if response[1] == 0 or response[2] == 0:
                mat[sphere.find_closest(sims.all_tensor_evecs(this_dir)[0]),
                    ii] = 1
            else:
                mat[:, ii] = sims.single_tensor_odf(
                    gtab.vertices, evals=response,
                    evecs=sims.all_tensor_evecs(this_dir))
    return mat


class SparseFascicleModel(ReconstModel, Cache):
    def __init__(self, gtab, sphere=None, response=(0.0015, 0.0005, 0.0005),
                 solver='ElasticNet', l1_ratio=0.5, alpha=0.001,
                 isotropic=None, seed=42):
        """
        Initialize a Sparse Fascicle Model

        Parameters
        ----------
        gtab : GradientTable class instance

        sphere : Sphere class instance, optional
            A sphere on which coefficients will be estimated. Default:
            symmetric sphere with 362 points (from :mod:`dipy.data`).

        response : (3,) array-like, optional
            The eigenvalues of a canonical tensor to be used as the response
            function of single-fascicle signals.
            Default:[0.0015, 0.0005, 0.0005]

        solver : string, or initialized linear model object.
            This will determine the algorithm used to solve the set of linear
            equations underlying this model. If it is a string it needs to be
            one of the following: {'ElasticNet', 'NNLS'}. Otherwise, it can be
            an object that inherits from `dipy.optimize.SKLearnLinearSolver`
            or an object with a similar interface from Scikit Learn:
            `sklearn.linear_model.ElasticNet`, `sklearn.linear_model.Lasso` or
            `sklearn.linear_model.Ridge` and other objects that inherit from
            `sklearn.base.RegressorMixin`.
            Default: 'ElasticNet'.

        l1_ratio : float, optional
            Sets the balance between L1 and L2 regularization in ElasticNet
            [Zou2005]_. Default: 0.5

        alpha : float, optional
            Sets the balance between least-squares error and L1/L2
            regularization in ElasticNet [Zou2005]_. Default: 0.001

        isotropic : IsotropicModel class instance
            This is a class that implements the function that calculates the
            value of the isotropic signal. This is a value of the signal that
            is independent of direction, and therefore removed from both sides
            of the SFM equation. The default is an instance of IsotropicModel,
            but other functions can be inherited from IsotropicModel to
            implement other fits to the aspects of the data that depend on
            b-value, but not on direction.

        Notes
        -----
        This is an implementation of the SFM, described in [Rokem2015]_.

        .. [Rokem2014] Ariel Rokem, Jason D. Yeatman, Franco Pestilli, Kendrick
           N. Kay, Aviv Mezer, Stefan van der Walt, Brian A. Wandell
           (2014). Evaluating the accuracy of diffusion MRI models in white
           matter. PLoS ONE 10(4): e0123272. doi:10.1371/journal.pone.0123272

        .. [Zou2005] Zou H, Hastie T (2005). Regularization and variable
           selection via the elastic net. J R Stat Soc B:301-320
        """
        ReconstModel.__init__(self, gtab)

        if sphere is None:
            sphere = dpd.get_sphere()
        self.sphere = sphere
        self.response = np.asarray(response)
        if isotropic is None:
            isotropic = IsotropicModel

        self.isotropic = isotropic
        if solver == 'ElasticNet':
            self.solver = lm.ElasticNet(l1_ratio=l1_ratio, alpha=alpha,
                                        positive=True, warm_start=False,
                                        random_state=seed)
        elif solver in ('NNLS', 'nnls'):
            self.solver = opt.NonNegativeLeastSquares()

        elif (isinstance(solver, opt.SKLearnLinearSolver) or
              has_sklearn and isinstance(solver, sklearn.base.RegressorMixin)):
            self.solver = solver

        else:
            # If sklearn is unavailable, we can fall back on nnls (but we also
            # warn the user that we are about to do that):
            if not has_sklearn:
                w = sklearn._msg + "\nAlternatively, you can use 'nnls' method "
                w += "to fit the SparseFascicleModel"
                warnings.warn(w)
            e_s = "The `solver` key-word argument needs to be: "
            e_s += "'ElasticNet', 'NNLS', or a "
            e_s += "`dipy.optimize.SKLearnLinearSolver` object"
            raise ValueError(e_s)

    @auto_attr
    def design_matrix(self):
        """
        The design matrix for a SFM.

        Returns
        -------
        ndarray
            The design matrix, where each column is a rotated version of the
            response function.
        """
        return sfm_design_matrix(self.gtab, self.sphere, self.response,
                                 'signal')

    def _fit_solver2voxels(self, isopredict, vox_data, vox, parallel=False):
        # In voxels in which S0 is 0, we just want to keep the
        # parameters at all-zeros, and avoid nasty sklearn errors:
        if not (np.any(~np.isfinite(vox_data)) or np.all(vox_data == 0)):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")

                if parallel:
                    coef = {vox: self.solver.fit(
                        self.design_matrix, vox_data - isopredict[vox]
                    ).coef_}
                else:
                    coef = self.solver.fit(
                        self.design_matrix, vox_data - isopredict[vox]
                    ).coef_
        else:
            if parallel:
                return {vox: np.zeros(self.design_matrix.shape[-1])}
            else:
                return np.zeros(self.design_matrix.shape[-1])
        return coef

    def fit(self, data, mask=None, num_processes=1,
            parallel_backend='multiprocessing'):
        """
        Fit the SparseFascicleModel object to data.

        Parameters
        ----------
        data : array
            The measured signal.

        mask : array, optional
            A boolean array used to mark the coordinates in the data that
            should be analyzed. Has the shape `data.shape[:-1]`. Default: None,
            which implies that all points should be analyzed.

        num_processes : int, optional
            Split the `fit` calculation to a pool of children processes using
            joblib. This only applies to 4D `data` arrays. Default is 1,
            which does not require joblib and will run `fit` serially.
            If < 0 the maximal number of cores minus ``num_processes + 1``
            is used (enter -1 to use as many cores as possible).
            0 raises an error.

        parallel_backend: str, ParallelBackendBase instance or None
            Specify the parallelization backend implementation.
            Supported backends are:
            - "loky" used by default, can induce some
              communication and memory overhead when exchanging input and
              output data with the worker Python processes.
            - "multiprocessing" previous process-based backend based on
              `multiprocessing.Pool`. Less robust than `loky`.
            - "threading" is a very low-overhead backend but it suffers
              from the Python Global Interpreter Lock if the called function
              relies a lot on Python objects. "threading" is mostly useful
              when the execution bottleneck is a compiled extension that
              explicitly releases the GIL (for instance a Cython loop wrapped
              in a "with nogil" block or an expensive call to a library such
              as NumPy).
            Default: 'multiprocessing'.

        Returns
        -------
        SparseFascicleFit object
        """

        if mask is None:
            # Flatten it to 2D either way:
            data_in_mask = np.reshape(data, (-1, data.shape[-1]))
        else:
            # Check for valid shape of the mask
            if mask.shape != data.shape[:-1]:
                raise ValueError("Mask is not the same shape as data.")
            mask = np.array(mask, dtype=bool, copy=False)
            data_in_mask = np.reshape(data[mask], (-1, data.shape[-1]))

        # Fitting is done on the relative signal (S/S0):
        flat_S0 = np.mean(data_in_mask[..., self.gtab.b0s_mask], -1)
        if not flat_S0.size or not flat_S0.max():
            flat_S = np.zeros(data_in_mask[..., ~self.gtab.b0s_mask].shape)
        else:
            flat_S = (data_in_mask[..., ~self.gtab.b0s_mask] /
                      flat_S0[..., None])
        isotropic = self.isotropic(self.gtab).fit(data, mask)
        flat_params = np.zeros((data_in_mask.shape[0],
                                self.design_matrix.shape[-1]))
        del data_in_mask
        gc.collect()

        isopredict = isotropic.predict()
        if mask is None:
            isopredict = np.reshape(isopredict, (-1, isopredict.shape[-1]))
        else:
            isopredict = isopredict[mask]

        if not num_processes:
            num_processes = determine_num_processes(num_processes)

        if num_processes > 1 and has_joblib:
            with joblib.Parallel(n_jobs=num_processes,
                                 backend=parallel_backend,
                                 mmap_mode='r+') as parallel:
                out = parallel(
                    joblib.delayed(self._fit_solver2voxels)(isopredict,
                                                            vox_data, vox,
                                                            True) for
                    vox, vox_data in enumerate(flat_S))

            del parallel

            flat_params_dict = {}
            for d in out:
                flat_params_dict.update(d)
            flat_params = np.concatenate(
                [np.array(i).reshape(1, flat_params.shape[1])
                 for i in list(OrderedDict(
                    sorted(flat_params_dict.items(),
                           key=lambda x: int(x[0]))).values())])
        else:
            for vox, vox_data in enumerate(flat_S):
                flat_params[vox] = self._fit_solver2voxels(isopredict,
                                                           vox_data, vox,
                                                           False)

        del isopredict, flat_S
        gc.collect()

        if mask is None:
            out_shape = data.shape[:-1] + (-1, )
            beta = flat_params.reshape(out_shape)
            S0 = flat_S0.reshape(data.shape[:-1])
        else:
            beta = np.zeros(data.shape[:-1] +
                            (self.design_matrix.shape[-1],))
            beta[mask, :] = flat_params
            S0 = np.zeros(data.shape[:-1])
            S0[mask] = flat_S0

        return SparseFascicleFit(self, beta, S0, isotropic)


class SparseFascicleFit(ReconstFit):

    def __init__(self, model, beta, S0, iso):
        """
        Initialize a SparseFascicleFit class instance

        Parameters
        ----------
        model : a SparseFascicleModel object.

        beta : ndarray
            The parameters of fit to data.

        S0 : ndarray
            The mean non-diffusion-weighted signal.

        iso : IsotropicFit class instance
            A representation of the isotropic signal, together with parameters
            of the isotropic signal in each voxel, that is capable of
            deriving/predicting an isotropic signal, based on a gradient-table.
        """
        super().__init__(self, model)

        self.model = model
        self.beta = beta
        self.S0 = S0
        self.iso = iso

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

        return np.dot(odf_matrix,
                      self.beta.reshape(-1,
                                        self.beta.shape[-1]).T).T.reshape(
            self.beta.shape[:-1] + (odf_matrix.shape[0], ))

    def predict(self, gtab=None, response=None, S0=None):
        """
        Predict the signal based on the SFM parameters

        Parameters
        ----------
        gtab : GradientTable, optional
            The bvecs/bvals to predict the signal on. Default: the gtab from
            the model object.

        response : list of 3 elements, optional
            The eigenvalues of a tensor which will serve as a kernel
            function. Default: the response of the model object. Default to use
            `model.response`.

        S0 : float or array, optional
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
        pred_weighted = np.dot(_matrix,
                               self.beta.reshape(
                                   -1, self.beta.shape[-1]).T).T.reshape(
            self.beta.shape[:-1] + (_matrix.shape[0],))

        if S0 is None:
            S0 = self.S0
        if isinstance(S0, np.ndarray):
            S0 = S0[..., None]

        pre_pred_sig = S0 * (pred_weighted +
                             self.iso.predict(gtab).reshape(
                                 pred_weighted.shape))
        pred_sig = np.zeros(pre_pred_sig.shape[:-1] + (gtab.bvals.shape[0],))
        pred_sig[..., ~gtab.b0s_mask] = pre_pred_sig
        pred_sig[..., gtab.b0s_mask] = S0
        return pred_sig.squeeze()
