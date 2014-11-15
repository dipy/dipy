"""
This is an implementation of the sparse fascicle model described in
[Rokem2014]_.


_[Rokem2014]

"""
import warnings

import numpy as np

from dipy.utils.optpkg import optional_package
import dipy.core.geometry as geo
import dipy.core.gradients as grad
import dipy.sims.voxel as sims
import dipy.reconst.dti as dti
import dipy.data as dpd
from dipy.reconst.base import ReconstModel, ReconstFit
from dipy.core.onetime import auto_attr
lm, has_sklearn, _ = optional_package('sklearn.linear_model')

# If sklearn is unavailable, we can fall back on nnls (but we also warn the
# user that we are about to do that):
if not has_sklearn:
    w = "sklearn is not available, we will fit the SFM using the KKT NNLS"
    w += " algorithm instead"
    warnings.warn(w)
    import scipy.optimize as opt

def sfm_design_matrix(gtab, sphere, response):
    """
    Construct the SFM design matrix

    Parameters
    ----------
    gtab : GradientTable
        Sets the rows of the matrix

    sphere : Sphere
        Sets the columns of the matrix

    response : list of 3 elements
        The eigenvalues of a tensor which will serve as a kernel function
    """
    # Preallocate:
    mat = np.empty((np.sum(~gtab.b0s_mask),
                   sphere.vertices.shape[0]))

    # Each column of the matrix is the signal in each measurement, as
    # predicted by a "canonical", symmetrical tensor rotated towards this
    # vertex of the sphere:
    canonical_tensor = np.array([[response[0], 0, 0],
                                     [0, response[1], 0],
                                     [0, 0, response[2]]])

    mat_gtab = grad.gradient_table(gtab.bvals[~gtab.b0s_mask],
                                   gtab.bvecs[~gtab.b0s_mask])
    # Calculate column-wise:
    for ii, this_dir in enumerate(sphere.vertices):
        # Rotate the canonical tensor towards this vertex and calculate the
        # signal you would have gotten in the direction
        rot_matrix = geo.vec2vec_rotmat(np.array([1,0,0]), this_dir)
        this_tensor = np.dot(rot_matrix, canonical_tensor)
        evals, evecs = dti.decompose_tensor(this_tensor)
        sig = sims.single_tensor(mat_gtab, evals=response, evecs=evecs)
        mat[:, ii] = sig - np.mean(sig)

    return mat


class SparseFascicleModel(ReconstModel):
    def __init__(self, gtab, sphere=None, response=[0.0015, 0.0005, 0.0005],
                 l1_ratio=0.5, alpha=0.001):
        """
        Initialize a Sparse Fascicle Model

        Parameters
        ----------
        gtab: GradienTable class instance
        sphere: Sphere class instance
        response : (3,) array-like
            The eigenvalues of a canonical tensor to be used as the response
            function of single-fascicle signals.
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
        if has_sklearn:
            self.solver = lm.ElasticNet(l1_ratio=l1_ratio, alpha=alpha,
                                        positive=True, warm_start=True)
        else:
            self.solver = opt.nnls

    @auto_attr
    def design_matrix(self):
        return sfm_design_matrix(self.gtab, self.sphere, self.response)

    def fit(self, data, mask=None):
        """

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
        # Fitting is done on the relative signal (S/S0):
        S0 = np.mean(data[..., self.gtab.b0s_mask], -1)
        S = data[..., ~self.gtab.b0s_mask]/S0[...,None]
        mean_signal = np.mean(data, -1)

        if len(mean_signal.shape) <= 1:
            mean_signal = np.reshape(mean_signal, (1,-1))
        if mask is not None:
            mask = np.array(mask, dtype=bool, copy=False)
            data_in_mask = S[mask]
            mean_in_mask = mean_signal[mask]
        else:
            data_in_mask = S
            mean_in_mask = mean_signal

        data_in_mask = data_in_mask.reshape((-1, data_in_mask.shape[-1]))
        mean_in_mask = mean_in_mask.reshape((-1, 1))
        S0_in_mask = S0.reshape((-1, 1))

        params_in_mask = np.zeros((data_in_mask.shape[0],
                                   self.design_matrix.shape[-1]))

        for vox, dd in enumerate(data_in_mask):
            if np.any(np.isnan(dd)):
                params_in_mask[vox] = (np.zeros(self.design_matrix.shape[-1]))
            else:
                fit_it = dd - mean_in_mask[vox]
                if has_sklearn:
                    params_in_mask[vox] = self.solver.fit(self.design_matrix,
                                                  fit_it).coef_
                else:
                    params_in_mask[vox], _ = self.solver(self.design_matrix,
                                                 fit_it)

        if mask is not None:
            beta = np.zeros(data.shape[:-1] +
                            (self.design_matrix.shape[-1], ))

            beta[mask, :] = params_in_mask
            mean_out = np.zeros(data.shape[:-1])
            mean_out[mask, ...] = mean_in_mask.squeeze()

        else:
            beta = params_in_mask.reshape(data.shape[:-1] + (-1, ))
            mean_out = mean_in_mask.reshape(data.shape[:-1] + (-1, ))

        return SparseFascicleFit(self, beta, S0, mean_out.squeeze())


class SparseFascicleFit(ReconstFit):
    def __init__(self, model, beta, S0, mean_signal):
        """
        Initalize a SparseFascicleFit class instance
        """
        self.model = model
        self.beta = beta
        self.S0 = S0
        self.mean_signal = mean_signal


    def odf(self, sphere):
        """
        The orientation distribution function (identical to the model parameters)
        """
        if self.beta.shape[-1] != sphere.x.shape[0]:
            e_s = "ODF must be evaluated on the sphere used for fitting "
            e_s = "The SFM model"
            return ValueError(e_s)

        return self.beta


    def predict(self, gtab=None, response=None, S0=None):
        """
        Predict the signal based on the SFM parameters

        Parameters
        ----------

        """
        if response is None:
            response=self.model.response
        if gtab is None:
            _matrix = self.model.design_matrix
            gtab = self.model.gtab

        # The only thing we can't change at this point iss the sphere we use
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

        pre_pred_sig = (S0 * pred_weighted) + mean_signal

        pred_sig = np.zeros(pre_pred_sig.shape[:-1] + (gtab.bvals.shape[0],))
        pred_sig[..., ~gtab.b0s_mask] = pre_pred_sig
        pred_sig[..., gtab.b0s_mask] = S0

        return pred_sig.squeeze()
