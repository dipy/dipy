"""
This is an implementation of the sparse fascicle model described in
[Rokem2014]_.


_[Rokem2014]

"""
import warnings

import numpy as np

from dipy.utils.optpkg import optional_package
import dipy.core.geometry as geo
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
    w += "algorithm instead"
    warnings.warn(w)
    import scipy.optimize as opt

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
                                        positive=True)
        else:
            self.solver = opt.nnls


    @auto_attr
    def design_matrix(self):
        # Preallocate:
        mat = np.empty((np.sum(~self.gtab.b0s_mask),
                       self.sphere.vertices.shape[0]))

        # Each column of the matrix is the signal in each measurement, as
        # predicted by a "canonical", symmetrical tensor rotated towards this
        # vertex of the sphere:
        canonical_tensor = np.array([[self.response[0], 0, 0],
                                         [0, self.response[1], 0],
                                         [0, 0, self.response[2]]])

        # Calculate column-wise:
        for ii, this_dir in enumerate(self.sphere.vertices):
            # Rotate the canonical tensor towards this vertex and calculate the
            # signal you would have gotten in the direction
            rot_matrix = geo.vec2vec_rotmat(np.array([1,0,0]), this_dir)
            this_tensor = np.dot(rot_matrix, canonical_tensor)
            evals, evecs = dti.decompose_tensor(this_tensor)
            sig = sims.single_tensor(self.gtab, evals=self.response)
            mat[:, ii] = sig - np.mean(sig)

        return mat


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
        if mask is not None:
            if mask.shape != data.shape[:-1]:
                raise ValueError("Mask is not the same shape as data.")
            mask = np.array(mask, dtype=bool, copy=False)
            data_in_mask = data[mask]
        else:
            data_in_mask = data

        data_in_mask = data_in_mask.reshape((-1, data_in_mask.shape[-1]))

        params_in_mask = np.zeros((data_in_mask.shape[0],
                                   self.design_matrix.shape[-1]))

        for vox, dd in enumerate(data_in_mask):
            fit_it = dd - np.mean(dd)
            if has_sklearn:
                params_in_mask[vox] = self.solver.fit(self.design_matrix,
                                              fit_it).coef_
            else:
                params_in_mask[vox], _ = self.solver(self.design_matrix,
                                             fit_it)

        beta = np.zeros(data.shape[:-1] +
                                (self.design_matrix.shape[-1], ))

        beta[mask, :] = params_in_mask
        return SparseFascicleFit(self, beta)


class SparseFascicleFit(ReconstFit):
    def __init__(self, model, beta):
        """
        Initalize a SparseFascicleFit class instance
        """
        self.model = model
        self.beta = beta


    def predict(self, gtab=None, S0=None):
        """
        Predict the signal based on the SFM parameters

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
        pred_weighted = np.reshape(opt.spdot(_matrix, self.beta),
                                   (self.vox_coords.shape[0],
                                    np.sum(~gtab.b0s_mask)))

        pred = np.empty((self.vox_coords.shape[0], gtab.bvals.shape[0]))
        if S0 is None:
            S0 = self.b0_signal

        pred[..., gtab.b0s_mask] = S0[:, None]
        pred[..., ~gtab.b0s_mask] =\
            (pred_weighted + self.mean_signal[:, None]) * S0[:, None]

        return pred
