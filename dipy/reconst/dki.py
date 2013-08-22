"""
Diffusion Kurtosis Imaging

"""
import numpy as np
from scipy.misc import factorial
import scipy.linalg as linalg
import dipy.reconst.dti as dti
import dipy.core.sphere as dps
import dipy.core.gradients as grad

def ols_matrix(A):
    """
    Generate the matrix used to solve OLS regression.

    Parameters
    ----------

    A: float array
        The design matrix

    Notes
    -----

    The matrix needed for OLS regression for the equation:

    ..math ::

        y = A \beta

   is given by:

    ..math ::

        \hat{\beta} = (A' x A)^{-1} A' y

    See also
    --------
    http://en.wikipedia.org/wiki/Ordinary_least_squares#Estimation
    """

    A = np.asarray(A)

    X = np.matrix(A.copy())

    return np.dot(linalg.pinv(np.dot(X.T, X)), X.T)


def dk_design_matrix(gtab):
    """
    Compute the DKI design matrix (according to appendix in [1]_)

    Parameters
    ----------
    gtab : a GradientTable class instance.

    Returns
    -------
    D : float array with shape (len(gtab.b0s_mask), 15)
        The design matrix containing interactions between gradients up to 4th
        order.

    Notes
    -----
    [1] Lu, H, Jensen, JH, Ramani, A, Helpern, JA (2006). Three-dimensional
    characterization of non-gaussian water diffusion in  humans using diffusion
    kurtosis imaging. NMR in Biomedicine 19: 236-247
    """

    bvecs = gtab.bvecs[~gtab.b0s_mask]
    D = np.zeros((bvecs.shape[0], 15))
    G = bvecs.T

    D[:, 0] = G[0, :] ** 4
    D[:, 1] = G[1, :] ** 4
    D[:, 2] = G[2, :] ** 4
    D[:, 3] = 4 * G[0, :] ** 3 * G[1, :]
    D[:, 4] = 4 * G[0, :] ** 3 * G[2, :]
    D[:, 5] = 4 * G[0, :] * G[1, :] ** 3
    D[:, 6] = 4 * G[1, :] ** 3 * G[2, :]
    D[:, 7] = 4 * G[0, :] * G[2, :] ** 3
    D[:, 8] = 4 * G[1, :] * G[2, :] ** 3
    D[:, 9] = 6 * G[0, :] ** 2 * G[1, :] ** 2
    D[:, 10] = 6 * G[0, :] ** 2 * G[2, :] ** 2
    D[:, 11] = 6 * G[1, :] ** 2 * G[2, :] ** 2
    D[:, 12] = 12 * G[0, :] ** 2 * G[1, :] * G[2, :]
    D[:, 13] = 12 * G[0, :] * G[1, :] ** 2 * G[2, :]
    D[:, 14] = 12 * G[0, :] * G[1, :] * G[2, :] ** 2

    return D


class DiffusionKurtosisModel(object):
    """
    The diffusion kurtosis model:

    Notes
    -----
    [1] Lu, H, Jensen, JH, Ramani, A, Helpern, JA (2006). Three-dimensional
    characterization of non-gaussian water diffusion in  humans using diffusion
    kurtosis imaging. NMR in Biomedicine 19: 236-247

    [2] Jensen, JH and Helpern JA (2010). MRI quantification of non-Gaussian
    water diffusion by kurtosis analysis. NMR in Biomedicine 23: 698-710.

    """
    def __init__(self, gtab, *args, **kwargs):
        """
        Initialize a DiffusionKurtosisModel class instance

        Parameters
        ----------

        gtab: GradientTable



        References
        ----------


        """
        self.gtab = gtab
        # We will have a separate tensor model for each shell:
        self.tensors = []
        self.sh_idx = []
        self.shells = np.unique(gtab.bvals[~gtab.b0s_mask])
        for shell in self.shells:
             self.sh_idx.append(np.where(gtab.bvals == shell)[0])
             sh_bvals = np.concatenate([gtab.bvals[self.sh_idx[-1]],
                                        gtab.bvals[self.gtab.b0s_mask]])
             sh_bvecs = np.concatenate([gtab.bvecs[self.sh_idx[-1]],
                                        gtab.bvecs[self.gtab.b0s_mask]])
             sh_gtab = grad.gradient_table(sh_bvals, sh_bvecs)
             self.tensors.append(dti.TensorModel(sh_gtab))

        self.dk_design_matrix = dk_design_matrix(gtab)

    def fit(self, data, mask=None):
        """

        """
        # Extract the ADC for the diffusion-weighted directions :
        sphere = dps.Sphere(xyz=self.gtab.bvecs[~self.gtab.b0s_mask])
        self.tensor_fits = []
        self.adc = []
        for idx, shell in enumerate(self.shells):
            sh_data = np.concatenate([data[...,self.sh_idx[idx]],
                                      data[...,self.gtab.b0s_mask]], -1)

            self.tensor_fits.append(self.tensors[idx].fit(sh_data, mask))
            # Get the ADC on the entire sphere in each b value:
            self.adc.append(self.tensor_fits[-1].apparent_diffusion_coef(sphere))

        # Following equations 38-39 in Jensen and Helpern 2010. We use the two
        # shells most different from each other:
        self.D=((self.shells[-1] * self.adc[0] - self.shells[0] * self.adc[-1])/
             (self.shells[-1] - self.shells[0]))  # Eq 38

        self.K = (6 * (self.adc[0] - self.adc[-1])/
             ((self.shells[-1] - self.shells[0]) * self.D **2)) # Eq 39

        to_fit = (self.K * self.D**2) / (np.mean(self.D, -1) ** 2)[..., None]

        # least-square estimation of the 15 DK params:
        model_params = np.tensordot(ols_matrix(self.dk_design_matrix), to_fit,
                                    axes=(1,-1)).T

        return DiffusionKurtosisFit(self, model_params)




class DiffusionKurtosisFit(object):
    """

    """
    def __init__(self, model, model_params):
        """
        """
        self.model = model
        self.model_params = model_params


    def mean_kurtosis(self):
       """

       Notes
       -----
       According to equation 55 in [1]_

       [1] Jensen and Helpern (2010). XXX

       """

       # This is the mean of the diagonal terms and the squared off-diagonals
       return np.mean(np.concatenate([self.model_params[..., :3],
                                      self.model_params[..., 9:12]]), -1)
