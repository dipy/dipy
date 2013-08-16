"""
Diffusion Kurtosis Imaging

"""
import numpy as np
from scipy.misc import factorial
import scipy.linalg as linalg
import dipy.reconst.dti as dti
import dipy.core.sphere as dps


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
        gtab: GradientTable


        References
        ----------


        """
        self.gtab = gtab
        self.dk_design_matrix = dk_design_matrix(gtab)
        # We'll estimate MD using DTI with the default params:
        self.TensorModel = dti.TensorModel(gtab)

    def fit(self, data, mask=None):
        """
        For now, single voxel
        """
        # Extract the ADC for the diffusion-weighted directions :
        sphere = dps.Sphere(xyz=self.gtab.bvecs[~self.gtab.b0s_mask])
        tensor_fit = self.TensorModel.fit(data, mask)

        # XXX Use only the lower b value, using equation 38 - 40 in Jensen and
        # Helpern ?
        self.adc = tensor_fit.apparent_diffusion_coef(sphere)

        # Calculate the AKC:
        logS0 = np.log(np.mean(data[self.gtab.b0s_mask]))
        logS = np.log(data[~self.gtab.b0s_mask])
        bv = self.gtab.bvals[~self.gtab.b0s_mask]
        # This is based on equation 1 in Lu et al:
        self.AKC = (logS - logS0 +  bv * self.adc) * (6 * bv**2 * self.adc**2)
        # This is based on equation 2 in Lu et al:
        to_fit = (self.AKC * self.adc**2)/(tensor_fit.md ** 2)
        # least-square estimation of the 15 DK params:
        model_params = linalg.lstsq(self.dk_design_matrix, to_fit)[0];
        return DiffusionKurtosisFit(self, model_params)

class DiffusionKurtosisFit(object):
    """

    """
    def __init__(self, model, model_params):
        """
        """
        self.model = model
        self.model_params = model_params

    def apparent_kurtosis_coef():
       """
       """
       pass
