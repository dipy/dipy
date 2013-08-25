"""
Diffusion Kurtosis Imaging

"""
import numpy as np
from scipy.misc import factorial
import scipy.linalg as linalg
import dipy.reconst.dti as dti
import dipy.core.sphere as dps
import dipy.core.gradients as grad


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
    def __init__(self, gtab):
        """
        Initialize a DiffusionKurtosisModel class instance

        Parameters
        ----------

        gtab: GradientTable

        Returns
        -------
        DiffusionKurtosisModel class instance

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
        # If a mask is provided, we will use it to access the data
        if mask is not None:
            # Make sure it's boolean, so that it can be used to mask
            mask = np.array(mask, dtype=bool, copy=False)
            data_in_mask = data[mask]
        else:
            data_in_mask = data

        return DiffusionKurtosisFit(self, data_in_mask)


class DiffusionKurtosisFit(object):
    """

    """
    def __init__(self, model, data):
        """
        Fit the DKI model to the provided data
        """
        # Extract the ADC for the diffusion-weighted directions :
        sphere = dps.Sphere(xyz=model.gtab.bvecs[~model.gtab.b0s_mask])
        tensor_fits = []
        adc = []
        md = []
        for idx, shell in enumerate(model.shells):
            sh_data = np.concatenate([data[...,model.sh_idx[idx]],
                                     data[...,model.gtab.b0s_mask]], -1)

            tensor_fits.append(model.tensors[idx].fit(sh_data))
            # Get the ADC on the entire sphere in each b value:
            adc.append(tensor_fits[-1].adc(sphere))
            md.append(tensor_fits[-1].md)
        # Following equations 38-39 in Jensen and Helpern 2010. We use the two
        # shells most different from each other:
        self.ADC = ((model.shells[-1] * adc[0] - model.shells[0] * adc[-1])/
                    (model.shells[-1] - model.shells[0]))  # Eq 38

        self.MD = ((model.shells[-1] * md[0] - model.shells[0] * md[-1])/
                    (model.shells[-1] - model.shells[0]))

        self.K = (6 * (adc[0] - adc[-1])/
             ((model.shells[-1] - model.shells[0]) * self.ADC **2)) # Eq 39

        # Kurtosis is not allowed to go below 0, but the noise sometimes drags
        # it there:
        self.K = np.where(self.K>0, self.K, 0)

        to_fit = (self.K * self.ADC**2) / (self.MD ** 2)[..., None]

        # least-square estimation of the 15 DK params. We use tensordot, so
        # that we can do this over all voxels at once:
        model_params = np.rollaxis(
            np.tensordot(np.linalg.pinv(model.dk_design_matrix), to_fit,
                                    axes=(1,-1)), 0, len(data.shape))
        # rollaxis is used to get it back into the shape (x,y,z, n_params)

        self.model = model
        self.model_params = model_params
        self.tensor_fits = tensor_fits

    def mean_kurtosis(self):
        """

        Notes
        -----
        According to equation 55 in [1]_

        [1] Jensen and Helpern (2010). XXX

        """
        # This is the mean of the diagonal terms and the squared off-diagonals
        # XXX This needs more work
        #np.mean(np.concatenate([self.model_params[..., :3],
        #                       self.model_params[..., 9:12]]), -1)

    def predict(self, gtab, S0=1):
        """
        Use the model parameters to predict the signal back

        """
        sphere = dps.Sphere(xyz=gtab.bvecs)
        shells = np.unique(gtab.bvals[~gtab.b0s_mask])
        # We use the tensor fits to predict ADC here, by linear
        # inter-/extra-polation:
        tensor_adc = np.array([t.adc(sphere) for t in self.tensor_fits])
        lin_design = np.vstack([self.model.shells, np.ones(len(shells))]).T

        # Get the parameters for the linear fit on tensor ADC by b-value:
        ten_lin = np.tensordot(np.linalg.pinv(lin_design.T),
                               tensor_adc,
                               axes=(0,0))

        # Now go figure out the ADC for the current gtab.
        # Pre-allocate the ADC as zeros:
        ADC = np.zeros(tuple(self.model_params.shape[:-1]) + gtab.bvals.shape)
        for idx, shell in enumerate(shells):
            sh_idx = np.where(gtab.bvals==shell)
            # This is simply: b * beta_0 + beta_1
            this_pred_adc = shell * ten_lin[0] + ten_lin[1]
            ADC[..., sh_idx] = this_pred_adc[..., sh_idx]

        dm = dk_design_matrix(gtab)
        AKC = np.zeros(ADC.shape)
        MD = self.MD[..., None]
        DKtensor = np.rollaxis(
            np.tensordot(dm, self.model_params, (1,-1)),
            0,len(MD.shape))

        AKC[..., ~gtab.b0s_mask] = ( (MD **2) / (ADC[..., ~gtab.b0s_mask])  *
                                     DKtensor)

        # Don't allow values below 0:
        AKC = np.where(AKC >= 0, AKC, 0)

        new_shape = tuple([ADC.shape[-1]] + [1] * (len(ADC.shape)-1))
        bvals = gtab.bvals.reshape(new_shape).T

        if np.iterable(S0):
            S0 = S0[...,None]

        pred_sig = S0 * np.exp(-ADC*bvals + (bvals**2 * ADC**2 * AKC)/6.0 )
        pred_sig[..., gtab.b0s_mask] = S0

        return pred_sig
