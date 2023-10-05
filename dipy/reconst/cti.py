#!/usr/bin/python
""" Classes and functions for fitting the correlation tensor model """

import numpy as np
from dipy.reconst.multi_voxel import multi_voxel_fit
from dipy.reconst.base import ReconstModel
from dipy.reconst.utils import cti_design_matrix as design_matrix
from dipy.reconst.dki import (
    DiffusionKurtosisFit,
)
from dipy.reconst.dti import (
    decompose_tensor,
    from_lower_triangular,
    lower_triangular,
    MIN_POSITIVE_SIGNAL)
from dipy.core.onetime import auto_attr


def from_qte_to_cti(C):
    """
    Rescales the qte C elements to the C elements used in CTI.

    Parameters
    ----------
    C: array(..., 21)
        Twenty-one elements of the covariance tensor in voigt notation plus
        some extra scaling factors.

    Returns
    -------
    ccti: array(..., 21)
        Covariance Tensor Elements with no hidden factors.
    """
    const = np.sqrt(2)
    ccti = np.zeros((21, 1))
    ccti[0] = C[0]
    ccti[1] = C[1]
    ccti[2] = C[2]
    ccti[3] = C[3] / const
    ccti[4] = C[4] / const
    ccti[5] = C[5] / const
    ccti[6] = C[6] / 2
    ccti[7] = C[7] / 2
    ccti[8] = C[8] / 2
    ccti[9] = C[9] / 2
    ccti[10] = C[10] / 2
    ccti[11] = C[11] / 2
    ccti[12] = C[12] / 2
    ccti[13] = C[13] / 2
    ccti[14] = C[14] / 2
    ccti[15] = C[15] / 2
    ccti[16] = C[16] / 2
    ccti[17] = C[17] / 2
    ccti[18] = C[18] / (2 * const)
    ccti[19] = C[19] / (2 * const)
    ccti[20] = C[20] / (2 * const)
    return ccti


def multi_gaussian_k_from_c(ccti, MD):
    """
    Computes the multiple Gaussian diffusion kurtosis tensor from the
    covariance tensor.

    Parameters
    ----------
    ccti: array(..., 21)
        Covariance Tensor Elements with no hidden factors.
    MD: Mean Diffusivity (MD) of a diffusion tensor.

    Returns
    -------
    K: array (..., 15)
        Fifteen elements of the kurtosis tensor
    """
    K = np.zeros((15, 1))
    K[0] = 3 * ccti[0] / (MD ** 2)
    K[1] = 3 * ccti[1] / (MD ** 2)
    K[2] = 3 * ccti[2] / (MD ** 2)
    K[3] = 3 * ccti[8] / (MD ** 2)
    K[4] = 3 * ccti[7] / (MD ** 2)
    K[5] = 3 * ccti[11] / (MD ** 2)
    K[6] = 3 * ccti[9] / (MD ** 2)
    K[7] = 3 * ccti[13] / (MD ** 2)
    K[8] = 3 * ccti[12] / (MD ** 2)
    K[9] = (ccti[5] + 2 * ccti[17]) / (MD**2)
    K[10] = (ccti[4] + 2 * ccti[16]) / (MD**2)
    K[11] = (ccti[3] + 2 * ccti[15]) / (MD**2)
    K[12] = (ccti[6] + 2 * ccti[19]) / (MD**2)
    K[13] = (ccti[10] + 2 * ccti[20]) / (MD**2)
    K[14] = (ccti[14] + 2 * ccti[18]) / (MD**2)
    return K


def split_cti_params(cti_params):
    r"""Splits CTI params into DTI, DKI, CTI portions.

    Extract the diffusion tensor eigenvalues, the diffusion tensor
    eigenvector matrix, and the 21 independent elements of the covariance
    tensor, and the 15 independent elements of the kurtosis tensor from the
    model parameters estimated from the CTI model

    Parameters
    ----------
        params: numpy.ndarray (..., 48)
        All parameters estimated from the correlation tensor model.
        Parameters are ordered as follows:

            1. Three diffusion tensor's eigenvalues
            2. Three lines of the eigenvector matrix each containing the
            first, second and third coordinates of the eigenvector
            3. Fifteen elements of the kurtosis tensor
            4. Twenty-One elements of the covariance tensor

    Returns
    -------
    evals : array (..., 3)
        Eigenvalues from eigen decomposition of the tensor.
    evecs : array (..., 3)
        Associated eigenvectors from eigen decomposition of the tensor.
        Eigenvectors are columnar (e.g. evecs[:,j] is associated with
        evals[j])
    kt : array (..., 15)
        Fifteen elements of the kurtosis tensor
    ct: array(..., 21)
        Twenty-one elements of the covariance tensor
       """
    evals = cti_params[..., :3]
    evecs = cti_params[..., 3:12].reshape(cti_params.shape[:-1] + (3, 3))
    kt = cti_params[..., 12:27]
    ct = cti_params[..., 27:48]
    return evals, evecs, kt, ct


def cti_prediction(cti_params, gtab1, gtab2, S0=1):
    """Predict a signal given correlation tensor imaging parameters.

        Parameters
        ----------
        cti_params: numpy.ndarray (..., 48)
        All parameters estimated from the correlation tensor model.
        Parameters are ordered as follows:
            1. Three diffusion tensor's eigenvalues
            2. Three lines of the eigenvector matrix each containing the
            first, second and third coordinates of the eigenvector
            3. Fifteen elements of the kurtosis tensor
            4. Twenty-One elements of the covariance tensor
        gtab1: dipy.core.gradients.GradientTable
            A GradientTable class instance for first DDE diffusion epoch

        gtab2: dipy.core.gradients.GradientTable
            A GradientTable class instance for second DDE diffusion epoch

        S0 : float or ndarray (optional)
            The non diffusion-weighted signal in every voxel, or across all
            voxels. Default: 1

        Returns
        -------
        S : ndarray
            Simulated signal based on the CTI model

    """
    evals, evecs, kt, ct = split_cti_params(cti_params)
    A = design_matrix(gtab1, gtab2)
    fevals = evals.reshape((-1, evals.shape[-1]))
    fevecs = evecs.reshape((-1,) + evecs.shape[-2:])
    fct = ct.reshape((-1, ct.shape[-1]))
    fkt = kt.reshape((-1, kt.shape[-1]))
    pred_sig = np.zeros((len(fevals), len(gtab1.bvals)))

    if isinstance(S0, np.ndarray):
        S0_vol = np.reshape(S0, (len(fevals)))
    else:
        S0_vol = S0
    for v in range(len(pred_sig)):
        DT = np.dot(np.dot(fevecs[v], np.diag(fevals[v])), fevecs[v].T)
        dt = lower_triangular(DT)
        MD = (dt[0] + dt[2] + dt[5]) / 3
        if isinstance(S0_vol, np.ndarray):
            this_S0 = S0_vol[v]
        else:
            this_S0 = S0_vol
        X = np.concatenate((dt, fkt[v] * MD * MD, fct[v],
                            np.array([-np.log(this_S0)])),
                           axis=0)
        pred_sig[v] = np.exp(np.dot(A, X))
    pred_sig = pred_sig.reshape(cti_params.shape[:-1] + (pred_sig.shape[-1],))

    return pred_sig


class CorrelationTensorModel(ReconstModel):
    """ Class for the Correlation Tensor Model
    """

    def __init__(self, gtab1, gtab2, fit_method="WLS", *args, **kwargs):
        """ Correlation Tensor Imaging Model [1]

        Parameters
        ----------
        gtab1: dipy.core.gradients.GradientTable
            A GradientTable class instance for first DDE diffusion epoch
        gtab2: dipy.core.gradients.GradientTable
            A GradientTable class instance for second DDE diffusion epoch
        fit_method : str or callable, optional
        args, kwargs :
            arguments and key-word arguments passed to the fit_method.

        """
        self.gtab1 = gtab1
        self.gtab2 = gtab2
        self.args = args
        self.kwargs = kwargs
        self.common_fit_method = not callable(fit_method)

        if self.common_fit_method:
            try:
                self.fit_method = common_fit_methods[fit_method]
            except KeyError:
                msg = '"' + str(fit_method) + \
                    '" is not a known fit method. The'
                msg += ' fit method should either be a function or one of the'
                msg += ' common fit methods.'
                raise ValueError(msg)

        self.args = args
        self.kwargs = kwargs

        self.min_signal = self.kwargs.pop('min_signal', None)
        if self.min_signal is None:
            self.min_signal = MIN_POSITIVE_SIGNAL
        elif self.min_signal <= 0:
            msg = "The `min_signal` key-word argument needs to be strictly"
            msg += " positive."
            raise ValueError(msg)

        self.design_matrix = design_matrix(self.gtab1, self.gtab2)
        self.inverse_design_matrix = np.linalg.pinv(self.design_matrix)

        tol = 1e-6
        self.min_diffusivity = tol / -self.design_matrix.min()
        self.weights = fit_method in {'WLS', 'WLLS', 'UWLLS'}

    @multi_voxel_fit
    def fit(self, data, mask=None):
        """ Fit method of the CTI model class.

        Parameters
        ----------
        data : array
            4D array of dMRI data.

        mask : array, optional
            A boolean array of the same shape as data.shape[-1]. It
            designates which coordinates in the data should be analyzed.
        """
        data_thres = np.maximum(data, self.min_signal)
        params = self.fit_method(self.design_matrix, data_thres,
                                 self.inverse_design_matrix,
                                 weights=self.weights,
                                 *self.args, **self.kwargs)

        return CorrelationTensorFit(self, params)

    def predict(self, cti_params, S0=1):
        """Predict a signal for the CTI model class instance given parameters

        Parameters
        ----------
        cti_params: numpy.ndarray (..., 48)
        All parameters estimated from the correlation tensor model.
        Parameters are ordered as follows:

            1. Three diffusion tensor's eigenvalues
            2. Three lines of the eigenvector matrix each containing the
            first, second and third coordinates of the eigenvector
            3. Fifteen elements of the kurtosis tensor
            4. Twenty-One elements of the covariance tensor
        gtab1: dipy.core.gradients.GradientTable
            A GradientTable class instance for first DDE diffusion epoch
        gtab2: dipy.core.gradients.GradientTable
            A GradientTable class instance for second DDE diffusion epoch
        S0 : float or ndarray (optional)
            The non diffusion-weighted signal in every voxel, or across all
            voxels. Default: 1

        Returns
        -------
        S : numpy.ndarray
            Predicted signal based on the CTI model
        """

        return cti_prediction(cti_params, self.gtab1, self.gtab2, S0)


class CorrelationTensorFit(DiffusionKurtosisFit):

    """ Class for fitting the Correlation Tensor Model """

    def __init__(self, model, model_params):
        """ Initialize a CorrelationTensorFit class instance.

        Parameters
        ----------
        model : CorrelationTensorModel Class instance
            Class instance containing the Correlation Tensor Model for the fit
        model_params : ndarray (x, y, z, 48) or (n, 48)
            All parameters estimated from the diffusion kurtosis model.
            Parameters are ordered as follows:
                1) Three diffusion tensor's eigenvalues
                2) Three lines of the eigenvector matrix each containing the
                   first, second and third coordinates of the eigenvector
                3) Fifteen elements of the kurtosis tensor
                4) Twenty One elements of the covariance tensor

        """
        DiffusionKurtosisFit.__init__(self, model, model_params)

    @property
    def ct(self):
        """
        Returns the 21 independent elements of the covariance tensor as an
        array
        """
        return self.model_params[..., 27:48]

    def predict(self, gtab1, gtab2, S0=1):
        """Given a CTI model fit, predict the signal on the vertices of a
        gradient table

        Parameters:
        -----------
        params: numpy.ndarray (...,43)
                All parameters estimated from the correlation tensor model.
                Parameters are ordered as follows:

                1. Three diffusion tensor's eigenvalues
                2. Three lines of the eigenvector matrix each containing the
                first, second and third coordinates of the eigenvector
                3. Fifteen elements of the kurtosis tensor
                4. Twenty-One elements of the covariance tensor
        gtab1: dipy.core.gradients.GradientTable
            A GradientTable class instance for first DDE diffusion epoch
        gtab2: dipy.core.gradients.GradientTable
            A GradientTable class instance for second DDE diffusion epoch
        S0 : float or ndarray (optional)
            The non diffusion-weighted signal in every voxel, or across all
            voxels. Default: 1

        Returns
        -------
        S : numpy.ndarray
            Predicted signal based on the CTI model
        """
        return cti_prediction(self.model_params, gtab1, gtab2, S0)

    @property
    def K_aniso(self):
        r""" Returns the anisotropic Source of Kurtosis (K_aniso)

            Notes
            -----
            The K_aniso is defined as[1]_:

            :math::

            \[K_{aniso} = \frac{6}{5} \cdot \frac{\langle V_{\lambda}(D_c)
                                                  \rangle}{\overline{D}^2}\]

        where: $K_{aniso}$ is the anisotropic kurtosis,
            $\langle V_{\lambda}(D_c) \rangle$ represents the mean of the
            variance of eigenvalues of the diffusion tensor,
            $\overline{D}$ is the mean of the diffusion tensor.

        References
        ----------
        .. [1]  [NetoHe2020] Henriques, R.N., Jespersen, S.N., Shemesh, N., 2020.
                Correlation tensor magnetic resonance imaging. Neuroimage 211.
                doi: 10.1016/j.neuroimage.2020.116605
        """
        C = self.ct
        D = self.quadratic_form

        Variance = 2/9 * (C[..., 0] + D[..., 0, 0] ** 2 + C[..., 1]
                          + D[..., 1, 1] ** 2 + C[..., 2]
                          + D[..., 2, 2]**2 - C[..., 5]
                          - D[..., 0, 0] * D[..., 1, 1] -
                          C[..., 4] - D[..., 0, 0] * D[..., 2, 2]
                          - C[..., 3] - D[..., 1, 1] * D[..., 2, 2]
                          + 3 * (C[..., 17] + D[..., 0, 1] ** 2 + C[..., 16]
                                 + D[..., 0, 2] ** 2
                                 + C[..., 15] + D[..., 1, 2] ** 2))
        mean_D = np.trace(D) / 3
        if mean_D == 0:
            K_aniso = 0
        else:
            K_aniso = (6/5) * (Variance / (mean_D ** 2))

        return K_aniso

    @property
    def K_iso(self):
        r""" Returns the isotropic Source of Kurtosis (K_iso)

        Notes
        -----
        The K_iso is defined as :

        :math::
            $$ K_{\text{iso}} = 3 \cdot \frac{V(\overline{D}^c)}{\overline{D}^2} $$

        where: $ K_{\text{iso}} $ is the isotropic kurtosis,
            \(V({\overline{D}^c})\) represents the variance of the diffusion
            tensor raised to the power c,
            \(\overline{D}\) is the mean of the diffusion tensor.

        """
        C = self.ct
        mean_D = self.md
        Variance = 1/9 * (C[..., 0] + C[..., 1] + C[..., 2] + 2 * C[..., 5]
                          + 2 * C[..., 4] + 2 * C[..., 3])
        if mean_D == 0:
            K_iso = 0
        else:
            K_iso = 3 * (Variance / (mean_D ** 2))
        return K_iso

    @auto_attr
    def K_total(self):
        r""" Returns the total excess kurtosis.

            Notes
            -----
            $K_total$ is defined as :

            :math::
                \[\Psi = \frac{2}{5} \cdot \frac{D_{11}^2 + D_{22}^2 + D_{33}^2
                                                 + 2D_{12}^2 + 2D_{13}^2 +
                                                 2D_{23}^2{\overline{D}^2} -
                                                 \frac{6}{5} \]
                \[{\overline{W}} = \frac{1}{5} \cdot (W_{1111} + W_{2222}
                                                      + W_{3333} + 2W_{1122}
                                                      + 2W_{1133} + 2W_{2233})\
                  ]

            where \(\Psi\) is a variable representing a part of the total
            excess kurtosis,
            \(D_{ij}\) are elements of the diffusion tensor,
            \(\overline{D}\) is the mean of the diffusion tensor.
            \{\overline{W}} is the mean kurtosis,
            \(W_{ijkl}\) are elements of the kurtosis tensor.
        """

        mean_K = self.mkt()
        D = self.quadratic_form
        mean_D = self.md
        if mean_D == 0:
            psi = 0
        else:
            psi = 2 / 5 * ((D[..., 0, 0]**2 + D[..., 1, 1]**2
                            + D[..., 2, 2]**2
                            + 2 * D[..., 0, 1]**2 + 2 * D[..., 0, 2]**2
                            + D[..., 1, 2]**2) / (mean_D ** 2)) - (6/5)

        return mean_K + psi

    @property
    def K_micro(self):
        r""" Returns Microscopic Source of Kurtosis.  """

        K_total = self.K_total
        K_aniso = self.K_aniso
        K_iso = self.K_iso
        micro_K = K_total - K_aniso - K_iso
        return micro_K


def params_to_cti_params(result, min_diffusivity=0):
    # Extracting the diffusion tensor parameters from solution
    DT_elements = result[:6]
    evals, evecs = decompose_tensor(from_lower_triangular(DT_elements),
                                    min_diffusivity=min_diffusivity)
    # Extracting kurtosis tensor parameters from solution
    MD_square = evals.mean(0)**2
    KT_elements = result[6:21] / MD_square if MD_square else 0.*result[6:21]

    # Extracting correlation tensor parameters from solution
    CT_elements = result[21:42]

    # Write output
    cti_params = np.concatenate((evals, evecs[0], evecs[1], evecs[2],
                                KT_elements, CT_elements), axis=0)

    return cti_params


def ls_fit_cti(design_matrix, data, inverse_design_matrix, weights=True,
               min_diffusivity=0):
    r""" Compute the diffusion kurtosis and covariance tensors using an
    ordinary or weighted linear least squares approach

    Parameters
    ----------
    design_matrix : array (g, 43)
        Design matrix holding the covariants used to solve for the regression
        coefficients.
    data : array (g)
        Data or response variables holding the data.
    inverse_design_matrix : array (43, g)
        Inverse of the design matrix.
    weights : bool, optional
        Parameter indicating whether weights are used. Default: True.
    min_diffusivity : float, optional
        Because negative eigenvalues are not physical and small eigenvalues,
        much smaller than the diffusion weighting, cause quite a lot of noise
        in metrics such as fa, diffusivity values smaller than `min_diffusivity`
        are replaced with `min_diffusivity`.

    Returns
    -------
    cti_params : array (48)
    All parameters estimated from the diffusion kurtosis model for all N
    voxels. Parameters are ordered as follows:
        1) Three diffusion tensor eigenvalues.
        2) Three blocks of three elements, containing the first second and
            third coordinates of the diffusion tensor eigenvectors.
        3) Fifteen elements of the kurtosis tensor.
        4) Twenty One elements of the covariance tensor.

    """
    A = design_matrix
    y = np.log(data)
    result = np.dot(inverse_design_matrix, y)
    if weights:
        W = np.diag(np.exp(2 * np.dot(A, result)))
        AT_W = np.dot(A.T, W)
        inv_AT_W_A = np.linalg.pinv(np.dot(AT_W, A))
        AT_W_LS = np.dot(AT_W, y)
        result = np.dot(inv_AT_W_A, AT_W_LS)
    cti_params = params_to_cti_params(result, min_diffusivity=min_diffusivity)

    return cti_params


common_fit_methods = {'WLS': ls_fit_cti,
                      'OLS': ls_fit_cti,
                      'UWLLS': ls_fit_cti,
                      'ULLS': ls_fit_cti,
                      'WLLS': ls_fit_cti,
                      'OLLS': ls_fit_cti
                      }
