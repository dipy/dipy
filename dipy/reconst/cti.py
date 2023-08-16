#!/usr/bin/python
""" Classes and functions for fitting the correlation tensor model """

import warnings
import functools
import numpy as np
import scipy.optimize as opt
from dipy.core.optimize import PositiveDefiniteLeastSquares
from dipy.reconst.multi_voxel import multi_voxel_fit

from dipy.reconst.base import ReconstModel
from dipy.reconst.utils import cti_design_matrix as design_matrix
from dipy.reconst.dki import (
    DiffusionKurtosisFit,
    apparent_kurtosis_coef,
    mean_kurtosis,
    axial_kurtosis
    )
from dipy.data import load_sdp_constraints
from dipy.reconst.dti import (
    decompose_tensor, from_lower_triangular, lower_triangular, mean_diffusivity, MIN_POSITIVE_SIGNAL)
from dipy.reconst.qti import from_6x1_to_3x3
# sources of kurtosis:
# we've formulats for Kaniso, Ksio, then get Ktotal. So for microscopic kurtosis, we subtract, Kt - K aniso




def split_cti_params(cti_params): #here cti_params.shape : (48, )
    r"""Extract the diffusion tensor eigenvalues, the diffusion tensor eigenvector matrix, and the 21 independent elements of the covariance tensor, and the 15 independent elements of the kurtosis tensor from the model parameters estimated from the CTI model
    Parameters:
         -----------
         params: numpy.ndarray (...,48)
                 All parameters estimated from the correlation tensor model.
                 Paramters are ordered as follows::

                 1. Three diffusion tensor's eigenvalues
                 2. Three lines of the eigenvector matrix each containing the
                 first, second and third coordinates of the eigenvector
                 3. Fifteen elements of the kurtosis tensor
                 4. Twenty-One elements of the covariance tensor
         S0 : float or ndarray (optional)
             The non diffusion-weighted signal in every voxel, or across all
             voxels. Default: 100

         Returns
         -------
         evals: Three diffusion tensor's eigenvalues
         evecs: Three lines of the eigenvector matrix each continaint ehf irst, second and third coordinates of the eigenvector
         kt: Fifteen elemnets of the kurtosis tensor
         cvt : Twenty-one elements of the covariance tensor

       """
    # DT_elements = np.squeeze(cti_params[:6, ...])
    # evals, evecs = decompose_tensor(from_lower_triangular(DT_elements))
    evals = cti_params[..., :3]
    evecs = cti_params[..., 3:12].reshape(cti_params.shape[:-1] + (3, 3))
    kt = cti_params[12:27, ...] #original
    # kt = cti_params[..., 12:27]
    cvt = cti_params[27:48, ...]
    return evals, evecs, kt, cvt


def cti_prediction(cti_params, gtab1, gtab2, S0=100):
    """Predict a signal given correlation tensor imaging parameters

        Parameters
        ----------
        cti_params: numpy.ndarray (...,48)
                All parameters estimated from the correlation tensor model.
                Paramters are ordered as follows::

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
            Simulated signal based on the CTI model:

    """
    evals, evecs, kt, cvt = split_cti_params(cti_params)
    # Define CTI design matrix according to given gtabs
    A = design_matrix(gtab1, gtab2)
    # Flat parameters and initialize pred_sig
    fevals = evals.reshape((-1, evals.shape[-1]))
    fevecs = evecs.reshape((-1,) + evecs.shape[-2:])
    fcvt = cvt.reshape((-1, cvt.shape[-1]))  # added line
    fkt = kt.reshape((-1, kt.shape[-1]))
    pred_sig = np.zeros((len(fevals), len(gtab1.bvals)))

    if isinstance(S0, np.ndarray):
        S0_vol = np.reshape(S0, (len(fevals)))
    else:
        S0_vol = S0
    # looping for all voxels
    for v in range(len(pred_sig)):
        DT = np.dot(np.dot(fevecs[v], np.diag(fevals[v])), fevecs[v].T)
        dt = lower_triangular(DT)
        MD = (dt[0] + dt[2] + dt[5]) / 3
        if isinstance(S0_vol, np.ndarray):
            this_S0 = S0_vol[v]
        else:
            this_S0 = S0_vol
        X = np.concatenate((dt, fkt[v] * MD * MD, fcvt[v],  # added line
                            np.array([-np.log(this_S0)])),
                           axis=0)
        # print("Under cti.py, cti_prediction function: ")
        # print("A shape:", A.shape) #(636, 43)
        # print("X shape:", X.shape) #(43, )
        # print("A", A) #very large values
        # print("X", X)
        # print("A dot X:", np.dot(A, X))
        # print("exp(A dot X):", np.exp(np.dot(A, X)))
        pred_sig[v] = np.exp(np.dot(A, X))

    # Reshape data according to the shape of cti_params
    pred_sig = pred_sig.reshape(cti_params.shape[:-1] + (pred_sig.shape[-1],))

    return pred_sig


class CorrelationTensorModel(ReconstModel):
    """ Class for the Correlation Tensor Model
    """

    # not sure about the fit method yet
    def __init__(self, gtab1, gtab2, fit_method="WLS", *args, **kwargs):
        """ Correlation Tensor Imaging Model [1]

        Parameters
        ----------
        gtab1: dipy.core.gradients.GradientTable
        A GradientTable class instance for first DDE diffusion epoch
        gtab2: dipy.core.gradients.GradientTable
        A GradientTable class instance for second DDE diffusion epoch

        fit_method : str or callable


        args, kwargs : arguments and key-word arguments passed to the
        fit_method.

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
                msg = '"' + str(fit_method) + '" is not a known fit method. The'
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
        # self.convexity_constraint = fit_method in {'CLS', 'CWLS'}
        # if self.convexity_constraint:
        #     self.cvxpy_solver = self.kwargs.pop('cvxpy_solver', None)
        #     self.convexity_level = self.kwargs.pop('convexity_level', 'full')
        #     msg = "convexity_level must be a positive, even number, or 'full'."
        #     if isinstance(self.convexity_level, str):
        #         if self.convexity_level == 'full':
        #             self.sdp_constraints = load_sdp_constraints('dki')
        #         else:
        #             raise ValueError(msg)
        #     elif self.convexity_level < 0 or self.convexity_level % 2:
        #         raise ValueError(msg)
        #     else:
        #         if self.convexity_level > 4:
        #             msg = "Maximum convexity_level supported is 4."
        #             warnings.warn(msg)
        #             self.convexity_level = 4
        #         self.sdp_constraints = load_sdp_constraints(
        #             'dki', self.convexity_level)
        #     self.sdp = PositiveDefiniteLeastSquares(22, A=self.sdp_constraints)

        self.weights = fit_method in {'WLS', 'WLLS', 'UWLLS'}
        self.is_multi_method = fit_method in ['WLS', 'OLS', 'UWLLS', 'ULLS',
                                              'WLLS', 'OLLS']
    @multi_voxel_fit 
    def fit(self, data, mask=None): #here data is cti_params of shape : (n, 48) 
        """ Fit method of the CTI model class

        Parameters
        ----------
        data : array
            The measured signal from one voxel.

        mask : array
            A boolean array used to mark the coordinates in the data that
            should be analyzed that has the shape data.shape[-1]

        """
        data_thres = np.maximum(data, self.min_signal)
        params = self.fit_method(self.design_matrix, data_thres,self.inverse_design_matrix,
                                 *self.args, **self.kwargs)
        #we need to somehow obtian cti_params from data (it's actually cti_pred_signals)
        # print('this is data.shape: ',data.shape) #this is always (81, )
        return CorrelationTensorFit(self, params) #ig there's a need to somehow define cti_params here. 
    
    # @multi_voxel_fit
    # def multi_fit(self, data_thres, mask=None):

    #     params = self.fit_method(self.design_matrix, data_thres,
    #                              self.inverse_design_matrix,
    #                              weights=self.weights,
    #                              min_diffusivity=self.min_diffusivity,
    #                             )

    #     return CorrelationTensorFit(self, params)
    
    def predict(self, cti_params, S0=100):  # created
        """Predict a signal for the CTI model class instance given parameteres

        Parameters:
        -----------
        cti_params: numpy.ndarray (...,48)
                All parameters estimated from the correlation tensor model.
                Paramters are ordered as follows::

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
            Signals.
        """

        return cti_prediction(cti_params, self.gtab1, self.gtab2, S0)


class CorrelationTensorFit(DiffusionKurtosisFit):

    """ Class for fitting the Correlation Tensor Model """

    def __init__(self, model, model_params):
        """ Initialize a CorrelationTensorFit class instance.

        Since CTI is an extension of DKI, class instance is defined as subclass
        of the DiffusionKurtosis from dki.py

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
    def kt(self):  # self.model_params.shape = (48, )
        """
        Return the 15 independent elements of the kurtosis tensor as an array
        """
        return self.model_params[12:27, ...]  

    @property
    def dft(self):  # created
        """
        Returns the 6 independent elements of the diffusion tensor as an array
        """
        # Extract the eigenvalues and the eigenvectors
        evals = self.model_params[:3]
        evecs = self.model_params[3:12].reshape((3, 3))

        # Construct the diffusion tensor
        diffusion_tensor = np.dot(np.dot(evecs, np.diag(evals)), evecs.T)

        # Extract the independent elements of the diffusion tensor
        dt_elements = np.array([diffusion_tensor[0, 0],
                                diffusion_tensor[1, 1],
                                diffusion_tensor[2, 2],
                                diffusion_tensor[0, 1],
                                diffusion_tensor[0, 2],
                                diffusion_tensor[1, 2]])

        return dt_elements
    
    @property
    def cvt(self):  # created
        """
        Returns the 21 independent elements of the covariance tensor as an array
        """
        return self.model_params[27:48, ...]

   # def cvt(__):  # calculates the mean of all covariance parameters. required ? Formula ?

        # There are 4 (radial,mean,axial,fractional) do we have something similar for covariance in qti ?

        # we separate the kurtosis in 3 parts: isotropic+anisotropoic+microscopic. Do we need methods for this? : REFER VIDEO ON THIS

    def predict(self, gtab1, gtab2, S0=100):  # created
        """Given a CTI model fit, predict the signal on the vertices of a gradient table

        Parameters:
        -----------
        params: numpy.ndarray (...,43)
                All parameters estimated from the correlation tensor model.
                Paramters are ordered as follows::

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
            Signals.
        """
        return cti_prediction(self.model_params, gtab1, gtab2, S0)
        
        #def calculate_K_aniso(D, C):  # D is a diffusion tensor 3x3, C is a 1D array.

    def K_aniso(self):
        r""" Returns the anisotropic Source of Kurtosis (K_aniso) 
            
            Notes 
            -----
            The K_aniso is defined as : 
            
            :math:: 
            
            \[K_{aniso} = \frac{6}{5} \cdot \frac{\langle V_{\lambda}(D_c) \rangle}{\overline{D}^2}\]
            
        where: \(K_{aniso}\) is the anisotropic kurtosis, 
            \(\langle V_{\lambda}(D_c) \rangle\) represents the mean of the variance of eigenvalues of the diffusion tensor,
            \(\overline{D}\) is the mean of the diffusion tensor.
        """ 

        D = self.dft
        C = self.cvt
        # print('this is D inside aniso, previously: ', D)
        # D = D.reshape(-1, 1)
        # #converting D
        # D = from_6x1_to_3x3(D)
        matrix = np.array([[D[0], D[3], D[4]],[D[3], D[1], D[5]],[D[4], D[5], D[2]]])
        D  = matrix 
        # print('this is D after: ', D)
        
        Variance = 2/9 * (C[0] + D[0, 0] ** 2 + C[1] + D[1, 1] ** 2 + C[2] + D[2, 2]**2 - C[5]
                 - D[0, 0] * D[1, 1] - C[4] - D[0, 0] * D[2, 2]
                 - C[3] - D[1, 1] * D[2, 2]
                 + 3 * (C[17] + D[0, 1] ** 2 + C[16] + D[0, 2] ** 2 + C[15] + D[1, 2] ** 2))

        mean_D =  np.trace(D) / 3#trace(D) / 3
        K_aniso = (6/5) * (Variance / (mean_D **2))
        return K_aniso 

    def K_iso(self): 
        r""" Returns the isotropic Source of Kurtosis (K_iso)
        
        Notes 
        -----
        The K_iso is defined as : 
        
        :math:: 
            \[K_{iso} = 3 \cdot \frac{V({\overline{D}^c})}{\overline{D}^2}\]
        
        where: \(K_{iso}\) is the isotropic kurtosis,
            \(V({\overline{D}^c})\) represents the variance of the diffusion tensor raised to the power c, 
            \(\overline{D}\) is the mean of the diffusion tensor.
                
        """ 
        D = self.dft #this is a (6, ) shaped array, 
        C = self.cvt
        #shouldn't we do : C = ccti ? or is this required only when 6x6 conversion is done?
        matrix = np.array([[D[0], D[3], D[4]],[D[3], D[1], D[5]],[D[4], D[5], D[2]]])
        D  = matrix 
        # print('this is D.shape inside K_iso: ', D.shape) #(6, )
        mean_D = np.trace(D) / 3
        Variance = 1/9 * (C[0] + C[1] + C[2] + 2 * C[5] + 2 * C[4] + 2 * C[3])
        K_iso = 3 * (Variance / mean_D)
        return K_iso

    def K_total(self):  #excess kurtosis. #W: kurtosis tenosr a 1D array, D: diffusionTensor: (3,3) matrix 
        #mean_K = mean_kurtosis_tensor(cti_params) 
        r""" Returns the total execess Kurtosis. (K_total)
            
            Notes
            -----
            The K_total is defined as :
            
            :math:: 
                \[\Psi = \frac{2}{5} \cdot \frac{D_{11}^2 + D_{22}^2 + D_{33}^2 + 2D_{12}^2 + 2D_{13}^2 + 2D_{23}^2{\overline{D}^2} - \frac{6}{5} \]
                \[{\overline{W}} = \frac{1}{5} \cdot (W_{1111} + W_{2222} + W_{3333} + 2W_{1122} + 2W_{1133} + 2W_{2233})\]
            
            where \(\Psi\) is a variable representing a part of the total excess kurtosis,
            \(D_{ij}\) are elements of the diffusion tensor,
            \(\overline{D}\) is the mean of the diffusion tensor.
            \{\overline{W}} is the mean kurtosis,
            \(W_{ijkl}\) are elements of the kurtosis tensor.
        """ 

        mean_K = self.mkt()
        D = self.dft
        matrix = np.array([[D[0], D[3], D[4]],[D[3], D[1], D[5]],[D[4], D[5], D[2]]])
        D  = matrix 
        mean_D = np.trace(D) / 3
        psi = 2/ 5 * ((np.sqrt(D[0,0]) + np.sqrt(D[1,1])+ np.sqrt(D[2,2])
                + 2 * np.sqrt(D[0,1]) + 2 * np.sqrt(D[0,2]) + np.sqrt(D[1,2])) / mean_D ** 2) - (6/5) 
        excess_K = 1/5 * mean_K  + psi 
        return excess_K  

    def K_micro(self):
        r""" Returns Microscopic Source of Kurtosis.  """ 

        K_excess = self.K_total()
        K_aniso = self.calculate_K_aniso()
        micro_K = K_excess - K_aniso 
        return micro_K


def params_to_cti_params(result, min_diffusivity=0):
    # Extracting the diffusion tensor parameters from solution
    DT_elements = result[:6]
    evals, evecs = decompose_tensor(from_lower_triangular(DT_elements),
                                    min_diffusivity=min_diffusivity)

    # Extracting kurtosis tensor parameters from solution
    MD_square = evals.mean(0)**2
    KT_elements = result[6:21] / MD_square if MD_square else 0.*result[6:21]

    # Extracting covariance tensor parameters from solution
    CT_elements = result[21:42]

    # Write output
    cti_params= np.concatenate((evals, evecs[0], evecs[1], evecs[2],
                                 KT_elements, CT_elements), axis=0)

    return cti_params

def ls_fit_cti(design_matrix, data, inverse_design_matrix, weights=True,  # shouldn't the effect of covariance tensor be obsvd ?
               min_diffusivity=0):
    r""" Compute the diffusion kurtosis and covariance tensors using an ordinary or
    weighted linear least squares approach [1]_

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
    # Set up least squares problem
    A = design_matrix
    y = np.log(data)

    # CTI ordinary linear least square solution
    result = np.dot(inverse_design_matrix, y)

    # Define weights as diag(yn**2)
    if weights:
        W = np.diag(np.exp(2 * np.dot(A, result)))
        AT_W = np.dot(A.T, W)
        inv_AT_W_A = np.linalg.pinv(np.dot(AT_W, A))
        AT_W_LS = np.dot(AT_W, y)
        result = np.dot(inv_AT_W_A, AT_W_LS)


    # Write output
    cti_params = params_to_cti_params(result, min_diffusivity=min_diffusivity)

    return cti_params

# def cls_fit_cti(design_matrix, data, inverse_design_matrix, sdp, weights=True,  # shouldn't the effect of covariance tensor be obsvd ?
#                min_diffusivity=0, cvxpy_solver=None):
#     r""" Compute the diffusion kurtosis and covariance tensors using a constrained ordinary or
#     weighted linear least squares approach [1]_

#     Parameters
#     ----------
#     design_matrix : array (g, 43)
#         Design matrix holding the covariants used to solve for the regression
#         coefficients.
#     data : array (g)
#         Data or response variables holding the data.
#     inverse_design_matrix : array (43, g)
#         Inverse of the design matrix.
#     sdp : PositiveDefiniteLeastSquares instance
#         A CVXPY representation of a regularized least squares optimization
#         problem.
#     weights : bool, optional
#         Parameter indicating whether weights are used. Default: True.
#     min_diffusivity : float, optional
#         Because negative eigenvalues are not physical and small eigenvalues,
#         much smaller than the diffusion weighting, cause quite a lot of noise
#         in metrics such as fa, diffusivity values smaller than `min_diffusivity`
#         are replaced with `min_diffusivity`.
#     cvxpy_solver : str, optional
#         cvxpy solver name. Optionally optimize the positivity constraint with a
#         particular cvxpy solver. See http://www.cvxpy.org/ for details.
#         Default: None (cvxpy chooses its own solver).

#     Returns
#     -------
#     cti_params : array (48)
#         All parameters estimated from the diffusion kurtosis model for all N
#         voxels. Parameters are ordered as follows:
#             1) Three diffusion tensor eigenvalues.
#             2) Three blocks of three elements, containing the first second and
#                third coordinates of the diffusion tensor eigenvectors.
#             3) Fifteen elements of the kurtosis tensor.
#             4) Twenty One elements of the covariance tensor.

#     """
#     # Set up least squares problem
#     A = design_matrix
#     y = np.log(data)

#     # Define sqrt weights as diag(yn)
#     if weights:
#         result = np.dot(inverse_design_matrix, y)
#         W = np.diag(np.exp(np.dot(A, result)))
#         A = np.dot(W, A)
#         y = np.dot(W, y)

#     # Solve sdp
#     result = sdp.solve(A, y, check=True, solver=cvxpy_solver)

#     # Write output
#     cti_params = params_to_cti_params(result, min_diffusivity=min_diffusivity)

#     return cti_params
# def params_to_dki_params(result, min_diffusivity=0):
    # takes kurtosis tensor parameters and returns a matrix


# def params_to_dti_params(result, min_diffusivity=0):


# def params_to_cvt_params(result, min_diffusivity=0):
def from_3x3_to_6x1_temp(T):
    """Convert symmetric 3 x 3 matrices into 6 x 1 vectors.

    Parameters
    ----------
    T : numpy.ndarray
        An array of size (..., 3, 3).

    Returns
    -------
    V : numpy.ndarray
        Converted vectors of size (..., 6).

    Notes
    -----
    The conversion of a matrix into a vector is defined as

        .. math::

            \mathbf{V} = \begin{bmatrix}
            T_{11} & T_{22} & T_{33} &
            \sqrt{2} T_{23} & \sqrt{2} T_{13} & \sqrt{2} T_{12}
            \end{bmatrix}^T
    """
    if T.shape[-2::] != (3, 3):
        raise ValueError('The shape of the input array must be (..., 3, 3).')
    if not np.all(np.isclose(T, np.swapaxes(T, -1, -2))):
        warnings.warn('All matrices converted to Voigt notation are not symmetric.')
    C = np.sqrt(2)
    V = np.stack((T[..., 0, 0],
                  T[..., 1, 1],
                  T[..., 2, 2],
                  C * T[..., 1, 2],
                  C * T[..., 0, 2],
                  C * T[..., 0, 1]), axis=-1)
    return V


common_fit_methods = {'WLS': ls_fit_cti,
                      'OLS': ls_fit_cti,
                      'UWLLS': ls_fit_cti,
                      'ULLS': ls_fit_cti,
                      'WLLS': ls_fit_cti,
                      'OLLS': ls_fit_cti
                      }
