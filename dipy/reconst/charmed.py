import numpy as np
from scipy.optimize import curve_fit, leastsq
from dipy.reconst import dti
from dipy.reconst.base import ReconstModel
import numpy as np
import dipy.data as dpd
import dipy.core.gradients as dpg
from dipy.segment.mask import median_otsu
import nibabel as nib
from dipy.core.gradients import GradientTable


""""
fdata, fbvals, fbvecs = dpd.get_data('small_101D')
img = nib.load(fdata)
data = img.get_data()
big_delta=150
small_delta=40
Dif_per = 1e-5
Tau = 100
R = 0.5
gtab = dpg.gradient_table(fbvals, fbvecs, big_delta=150,
                          small_delta=40, b0_threshold=1000)
a = GradientTable(gtab.gradients, big_delta=150,
small_delta=40, b0_threshold=1000)
a.bvals = gtab.bvals[gtab.b0s_mask]
a.bvecs = gtab.bvecs[gtab.b0s_mask]
a.gradients = gtab.gradients[gtab.b0s_mask]
a.b0s_mask = gtab.b0s_mask[gtab.b0s_mask]
print(a.b0s_mask)
x = GradientTable(gtab.gradients, big_delta=150,
small_delta=40, b0_threshold=1000)
x.bvals = gtab.bvals[~gtab.b0s_mask]
x.bvecs = gtab.bvecs[~gtab.b0s_mask]
x.gradients = gtab.gradients[~gtab.b0s_mask]
x.b0s_mask = gtab.b0s_mask[~gtab.b0s_mask]
maskdata, mask = median_otsu(data, 3, 1, True,
                             vol_idx=range(10, 50), dilate=2)
print('maskdata.shape (%d, %d, %d, %d)' % maskdata.shape)
voxels = maskdata[:,:,:,-1].flatten().shape[0]
"""


class CharmedModel(ReconstModel):
    """ CHARMED Model
    """
    def __init__(self, gtab, fit_method="WLS", *args, **kwargs):
        """ A CHARMED Model [1]_, [2]_.

        Parameters
        ----------
        gtab : GradientTable class instance

        fit_method : str or callable
            str can be one of the following:
            'WLS' for weighted least squares
                dti.wls_fit_tensor
            'LS' or 'OLS' for ordinary least squares
                dti.ols_fit_tensor
            'NLLS' for non-linear least-squares
                dti.nlls_fit_tensor
            'RT' or 'restore' or 'RESTORE' for RESTORE robust tensor
                fitting [3]_
                dti.restore_fit_tensor

            callable has to have the signature:
              fit_method(design_matrix, data, *args, **kwargs)

        args, kwargs : arguments and key-word arguments passed to the
           fit_method. See dti.wls_fit_tensor, dti.ols_fit_tensor for details

        min_signal : float
            The minimum signal value. Needs to be a strictly positive
            number. Default: minimal signal in the data provided to `fit`.

        References
        ----------

        """
        ReconstModel.__init__(self, gtab)

        if not callable(fit_method):
            try:
                fit_method = common_fit_methods[fit_method]
            except KeyError:
                e_s = '"' + str(fit_method) + '" is not a known fit '
                e_s += 'method, the fit method should either be a '
                e_s += 'function or one of the common fit methods'
                raise ValueError(e_s)
        self.fit_method = fit_method
        self.design_matrix = design_matrix(self.gtab)
        self.args = args
        self.kwargs = kwargs
        self.min_signal = self.kwargs.pop('min_signal', None)
        if self.min_signal is not None and self.min_signal <= 0:
            e_s = "The `min_signal` key-word argument needs to be strictly"
            e_s += " positive."
            raise ValueError(e_s)

    def fit(self, data, mask=None):
        """ Fit method of the CHARMED model class

        Parameters
        ----------
        data : array
            The measured signal from one voxel.

        mask : array
            A boolean array used to mark the coordinates in the data that
            should be analyzed that has the shape data.shape[:-1]

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

        if self.min_signal is None:
            min_signal = _min_positive_signal(data)
        else:
            min_signal = self.min_signal

        data_in_mask = np.maximum(data_in_mask, min_signal)
        params_in_mask = self.fit_method(self.design_matrix, data_in_mask,
                                         *self.args, **self.kwargs)

        if mask is None:
            out_shape = data.shape[:-1] + (-1, )
            dti_params = params_in_mask.reshape(out_shape)
        else:
            charmed_params = np.zeros(data.shape[:-1] + (12,))
            charmed_params[mask, :] = params_in_mask

        return TensorFit(self, charmed_params)

    def predict(self, dti_params, S0=1):
        """
        Predict a signal for this TensorModel class instance given parameters.

        Parameters
        ----------
        dti_params : ndarray
            The last dimension should have 12 tensor parameters: 3
            eigenvalues, followed by the 3 eigenvectors

        S0 : float or ndarray
            The non diffusion-weighted signal in every voxel, or across all
            voxels. Default: 1
        """
        return tensor_prediction(charmed_params, self.gtab)


def intial_conditions_prediction(gtab, maskdata):
    r""" This function calculates the intial parameters at low b values
    Parameters
    ----------
    gtab : a GradientTable class instance
    The gradient table for this prediction at low b values
    maskdata : 4D array having diffusion data in last dimension
        The 4D array of diffusion data with background masked
    Notes
    -----
    References
    ----------
    Returns
    -------
    Parameters estimated from DTI are as follows:
        1) The axial diffusivity in the voxel
        2) The radial diffusivity in the voxel
    """
    tenmodel = dti.TensorModel(gtab)
    tenfit = tenmodel.fit(maskdata[:, :, :, gtab.b0s_mask])

    intial_params = {}	 # intialising a dictionary
    #                      for storing intial parameters

    intial_params['lambda_per'] = (
        np.reshape(dti.axial_diffusivity(tenfit.evals), (voxels, -1)))
    #           lambda_per axial diffusivity

    intial_params['lambda_par'] = (
        np.reshape(dti.radial_diffusivity(tenfit.evals), (voxels, -1)))
    #  lambda_par is radial
    #  diffusivity

    return intial_params


def create_qtable(gtab, origin=np.array([0])):
    """ create a normalized version of gradients

    Parameters
    ----------
    gtab : GradientTable
    origin : (3,) ndarray
        center of qspace

    Returns
    -------
    qtable : ndarray
    """

    bv = gtab.bvals
    bsorted = np.sort(bv)
    for i in range(len(bsorted)):
        bmin = bsorted[i]
        try:
            if np.sqrt(bv.max() / bmin) > origin + 1:
                continue
            else:
                break
        except ZeroDivisionError:
            continue

    bv = np.sqrt(bv / bmin)
    qtable = np.vstack((bv, bv, bv)).T * gtab.bvecs
    return qtable


def hindered_signal(x, theta, phi):
    r""" Signal prediction at low b values for estimating the angles
    Parameters
    ----------
    x : an array containing a GradientTable class, axial diffusivities and
        radial diffusivities
    theta , phi : angles
        The spherical coordinates of the axon or nerve fasicles
    Notes
    -----

    References
    ----------
    """
    # create q vectors from b vectors
    qvec_H = create_qtable(x[0])
    # calculating spherical coordiantes for q vector
    phi_Q = np.arctan(qvec_H[:, 1]/qvec_H[:, 0])
    theta_Q = np.sqrt(qvec_H[:, 1]**2 + qvec_H[:, 0]**2)
    theta_Q = np.arctan(phi_Q/qvec_H[:, 2])
    # Estimating intial parameters from DTI model
    intial_params = intial_conditions_prediction(gtab, maskdata)
    # Calculating squares of perpendicular and parallel components of q vectors
    Qper2_H = (x[o].qvals**2)*(1 -
                               (np.sin(theta_Q)*np.sin(theta)*np.cos(phi_Q -
                                                                     phi) +
                                np.cos(theta_Q)*np.cos(theta))**2)
    Qpar2_H = (x[o].qvals**2)*((np.sin(theta_Q))*np.sin(theta)*np.cos(phi_Q -
                                                                      phi) +
                               np.cos(theta_Q)*np.cos(theta))**2
    # Calulate the predicted signal
    E_H = np.exp(-4 * (np.pi**2) * (big_delta - (small_delta/3)) *
                 (Qper2_H * x[1] + Qpar2_H * x[2]))
    return E_H


def hindered_fit(maskdata, gtab):
    r""" Fit the data using non-linear least squares
    Parameters
    ----------
    gtab : a GradientTable class instance
        The gradient table for this prediction at low b values
    maskdata : 4D array having diffusion data in last dimension
        The 4D array of diffusion data with background masked
    Returns
    -------
    param : For each voxel, the spherical coordinates of axons or nerve
        fasicles are estimated.
    """
    # Reshaping diffusion data at low b values into a 2-dimensional array
    ydata = np.reshape(maskdata[:, :, :, gtab.b0s_mask], (voxels, -1))
    # intialize a empty array to store the estimated parameters
    param = np.empty(voxels, dtype=object)
    intial_params = intial_conditions_prediction(a, maskdata)
    # Running the iterative fitting for each voxel
    for vox in range(voxels):
        x = [a, intial_params['lambda_per'][vox],
             intial_params['lambda_par'][vox]]
        param[vox], popt = curve_fit(hindered_signal, x, ydata[vox],
                                     bounds=([-np.pi, 0], [np.pi, np.pi]),
                                     method='trf')
    return param


def hindered_and_restricted_signal(xdata, theta_H, phi_H, theta_R, phi_R,
                                   lambda_per, lambda_par, Dif_par, f, noise):
    r""" Signal prediction at high b values
    Parameters
    ----------
    xdata : a GradientTable class instance
        The gradient table for this prediction at high b values
    theta_H, phi_H : angles
        Spherical coordinates of axons in hindered compartments
    theta_R, phi_R : angles
        Spherical coordinates of axons in restricted compartments
    lambda_per : axial diffusivity in hindered compartments
    lambda_par : radial diffusivity in hindered compartments
    Dif_par : axial diffusivity in redtricted compartments
    f : restricted compartment volume fraction
    Notes
    -----
    Refernces
    ---------
    """
    # create q vectors from b vectors
    qvec_R = create_qtable(xdata[0])
    # calculating spherical coordiantes for q vector
    phi_Q = np.arctan(qvec_R[:, 1]/qvec_R[:, 0])
    theta_Q = np.sqrt(qvec_R[:, 1]**2 + qvec_R[:, 0]**2)
    theta_Q = np.arctan(phi_Q/qvec_R[:, 2])
    # Calculating squares of perpendicular and parallel components of q vectors
    # in both hindered and restricted components
    Qper2_H = (xdata[0].qvals**2)*(1-(np.sin(theta_Q) * np.sin(theta_H) *
                                      np.cos(phi_Q - phi_H) +
                                      np.cos(theta_Q)*np.cos(theta_H))**2)
    Qpar2_H = (xdata[0].qvals**2)*((np.sin(theta_Q)) * np.sin(theta_H) *
                                   np.cos(phi_Q - phi_H) +
                                   np.cos(theta_Q)*np.cos(theta_H))**2
    Qper2_R = (xdata[0].qvals**2)*(1-(np.sin(theta_Q)*np.sin(theta_R) *
                                      np.cos(phi_Q - phi_R) +
                                      np.cos(theta_Q)*np.cos(theta_R))**2)
    Qpar2_R = (xdata[0].qvals**2)*((np.sin(theta_Q))*np.sin(theta_R) *
                                   np.cos(phi_Q - phi_R) +
                                   np.cos(theta_Q)*np.cos(theta_R))**2
    # Calulate the predicted signal from hindered compartment
    E_H = np.exp(-4 * (np.pi**2) * (big_delta - (small_delta/3)) *
                 (Qper2_R * lambda_per + Qpar2_R * lambda_par))
    # Calulate the predicted signal from restricted compartment
    E_R = np.exp(-4 * (np.pi**2) * (Qpar2_R * (big_delta - (small_delta/3)) *
                                    Dif_par + ((((R**4) * Qper2_R)/(Dif_per *
                                                                    Tau)) *
                                               (2 - ((99/112) *
                                                     ((R**2)/(Dif_per *
                                                              Tau)))))))
    E = (f)*E_R + (1-f)*E_H
    E = np.sqrt((E**2) + (noise**2))
    return E


def hind_and_rest_fit(maskdata, gtab, hind_param):
    r""" Fitting the data using non-linear least squares
    Parameters
    ----------
    gtab : a GradientTable class instance
        The gradient table for this prediction at high b values
    maskdata : 4D array having diffusion data in last dimension
        The 4D array of diffusion data with background masked
    hind_param : Spherical coordiantes of axons in hindered compartments
    Returns
    -------
    All the parameters estimated from CHARMED model
    Parameters are ordered as follows:
        1) Spherical coordiantes of axons in hindered compartments
        2) Spherical coordiantes of axons in restricted compartments
        3) Axial and radial diffusivities in hindered compartment
        4) Axial diffusivity in restricted compartment
        5) Volume fraction of restricted compartment
        6) Noise
    Notes
    -----
    References
    ----------
    """
    # Reshaping diffusion data at high b values into a 2-dimensional array
    ydata = np.reshape(maskdata[:, :, :, ~gtab.b0s_mask], (voxels, -1))
    # intial paramters  estimated from DTI model
    intial_params = intial_conditions_prediction(a, maskdata)
    charmed_params = np.empty(voxels, dtype=object)
    # Specifying the boundaries
    lb = np.array([-np.pi, 0, -np.pi, 0, 1e-10, 1e-10, 1e-10, 0, 0])
    ub = np.array([np.pi, np.pi, np.pi, np.pi, 1e5, 1e5, 1e5, 1, 1])
    for vox in range(voxels):
        xdata = [x]
        x0 = [hind_param[vox][0], hind_param[vox][1], hind_param[vox][0],
              hind_param[vox][1], intial_params['lambda_per'][vox],
              intial_params['lambda_par'][vox], Dif_per, 0.3, 0.03]
        charmed_params[vox], popt = curve_fit(hindered_and_restricted_signal,
                                              xdata, ydata[vox], p0=x0,
                                              bounds=(lb, ub), method='trf')
    return charmed_params
"""
def noise_function(E_est, noise):
    E = np.sqrt(E_est**2 + noise**2)
    return E


def noise_fit(data, E_est, n0):
    noise_param , flag = leastsq(noise_residual, n0, args=(data,E_est))
    return noise_param



intial_params = intial_conditions_prediction(a, maskdata)
#print(intial_params['lambda_per'])
print(maskdata[:,:,:,-1].flatten().shape)
hind_param = hindered_fit(maskdata,gtab)
high_b_param = hind_and_rest_fit(maskdata, gtab, hind_param)

print(hind_param)
"""
