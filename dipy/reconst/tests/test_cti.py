import numpy as np
import math

from dipy.core.sphere import disperse_charges, HemiSphere
from dipy.reconst.utils import cti_design_matrix as design_matrix
from numpy.testing import (assert_array_almost_equal, assert_raises)
from dipy.reconst.tests.test_qti import _anisotropic_DTD, _isotropic_DTD
from dipy.core.gradients import gradient_table
import dipy.reconst.qti as qti
import dipy.reconst.cti as cti
from dipy.reconst.dti import (
    decompose_tensor, mean_diffusivity)
from dipy.reconst.cti import (split_cti_params, ls_fit_cti,
                              multi_gaussian_k_from_c, from_qte_to_cti)
from dipy.reconst.dki import (mean_kurtosis,
                              axial_kurtosis, radial_kurtosis,
                              mean_kurtosis_tensor,
                              kurtosis_fractional_anisotropy)

gtab1, gtab2, gtab, DTDs, S0 = None, None, None, None, None


def setup_module():
    global gtab1, gtab2, gtab, DTDs, S0
    # Generating the DDE acquisition parameters (gtab1 and gtab2) for CTI
    # based on the minimal requirements.
    # This code then further generates the corresponding QTE gtab to test
    # simulated signals based on multiple Gaussian components.
    # In this scenario, CTI should yield analogous Kaniso and Kiso values as
    # compared to QTE, while Kmicro would be zero.

    rng = np.random.default_rng(1234)
    n_pts = 20
    theta = np.pi * rng.random(n_pts)
    phi = 2 * np.pi * rng.random(n_pts)
    hsph_initial = HemiSphere(theta=theta, phi=phi)
    hsph_updated, potential = disperse_charges(hsph_initial, 5000)

    # Generating gtab1
    bvecs1 = np.concatenate([hsph_updated.vertices] * 4)
    bvecs1 = np.append(bvecs1, [[0, 0, 0]], axis=0)
    bvals1 = np.array([2] * 20 + [1] * 20 + [1] * 20 + [1] * 20 + [0])
    gtab1 = gradient_table(bvals1, bvecs1)

    # Generating perpendicular directions to hsph_updated
    hsph_updated90 = _perpendicular_directions_temp(hsph_updated.vertices)
    dot_product = np.sum(hsph_updated.vertices * hsph_updated90, axis=1)
    are_perpendicular = np.isclose(dot_product, 0)

    # Generating gtab2
    bvecs2 = np.concatenate(([hsph_updated.vertices] * 2) +
                            [hsph_updated90] + ([hsph_updated.vertices]))
    bvecs2 = np.append(bvecs2, [[0, 0, 0]], axis=0)
    bvals2 = np.array([0] * 20 + [1] * 20 + [1] * 20 + [0] * 20 + [0])
    gtab2 = gradient_table(bvals2, bvecs2)

    e1 = bvecs1
    e2 = bvecs2
    e3 = np.cross(e1, e2)
    V = np.stack((e1, e2, e3), axis=-1)
    V_transpose = np.transpose(V, axes=(0, 2, 1))
    B = np.zeros((81, 3, 3))
    b = np.zeros((3, 3))
    for i in range(81):
        b[0, 0] = bvals1[i]
        b[1, 1] = bvals2[i]
        B[i] = np.matmul(V[i], np.matmul(b, V_transpose[i]))
    gtab = gradient_table(bvals1, bvecs1, btens=B)
    S0 = 100
    anisotropic_DTD = _anisotropic_DTD()
    isotropic_DTD = _isotropic_DTD()

    DTDs = [
        anisotropic_DTD,
        isotropic_DTD,
        np.concatenate((anisotropic_DTD, isotropic_DTD))
    ]
    DTD_labels = ['Anisotropic DTD', 'Isotropic DTD', 'Combined DTD']


def teardown_module():
    global gtab1, gtab2, gtab, DTDs, S0
    gtab1, gtab2, gtab, DTDs, S0 = None, None, None, None, None


def _perpendicular_directions_temp(v, num=20, half=False):
    """
    Computes a set of perpendicular directions relative to the direction in v.

    The perpendicular directions are computed by sampling on a unit
    circumference that is perpendicular to `v`. The computation depends on
    whether the vector is aligned with the x-axis or not.

    Parameters
    ----------
    v : array (3,)
        Array containing the three cartesian coordinates of vector v.
    num : int, optional
        Number of perpendicular directions to generate. Default is 20.
    half : bool, optional
        If True, perpendicular directions are sampled on half of the unit
        circumference perpendicular to v, otherwise they are sampled on the
        full circumference. Default is False.

    Returns
    -------
    psamples : array (n, 3)
        Array of vectors perpendicular to v.
    """
    v = np.array(v, dtype=np.float64)
    v = v.T
    er = np.finfo(v[0].dtype).eps * 1e3
    if half is True:
        a = np.linspace(0., math.pi, num=num, endpoint=False)
    else:
        a = np.linspace(0., 2 * math.pi, num=num, endpoint=False)
    cosa = np.cos(a)
    sina = np.sin(a)
    if np.any(abs(v[0] - 1.) > er):
        sq = np.sqrt(v[1]**2 + v[2]**2)
        psamples = np.array([- sq*sina, (v[0]*v[1]*sina - v[2]*cosa) / sq,
                             (v[0]*v[2]*sina + v[1]*cosa) / sq])
    else:
        sq = np.sqrt(v[0]**2 + v[2]**2)
        psamples = np.array([- (v[2]*cosa + v[0]*v[1]*sina) / sq, sina*sq,
                             (v[0]*cosa - v[2]*v[1]*sina) / sq])
    return psamples.T


def construct_cti_params(evals, evecs, kt, fct):
    """
    Combines all components to generate Correlation Tensor Model Parameter.

    Parameters
    ----------
    evals : array (..., 3)
        Eigenvalues from eigen decomposition of the tensor.
    evecs : array (..., 3)
        Associated eigenvectors from eigen decomposition of the tensor.
        Eigenvectors are columnar (e.g. evecs[:,j] is associated with
        evals[j])
    kt : array (..., 15)
        Fifteen elements of the kurtosis tensor
    fct: array(..., 21)
        Twenty-one elements of the covariance tensor

    Returns
    -------
    cti_params :  numpy.ndarray (..., 48)
    All parameters estimated from the correlation tensor model.
    Parameters are ordered as follows:

        1. Three diffusion tensor's eigenvalues
        2. Three lines of the eigenvector matrix each containing the
        first, second and third coordinates of the eigenvector
        3. Fifteen elements of the kurtosis tensor
        4. Twenty-One elements of the covariance tensor
    """
    fevals = evals.reshape((-1, evals.shape[-1]))
    fevecs = evecs.reshape((-1,) + evecs.shape[-2:])
    fevecs = fevecs.reshape((1, -1))
    fkt = kt.reshape((-1, kt.shape[-1]))
    cti_params = np.concatenate((fevals.T, fevecs.T, fkt, fct), axis=0)
    return np.squeeze(cti_params)


def test_cti_prediction():
    ctiM = cti.CorrelationTensorModel(gtab1, gtab2)
    for DTD in DTDs:
        D = np.mean(DTD, axis=0)
        evals, evecs = decompose_tensor(D)
        C = qti.dtd_covariance(DTD)
        C = qti.from_6x6_to_21x1(C)
        ccti = from_qte_to_cti(C)
        MD = mean_diffusivity(evals)
        K = multi_gaussian_k_from_c(ccti, MD)
        cti_params = construct_cti_params(evals, evecs, K, ccti)
        cti_pred_signals = ctiM.predict(cti_params, S0=S0)
        qti_pred_signals = qti.qti_signal(gtab, D, C, S0=S0)[
            np.newaxis, :]
        assert np.allclose(cti_pred_signals, qti_pred_signals), (
            "CTI and QTI signals do not match!"
        )

        # check the function predict of the CorrelationTensorFit object
        ctiF = ctiM.fit(cti_pred_signals)
        ctiF_pred = ctiF.predict(gtab1, gtab2, S0=S0)
        assert_array_almost_equal(ctiF_pred, cti_pred_signals)


def test_split_cti_param():
    ctiM = cti.CorrelationTensorModel(gtab1, gtab2)
    for DTD in DTDs:
        D = np.mean(DTD, axis=0)
        evals, evecs = decompose_tensor(D)
        C = qti.dtd_covariance(DTD)
        C = qti.from_6x6_to_21x1(C)
        ccti = from_qte_to_cti(C)

        MD = mean_diffusivity(evals)
        K = multi_gaussian_k_from_c(ccti, MD)

        cti_params = construct_cti_params(evals, evecs, K, ccti)
        ctiM = cti.CorrelationTensorModel(gtab1, gtab2)
        cti_pred_signals = ctiM.predict(cti_params, S0=S0)
        ctiF = ctiM.fit(cti_pred_signals)
        evals, evecs, kt, ct = cti.split_cti_params(ctiF.model_params)

        assert_array_almost_equal(evals, ctiF.evals)
        assert_array_almost_equal(evecs, ctiF.evecs)
        assert np.allclose(
            kt, ctiF.kt), "kt doesn't match in test_split_cti_param "
        assert np.allclose(
            ct, ctiF.ct), "ct doesn't match in test_split_cti_param"


def test_cti_fits():
    ctiM = cti.CorrelationTensorModel(gtab1, gtab2)
    CTI_data = np.zeros((2, 2, 1, len(gtab1.bvals)))
    for i, DTD in enumerate(DTDs):
        D = np.mean(DTD, axis=0)
        evals, evecs = decompose_tensor(D)
        C = qti.dtd_covariance(DTD)
        C = qti.from_6x6_to_21x1(C)
        ccti = from_qte_to_cti(C)
        MD = mean_diffusivity(evals)
        K = multi_gaussian_k_from_c(ccti, MD)
        cti_params = construct_cti_params(evals, evecs, K, ccti)
        cti_pred_signals = ctiM.predict(cti_params, S0=S0)
        evals, evecs, kt, ct = split_cti_params(cti_params)

        # Testing the model with correct min_signal value
        ctiM = cti.CorrelationTensorModel(gtab1, gtab2, min_signal=1)
        cti_pred_signals = ctiM.predict(cti_params, S0=S0)
        ctiF = ctiM.fit(cti_pred_signals)
        evals, evecs, kt, ct = cti.split_cti_params(ctiF.model_params)
        assert_array_almost_equal(evals, ctiF.evals)
        assert_array_almost_equal(evecs, ctiF.evecs)
        assert np.allclose(
            kt, ctiF.kt), "kt doesn't match in test_split_cti_param "
        assert np.allclose(
            ct, ctiF.ct), "ct doesn't match in test_split_cti_param"

        # Testing Multi-Voxel Fit
        CTI_data[0, 0, 0] = cti_pred_signals
        CTI_data[0, 1, 0] = cti_pred_signals
        CTI_data[1, 0, 0] = cti_pred_signals
        CTI_data[1, 1, 0] = cti_pred_signals
        multi_params = np.zeros((2, 2, 1, 48))
        multi_params[0, 0, 0] = multi_params[0, 1, 0] = cti_params
        multi_params[1, 0, 0] = multi_params[1, 1, 0] = cti_params
        ctiF_multi = ctiM.fit(CTI_data)

        multi_evals, _, multi_kt, multi_ct = split_cti_params(
            ctiF_multi.model_params)
        assert np.allclose(evals, multi_evals), "Evals don't match"
        assert np.allclose(kt, multi_kt), "K doesn't match"
        assert np.allclose(ct, multi_ct), "C doesn't match"

        # Check that it works with more than one voxel, and with a different S0
        # in each voxel:
        cti_multi_pred_signals = ctiM.predict(multi_params,
                                              S0=100*np.ones(
                                                  ctiF_multi.shape[:3])
                                              )
        CTI_data = cti_multi_pred_signals
        ctiF_multi_pred_signals = ctiM.fit(CTI_data)
        multi_evals, _, multi_kt, multi_ct = split_cti_params(
            ctiF_multi_pred_signals.model_params)
        assert np.allclose(evals, multi_evals), "Evals don't match"
        assert np.allclose(kt, multi_kt), "K doesn't match"
        assert np.allclose(ct, multi_ct), "C doesn't match"

        # Testing ls_fit_cti
        inverse_design_matrix = np.linalg.pinv(design_matrix(gtab1, gtab2))
        cti_return = ls_fit_cti(design_matrix(
            gtab1, gtab2), cti_pred_signals, inverse_design_matrix)
        evals_return, _, kt_return, ct_return = split_cti_params(cti_return)
        assert np.allclose(evals, evals_return), "evals do not match!"
        assert np.allclose(kt, kt_return), "K do not match!"
        assert np.allclose(ct, ct_return), "C do not match!"

        # OLS fitting
        ctiM = cti.CorrelationTensorModel(gtab1, gtab2, fit_method="OLS")
        ctiF = ctiM.fit(cti_pred_signals)
        ols_evals, _, ols_kt, ols_ct = split_cti_params(ctiF.model_params)
        assert np.allclose(evals, ols_evals), "evals do not match!"
        assert np.allclose(kt, ols_kt), "K do not match!"
        assert np.allclose(ct, ols_ct), "C do not match!"

        # WLS fitting
        cti_wlsM = cti.CorrelationTensorModel(gtab1, gtab2, fit_method="WLS")
        cti_wlsF = cti_wlsM.fit(cti_pred_signals)
        wls_evals, _, wls_kt, wls_ct = split_cti_params(cti_wlsF.model_params)
        assert np.allclose(evals, wls_evals), "evals do not match!"
        assert np.allclose(kt, wls_kt), "K do not match!"
        assert np.allclose(ct, wls_ct), "C do not match!"

        # checking Mean Kurtosis Values
        mk_result = ctiF.mk(min_kurtosis=-3./7,
                            max_kurtosis=10, analytical=True)
        mean_kurtosis_result = mean_kurtosis(
            cti_params, min_kurtosis=-3./7, max_kurtosis=10, analytical=True)
        assert np.allclose(mk_result, mean_kurtosis_result), (
            "The results of the mk function from CorrelationTensorFit and the "
            "mean_kurtosis function from dki.py are not equal."
        )
        # checking Axial Kurtosis Values
        ak_result = ctiF.ak(min_kurtosis=-3./7,
                            max_kurtosis=10, analytical=True)
        axial_kurtosis_result = axial_kurtosis(
            cti_params, min_kurtosis=-3./7, max_kurtosis=10, analytical=True)

        assert np.allclose(ak_result, axial_kurtosis_result), (
            "The results of the ak function from CorrelationTensorFit and the "
            "axial_kurtosis function from dki.py are not equal."
        )
        # checking Radial kurtosis values
        rk_result = ctiF.rk(min_kurtosis=-3./7,
                            max_kurtosis=10, analytical=True)
        radial_kurtosis_result = radial_kurtosis(
            cti_params, min_kurtosis=-3./7, max_kurtosis=10, analytical=True)

        assert np.allclose(rk_result, radial_kurtosis_result), (
            "The results of the rk function from CorrelationTensorFit and the "
            "radial_kurtosis function from DKI.py are not equal."
        )
        # checking Anisotropic values.
        kfa_result = ctiF.kfa
        kurtosis_fractional_anisotropy_result = kurtosis_fractional_anisotropy(
            cti_params)
        assert np.allclose(kfa_result,
                           kurtosis_fractional_anisotropy_result), (
            "The results of the kfa function from CorrelationTensorFit and the"
            "kurtosis_fractional_anisotropy function from dki.py are not equal"
        )
        # checking mean Kurtosis tensor
        mkt_result = ctiF.mkt(min_kurtosis=-3./7, max_kurtosis=10)
        mean_kurtosis_result = mean_kurtosis_tensor(
            cti_params, min_kurtosis=-3./7, max_kurtosis=10)
        assert np.allclose(mkt_result, mean_kurtosis_result), (
            "The results of the mkt function from CorrelationTensorFit and the"
            "mean_kurtosis_tensor function from dki.py are not equal."
        )
        # checking anisotropic source of kurtosis.
        K_aniso = ctiF.K_aniso
        variance_of_eigenvalues = []
        for tensor in DTD:
            evals_tensor, _ = decompose_tensor(tensor)
            variance_of_eigenvalues.append(np.var(evals_tensor))
        mean_variance_of_eigenvalues = np.mean(variance_of_eigenvalues)
        mean_D = np.trace(np.mean(DTD, axis=0)) / 3

        ground_truth_K_aniso = (
            6/5) * (mean_variance_of_eigenvalues / (mean_D ** 2))
        error_msg = (
            f"Calculated K_iso {K_aniso} for isotropicDTD does not match the "
            f"ground truth {ground_truth_K_aniso}"
        )
        assert np.isclose(K_aniso, ground_truth_K_aniso), error_msg

        # checking isotropic source of kurtosis.
        K_iso = ctiF.K_iso

        mean_diffusivities = []

        for tensor in DTD:
            evals_tensor, _ = decompose_tensor(tensor)
            mean_diffusivities.append(np.mean(evals_tensor))
        variance_of_mean_diffusivities = np.var(mean_diffusivities)
        mean_D = np.mean(mean_diffusivities)
        ground_truth_K_iso = 3 * variance_of_mean_diffusivities / (mean_D ** 2)

        error_msg = (
            f"Calculated K_iso {K_iso} for anisotropicDTD does not match the "
            f"ground truth {ground_truth_K_iso}"
        )
        assert np.isclose(K_iso, ground_truth_K_iso), error_msg

        # checking microscopic source of kurtosis
        ground_truth_K_micro = 0
        K_micro = ctiF.K_micro
        assert np.allclose(K_micro, ground_truth_K_micro), (
            "K_micro values don't match ground truth values"
            )


def test_cti_errors():

    # first error of CTI module is if a unknown fit method is given
    assert_raises(ValueError, cti.CorrelationTensorModel, gtab1, gtab2,
                  fit_method="")

    # second error of CTI module is if a min_signal is defined as negative
    assert_raises(ValueError, cti.CorrelationTensorModel, gtab1, gtab2,
                  min_signal=-1)


def test_cti_design_matrix():
    A1 = design_matrix(gtab1, gtab2)
    A2 = design_matrix(gtab2, gtab1)
    # Check if the two matrices are the same
    assert np.allclose(A1, A2), (
        "The design matrices are not symmetric for different gradient"
        "directions order."
        )
