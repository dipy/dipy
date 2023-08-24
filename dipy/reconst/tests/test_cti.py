import numpy as np
import math

from dipy.core.sphere import disperse_charges, HemiSphere
from dipy.reconst.utils import cti_design_matrix as design_matrix
from numpy.testing import (assert_array_almost_equal)
from dipy.reconst.tests.test_qti import _anisotropic_DTD, _isotropic_DTD
from dipy.core.gradients import gradient_table
import dipy.reconst.qti as qti
import dipy.reconst.cti as cti
from dipy.reconst.dti import (
    decompose_tensor, mean_diffusivity)
from dipy.reconst.cti import (split_cti_params, ls_fit_cti)
from dipy.reconst.dki import (mean_kurtosis,
                              axial_kurtosis, radial_kurtosis,
                              mean_kurtosis_tensor,
                              kurtosis_fractional_anisotropy)
from dipy.utils.optpkg import optional_package
_, have_cvxpy, _ = optional_package("cvxpy")


def _perpendicular_directions_temp(v, num=20, half=False):
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


# Simulation: signals of two crossing fibers are simulated
n_pts = 20
theta = np.pi * np.random.rand(n_pts)
phi = 2 * np.pi * np.random.rand(n_pts)
hsph_initial = HemiSphere(theta=theta, phi=phi)
hsph_updated, potential = disperse_charges(hsph_initial, 5000)

bvecs1 = np.concatenate([hsph_updated.vertices] * 4)
bvecs1 = np.append(bvecs1, [[0, 0, 0]], axis=0)
bvals1 = np.array([2] * 20 + [1] * 20 + [1] * 20 + [1] * 20 + [0])
gtab1 = gradient_table(bvals1, bvecs1)
hsph_updated90 = _perpendicular_directions_temp(hsph_updated.vertices)
dot_product = np.sum(hsph_updated.vertices * hsph_updated90, axis=1)
are_perpendicular = np.isclose(dot_product, 0)
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

CTI_data = np.zeros((2, 2, 1, len(gtab1.bvals)))


def construct_cti_params(evals, evecs, kt, fct):
    fevals = evals.reshape((-1, evals.shape[-1]))
    fevecs = evecs.reshape((-1,) + evecs.shape[-2:])
    fevecs = fevecs.reshape((1, -1))
    fkt = kt.reshape((-1, kt.shape[-1]))
    cti_params = np.concatenate((fevals.T, fevecs.T, fkt, fct), axis=0)
    return np.squeeze(cti_params)


def modify_C_params(C):
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


def generate_K(ccti, MD):
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


def test_cti_prediction():
    ctiM = cti.CorrelationTensorModel(gtab1, gtab2)
    anisotropic_DTD = _anisotropic_DTD()
    isotropic_DTD = _isotropic_DTD()

    DTDs = [
        anisotropic_DTD,
        isotropic_DTD,
        np.concatenate((anisotropic_DTD, isotropic_DTD))
    ]

    for DTD in DTDs:
        D = np.mean(DTD, axis=0)
        evals, evecs = decompose_tensor(D)
        C = qti.dtd_covariance(DTD)
        C = qti.from_6x6_to_21x1(C)
        ccti = modify_C_params(C)
        MD = mean_diffusivity(evals)
        K = generate_K(ccti, MD)
        cti_params = construct_cti_params(evals, evecs, K, ccti)
        cti_pred_signals = ctiM.predict(cti_params, S0 = S0)
        qti_pred_signals = qti.qti_signal(gtab, D, C, S0=S0)[
            np.newaxis, :]
        assert np.allclose(cti_pred_signals, qti_pred_signals), (
            "CTI and QTI signals do not match!"
        )


def test_split_cti_param():
    ctiM = cti.CorrelationTensorModel(gtab1, gtab2)
    for DTD in DTDs:
        D = np.mean(DTD, axis=0)
        evals, evecs = decompose_tensor(D)
        C = qti.dtd_covariance(DTD)
        C = qti.from_6x6_to_21x1(C)
        ccti = modify_C_params(C)

        MD = mean_diffusivity(evals)
        K = generate_K(ccti, MD)

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
    DTDs = [
        anisotropic_DTD,
        isotropic_DTD,
        np.concatenate((anisotropic_DTD, isotropic_DTD))
    ]
    ctiM = cti.CorrelationTensorModel(gtab1, gtab2)
    for i, DTD in enumerate(DTDs):
        D = np.mean(DTD, axis=0)
        evals, evecs = decompose_tensor(D)
        C = qti.dtd_covariance(DTD)
        C = qti.from_6x6_to_21x1(C)
        ccti = modify_C_params(C)
        MD = mean_diffusivity(evals)
        K = generate_K(ccti, MD)
        cti_params = construct_cti_params(evals, evecs, K, ccti)
        cti_pred_signals = ctiM.predict(cti_params, S0=S0)
        evals, evecs, kt, ct = split_cti_params(cti_params)

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
        # assert_array_almost_equal(ctiF_multi.model_params, multi_params)
        assert np.allclose(ctiF_multi.model_params, multi_params), (
            "multi voxel fit doesn't pass"
            )
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
        assert np.allclose(kfa_result, kurtosis_fractional_anisotropy_result), (
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

# def test_isotropic_source():
#     ctiM = cti.CorrelationTensorModel(gtab1, gtab2)
#     isotropic_DTD = _isotropic_DTD()
#     anisotropic_DTD = _anisotropic_DTD()
#     DTD = np.concatenate((anisotropic_DTD, isotropic_DTD))
#     DTD = _isotropic_DTD()
#     D = np.mean(DTD, axis=0)
#     evals, evecs = decompose_tensor(D)
#     C = qti.dtd_covariance(DTD)
#     C = qti.from_6x6_to_21x1(C)
#     ccti = modify_C_params(C)
#     MD = mean_diffusivity(evals)
#     K = generate_K(ccti, MD)
#     cti_params = construct_cti_params(evals, evecs, K, ccti)
#     cti_pred_signals = ctiM.predict(cti_params, S0=S0)
#     ctiF = ctiM.fit(cti_pred_signals)
#     K_iso = ctiF.K_iso

    
#     # variance_of_mean_diffusivities = np.var(MD)
#     # print('this is variance_of_mean_diffusivities: ', variance_of_mean_diffusivities )
#     # # print('And this is np.mean(MD): ', np.mean(MD))
#     # ground_truth_K_iso = 3 * variance_of_mean_diffusivities / np.mean(MD)**2

#     mean_diffusivities = []

#     for tensor in DTD:
#         evals_tensor, _ = decompose_tensor(tensor)
#         mean_diffusivities.append(np.mean(evals_tensor))

#     # Variance of individual tensor's mean diffusivities
#     variance_of_mean_diffusivities = np.var(mean_diffusivities)
#     mean_D = np.mean(mean_diffusivities)  # Or use the existing computation: mean_D = np.trace(np.mean(DTD, axis=0)) / 3
#     ground_truth_K_iso = 3 * variance_of_mean_diffusivities / (mean_D ** 2)


#     error_msg = (
#         f"Calculated K_iso {K_iso} for anisotropicDTD does not match the "
#         f"ground truth {ground_truth_K_iso}"
#     )
#     assert np.isclose(K_iso, ground_truth_K_iso), error_msg
