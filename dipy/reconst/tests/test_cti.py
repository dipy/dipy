import matplotlib.pyplot as plt
import numpy as np
import math

from dipy.reconst.utils import cti_design_matrix as design_matrix
from numpy.testing import (assert_array_almost_equal, assert_array_equal,
                           assert_almost_equal, assert_raises)
from dipy.reconst.tests.test_qti import _anisotropic_DTD, _isotropic_DTD
from dipy.core.gradients import gradient_table
from dipy.core.sphere import disperse_charges, Sphere, HemiSphere
from dipy.sims.voxel import multi_tensor
import dipy.reconst.qti as qti
import dipy.reconst.cti as cti
from dipy.reconst.qti import (
    from_3x3_to_6x1, from_6x1_to_3x3, dtd_covariance, qti_signal)
from dipy.reconst.dti import (
    decompose_tensor, from_lower_triangular, mean_diffusivity)
from dipy.reconst.cti import (cti_prediction, split_cti_params, ls_fit_cti)
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
    cosa = np.cos(a)  # (20,)
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
n_pts = 20  # points are assumed to be on a sphere
theta = np.pi * np.random.rand(n_pts)  # theta: angle betn point P and z-axis
phi = 2 * np.pi * np.random.rand(n_pts)  # value ranges between 0 to n
hsph_initial = HemiSphere(theta=theta, phi=phi)
hsph_updated, potential = disperse_charges(hsph_initial, 5000)
# defining bvecs1, bvals1
# total 4 x 20 + 1 = 81 vectors
bvecs1 = np.concatenate([hsph_updated.vertices] * 4)
bvecs1 = np.append(bvecs1, [[0, 0, 0]], axis=0)
bvals1 = np.array([2] * 20 + [1] * 20 + [1] * 20 + [1] * 20 + [0])
# in order to create 2 gtabs,
gtab1 = gradient_table(bvals1, bvecs1)
# Now in order to create perpendicular vector, we'll use a method: perpendicular_directions
hsph_updated90 = _perpendicular_directions_temp(hsph_updated.vertices)
dot_product = np.sum(hsph_updated.vertices * hsph_updated90, axis=1)
are_perpendicular = np.isclose(dot_product, 0)
bvecs2 = np.concatenate(([hsph_updated.vertices] * 2) +
                        [hsph_updated90] + ([hsph_updated.vertices]))
bvecs2 = np.append(bvecs2, [[0, 0, 0]], axis=0)
bvals2 = np.array([0] * 20 + [1] * 20 + [1] * 20 + [0] * 20 + [0])
# Creating the second gtab table:
gtab2 = gradient_table(bvals2, bvecs2)
# Defining Btens:
e1 = bvecs1  # (81,3)
e2 = bvecs2  # (81,3)
e3 = np.cross(e1, e2)
V = np.stack((e1, e2, e3), axis=-1)
# transposing along 2nd and 3rd axis.
V_transpose = np.transpose(V, axes=(0, 2, 1))
B = np.zeros((81, 3, 3))  # initializing a btensor
b = np.zeros((3, 3))
for i in range(81):
    b[0, 0] = bvals1[i]
    b[1, 1] = bvals2[i]
    B[i] = np.matmul(V[i], np.matmul(b, V_transpose[i]))

# on providing btens, (bvals1,bvecs1) is ignored.
gtab = gradient_table(bvals1, bvecs1, btens=B)
S0 = 100
# we've isotropic and anisotropic diffusion tensor distribution (DTD)
anisotropic_DTD = _anisotropic_DTD()  # assuming these functions work correctly
isotropic_DTD = _isotropic_DTD()

DTDs = [
    anisotropic_DTD,
    isotropic_DTD,
    np.concatenate((anisotropic_DTD, isotropic_DTD))
]

# label for each DTD, for the plot
DTD_labels = ['Anisotropic DTD', 'Isotropic DTD', 'Combined DTD']


def construct_cti_params(evals, evecs, kt, fcvt):
    fevals = evals.reshape((-1, evals.shape[-1]))  # has shape: (1, 3)
    fevecs = evecs.reshape((-1,) + evecs.shape[-2:])
    fevecs = fevecs.reshape((1, -1))  # shape: (3, 3)
    fkt = kt.reshape((-1, kt.shape[-1]))
    # Concatenate all the flattened tensors
    cti_params = np.concatenate((fevals.T, fevecs.T, fkt, fcvt), axis=0)
    return np.squeeze(cti_params)  # returns shape: (48, )


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

def convert_E_bulk(Ebulk): 
    const = np.sqrt(2)
    E_bulk = np.zeros((21, 1))
    E_bulk[0] = Ebulk[0]
    E_bulk[1] = Ebulk[1]
    E_bulk[2] = Ebulk[2]
    E_bulk[3] = Ebulk[3] / const
    E_bulk[4] = Ebulk[4] / const
    E_bulk[5] = Ebulk[5] / const
    E_bulk[6] = Ebulk[6] / 2
    E_bulk[7] = Ebulk[7] / 2
    E_bulk[8] = Ebulk[8] / 2
    E_bulk[9] = Ebulk[9] / 2
    E_bulk[10] = Ebulk[10] / 2
    E_bulk[11] = Ebulk[11] / 2
    E_bulk[12] = Ebulk[12] / 2
    E_bulk[13] = Ebulk[13] / 2
    E_bulk[14] = Ebulk[14] / 2
    E_bulk[15] = Ebulk[15] / 2
    E_bulk[16] = Ebulk[16] / 2
    E_bulk[17] = Ebulk[17] / 2
    E_bulk[18] = Ebulk[18] / (2 * const)
    E_bulk[19] = Ebulk[19] / (2 * const)
    E_bulk[20] = Ebulk[20] / (2 * const)
    return E_bulk

def convert_E_shear(Eshear): 
    const = np.sqrt(2)
    E_shear = np.zeros((21, 1))
    E_shear[0] = Eshear[0]
    E_shear[1] = Eshear[1]
    E_shear[2] = Eshear[2]
    E_shear[3] = Eshear[3] / const
    E_shear[4] = Eshear[4] / const
    E_shear[5] = Eshear[5] / const
    E_shear[6] = Eshear[6] / 2
    E_shear[7] = Eshear[7] / 2
    E_shear[8] = Eshear[8] / 2
    E_shear[9] = Eshear[9] / 2
    E_shear[10] = Eshear[10] / 2
    E_shear[11] = Eshear[11] / 2
    E_shear[12] = Eshear[12] / 2
    E_shear[13] = Eshear[13] / 2
    E_shear[14] = Eshear[14] / 2
    E_shear[15] = Eshear[15] / 2
    E_shear[16] = Eshear[16] / 2
    E_shear[17] = Eshear[17] / 2
    E_shear[18] = Eshear[18] / (2 * const)
    E_shear[19] = Eshear[19] / (2 * const)
    E_shear[20] = Eshear[20] / (2 * const)
    return E_shear

def convert_d_sq(dsq): 
    const = np.sqrt(2)
    d_sq = np.zeros((21, 1))
    d_sq[0] = dsq[0]
    d_sq[1] = dsq[1]
    d_sq[2] = dsq[2]
    d_sq[3] = dsq[3] / const
    d_sq[4] = dsq[4] / const
    d_sq[5] = dsq[5] / const
    d_sq[6] = dsq[6] / 2
    d_sq[7] = dsq[7] / 2
    d_sq[8] = dsq[8] / 2
    d_sq[9] = dsq[9] / 2
    d_sq[10] = dsq[10] / 2
    d_sq[11] = dsq[11] / 2
    d_sq[12] = dsq[12] / 2
    d_sq[13] = dsq[13] / 2
    d_sq[14] = dsq[14] / 2
    d_sq[15] = dsq[15] / 2
    d_sq[16] = dsq[16] / 2
    d_sq[17] = dsq[17] / 2
    d_sq[18] = dsq[18] / (2 * const)
    d_sq[19] = dsq[19] / (2 * const)
    d_sq[20] = dsq[20] / (2 * const)
    return d_sq

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
        # has shape:(6, )    #pretty useless, not needed
        D_flat = np.squeeze(from_3x3_to_6x1(D))
        # evals:shape: (3, ) & evecs.shape: (3, 3)
        evals, evecs = decompose_tensor(D)
        C = qti.dtd_covariance(DTD)
        C = qti.from_6x6_to_21x1(C)

        # getting C_params from voigt notation
        ccti = modify_C_params(C)

        MD = mean_diffusivity(evals)  # is a sclar
        # Compute kurtosis tensor (K)
        K = generate_K(ccti, MD)

        cti_params = construct_cti_params(evals, evecs, K, ccti)
        # Generate predicted signals using cti_prediction function
        cti_pred_signals = ctiM.predict(cti_params)  # shape: (81, )

        # Generate predicted signals using QTI model
        qti_pred_signals = qti.qti_signal(gtab, D, C, S0=S0)[
            np.newaxis, :]  # shape:(81, )

        # Compare CTI and QTI predicted signals
        assert np.allclose(
            cti_pred_signals, qti_pred_signals), "CTI and QTI signals do not match!"


def test_split_cti_param():
    ctiM = cti.CorrelationTensorModel(gtab1, gtab2) #fit_method is WLS by default.
    for DTD in DTDs: #generating cti_pred_signals for all DTDs
        D = np.mean(DTD, axis=0)
        evals, evecs = decompose_tensor(D)
        C = qti.dtd_covariance(DTD)
        C = qti.from_6x6_to_21x1(C)

        # getting C_params from voigt notation
        ccti = modify_C_params(C)

        MD = mean_diffusivity(evals)  # is a sclar
        # Compute kurtosis tensor (K)
        K = generate_K(ccti, MD)

        cti_params = construct_cti_params(evals, evecs, K, ccti)
        # Generate predicted signals using cti_prediction function
        ctiM = cti.CorrelationTensorModel(gtab1, gtab2) 
        cti_pred_signals = ctiM.predict(cti_params)
        #DWI = np.zeros((2, 2, 1, len(gtab.bvals)))
        #DWI[0, 0, 0] = DWI[0, 1, 0] = DWI[1, 0, 0] = DWI[1, 1, 0] = cti_pred_signals
        ctiF = ctiM.fit(cti_pred_signals) #should pass cti_pred_signal, in our case we don't have DWI
        # ctiF = ctiM.fit(cti_params)
        evals, evecs, kt, cvt = cti.split_cti_params(ctiF.model_params)
        print('this is ctiF.kt: and its type: ', ctiF.kt, type(ctiF.kt))
        assert_array_almost_equal(evals, ctiF.evals)
        assert_array_almost_equal(evecs, ctiF.evecs)
        assert_array_almost_equal(kt, ctiF.kt)
        assert_array_almost_equal(cvt, ctiF.cvt)


def test_cti_fits():
    ctiM = cti.CorrelationTensorModel(gtab1, gtab2)
    for DTD in DTDs:  # trying out all fits for each DTD.
        D = np.mean(DTD, axis=0)
        evals, evecs = decompose_tensor(D)
        C = qti.dtd_covariance(DTD)
        C = qti.from_6x6_to_21x1(C)

        # getting C_params from voigt notation
        ccti = modify_C_params(C)

        MD = mean_diffusivity(evals)  # is a sclar
        # Compute kurtosis tensor (K)
        K = generate_K(ccti, MD)

        cti_params = construct_cti_params(evals, evecs, K, ccti)
        # Generate predicted signals using cti_prediction function
        cti_pred_signals = ctiM.predict(cti_params)

        # def ls_fit_cti(design_matrix, data, inverse_design_matrix, weights=True,  # shouldn't the effect of covariance tensor be obsvd ?
        #        min_diffusivity=0):
        inverse_design_matrix = np.linalg.pinv(design_matrix(gtab1, gtab2))
        cti_return = ls_fit_cti(design_matrix(gtab1, gtab2), cti_pred_signals, inverse_design_matrix )
        # OLS fitting
        ctiM = cti.CorrelationTensorModel(gtab1, gtab2, fit_method="OLS")
        # ctiF = ctiM.fit(cti_pred_signals)
        # ctiF = ctiM.fit(cti_params) #this is turning out to be NoneType                     #error 

        # assert_array_almost_equal(ctiF.model_params, cti_params)
        assert_array_almost_equal(cti_return, cti_params)

        # WLS fitting
        cti_wlsM = cti.CorrelationTensorModel(gtab1, gtab2, fit_method="WLS")
        # signal_cross ---> cti_pred_signals, crossing_ref --> cti_params
        cti_wlsF = cti_wlsM.fit(cti_pred_signals)
        # cti_wlsF = cti_wlsM.fit(cti_params)

        assert_array_almost_equal(cti_wlsF.model_params, cti_params)

        if have_cvxpy:
            # CLS fitting
            cti_clsM = cti.CorrelationTensorModel(
                gtab1, gtab2, fit_method="CLS")
            cti_clsF = cti_clsM.fit(cti_params)

            assert_array_almost_equal(cti_clsF.model_params, cti_params)

            # CWLS fitting
            cti_cwlsM = cti.CorrelationTensorModel(
                gtab1, gtab2, fit_method="CWLS")
            cti_cwlsF = cti_cwlsM.fit(cti_params)

            assert_array_almost_equal(cti_clsF.model_params, cti_params)
        else:
            assert_raises(ValueError, cti.CorrelationTensorModel,
                          gtab1, gtab2, fit_method="CLS")
            assert_raises(ValueError, cti.CorrelationTensorModel,
                          gtab1, gtab2, fit_method="CWLS")

        # checking Mean Kurtosis values:
        mk_result = ctiF.mk(min_kurtosis=-3./7,
                            max_kurtosis=10, analytical=True)
        mean_kurtosis_result = mean_kurtosis(
            cti_params, min_kurtosis=-3./7, max_kurtosis=10, analytical=True)
        assert mk_result == mean_kurtosis_result, "The results of the mk function from CorrelationTensorFit and the mean_kurtosis function from dki.py are not equal."

        # checking Axial Kurtosis Values
        ak_result = ctiF.ak(min_kurtosis=-3./7,
                            max_kurtosis=10, analytical=True)
        axial_kurtosis_result = axial_kurtosis(
            cti_params, min_kurtosis=-3./7, max_kurtosis=10, analytical=True)
        assert ak_result == axial_kurtosis_result, "The result of the ak function from CorrealtionTensorFit and the axial_kurtosis function from dki.py are not equal."

        # checking Radial kurtosis values
        rk_result = ctiF.rk(min_kurtosis=-3./7,
                            max_kurtosis=10, analytical=True)
        radial_kurtosis_result = radial_kurtosis(
            cti_params, min_kurtosis=-3./7, max_kurtosis=10, analytical=True)
        assert rk_result == radial_kurtosis_result, "The results of the rk function from CorrelationTensorfit and the radial_kurtosis function from dki.py are not equal. "

        # checking Anisotropic values.
        kfa_result = ctiF.kfa()
        kurtosis_fractional_anisotropy_result = kurtosis_fractional_anisotropy(
            cti_params)
        assert kfa_result == kurtosis_fractional_anisotropy_result, "the reuslts of the kfa function from CorrelationTensorFit and the kurtosis_fractional_anisotropy function from dki.py are not equal. "

        # checking mean Kurtosis tensor
        mkt_result = ctiF.mkt(min_kurtosis=-3./7, max_kurtosis=10)
        mkt_kurtosis_result = ctiF.mean_kurtosis_tensor(
            cti_params, min_kurtosis=-3./7, max_kurtosis=10)
        assert mkt_result == mkt_kurtosis_result, "The results of mkt function from CorrelationTensorFit and the mean_kurtosis_tensor function from dki.py are not equal. "

        #checking sources of kurtosis : 
        d_sq = qti.from_3x3_to_6x1(D) @ qti.from_3x3_to_6x1(D).T
        e_iso = np.eye(3) / 3
        E_bulk = from_3x3_to_6x1(e_iso) @ from_3x3_to_6x1(e_iso).T                  #this is a 6x6 matrix.

        #defining test for K_iso
        k_bulk = (3 * np.matmul(                                                       #deal wE_bulk
            np.swapaxes(ccti, -1, -2),                                                 #also deal with d_sq conversion
            convert_E_bulk(qti.from_6x6_to_21x1(E_bulk))) / np.matmul(
                np.swapaxes(convert_d_sq(qti.from_6x6_to_21x1(d_sq)), -1, -2),          #define convert_d_sq
                convert_E_bulk(qti.from_6x6_to_21x1(E_bulk))))[0, 0]                   #define convert_E_bulk
        K_iso = ctiF.calculate_K_iso() 

        #defining test for K_aniso 
        k_shear = (6 / 5 * np.matmul(
            np.swapaxes(ccti, -1, -2),
            convert_E_shear(qti.from_6x6_to_21x1(qti.E_shear))) / np.matmul(            #define convert_E_shear
                np.swapaxes(convert_d_sq(qti.from_6x6_to_21x1(d_sq)), -1, -2),
                convert_E_bulk(qti.from_6x6_to_21x1(E_bulk))))[0, 0]
        K_aniso  = ctiF.calculate_K_aniso() 


        assert k_bulk == K_iso, "The results of calculate_K_iso function from CorrelationTensorFit and the k_bulk from qti.py are not equal. "
        assert k_shear == K_aniso, "The results of calculate_K_aniso function from CorrelationTensorFit and the k_shear from qti.py are not equal. "