import matplotlib.pyplot as plt
import numpy as np
import math

from dipy.reconst.tests.test_qti import _anisotropic_DTD, _isotropic_DTD
from dipy.core.gradients import gradient_table
from dipy.core.sphere import disperse_charges, Sphere, HemiSphere
from dipy.sims.voxel import multi_tensor
from dipy.reconst.tests.test_qti import _anisotropic_DTD, _isotropic_DTD
import dipy.reconst.qti as qti
import dipy.reconst.cti as cti 
from dipy.reconst.qti import (from_3x3_to_6x1, from_6x1_to_3x3, dtd_covariance, qti_signal)
from dipy.reconst.dti import (decompose_tensor, from_lower_triangular, mean_diffusivity)
from dipy.reconst.cti import (cti_prediction, split_cti_params)
from dipy.reconst.qti import (qti_signal)
import dipy.reconst.cti as cti 
import dipy.reconst.qti as qti 
from dipy.reconst.qti import from_3x3_to_6x1

def perpendicular_directions_temp(v, num=20, half=False):
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
hsph_updated90 = perpendicular_directions_temp(hsph_updated.vertices)
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

for DTD in DTDs:
    D = np.mean(DTD, axis=0)
    C = qti.dtd_covariance(DTD)
    params = np.concatenate(
        (np.zeros((1, 1)),  # replaced np.log(1)[np.newaxis, np.newaxis] with 0
            qti.from_3x3_to_6x1(D),
            qti.from_6x6_to_21x1(C))).T
    data = qti.qti_signal(gtab, D, C, S0=S0)[np.newaxis, :]

    # use labels that describe the signal
    plt.plot(data.ravel(), label=DTD_labels[i])

plt.legend()
plt.show()


def construct_cti_params(evals, evecs, kt, fcvt):
    fevals = evals.reshape((-1, evals.shape[-1])) #has shape: (1, 3)
    fevecs = evecs.reshape((-1,) + evecs.shape[-2:])
    fevecs = fevecs.reshape((1, -1)) #shape: (3, 3)
    fkt = kt.reshape((-1, kt.shape[-1]))
    # Concatenate all the flattened tensors
    cti_params = np.concatenate((fevals.T, fevecs.T, fkt, fcvt), axis = 0)    
    return np.squeeze(cti_params) #returns shape: (48, )


def test_cti_prediction():
    ctiM = cti.CorrelationTensorModel(gtab1, gtab2)
    anisotropic_DTD = _anisotropic_DTD()
    isotropic_DTD = _isotropic_DTD()

    DTDs = [
        anisotropic_DTD,
        isotropic_DTD,
        np.concatenate((anisotropic_DTD, isotropic_DTD))
    ]
    DTD_labels = ["anisotropic_DTD", "isotropic_DTD", "combined_DTD"]


    for DTD in DTDs:
        D = np.mean(DTD, axis=0)
        D_flat = np.squeeze(from_3x3_to_6x1(D)) #has shape:(6, )
        evals, evecs = decompose_tensor(
            from_lower_triangular(D_flat))  #evals:shape: (3, ) & evecs.shape: (3, 3)
        C = qti.dtd_covariance(DTD)
        C = qti.from_6x6_to_21x1(C)
        MD = mean_diffusivity(evals) #is a sclar
        # Compute kurtosis tensor (K)
        K = np.zeros((15,1))
        K[0] = 3 * C[0] / (MD ** 2)
        K[1] = 3 * C[1] / (MD ** 2)
        K[2] = 3 * C[2] / (MD ** 2)
        K[3] = 3 * C[8] / (MD ** 2)
        K[4] = 3 * C[7] / (MD ** 2)
        K[5] = 3 * C[11] / (MD ** 2)
        K[6] = 3 * C[9] / (MD ** 2)
        K[7] = 3 * C[13] / (MD ** 2)
        K[8] = 3 * C[12] / (MD ** 2)
        K[9] = (C[5] + 2 * C[17]) / (MD**2)
        K[10] = (C[4] + 2 * C[16]) / (MD**2)
        K[11] = (C[3] + 2 * C[15]) / (MD**2)
        K[12] = (C[6] + 2 * C[19]) / (MD**2)
        K[13] = (C[10] + 2 * C[20]) / (MD**2)
        K[14] = (C[14] + 2 * C[18]) / (MD**2)

        cti_params = construct_cti_params(evals, evecs, K, C)
        # Generate predicted signals using cti_prediction function
        cti_pred_signals = ctiM.predict(cti_params) #shape: (81, )

        # Generate predicted signals using QTI model
        qti_pred_signals = qti.qti_signal(gtab, D, C, S0=S0)[np.newaxis, :] #shape:(81, )
        
        # plt.plot(cti_pred_signals.ravel(), label=DTD_labels[i])
        # plt.plot(qti_pred_signals.ravel(), label=DTD_labels[i])

        # Compare CTI and QTI predicted signals
        assert np.allclose(
            cti_pred_signals, qti_pred_signals), "CTI and QTI signals do not match!"
