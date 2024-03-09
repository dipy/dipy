import warnings

from dipy.denoise.enhancement_kernel import EnhancementKernel
from dipy.denoise.shift_twist_convolution import convolve, convolve_sf
from dipy.reconst.shm import sh_to_sf, sf_to_sh, descoteaux07_legacy_msg
from dipy.core.sphere import Sphere

import numpy as np
import numpy.testing as npt

def test_enhancement_kernel():
    """ Test if the kernel values are correct by comparison against the values
    originally calculated by implementation in Mathematica, and at the same time
    checks the symmetry of the kernel."""

    D33 = 1.0
    D44 = 0.04
    t = 1
    k = EnhancementKernel(D33, D44, t, orientations=0, force_recompute=True)

    y = np.array([0., 0., 0.])
    v = np.array([0., 0., 1.])
    orientationlist=[[0., 0., 1.], [-0.0527864, 0.688191, 0.723607],
      [-0.67082, -0.16246, 0.723607], [-0.0527864, -0.688191, 0.723607],
      [0.638197, -0.262866, 0.723607], [0.831052, 0.238856, 0.502295],
      [0.262866, -0.809017, -0.525731], [0.812731, 0.295242, -0.502295],
      [-0.029644, 0.864188, -0.502295], [-0.831052, 0.238856, -0.502295],
      [-0.638197, -0.262866, -0.723607], [-0.436009, 0.864188, -0.251148],
      [-0.687157, -0.681718, 0.251148], [0.67082, -0.688191, 0.276393],
      [0.67082, 0.688191, 0.276393], [0.947214, 0.16246, -0.276393],
      [-0.861803, -0.425325, -0.276393]]
    positionlist= [[-0.108096, 0.0412229, 0.339119], [0.220647, -0.422053, 0.427524],
      [-0.337432, -0.0644619, -0.340777], [0.172579, -0.217602, -0.292446],
      [-0.271575, -0.125249, -0.350906], [-0.483807, 0.326651, 0.191993],
      [-0.480936, -0.0718426, 0.33202], [0.497193, -0.00585659, -0.251344],
      [0.237737, 0.013634, -0.471988], [0.367569, -0.163581, 0.0723955],
      [0.47859, -0.143252, 0.318579], [-0.21474, -0.264929, -0.46786],
      [-0.0684234, 0.0342464, 0.0942475], [0.344272, 0.423119, -0.303866],
      [0.0430714, 0.216233, -0.308475], [0.386085, 0.127333, 0.0503609],
      [0.334723, 0.071415, 0.403906]]
    kernelvalues = [0.10701063104295713, 0.0030052117308328923, 0.003125410084676201,
      0.0031765819772012613, 0.003127254657020615, 0.0001295130396491743,
      6.882352014430076e-14, 1.3821277371353332e-13, 1.3951939946082493e-13,
      1.381612071786285e-13, 5.0861109163441125e-17, 1.0722120295517027e-10,
      2.425145934791457e-6, 3.557919265806602e-6, 3.6669510385105265e-6,
      5.97473789679846e-11, 6.155412262223178e-11]

    for p in range(len(orientationlist)):
        r = np.array(orientationlist[p])
        x = np.array(positionlist[p])
        npt.assert_almost_equal(k.evaluate_kernel(x, y, r, v), kernelvalues[p])


def test_spike():
    """ Test if a convolution with a delta spike is equal to the kernel
    saved in the lookup table."""

    # create kernel
    D33 = 1.0
    D44 = 0.04
    t = 1
    num_orientations = 5
    k = EnhancementKernel(D33, D44, t, orientations=num_orientations, force_recompute=True)

    # create a delta spike
    numorientations = k.get_orientations().shape[0]
    spike = np.zeros((7, 7, 7, numorientations), dtype=np.float64)
    spike[3, 3, 3, 0] = 1

    # convolve kernel with delta spike
    csd_enh = convolve_sf(spike, k, test_mode=True, normalize=False)

    # check if kernel matches with the convolved delta spike
    totalsum = 0.0
    for i in range(0, numorientations):
        totalsum += np.sum(np.array(k.get_lookup_table())[i, 0, :, :, :] - \
                    np.array(csd_enh)[:, :, :, i])
    npt.assert_equal(totalsum, 0.0)

def test_normalization():
    """ Test the normalization routine applied after a convolution"""
    # create kernel
    D33 = 1.0
    D44 = 0.04
    t = 1
    num_orientations = 5
    k = EnhancementKernel(D33, D44, t, orientations=num_orientations, force_recompute=True)

    # create a constant dataset
    numorientations = k.get_orientations().shape[0]
    spike = np.ones((7, 7, 7, numorientations), dtype=np.float64)

    # convert dataset to SH
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", message=descoteaux07_legacy_msg,
            category=PendingDeprecationWarning)
        spike_sh = sf_to_sh(spike, k.get_sphere(), sh_order_max=8)

        # convolve kernel with delta spike and apply normalization
        csd_enh = convolve(
            spike_sh, k, sh_order_max=8, test_mode=True, normalize=True)

        # convert dataset to DSF
        csd_enh_dsf = sh_to_sf(
            csd_enh, k.get_sphere(), sh_order_max=8, basis_type=None)

    # test if the normalization is performed correctly
    npt.assert_almost_equal(np.amax(csd_enh_dsf), np.amax(spike))

def test_kernel_input():
    """ Test the kernel for inputs of type Sphere, type int and for input None"""

    sph = Sphere(1, 0, 0)
    D33 = 1.0
    D44 = 0.04
    t = 1
    k = EnhancementKernel(D33, D44, t, orientations=sph, force_recompute=True)
    npt.assert_equal(k.get_lookup_table().shape, (1, 1, 7, 7, 7))

    num_orientations = 2
    k = EnhancementKernel(D33, D44, t, orientations=num_orientations, force_recompute=True)
    npt.assert_equal(k.get_lookup_table().shape, (2, 2, 7, 7, 7))

    k = EnhancementKernel(D33, D44, t, orientations=0, force_recompute=True)
    npt.assert_equal(k.get_lookup_table().shape, (0, 0, 7, 7, 7))
