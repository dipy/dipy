"""
Testing the Intravoxel incoherent motion module

The values of the various parameters used in the tests are inspired by
the study of the IVIM model applied to MR images of the brain by
Federau, Christian, et al. [1].

References
----------
.. [1] Federau, Christian, et al. "Quantitative measurement
       of brain perfusion with intravoxel incoherent motion
       MR imaging." Radiology 265.3 (2012): 874-881.
"""
import numpy as np
from numpy.testing import (assert_array_equal, assert_array_almost_equal,
                           assert_raises)

from dipy.reconst.ivim import ivim_function, IvimModel
from dipy.core.gradients import gradient_table, generate_bvecs
from dipy.sims.voxel import multi_tensor

from distutils.version import LooseVersion
import scipy

SCIPY_VERSION = LooseVersion(scipy.version.short_version)

# Let us generate some data for testing.
bvals = np.array([0., 10., 20., 30., 40., 60., 80., 100.,
                  120., 140., 160., 180., 200., 220., 240.,
                  260., 280., 300., 350., 400., 500., 600.,
                  700., 800., 900., 1000.])
N = len(bvals)
bvecs = generate_bvecs(N)
gtab = gradient_table(bvals, bvecs.T)

S0, f, D_star, D = 1000.0, 0.132, 0.00885, 0.000921
# params for a single voxel
params = np.array([S0, f, D_star, D])

mevals = np.array(([D_star, D_star, D_star], [D, D, D]))
# This gives an isotropic signal.
signal = multi_tensor(gtab, mevals, snr=None, S0=S0,
                      fractions=[f * 100, 100 * (1 - f)])
# Single voxel data
data_single = signal[0]

data_multi = np.zeros((2, 2, 1, len(gtab.bvals)))
data_multi[0, 0, 0] = data_multi[0, 1, 0] = data_multi[
    1, 0, 0] = data_multi[1, 1, 0] = data_single

ivim_params = np.zeros((2, 2, 1, 4))
ivim_params[0, 0, 0] = ivim_params[0, 1, 0] = params
ivim_params[1, 0, 0] = ivim_params[1, 1, 0] = params


def test_single_voxel_fit():
    """
    Test the implementation of the fitting for a single voxel.

    Here, we will use the multi_tensor function to generate a
    biexponential signal. The multi_tensor generates a multi
    tensor signal and expects eigenvalues of each tensor in mevals.
    Our basic test requires a scalar signal isotropic signal and
    hence we set the same eigenvalue in all three directions to
    generate the required signal.

    The bvals, f, D_star and D are inspired from the paper by
    Federau, Christian, et al. We use the function "generate_bvecs"
    to simulate bvectors corresponding to the bvalues.

    In the two stage fitting routine, initially we fit the signal
    values for bvals less than the specified split_b using the
    TensorModel and get an intial guess for f and D. Then, using
    these parameters we fit the entire data for all bvalues.
    """
    ivim_model = IvimModel(gtab)
    ivim_fit = ivim_model.fit(data_single)

    est_signal = ivim_function(ivim_fit.model_params, bvals)

    assert_array_equal(est_signal.shape, data_single.shape)
    assert_array_almost_equal(est_signal, data_single)
    assert_array_almost_equal(ivim_fit.model_params, params)

    # Test predict function for single voxel
    p = ivim_fit.predict(gtab)
    assert_array_equal(p.shape, data_single.shape)
    assert_array_almost_equal(p, data_single)


def test_multivoxel():
    """Test fitting with multivoxel data.

    We generate a multivoxel signal to test the fitting for multivoxel data.
    This is to ensure that the fitting routine takes care of signals packed as
    1D, 2D or 3D arrays.
    """
    ivim_model = IvimModel(gtab)
    ivim_fit = ivim_model.fit(data_multi)

    est_signal = ivim_fit.predict(gtab, S0=1.)
    assert_array_equal(est_signal.shape, data_multi.shape)
    assert_array_almost_equal(est_signal, data_multi)
    assert_array_almost_equal(ivim_fit.model_params, ivim_params)


def test_ivim_errors():
    """
    Test if errors raised in the module are working correctly.

    Scipy introduced bounded least squares fitting in the version 0.17
    and is not supported by the older versions. Initializing an IvimModel
    with bounds for older Scipy versions should raise an error.
    """
    # Run the test for Scipy versions less than 0.17
    if SCIPY_VERSION < '0.17':
        assert_raises(ValueError, IvimModel, gtab,
                      bounds=([0., 0., 0., 0.], [np.inf, 1., 1., 1.]))
    else:
        ivim_model = IvimModel(gtab,
                               bounds=([0., 0., 0., 0.], [np.inf, 1., 1., 1.]))
        ivim_fit = ivim_model.fit(data_multi)
        est_signal = ivim_fit.predict(gtab, S0=1.)
        assert_array_equal(est_signal.shape, data_multi.shape)
        assert_array_almost_equal(est_signal, data_multi)
        assert_array_almost_equal(ivim_fit.model_params, ivim_params)


def test_mask():
    """
    Test whether setting incorrect mask raises and error
    """
    ivim_model = IvimModel(gtab)
    mask_correct = data_multi[..., 0] > 0.2
    mask_not_correct = np.array([[False, True, False], [True, False]])

    ivim_fit = ivim_model.fit(data_multi, mask_correct)
    est_signal = ivim_fit.predict(gtab, S0=1.)
    assert_array_equal(est_signal.shape, data_multi.shape)
    assert_array_almost_equal(est_signal, data_multi)
    assert_array_almost_equal(ivim_fit.model_params, ivim_params)
    assert_raises(ValueError, ivim_model.fit, data_multi,
                  mask=mask_not_correct)


def test_with_higher_S0():
    """
    Test whether fitting works for S0 > 1.
    """
    # params for a single voxel
    S0_2 = 1000.
    params2 = np.array([S0_2, f, D_star, D])
    mevals2 = np.array(([D_star, D_star, D_star], [D, D, D]))
    # This gives an isotropic signal.
    signal2 = multi_tensor(gtab, mevals2, snr=None, S0=S0_2,
                           fractions=[f * 100, 100 * (1 - f)])
    # Single voxel data
    data_single2 = signal2[0]

    ivim_model = IvimModel(gtab)
    ivim_fit = ivim_model.fit(data_single2)

    est_signal = ivim_fit.predict(gtab)
    assert_array_equal(est_signal.shape, data_single2.shape)
    assert_array_almost_equal(est_signal, data_single2)
    assert_array_almost_equal(ivim_fit.model_params, params2)


def test_bounds_x0():
    """
    Test to check if setting bounds for signal where initial value is
    higer than subsequent values works. These values are from the
    IVIM dataset which can be obtained by using the `read_ivim` function
    from dipy.data.fetcher. These are values from the voxel [160, 98, 33]
    which can be obtained by :

    .. code-block:: python

       from dipy.data.fetcher import read_ivim
       img, gtab = read_ivim()
       data = img.get_data()
       signal = data[160, 98, 33, :]

    """
    test_signal = np.array([4574.34814453, 4745.18164062,  4759.51806641,
                            4618.24951172, 4665.63623047, 4568.046875,
                            4525.90478516, 4734.54785156, 4526.41357422,
                            4299.99414062, 4256.61279297, 4254.50292969,
                            4098.74707031, 3776.10375977,  3614.0769043,
                            3440.56445312, 3146.52294922, 3006.94287109,
                            2879.69580078, 2728.44018555, 2600.09472656,
                            2570., 2440., 2400., 2380., 2370.])

    ivim_model = IvimModel(gtab)
    ivim_fit = ivim_model.fit(test_signal)

    est_signal = ivim_fit.predict(gtab)
    assert_array_equal(est_signal.shape, test_signal.shape)
