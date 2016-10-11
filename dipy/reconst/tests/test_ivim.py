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
                           assert_raises, assert_array_less, run_module_suite,
                           assert_warns, dec)

from dipy.reconst.ivim import ivim_prediction, IvimModel
from dipy.core.gradients import gradient_table, generate_bvecs
from dipy.sims.voxel import multi_tensor

from distutils.version import LooseVersion
import scipy

SCIPY_VERSION = LooseVersion(scipy.version.short_version)

# Let us generate some data for testing.
bvals = np.array([0., 10., 20., 30., 40., 60., 80., 100.,
                  120., 140., 160., 180., 200., 300., 400.,
                  500., 600., 700., 800., 900., 1000.])
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

ivim_model = IvimModel(gtab)
ivim_model_one_stage = IvimModel(gtab)
ivim_fit_single = ivim_model.fit(data_single)
ivim_fit_multi = ivim_model.fit(data_multi)

ivim_fit_single_one_stage = ivim_model_one_stage.fit(data_single)
ivim_fit_multi_one_stage = ivim_model_one_stage.fit(data_multi)

bvals_no_b0 = np.array([5., 10., 20., 30., 40., 60., 80., 100.,
                        120., 140., 160., 180., 200., 300., 400.,
                        500., 600., 700., 800., 900., 1000.])

bvecs_no_b0 = generate_bvecs(N)
gtab_no_b0 = gradient_table(bvals_no_b0, bvecs.T)

bvals_with_multiple_b0 = np.array([0., 0., 0., 0., 40., 60., 80., 100.,
                                   120., 140., 160., 180., 200., 300., 400.,
                                   500., 600., 700., 800., 900., 1000.])

bvecs_with_multiple_b0 = generate_bvecs(N)
gtab_with_multiple_b0 = gradient_table(bvals_with_multiple_b0,
                                       bvecs_with_multiple_b0.T)

noisy_single = np.array([4243.71728516, 4317.81298828, 4244.35693359,
                         4439.36816406, 4420.06201172, 4152.30078125, 4114.34912109, 4104.59375, 4151.61914062,
                         4003.58374023, 4013.68408203, 3906.39428711,
                         3909.06079102, 3495.27197266, 3402.57006836,
                         3163.10180664, 2896.04003906, 2663.7253418,
                         2614.87695312, 2316.55371094, 2267.7722168])

noisy_multi = np.zeros((2, 2, 1, len(gtab.bvals)))
noisy_multi[0, 1, 0] = noisy_multi[
    1, 0, 0] = noisy_multi[1, 1, 0] = noisy_single
noisy_multi[0, 0, 0] = data_single


def single_exponential(S0, D, bvals):
    return S0 * np.exp(-bvals * D)


def test_single_voxel_fit():
    """
    Test the implementation of the fitting for a single voxel.

    Here, we will use the multi_tensor function to generate a
    bi-exponential signal. The multi_tensor generates a multi
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
    est_signal = ivim_prediction(ivim_fit_single.model_params, gtab)

    assert_array_equal(est_signal.shape, data_single.shape)

    assert_array_almost_equal(ivim_fit_single.model_params, params)
    assert_array_almost_equal(est_signal, data_single)

    # Test predict function for single voxel
    p = ivim_fit_single.predict(gtab)
    assert_array_equal(p.shape, data_single.shape)
    assert_array_almost_equal(p, data_single)


def test_multivoxel():
    """Test fitting with multivoxel data.

    We generate a multivoxel signal to test the fitting for multivoxel data.
    This is to ensure that the fitting routine takes care of signals packed as
    1D, 2D or 3D arrays.
    """
    ivim_fit_multi = ivim_model.fit(data_multi)

    est_signal = ivim_fit_multi.predict(gtab, S0=1.)
    assert_array_equal(est_signal.shape, data_multi.shape)
    assert_array_almost_equal(ivim_fit_multi.model_params, ivim_params)
    assert_array_almost_equal(est_signal, data_multi)


def test_ivim_errors():
    """
    Test if errors raised in the module are working correctly.

    Scipy introduced bounded least squares fitting in the version 0.17
    and is not supported by the older versions. Initializing an IvimModel
    with bounds for older Scipy versions should raise an error.
    """
    # Run the test for Scipy versions less than 0.17
    if SCIPY_VERSION < LooseVersion('0.17'):
        assert_raises(ValueError, IvimModel, gtab,
                      bounds=([0., 0., 0., 0.], [np.inf, 1., 1., 1.]))
    else:
        ivim_model = IvimModel(gtab,
                               bounds=([0., 0., 0., 0.], [np.inf, 1., 1., 1.]))
        ivim_fit = ivim_model.fit(data_multi)
        est_signal = ivim_fit.predict(gtab, S0=1.)
        assert_array_equal(est_signal.shape, data_multi.shape)
        assert_array_almost_equal(ivim_fit.model_params, ivim_params)
        assert_array_almost_equal(est_signal, data_multi)


def test_mask():
    """
    Test whether setting incorrect mask raises and error
    """
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

    ivim_fit = ivim_model.fit(data_single2)

    est_signal = ivim_fit.predict(gtab)
    assert_array_equal(est_signal.shape, data_single2.shape)
    assert_array_almost_equal(est_signal, data_single2)
    assert_array_almost_equal(ivim_fit.model_params, params2)


def test_bounds_x0():
    """
    Test to check if setting bounds for signal where initial value is
    higher than subsequent values works.

    These values are from the IVIM dataset which can be obtained by using
    the `read_ivim` function from dipy.data.fetcher. These are values from
    the voxel [160, 98, 33] which can be obtained by :

    .. code-block:: python

       from dipy.data.fetcher import read_ivim
       img, gtab = read_ivim()
       data = img.get_data()
       signal = data[160, 98, 33, :]

    """
    test_signal = np.array([4574.34814453, 4745.18164062, 4759.51806641,
                            4618.24951172, 4665.63623047, 4568.046875,
                            4525.90478516, 4734.54785156, 4526.41357422,
                            4299.99414062, 4256.61279297, 4254.50292969,
                            4098.74707031, 3776.10375977, 3614.0769043,
                            3440.56445312, 3146.52294922, 3006.94287109,
                            2879.69580078, 2728.44018555, 2600.09472656])
    x0_test = np.array([1., 0.13, 0.001, 0.0001])
    test_signal = ivim_prediction(x0_test, gtab)

    ivim_fit = ivim_model.fit(test_signal)

    est_signal = ivim_fit.predict(gtab)
    assert_array_equal(est_signal.shape, test_signal.shape)


def test_predict():
    """
    Test the model prediction API.
    The predict method is already used in previous tests for estimation of the
    signal. But here, we will test is separately.
    """
    assert_array_almost_equal(ivim_fit_single.predict(gtab),
                              data_single)
    assert_array_almost_equal(ivim_model.predict(ivim_fit_single.model_params,
                                                 gtab),
                              data_single)

    ivim_fit_multi = ivim_model.fit(data_multi)
    assert_array_almost_equal(ivim_fit_multi.predict(gtab),
                              data_multi)


def test_fit_object():
    """
    Test the method of IvimFit class
    """
    assert_raises(IndexError, ivim_fit_single.__getitem__, (-.1, 0, 0))
    # Check if the S0 called is matching
    assert_array_almost_equal(
        ivim_fit_single.__getitem__(0).model_params, 1000.)

    ivim_fit_multi = ivim_model.fit(data_multi)
    # Should raise a TypeError if the arguments are not passed as tuple
    assert_raises(TypeError, ivim_fit_multi.__getitem__, -.1, 0)
    # Should return IndexError if invalid indices are passed
    assert_raises(IndexError, ivim_fit_multi.__getitem__, (100, -0))
    assert_raises(IndexError, ivim_fit_multi.__getitem__, (100, -0, 2))
    assert_raises(IndexError, ivim_fit_multi.__getitem__, (-100, 0))
    assert_raises(IndexError, ivim_fit_multi.__getitem__, [-100, 0])
    assert_raises(IndexError, ivim_fit_multi.__getitem__, (1, 0, 0, 3, 4))
    # Check if the get item returns the S0 value for voxel (1,0,0)
    assert_array_almost_equal(
        ivim_fit_multi.__getitem__((1, 0, 0)).model_params[0],
        data_multi[1, 0, 0][0])


def test_shape():
    """
    Test if `shape` in `IvimFit` class gives the correct output.
    """
    assert_array_equal(ivim_fit_single.shape, ())
    ivim_fit_multi = ivim_model.fit(data_multi)
    assert_array_equal(ivim_fit_multi.shape, (2, 2, 1))


def test_multiple_b0():
    # Generate a signal with multiple b0

    # This gives an isotropic signal.
    signal = multi_tensor(gtab_with_multiple_b0, mevals, snr=None, S0=S0,
                          fractions=[f * 100, 100 * (1 - f)])
    # Single voxel data
    data_single = signal[0]

    ivim_model_multiple_b0 = IvimModel(gtab_with_multiple_b0)

    x0_estimated = ivim_model_multiple_b0.fit(data_single)
    # Test if all signals are positive


def test_no_b0():
    assert_raises(ValueError, IvimModel, gtab_no_b0)


@dec.skipif(SCIPY_VERSION < LooseVersion('0.17'),
            "Gives wrong value for f")
def test_noisy_fit():
    """
    Test fitting for noisy signals. This tests whether the threshold condition
    applies correctly and returns the linear fitting parameters.

    For older scipy versions, the returned value of `f` from a linear fit is around 135
    and D and D_star values are equal. Hence doing a test based on Scipy version.
    """
    model_one_stage = IvimModel(gtab)
    fit_one_stage = model_one_stage.fit(noisy_single)
    assert_array_less(fit_one_stage.model_params, [10000., 0.3, .01, 0.001])


def test_S0():
    """
    Test if the `IvimFit` class returns the correct S0
    """
    assert_array_almost_equal(ivim_fit_single.S0_predicted, S0)
    assert_array_almost_equal(ivim_fit_multi.S0_predicted, ivim_params[..., 0])


def test_perfusion_fraction():
    """
    Test if the `IvimFit` class returns the correct f
    """
    assert_array_almost_equal(ivim_fit_single.perfusion_fraction, f)
    assert_array_almost_equal(
        ivim_fit_multi.perfusion_fraction, ivim_params[..., 1])


def test_D_star():
    """
    Test if the `IvimFit` class returns the correct D_star
    """
    assert_array_almost_equal(ivim_fit_single.D_star, D_star)
    assert_array_almost_equal(ivim_fit_multi.D_star, ivim_params[..., 2])


def test_D():
    """
    Test if the `IvimFit` class returns the correct D
    """
    assert_array_almost_equal(ivim_fit_single.D, D)
    assert_array_almost_equal(ivim_fit_multi.D, ivim_params[..., 3])


def test_estimate_linear_fit():
    """
    Test the linear estimates considering a single exponential fit.
    """
    data_single_exponential_D = single_exponential(S0, D, gtab.bvals)
    assert_array_almost_equal(ivim_model.estimate_linear_fit(
        data_single_exponential_D,
        split_b=500.,
        less_than=False),
        (S0, D))
    data_single_exponential_D_star = single_exponential(S0, D_star, gtab.bvals)
    assert_array_almost_equal(ivim_model.estimate_linear_fit(
        data_single_exponential_D_star,
        split_b=100.,
        less_than=True),
        (S0, D_star))


def test_estimate_f_D_star():
    """
    Test if the `estimate_f_D_star` returns the correct parameters after a
    non-linear fit.
    """
    params_f_D = f + 0.001, D + 0.0001
    assert_array_almost_equal(ivim_model.estimate_f_D_star(params_f_D,
                                                           data_single, S0, D),
                              (f, D_star))


def test_fit_one_stage():
    """
    Test to check the results for the one_stage linear fit.
    """
    model = IvimModel(gtab, two_stage=False)
    fit = model.fit(data_single)
    # assert_array_almost_equal()
    linear_fit_params = [9.88834140e+02, 1.19707191e-01, 7.91176970e-03,
                         9.30095210e-04]

    linear_fit_signal = [988.83414044, 971.77122546, 955.46786293,
                         939.87125905, 924.93258982, 896.85182201,
                         870.90346447, 846.81187693, 824.34108781,
                         803.28900104, 783.48245048, 764.77297789,
                         747.03322866, 669.54798887, 605.03328304,
                         549.00852235, 499.21077611, 454.40299244,
                         413.83192296, 376.98072773, 343.45531017]

    assert_array_almost_equal(fit.model_params, linear_fit_params)
    assert_array_almost_equal(fit.predict(gtab), linear_fit_signal)


def test_leastsq_failing():
    """
    Test for cases where leastsq fitting fails and the results from a linear
    fit is returned.
    """
    fit_single = ivim_model.fit(noisy_single)
    # Test for the S0 and D values
    assert_array_almost_equal(fit_single.S0_predicted, 4356.268901117833)
    assert_array_almost_equal(fit_single.D, 6.936684e-04)


def test_leastsq_error():
    """
    Test error handling of the `_leastsq` method works when unfeasible x0 is
    passed. If an unfeasible x0 value is passed using which leastsq fails, the
    x0 value is returned as it is.
    """
    fit = ivim_model._leastsq(data_single, [-1, -1, -1, -1])
    assert_array_almost_equal(fit, [-1, -1, -1, -1])


if __name__ == '__main__':
    run_module_suite()
