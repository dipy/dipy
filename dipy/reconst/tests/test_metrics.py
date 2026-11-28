import numpy as np
from numpy.exceptions import AxisError
from numpy.testing import assert_allclose, assert_array_equal
import pytest

from dipy.reconst.metrics import nrmse, rmse


def test_rmse_one_dimensional():
    actual = rmse([1, -3, 5, 1], [1, -1, 1, 1])
    assert np.isscalar(actual)
    assert_allclose(actual, np.sqrt(5))


def test_nrmse_one_dimensional():
    observed = np.array([1, -3])
    predicted = np.array([2, -2])
    actual = nrmse(observed, predicted)
    assert np.isscalar(actual)
    assert_allclose(actual, 1 / np.sqrt(5))
    assert_allclose(nrmse(observed, predicted, normalization="mean_absolute"), 0.5)


@pytest.mark.parametrize("axis", [-1, 0, np.int64(1), -2])
def test_multidimensional_axes(axis):
    observed = np.arange(1, 25).reshape(2, 3, 4)
    predicted = np.linspace(1, 20, 24).reshape(observed.shape)
    expected_rmse = np.linalg.norm(observed - predicted, axis=axis) / np.sqrt(
        observed.shape[axis]
    )
    expected_rms = np.linalg.norm(observed, axis=axis) / np.sqrt(observed.shape[axis])
    expected_mean_absolute = np.mean(np.abs(observed), axis=axis)

    assert_allclose(rmse(observed, predicted, axis=axis), expected_rmse)
    assert_allclose(nrmse(observed, predicted, axis=axis), expected_rmse / expected_rms)
    assert_allclose(
        nrmse(observed, predicted, axis=axis, normalization="mean_absolute"),
        expected_rmse / expected_mean_absolute,
    )


@pytest.mark.parametrize("dtype", [np.int16, np.uint16, np.int64, np.uint64])
def test_integer_promotion(dtype):
    observed = np.array([1, 300], dtype=dtype)
    predicted = np.array([301, 0], dtype=dtype)

    assert_allclose(rmse(observed, predicted), 300.0)
    assert_allclose(nrmse(observed, predicted), 300.0 / np.sqrt(90001 / 2))


@pytest.mark.parametrize("dtype", [np.float16, np.float32, np.float64, np.longdouble])
def test_floating_precision_and_dtype(dtype):
    observed = np.array([0.01, -1.5, 600], dtype=dtype)
    predicted = np.array([0, 0, -600], dtype=np.float32)
    reference_dtype = np.result_type(dtype, np.float64)
    reference = observed.astype(reference_dtype)
    expected_error = np.sqrt(np.mean((reference - predicted) ** 2))
    expected_scale = np.sqrt(np.mean(reference**2))

    error = rmse(observed, predicted)
    normalized_error = nrmse(observed, predicted)
    assert error.dtype == normalized_error.dtype == reference_dtype
    assert_allclose(error, expected_error, rtol=1e-14)
    assert_allclose(normalized_error, expected_error / expected_scale, rtol=1e-14)


@pytest.mark.parametrize("magnitude", [1e-200, 1e200])
def test_extreme_signal_magnitudes(magnitude):
    observed = np.array([magnitude, -magnitude])
    predicted = np.zeros_like(observed)
    with np.errstate(over="raise", invalid="raise", divide="raise"):
        assert_allclose(rmse(observed, predicted), magnitude, atol=0)
        assert_allclose(nrmse(observed, predicted), 1.0)
        assert_allclose(nrmse(observed, predicted, normalization="mean_absolute"), 1.0)


@pytest.mark.parametrize("metric", [rmse, nrmse])
def test_shape_validation(metric):
    with pytest.raises(ValueError, match="identical shapes"):
        metric(np.zeros((2, 3)), np.zeros(3))


@pytest.mark.parametrize("metric", [rmse, nrmse])
@pytest.mark.parametrize("axis", [2, -3])
def test_out_of_bounds_axis(metric, axis):
    with pytest.raises(AxisError, match="axis"):
        metric(np.ones((2, 3)), np.ones((2, 3)), axis=axis)


@pytest.mark.parametrize("metric", [rmse, nrmse])
@pytest.mark.parametrize("axis", [None, (0, 1), 1.5, True, np.bool_(False)])
def test_non_integer_axis(metric, axis):
    with pytest.raises(TypeError):
        metric(np.ones((2, 3)), np.ones((2, 3)), axis=axis)


@pytest.mark.parametrize("metric", [rmse, nrmse])
@pytest.mark.parametrize("shape", [(0,), (2, 0), (0, 2)])
def test_empty_signal(metric, shape):
    with pytest.raises(ValueError, match="nonempty"):
        metric(np.empty(shape), np.empty(shape))


@pytest.mark.parametrize("normalization", ["range", "RMS", None, ["rms"]])
def test_invalid_normalization(normalization):
    with pytest.raises(ValueError, match="normalization must be"):
        nrmse([1, 2], [2, 1], normalization=normalization)


@pytest.mark.parametrize("normalization", ["rms", "mean_absolute"])
def test_zero_normalization(normalization):
    observed = np.array([[0, 0], [0, 0], [2, 2]])
    predicted = np.array([[0, 0], [1, 1], [1, 1]])
    with np.errstate(divide="raise", invalid="raise"):
        actual = nrmse(observed, predicted, normalization=normalization)
        scalar = nrmse(observed[0], predicted[0], normalization=normalization)

    assert_array_equal(actual, [np.nan, np.nan, 0.5])
    assert np.isscalar(scalar) and np.isnan(scalar)


@pytest.mark.parametrize("metric", [rmse, nrmse])
@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf])
def test_nonfinite_signals(metric, value):
    valid = np.array([1.0, 2.0])
    invalid = np.array([1.0, value])
    with pytest.raises(ValueError, match="finite values"):
        metric(invalid, valid)
    with pytest.raises(ValueError, match="finite values"):
        metric(valid, invalid)


@pytest.mark.parametrize("metric", [rmse, nrmse])
@pytest.mark.parametrize("dtype", [complex, str, object, bool, "timedelta64[s]"])
def test_invalid_signals(metric, dtype):
    valid = np.array([1, 2])
    invalid = np.array([1, 2], dtype=dtype)
    with pytest.raises(TypeError, match="real numbers"):
        metric(invalid, valid)
    with pytest.raises(TypeError, match="real numbers"):
        metric(valid, invalid)


@pytest.mark.parametrize("model_name", ["DTI", "DKI"])
def test_reconstruction_signals(model_name):
    from dipy.core.gradients import gradient_table
    from dipy.data import get_sphere
    from dipy.reconst.dki import DiffusionKurtosisModel
    from dipy.reconst.dti import TensorModel
    from dipy.sims.voxel import multi_tensor_dki, single_tensor

    directions = get_sphere(name="symmetric362").vertices
    shells = [1000.0] if model_name == "DTI" else [1000.0, 2000.0]
    bvals = np.r_[0, np.repeat(shells, len(directions))]
    bvecs = np.vstack((np.zeros((1, 3)), np.tile(directions, (len(shells), 1))))
    gtab = gradient_table(bvals, bvecs=bvecs)
    s0 = 100.0
    evals = [0.0015, 0.0004, 0.0003]
    if model_name == "DTI":
        model = TensorModel(gtab, fit_method="LS")
        signal = single_tensor(gtab, S0=s0, evals=evals, snr=None)
    else:
        model = DiffusionKurtosisModel(gtab, fit_method="OLS")
        signal, _, _ = multi_tensor_dki(
            gtab,
            [evals, evals],
            S0=s0,
            angles=[(0, 0), (90, 0)],
            fractions=[50, 50],
            snr=None,
        )

    # Perturb one measurement so the test compares meaningful nonzero residuals.
    observed = np.stack([signal, signal])
    observed[:, -1] += [2.0, -4.0]
    predicted = model.fit(observed).predict(gtab, S0=s0)
    assert predicted.shape == observed.shape

    expected_rmse = np.linalg.norm(observed - predicted, axis=-1) / np.sqrt(len(bvals))
    expected_rms = np.linalg.norm(observed, axis=-1) / np.sqrt(len(bvals))
    assert np.all(expected_rmse > 0)
    assert_allclose(rmse(observed, predicted), expected_rmse)
    assert_allclose(nrmse(observed, predicted), expected_rmse / expected_rms)
    assert_allclose(
        nrmse(observed, predicted, normalization="mean_absolute"),
        expected_rmse / np.mean(np.abs(observed), axis=-1),
    )
