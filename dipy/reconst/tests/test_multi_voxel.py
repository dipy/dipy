from functools import reduce
import warnings

import numpy as np
import numpy.testing as npt
import pytest

from dipy.core.gradients import gradient_table
from dipy.core.sphere import unit_icosahedron
from dipy.data import get_fnames
from dipy.io.gradients import read_bvals_bvecs
from dipy.reconst.csdeconv import ConstrainedSphericalDeconvModel
from dipy.reconst.dki import DiffusionKurtosisModel
from dipy.reconst.multi_voxel import CallableArray, _squash, multi_voxel_fit, ORCHESTRATION_KWARGS
from dipy.sims.voxel import multi_tensor_dki, single_tensor
from dipy.testing.decorators import set_random_number_generator, warning_for_keywords
from dipy.utils.optpkg import optional_package

joblib, has_joblib, _ = optional_package("joblib")
dask, has_dask, _ = optional_package("dask")
ray, has_ray, _ = optional_package("ray")

PARALLEL_ENGINES = ["serial"]
if has_joblib:
    PARALLEL_ENGINES.append("joblib")
if has_dask:
    PARALLEL_ENGINES.append("dask")
if has_ray:
    PARALLEL_ENGINES.append("ray")

NONSERIAL_ENGINES = [e for e in PARALLEL_ENGINES if e != "serial"]
INPROCESS_ENGINES = [e for e in PARALLEL_ENGINES if e != "ray"]


def test_squash():
    A = np.ones((3, 3), dtype=float)
    B = np.asarray(A, object)
    npt.assert_array_equal(A, _squash(B))
    npt.assert_equal(_squash(B).dtype, A.dtype)

    B[2, 2] = None
    A[2, 2] = 0
    npt.assert_array_equal(A, _squash(B))
    npt.assert_equal(_squash(B).dtype, A.dtype)

    for ijk in np.ndindex(*B.shape):
        B[ijk] = np.ones((2,))
    A = np.ones((3, 3, 2))
    npt.assert_array_equal(A, _squash(B))
    npt.assert_equal(_squash(B).dtype, A.dtype)

    B[2, 2] = None
    A[2, 2] = 0
    npt.assert_array_equal(A, _squash(B))
    npt.assert_equal(_squash(B).dtype, A.dtype)

    # sub-arrays have different shapes ( (3,) and (2,) )
    B[0, 0] = np.ones((3,))
    npt.assert_(_squash(B) is B)

    # Check dtypes for arrays and scalars
    arr_arr = np.zeros((2,), dtype=object)
    scalar_arr = np.zeros((2,), dtype=object)
    numeric_types = [
        getattr(np, dtype)
        for dtype in (
            "int8",
            "byte",
            "int16",
            "short",
            "int32",
            "intc",
            "int_",
            "int64",
            "longlong",
            "uint8",
            "ubyte",
            "uint16",
            "ushort",
            "uint32",
            "uintc",
            "uint",
            "uint64",
            "ulonglong",
            "float16",
            "half",
            "float32",
            "single",
            "float64",
            "double",
            "float96",
            "float128",
            "longdouble",
            "complex64",
            "csingle",
            "complex128",
            "cdouble",
            "complex192",
            "complex256",
            "clongdouble",
        )
        if hasattr(np, dtype)
    ] + [bool]
    for dt0 in numeric_types:
        arr_arr[0] = np.zeros((3,), dtype=dt0)
        scalar_arr[0] = dt0(0)
        for dt1 in numeric_types:
            arr_arr[1] = np.zeros((3,), dtype=dt1)
            npt.assert_equal(_squash(arr_arr).dtype, reduce(np.add, arr_arr).dtype)
            scalar_arr[1] = dt0(1)
            npt.assert_equal(
                _squash(scalar_arr).dtype, reduce(np.add, scalar_arr).dtype
            )

    # Check masks and Nones
    arr = np.ones((3, 4), dtype=float)
    obj_arr = arr.astype(object)
    arr[1, 1] = 99
    obj_arr[1, 1] = None
    npt.assert_array_equal(_squash(obj_arr, mask=None, fill=99), arr)
    msk = arr == 1
    npt.assert_array_equal(_squash(obj_arr, mask=msk, fill=99), arr)
    msk[1, 1] = 1  # unmask None - object array back
    npt.assert_array_equal(_squash(obj_arr, mask=msk, fill=99), obj_arr)
    msk[1, 1] = 0  # remask, back to fill again
    npt.assert_array_equal(_squash(obj_arr, mask=msk, fill=99), arr)
    obj_arr[2, 3] = None  # add another unmasked None, object again
    npt.assert_array_equal(_squash(obj_arr, mask=msk, fill=99), obj_arr)

    # Check array of arrays
    obj_arrs = np.zeros((3,), dtype=object)
    for i in range(3):
        obj_arrs[i] = np.ones((4, 5))
    arr_arrs = np.ones((3, 4, 5))
    # No Nones
    npt.assert_array_equal(_squash(obj_arrs, mask=None, fill=99), arr_arrs)
    # None, implicit masking
    obj_masked = obj_arrs.copy()
    obj_masked[1] = None
    arr_masked = arr_arrs.copy()
    arr_masked[1] = 99
    npt.assert_array_equal(_squash(obj_masked, mask=None, fill=99), arr_masked)
    msk = np.array([1, 0, 1], dtype=bool)  # explicit mask
    npt.assert_array_equal(_squash(obj_masked, mask=msk, fill=99), arr_masked)


def test_CallableArray():
    callarray = CallableArray((2, 3), dtype=object)

    # Test without Nones
    callarray[:] = np.arange
    expected = np.empty([2, 3, 4])
    expected[:] = range(4)
    npt.assert_array_equal(callarray(4), expected)

    # Test with Nones
    callarray[0, 0] = None
    expected[0, 0] = 0
    npt.assert_array_equal(callarray(4), expected)


@set_random_number_generator()
def test_multi_voxel_fit(rng):
    class SillyModel:
        @warning_for_keywords()
        @multi_voxel_fit
        def fit(
            self, data, *, mask=None, another_kwarg=None, kwarg_untouched=True, **kwargs
        ):
            # We want to make sure that all kwargs are passed through to the
            # the fitting procedure
            assert another_kwarg is not None
            # Make sure that an argument that is not passed is still
            # usable in the fitting procedure:
            assert kwarg_untouched
            # ``SillyModel.fit`` declares no orchestration flags, so the decorator
            # must strip every one before dispatch.
            leaked = [k for k in ORCHESTRATION_KWARGS if k in kwargs]
            assert not leaked, (
                f"orchestration kwargs leaked into per-voxel fit: {leaked}"
            )
            return SillyFit(model, data)

        def predict(self, S0):
            return np.ones(10) * S0

    class SillyFit:
        def __init__(self, model, data):
            self.model = model
            self.data = data

        model_attr = 2.0

        def odf(self, sphere):
            return np.ones(len(sphere.phi))

        @property
        def directions(self):
            n = rng.integers(0, 10)
            return np.zeros((n, 3))

        def predict(self, S0):
            return np.ones(self.data.shape) * S0

    # Test the single voxel case
    model = SillyModel()
    single_voxel = np.zeros(64)
    fit = model.fit(single_voxel, another_kwarg="foo")
    npt.assert_equal(type(fit), SillyFit)

    # Test without a mask
    many_voxels = np.zeros((2, 3, 4, 64))
    for verbose in [True, False]:
        fit = model.fit(many_voxels, verbose=verbose, another_kwarg="foo")
        expected = np.empty((2, 3, 4))
        expected[:] = 2.0
        npt.assert_array_equal(fit.model_attr, expected)
        expected = np.ones((2, 3, 4, 12))
        npt.assert_array_equal(fit.odf(unit_icosahedron), expected)
        npt.assert_equal(fit.directions.shape, (2, 3, 4))
        S0 = 100.0
        npt.assert_equal(fit.predict(S0=S0), np.ones(many_voxels.shape) * S0)

    # Test with parallelization (using the "serial" dummy engine)
    fit = model.fit(many_voxels, another_kwarg="foo", engine="serial")

    for verbose in [True, False]:
        # Test with single value kwarg, or sequence of kwarg values
        for another_kwarg in ["foo", len(many_voxels) * ["foo"]]:
            # If parallelization engines are installed use them to test:
            if has_joblib:
                fit = model.fit(
                    many_voxels,
                    verbose=verbose,
                    another_kwarg=another_kwarg,
                    engine="joblib",
                )
                npt.assert_equal(fit.predict(S0=S0), np.ones(many_voxels.shape) * S0)

            if has_dask:
                fit = model.fit(
                    many_voxels,
                    verbose=verbose,
                    another_kwarg=another_kwarg,
                    engine="dask",
                )
                npt.assert_equal(fit.predict(S0=S0), np.ones(many_voxels.shape) * S0)

            if has_ray:
                fit = model.fit(
                    many_voxels,
                    verbose=verbose,
                    another_kwarg=another_kwarg,
                    engine="ray",
                )
                npt.assert_equal(fit.predict(S0=S0), np.ones(many_voxels.shape) * S0)

    # Test with a mask
    mask = np.zeros((3, 3, 3)).astype("bool")
    mask[0, 0] = 1
    mask[1, 1] = 1
    mask[2, 2] = 1
    data = np.zeros((3, 3, 3, 64))
    fit = model.fit(data, mask=mask, another_kwarg="foo")
    expected = np.zeros((3, 3, 3))
    expected[0, 0] = 2
    expected[1, 1] = 2
    expected[2, 2] = 2
    npt.assert_array_equal(fit.model_attr, expected)
    odf = fit.odf(unit_icosahedron)
    npt.assert_equal(odf.shape, (3, 3, 3, 12))
    npt.assert_array_equal(odf[~mask], 0)
    npt.assert_array_equal(odf[mask], 1)
    predicted = np.zeros(data.shape)
    predicted[mask] = S0
    npt.assert_equal(fit.predict(S0=S0), predicted)

    # Test fit.shape
    npt.assert_equal(fit.shape, (3, 3, 3))

    # Test indexing into a fit
    npt.assert_equal(type(fit[0, 0, 0]), SillyFit)
    npt.assert_equal(fit[:2, :2, :2].shape, (2, 2, 2))


# ---------------------------------------------------------------------------
# Regression tests for dipy#4053
# ---------------------------------------------------------------------------

class _RecordingFit:
    """Minimal fit object that knows how to predict a constant signal."""

    def __init__(self, model, data):
        self.model = model
        self.data = data

    def predict(self, S0):
        return np.ones(self.data.shape) * S0


def _recording_model(received):
    """A model whose per-voxel fit records the kwargs it actually receives.

    ``fit_method`` is a legitimate, explicitly declared model kwarg used to
    check that real model kwargs are *not* stripped.  Everything else lands in
    ``**kwargs``, where any leaked orchestration kwarg would show up.
    """

    class RecordingModel:
        @multi_voxel_fit
        def fit(self, data, *, fit_method="LS", **kwargs):
            received.append((fit_method, dict(kwargs)))
            return _RecordingFit(self, data)

    return RecordingModel()


def _verbose_declaring_model(seen_verbose):
    """A model whose per-voxel fit *declares* ``verbose`` (as MCSD does)."""

    class VerboseModel:
        @multi_voxel_fit
        def fit(self, data, *, verbose=False, **kwargs):
            seen_verbose.append(verbose)
            # Keys this fit does NOT declare are still stripped:
            assert "engine" not in kwargs
            assert "n_jobs" not in kwargs
            return _RecordingFit(self, data)

    return VerboseModel()


@set_random_number_generator()
def test_multi_voxel_fit_no_orchestration_leak(rng):
    """Orchestration kwargs never reach a per-voxel fit that doesn't declare them.

    Covers the serial (explicit ``engine="serial"``) and the in-process parallel
    paths (the recorder lives in the calling process). A legitimate model kwarg
    (``fit_method``) and a per-voxel ``weights`` array must still pass through.
    """
    n_vox, n_grad = 6, 8
    data = rng.random((n_vox, n_grad))

    for engine in INPROCESS_ENGINES:
        received = []
        model = _recording_model(received)
        weights = rng.random((n_vox, n_grad))
        with warnings.catch_warnings():
            # warn-on-drop is asserted separately; silence it here
            warnings.simplefilter("ignore")
            model.fit(
                data,
                engine=engine,
                n_jobs=1,
                vox_per_chunk=2,
                verbose=False,
                fit_method="WLS",
                weights=weights,
            )

        assert received, f"per-voxel fit was never called (engine={engine})"
        for fit_method, received_kwargs in received:
            # Legitimate, explicitly declared model kwarg survived:
            npt.assert_equal(fit_method, "WLS")
            # No orchestration kwarg leaked into the per-voxel fit:
            for key in ORCHESTRATION_KWARGS:
                assert key not in received_kwargs, (
                    f"{key!r} leaked into the per-voxel fit (engine={engine})"
                )
            # A per-voxel model kwarg (weights) was forwarded and sliced:
            assert "weights" in received_kwargs
            assert isinstance(received_kwargs["weights"], np.ndarray)


@set_random_number_generator()
def test_multi_voxel_fit_forwards_declared_kwarg(rng):
    """A reserved key the fit *declares* (e.g. ``verbose``, as in MCSD) is
    forwarded — and is not warned about — while undeclared keys are stripped.
    """
    n_vox, n_grad = 6, 8
    data = rng.random((n_vox, n_grad))

    for engine in INPROCESS_ENGINES:
        seen_verbose = []
        model = _verbose_declaring_model(seen_verbose)

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            model.fit(data, engine=engine, n_jobs=1, verbose=True)

        # The declared ``verbose`` reached every per-voxel fit:
        assert seen_verbose and all(v is True for v in seen_verbose)
        # ... and was not reported as a dropped kwarg:
        assert not any("verbose" in str(w.message) for w in caught), (
            "verbose was dropped/warned despite being declared by the fit"
        )


@set_random_number_generator()
def test_multi_voxel_fit_warns_on_dropped_kwargs(rng):
    """Dropping orchestration kwargs is never silent: a warning names them."""
    data = rng.random((6, 8))
    received = []
    model = _recording_model(received)
    with pytest.warns(UserWarning, match="engine"):
        model.fit(data, engine="serial", n_jobs=1)


def test_multi_voxel_fit_orchestration_reaches_paramap(monkeypatch):
    """Orchestration kwargs are forwarded to ``paramap`` while the per-chunk
    kwargs are stripped. ``paramap`` is replaced by an in-process spy so the
    routing is checked deterministically without spawning workers.
    """
    import dipy.reconst.multi_voxel as mv

    captured = {}

    def spy(func, in_list, *, func_args=None, func_kwargs=None, **kwargs):
        captured["parallel_kwargs"] = kwargs
        captured["per_chunk_kwargs"] = func_kwargs
        func_args = func_args or []
        if isinstance(func_kwargs, (list, tuple)):
            return [func(x, *func_args, **fk) for x, fk in zip(in_list, func_kwargs)]
        return [func(x, *func_args, **(func_kwargs or {})) for x in in_list]

    monkeypatch.setattr(mv, "paramap", spy)

    received = []
    model = _recording_model(received)
    data = np.zeros((6, 8))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit(data, engine="joblib", n_jobs=2, vox_per_chunk=3, verbose=False)

    # The orchestration kwargs were handed to paramap / the engine:
    for key in ("engine", "n_jobs", "vox_per_chunk", "verbose", "inflight_cap"):
        assert key in captured["parallel_kwargs"], f"{key} did not reach paramap"
    # ... but none of them leaked into the per-chunk (per-voxel) kwargs:
    for chunk_kwargs in captured["per_chunk_kwargs"]:
        for key in ORCHESTRATION_KWARGS:
            assert key not in chunk_kwargs


# --- Real-model parity fixtures (built once) --------------------------------
_GTAB_2S = _DKI_DATA = _DKI_REF = None
_CSD_MODEL = _CSD_DATA = _CSD_REF = None


def _unwrap(fit):
    """Some decorated fits return ``(MultiVoxelFit, extra)``; take the fit."""
    return fit[0] if isinstance(fit, tuple) else fit


def setup_module():
    """Build small, per-voxel-distinct datasets and serial-reference fits."""
    global _GTAB_2S, _DKI_DATA, _DKI_REF
    global _CSD_MODEL, _CSD_DATA, _CSD_REF

    _, fbvals, fbvecs = get_fnames(name="small_64D")
    bvals, bvecs = read_bvals_bvecs(fbvals, fbvecs)

    # Distinct per-voxel scaling so a chunk-ordering bug would surface.
    scale = (1.0 + 0.02 * np.arange(6))[:, None]

    # Two-shell table for DKI (multi-shell model).
    bvals_2s = np.concatenate((bvals, bvals * 2))
    bvecs_2s = np.concatenate((bvecs, bvecs))
    _GTAB_2S = gradient_table(bvals_2s, bvecs=bvecs_2s)
    mevals = np.array([[0.00099, 0, 0], [0.00226, 0.00087, 0.00087]])
    signal, _, _ = multi_tensor_dki(
        _GTAB_2S, mevals, S0=100, fractions=[50, 50], snr=None
    )
    _DKI_DATA = (signal[None, :] * scale).astype(float)
    # Reference: default serial path (no engine kwarg -> no leak).
    _DKI_REF = _unwrap(DiffusionKurtosisModel(_GTAB_2S).multi_fit(_DKI_DATA))

    # Single-shell table for CSD.
    gtab_1s = gradient_table(bvals, bvecs=bvecs)
    response = (np.array([0.0015, 0.0003, 0.0003]), 100.0)
    _CSD_MODEL = ConstrainedSphericalDeconvModel(gtab_1s, response)
    csd_signal = single_tensor(
        gtab_1s,
        100.0,
        evals=np.array([0.0015, 0.0003, 0.0003]),
        evecs=np.eye(3),
        snr=None,
    )
    _CSD_DATA = (csd_signal[None, :] * scale).astype(float)
    _CSD_REF = _unwrap(_CSD_MODEL.fit(_CSD_DATA))


@pytest.mark.skipif(not NONSERIAL_ENGINES, reason="no parallel engine installed")
@pytest.mark.parametrize("engine", NONSERIAL_ENGINES)
@pytest.mark.parametrize("vox_per_chunk", [1, 4, 100])  # ==1, <n_vox, >n_vox
@pytest.mark.parametrize("n_jobs", [1, 2])
def test_dki_multi_fit_parallel_matches_serial(engine, vox_per_chunk, n_jobs):
    """DKI (single-voxel fit rejects ``engine``) parallel == serial, exactly."""
    model = DiffusionKurtosisModel(_GTAB_2S)
    got = _unwrap(
        model.multi_fit(
            _DKI_DATA, engine=engine, n_jobs=n_jobs, vox_per_chunk=vox_per_chunk
        )
    )
    npt.assert_allclose(
        np.asarray(got.model_params),
        np.asarray(_DKI_REF.model_params),
        rtol=1e-6,
        atol=1e-8,
    )
    npt.assert_allclose(
        np.asarray(got.kt), np.asarray(_DKI_REF.kt), rtol=1e-6, atol=1e-8
    )


@pytest.mark.skipif(not NONSERIAL_ENGINES, reason="no parallel engine installed")
@pytest.mark.parametrize("engine", NONSERIAL_ENGINES)
@pytest.mark.parametrize("vox_per_chunk", [1, 4, 100])
def test_csd_fit_parallel_matches_serial(engine, vox_per_chunk):
    """CSD parallel == serial (guards against regressions on the absorb path)."""
    got = _unwrap(
        _CSD_MODEL.fit(_CSD_DATA, engine=engine, n_jobs=1, vox_per_chunk=vox_per_chunk)
    )
    npt.assert_allclose(
        np.asarray(got.shm_coeff),
        np.asarray(_CSD_REF.shm_coeff),
        rtol=1e-6,
        atol=1e-8,
    )
