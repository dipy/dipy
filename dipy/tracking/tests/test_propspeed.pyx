import warnings

import numpy as np
import numpy.testing as npt

from dipy.core.sphere import unit_octahedron
from dipy.data import default_sphere
from dipy.direction.pmf import SHCoeffPmfGen, SimplePmfGen
from dipy.reconst.shm import (
    SphHarmFit,
    SphHarmModel,
    descoteaux07_legacy_msg,
)
from dipy.tracking.propspeed import ndarray_offset, eudx_both_directions
from dipy.tracking.propspeed cimport (
    deterministic_propagator,
    probabilistic_propagator,
    parallel_transport_propagator,
)

from dipy.tracking.tracker_parameters import generate_tracking_parameters, TrackerStatus
from dipy.tracking.tests.test_tractogen import get_fast_tracking_performances
from dipy.utils.fast_numpy cimport RNGState, seed_rng


def test_tracker_deterministic():
    # Test the probabilistic tracker function
    cdef double[:] stream_data = np.zeros(3, dtype=float)
    cdef double[:] point
    cdef double[:] direction
    cdef RNGState rng

    seed_rng(&rng, 12345)
    class SillyModel(SphHarmModel):
        sh_order_max = 4

        def fit(self, data, mask=None):
            coeff = np.zeros(data.shape[:-1] + (15,))
            return SphHarmFit(self, coeff, mask=None)

    model = SillyModel(gtab=None)
    data = np.zeros((3, 3, 3, 7))
    sphere = unit_octahedron

    params = generate_tracking_parameters("det",
                                          max_len=500,
                                          step_size=0.2,
                                          voxel_size=np.ones(3),
                                          max_angle=20)

    # Test if the tracking works on different dtype of the same data.
    for dtype in [np.float32, np.float64]:
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message=descoteaux07_legacy_msg,
                category=PendingDeprecationWarning,
            )
            fit = model.fit(data.astype(dtype))
            sh_pmf_gen = SHCoeffPmfGen(fit.shm_coeff, sphere, 'descoteaux07')
            sf_pmf_gen = SimplePmfGen(fit.odf(sphere), sphere)

        point = np.zeros(3)
        direction = unit_octahedron.vertices[0].copy()

        # Test using SH pmf
        status = deterministic_propagator(&point[0],
                                          &direction[0],
                                          params,
                                          &stream_data[0],
                                          sh_pmf_gen,
                                          &rng)
        npt.assert_equal(status, TrackerStatus.FAIL)

        # Test using SF pmf
        status = deterministic_propagator(&point[0],
                                          &direction[0],
                                          params,
                                          &stream_data[0],
                                          sf_pmf_gen,
                                          &rng)
        npt.assert_equal(status, TrackerStatus.FAIL)


def test_deterministic_performances():
    # Test deterministic tracker on the DiSCo dataset
    params = generate_tracking_parameters("det",
                                          max_len=500,
                                          step_size=0.2,
                                          voxel_size=np.ones(3),
                                          max_angle=20)
    r = get_fast_tracking_performances(params, nbr_seeds=5000)
    npt.assert_(r > 0.85, msg="Deterministic tracker has a low performance "
                              "score: " + str(r))


def test_tracker_probabilistic():
    # Test the probabilistic tracker function
    cdef double[:] stream_data = np.zeros(3, dtype=float)
    cdef double[:] point
    cdef double[:] direction
    cdef RNGState rng

    seed_rng(&rng, 12345)

    class SillyModel(SphHarmModel):
        sh_order_max = 4

        def fit(self, data, mask=None):
            coeff = np.zeros(data.shape[:-1] + (15,))
            return SphHarmFit(self, coeff, mask=None)

    model = SillyModel(gtab=None)
    data = np.zeros((3, 3, 3, 7))
    sphere = unit_octahedron

    params = generate_tracking_parameters("prob",
                                          max_len=500,
                                          step_size=0.2,
                                          voxel_size=np.ones(3),
                                          max_angle=20)

    # Test if the tracking works on different dtype of the same data.
    for dtype in [np.float32, np.float64]:
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message=descoteaux07_legacy_msg,
                category=PendingDeprecationWarning,
            )
            fit = model.fit(data.astype(dtype))
            sh_pmf_gen = SHCoeffPmfGen(fit.shm_coeff, sphere, 'descoteaux07')
            sf_pmf_gen = SimplePmfGen(fit.odf(sphere), sphere)

        point = np.zeros(3)
        direction = unit_octahedron.vertices[0].copy()

        # Test using SH pmf
        status = probabilistic_propagator(&point[0],
                                          &direction[0],
                                          params,
                                          &stream_data[0],
                                          sh_pmf_gen,
                                          &rng)
        npt.assert_equal(status, TrackerStatus.FAIL)

        # Test using SF pmf
        status = probabilistic_propagator(&point[0],
                                          &direction[0],
                                          params,
                                          &stream_data[0],
                                          sf_pmf_gen,
                                          &rng)
        npt.assert_equal(status, TrackerStatus.FAIL)


def test_probabilistic_performances():
    # Test probabilistic tracker on the DiSCo dataset
    params = generate_tracking_parameters("prob",
                                          max_len=500,
                                          step_size=0.2,
                                          voxel_size=np.ones(3),
                                          max_angle=20)
    r = get_fast_tracking_performances(params, nbr_seeds=10000)
    npt.assert_(r > 0.85, msg="Probabilistic tracker has a low performance "
                              "score: " + str(r))


def test_tracker_ptt():
    # Test the probabilistic tracker function
    cdef double[:] stream_data = np.zeros(100, dtype=float)
    cdef double[:] point
    cdef double[:] direction
    cdef RNGState rng

    seed_rng(&rng, 12345)

    class SillyModel(SphHarmModel):
        sh_order_max = 4

        def fit(self, data, mask=None):
            coeff = np.zeros(data.shape[:-1] + (15,))
            return SphHarmFit(self, coeff, mask=None)

    model = SillyModel(gtab=None)
    data = np.zeros((3, 3, 3, 7))
    sphere = unit_octahedron

    params = generate_tracking_parameters("ptt",
                                          max_len=500,
                                          step_size=0.2,
                                          voxel_size=np.ones(3),
                                          max_angle=20,
                                          probe_quality=3)

    # Test if the tracking works on different dtype of the same data.
    for dtype in [np.float32, np.float64]:
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message=descoteaux07_legacy_msg,
                category=PendingDeprecationWarning,
            )
            fit = model.fit(data.astype(dtype))
            sh_pmf_gen = SHCoeffPmfGen(fit.shm_coeff, sphere, 'descoteaux07')
            sf_pmf_gen = SimplePmfGen(fit.odf(sphere), sphere)

        point = np.zeros(3)
        direction = unit_octahedron.vertices[0].copy()

        # Test using SH pmf
        status = parallel_transport_propagator(&point[0],
                                               &direction[0],
                                               params,
                                               &stream_data[0],
                                               sh_pmf_gen,
                                               &rng)
        npt.assert_equal(status, TrackerStatus.FAIL)

        # Test using SF pmf
        status = parallel_transport_propagator(&point[0],
                                               &direction[0],
                                               params,
                                               &stream_data[0],
                                               sf_pmf_gen,
                                               &rng)
        npt.assert_equal(status, TrackerStatus.FAIL)


def test_ptt_performances():
    # Test ptt tracker on the DiSCo dataset

    params = generate_tracking_parameters("ptt",
                                          max_len=500,
                                          step_size=0.2,
                                          voxel_size=np.ones(3),
                                          max_angle=15,
                                          probe_quality=4)

    r = get_fast_tracking_performances(params, nbr_seeds=5000)
    npt.assert_(r > 0.85, msg="PTT tracker has a low performance "
                              "score: " + str(r))


def stepped_1d(arr_1d):
    # Make a version of `arr_1d` which is not contiguous
    return np.vstack((arr_1d, arr_1d)).ravel(order="F")[::2]


def test_offset():
    # Test ndarray_offset function
    for dt in (np.int32, np.float64):
        index = np.array([1, 1], dtype=np.intp)
        A = np.array([[1, 0, 0], [0, 2, 0], [0, 0, 3]], dtype=dt)
        strides = np.array(A.strides, np.intp)
        i_size = A.dtype.itemsize
        npt.assert_equal(ndarray_offset(index, strides, 2, i_size), 4)
        npt.assert_equal(A.ravel()[4], A[1, 1])
        # Index and strides arrays must be C-continuous. Test this is enforced
        # by using non-contiguous versions of the input arrays.
        npt.assert_raises(ValueError, ndarray_offset, stepped_1d(index), strides, 2, i_size)
        npt.assert_raises(ValueError, ndarray_offset, index, stepped_1d(strides), 2, i_size)


def test_eudx_both_directions_errors():
    # Test error conditions for both directions function
    sphere = default_sphere
    seed = np.zeros(3, np.float64)
    qa = np.zeros((4, 5, 6, 7), np.float64)
    ind = qa.copy()
    # All of seed, qa, ind, odf_vertices must be C-contiguous.  Check by
    # passing in versions that aren't C contiguous
    npt.assert_raises(
        ValueError,
        eudx_both_directions,
        stepped_1d(seed),
        0,
        qa,
        ind,
        sphere.vertices,
        0.5,
        0.1,
        1.0,
        1.0,
        2,
    )
    npt.assert_raises(
        ValueError,
        eudx_both_directions,
        seed,
        0,
        qa[..., ::2],
        ind,
        sphere.vertices,
        0.5,
        0.1,
        1.0,
        1.0,
        2,
    )
    npt.assert_raises(
        ValueError,
        eudx_both_directions,
        seed,
        0,
        qa,
        ind[..., ::2],
        sphere.vertices,
        0.5,
        0.1,
        1.0,
        1.0,
        2,
    )
    npt.assert_raises(
        ValueError,
        eudx_both_directions,
        seed,
        0,
        qa,
        ind,
        sphere.vertices[::2],
        0.5,
        0.1,
        1.0,
        1.0,
        2,
    )
