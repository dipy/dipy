import numpy as np
import numpy.testing as npt
import pytest

import dipy.utils.parallel as para


def power_it(num, n=2):
    # We define a function of the right form for parallelization
    return num**n


ENGINES = ["serial"]
if para.has_dask:
    ENGINES.append("dask")
if para.has_joblib:
    ENGINES.append("joblib")
if para.has_ray:
    ENGINES.append("ray")


def test_paramap():
    my_array = np.arange(100).reshape(10, 10)
    my_list = list(my_array.ravel())
    for engine in ENGINES:
        for backend in ["threading", "multiprocessing"]:
            npt.assert_array_equal(
                para.paramap(
                    power_it,
                    my_list,
                    engine=engine,
                    backend=backend,
                    out_shape=my_array.shape,
                ),
                power_it(my_array),
            )

            # If it's not reshaped, the first item should be the item 0, 0:
            npt.assert_equal(
                para.paramap(power_it, my_list, engine=engine, backend=backend)[0],
                power_it(my_array[0, 0]),
            )


def test_paramap_sequence_kwargs():
    my_array = np.arange(10)
    kwargs_sequence = [{"n": i} for i in range(len(my_array))]
    my_list = list(my_array.ravel())
    for engine in ENGINES:
        for backend in ["threading", "multiprocessing"]:
            npt.assert_array_equal(
                para.paramap(
                    power_it,
                    my_list,
                    engine=engine,
                    backend=backend,
                    out_shape=my_array.shape,
                    func_kwargs=kwargs_sequence,
                ),
                [
                    power_it(ii, **kwargs)
                    for ii, kwargs in zip(my_array, kwargs_sequence)
                ],
            )


def test_paramap_ray_respects_n_jobs(monkeypatch):
    """Validate Ray init uses num_cpus=n_jobs without requiring ray install."""

    class _FakeRemoteFunc:
        def __init__(self, func):
            self._func = func

        def remote(self, *args, **kwargs):
            return self._func(*args, **kwargs)

    class _FakeRay:
        def __init__(self):
            self._initialized = False
            self.init_kwargs = None

        def is_initialized(self):
            return self._initialized

        def init(self, **kwargs):
            self._initialized = True
            self.init_kwargs = kwargs

        def remote(self, func):
            return _FakeRemoteFunc(func)

        def get(self, refs):
            return refs

    fake_ray = _FakeRay()

    monkeypatch.setattr(para, "ray", fake_ray)
    monkeypatch.setattr(para, "has_ray", True)

    result = para.paramap(power_it, [1, 2, 3], engine="ray", n_jobs=3, clean_spill=False)
    npt.assert_array_equal(result, [1, 4, 9])
    assert fake_ray.init_kwargs is not None
    assert fake_ray.init_kwargs.get("num_cpus") == 3


@pytest.mark.parametrize("clean_spill", [True, False])
def test_paramap_ray_initializes(monkeypatch, clean_spill):
    """Ray backend should call init() if not already initialized."""

    class _FakeRemoteFunc:
        def __init__(self, func):
            self._func = func

        def remote(self, *args, **kwargs):
            return self._func(*args, **kwargs)

    class _FakeRay:
        def __init__(self):
            self._initialized = False
            self.init_calls = 0

        def is_initialized(self):
            return self._initialized

        def init(self, **kwargs):
            self._initialized = True
            self.init_calls += 1

        def remote(self, func):
            return _FakeRemoteFunc(func)

        def get(self, refs):
            return refs

    fake_ray = _FakeRay()
    monkeypatch.setattr(para, "ray", fake_ray)
    monkeypatch.setattr(para, "has_ray", True)

    _ = para.paramap(power_it, [1], engine="ray", n_jobs=1, clean_spill=clean_spill)
    assert fake_ray.init_calls == 1
