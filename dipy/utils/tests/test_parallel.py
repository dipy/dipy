import numpy as np
import numpy.testing as npt

import dipy.utils.parallel as para
from dipy.utils.parallel import _available_ram, auto_ray_chunk_size


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


def test_available_ram():
    ram = _available_ram()
    assert isinstance(ram, int)
    assert ram > 0
    # Sanity: at least 64 MB, at most 64 TiB
    assert 64 * 1024**2 <= ram <= 64 * 1024**4


def test_auto_ray_chunk_size_returns_positive_int():
    chunk = auto_ray_chunk_size(n_jobs=4, n_gradients=160)
    assert isinstance(chunk, int)
    assert chunk > 0


def test_auto_ray_chunk_size_efficiency_floor():
    # Even with zero RAM budget the result must not fall below N_min
    from dipy.utils.parallel import _RAY_TASK_OVERHEAD_MS, _RAY_T_VOXEL_MS

    n_min = int(_RAY_TASK_OVERHEAD_MS / (0.10 * _RAY_T_VOXEL_MS))
    chunk = auto_ray_chunk_size(
        n_jobs=8,
        n_gradients=160,
        shared_obj_nbytes=10 * 1024**4,  # 10 TiB — exhausts any RAM budget
    )
    assert chunk >= n_min


def test_auto_ray_chunk_size_parallelism_cap():
    # With n_vox given, chunk <= n_vox // n_jobs  (enough chunks for all workers)
    n_vox, n_jobs = 10_000, 4
    chunk = auto_ray_chunk_size(n_jobs=n_jobs, n_gradients=160, n_vox=n_vox)
    assert chunk <= n_vox // n_jobs


def test_auto_ray_chunk_size_memory_scales_with_ram():
    # More RAM budget  →  larger (or equal) chunk when n_par is not binding
    large_n_vox = 50_000_000  # large enough that n_par does not cap
    chunk_tight = auto_ray_chunk_size(
        n_jobs=8,
        n_gradients=160,
        n_vox=large_n_vox,
        shared_obj_nbytes=0,
        ram_fraction=0.001,  # very small budget
    )
    chunk_generous = auto_ray_chunk_size(
        n_jobs=8,
        n_gradients=160,
        n_vox=large_n_vox,
        shared_obj_nbytes=0,
        ram_fraction=0.9,  # large budget
    )
    assert chunk_tight <= chunk_generous


def test_auto_ray_chunk_size_negative_n_jobs():
    # n_jobs=-1 must not collapse to 1 worker (which would yield 1 giant chunk)
    from dipy.utils.multiproc import determine_num_processes

    n_vox = 100_000
    chunk_neg = auto_ray_chunk_size(n_jobs=-1, n_gradients=160, n_vox=n_vox)
    chunk_pos = auto_ray_chunk_size(
        n_jobs=determine_num_processes(-1), n_gradients=160, n_vox=n_vox
    )
    assert chunk_neg == chunk_pos
    # Must produce more than one chunk
    assert chunk_neg < n_vox


def test_auto_ray_chunk_size_more_jobs_smaller_chunks():
    # More workers → smaller per-worker budget → smaller chunks
    n_vox = 1_000_000
    chunk_few = auto_ray_chunk_size(n_jobs=2, n_gradients=160, n_vox=n_vox)
    chunk_many = auto_ray_chunk_size(n_jobs=32, n_gradients=160, n_vox=n_vox)
    assert chunk_few >= chunk_many
