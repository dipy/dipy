from collections.abc import Sequence
import json
import multiprocessing
import shutil
import tempfile

import numpy as np
from tqdm.auto import tqdm

from dipy.testing.decorators import warning_for_keywords
from dipy.utils.multiproc import determine_num_processes
from dipy.utils.optpkg import optional_package

ray, has_ray, _ = optional_package("ray")
joblib, has_joblib, _ = optional_package("joblib")
dask, has_dask, _ = optional_package("dask")
_RAY_TASK_OVERHEAD_MS = 3.0  # empirical fixed cost per Ray task (ms)
_RAY_T_VOXEL_MS = 0.015  # conservative compute time per voxel (ms)


def _available_ram():
    """Estimate available RAM in bytes using stdlib only.

    * **Linux** — reads ``MemAvailable`` from ``/proc/meminfo`` (accurate).
    * **Windows** — queries ``GlobalMemoryStatusEx`` via ``ctypes`` (accurate).
    * **macOS / other POSIX** — returns half of total physical RAM via
      ``os.sysconf`` (conservative approximation).
    * **Fallback** — 4 GiB when none of the above succeed.
    """
    import platform

    system = platform.system()
    try:
        if system == "Linux":
            with open("/proc/meminfo") as fh:
                for line in fh:
                    if line.startswith("MemAvailable:"):
                        return int(line.split()[1]) * 1024
        elif system == "Windows":
            import ctypes

            class _MEMORYSTATUSEX(ctypes.Structure):
                _fields_ = [
                    ("dwLength", ctypes.c_ulong),
                    ("dwMemoryLoad", ctypes.c_ulong),
                    ("ullTotalPhys", ctypes.c_ulonglong),
                    ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong),
                    ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong),
                    ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
                ]

            mem = _MEMORYSTATUSEX()
            mem.dwLength = ctypes.sizeof(_MEMORYSTATUSEX)
            ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(mem))
            return mem.ullAvailPhys
        else:
            import os

            pages = os.sysconf("SC_PHYS_PAGES")
            page_size = os.sysconf("SC_PAGE_SIZE")
            return (pages * page_size) // 2
    except Exception:
        pass
    return 4 * 1024**3


def auto_ray_chunk_size(
    *,
    n_jobs,
    n_gradients,
    n_vox=None,
    shared_obj_nbytes=0,
    ram_fraction=0.4,
    min_efficiency=0.10,
    output_bytes_per_voxel=4096,
):
    """Compute a memory- and efficiency-aware Ray chunk size.

    Three constraints are balanced:

    * **Efficiency lower bound** — chunk large enough that Ray task overhead
      stays below ``min_efficiency`` of total compute time::

          N_min = ray_overhead_ms / (min_efficiency × t_voxel_ms)
                ≈ 3 ms / (0.10 × 0.015 ms) ≈ 2 000 voxels

    * **Memory upper bound** — all ``n_jobs`` in-flight chunks fit in the
      available RAM budget::

          N_mem = (available_RAM × ram_fraction − shared_obj_bytes)
                  ────────────────────────────────────────────────────
                          n_jobs × bytes_per_voxel

      where ``bytes_per_voxel = n_gradients × 4 + output_bytes_per_voxel``.

    * **Parallelism upper bound** — at least ``n_jobs`` chunks are produced
      so all workers stay busy::

          N_par = n_vox // n_jobs   (only applied when n_vox is given)

    Final result: ``clamp(min(N_mem, N_par), lo=N_min)``.  Available RAM
    is detected via :func:`_available_ram` (stdlib only, no extra
    dependencies).

    Parameters
    ----------
    n_jobs : int
        Number of parallel Ray workers (concurrent in-flight tasks).
    n_gradients : int
        Number of diffusion gradients — sets the input chunk byte size.
    n_vox : int, optional
        Total number of voxels to fit.  Used to cap chunk size so that
        at least ``n_jobs`` chunks are produced.
    shared_obj_nbytes : int, optional
        Total bytes already placed in the Ray object store (e.g. the
        simulation library).  Deducted from the available RAM budget.
    ram_fraction : float, optional
        Fraction of available RAM to allocate to the Ray object store.
    min_efficiency : float, optional
        Target minimum ratio ``compute_time / total_task_time``.
    output_bytes_per_voxel : int, optional
        Estimated output bytes per voxel in the raw result dict returned
        by workers.  Defaults to 4 096 (≈ FORCE with ODF enabled).

    Returns
    -------
    chunk_size : int
    """
    n_min = int(_RAY_TASK_OVERHEAD_MS / (min_efficiency * _RAY_T_VOXEL_MS))

    n_jobs = determine_num_processes(n_jobs if n_jobs != 0 else 1)

    available = _available_ram()

    ray_budget = max(available * ram_fraction - shared_obj_nbytes, 0)
    bytes_per_voxel = n_gradients * 4 + output_bytes_per_voxel
    n_mem = int(ray_budget / n_jobs / bytes_per_voxel)

    n_par = (n_vox // n_jobs) if n_vox is not None else n_mem

    return max(n_min, min(n_mem, n_par))


@warning_for_keywords()
def paramap(
    func,
    in_list,
    *,
    out_shape=None,
    n_jobs=-1,
    engine="ray",
    backend=None,
    func_args=None,
    clean_spill=True,
    func_kwargs=None,
    shared_objects=None,
    inflight_cap=None,
    verbose=False,
    **kwargs,
):
    # FIXME: but several fitting functions return "extra" as well
    #        is this being handled properly, or not?
    #        perhaps the 'func' docs below are misleading?
    """
    Maps a function to a list of inputs in parallel.

    Parameters
    ----------
    func : callable
        The function to apply to each item in the array. Must have the form:
        ``func(arr, idx, *args, *kwargs)`` where arr is an ndarray and idx is
        an index into that array (a tuple). The Return of `func` needs to be
        one item (e.g. float, int) per input item.
    in_list : list
       A sequence of items each of which can be an input to ``func``.
    out_shape : tuple, optional
         The shape of the output array. If not specified, the output shape will
         be `(len(in_list),)`.
    n_jobs : int, optional
        The number of jobs to perform in parallel. Use -1 to use all but one
        cpu.
    engine : str, optional
        {"dask", "joblib", "ray", "serial"}
        The last one is useful for debugging -- runs the code without any
        parallelization.
    backend : str, optional
        The joblib or dask backend to use. For joblib the options are "loky",
        "threading", and "multiprocessing". For dask the options are
        "threading" and "multiprocessing".
    clean_spill : bool, optional
        If True, clean up the spill directory after the computation is done.
        Only applies to "ray" engine.
    func_args : list, optional
        Positional arguments to `func`.
    func_kwargs : dict or sequence, optional
        Keyword arguments to `func` or sequence of keyword arguments
        to `func`: one item for each item in the input list.
    shared_objects : dict, optional
        Dictionary of ``{name: array}`` to place in the Ray object store
        once via ``ray.put()``.  Each ``ObjectRef`` is then passed to
        workers as a direct top-level keyword argument prefixed with
        ``_sobj_`` (e.g. ``_sobj__index``).  Ray auto-resolves top-level
        ``ObjectRef`` arguments before invoking the worker — no explicit
        ``ray.get()`` is required on the worker side.  Only used with the
        ``"ray"`` engine; ignored for other engines.
    inflight_cap : int, optional
        Maximum number of pending Ray tasks before draining results.
        Limits memory pressure when processing many chunks.
        Only used with the ``"ray"`` engine; ignored for other engines.
        When None, all tasks are submitted at once.
    verbose : bool, optional
        Show a tqdm progress bar while processing chunks.
    kwargs : dict, optional
        Additional arguments to pass to either joblib.Parallel
        or dask.compute, or ray.remote depending on the engine used.

    Returns
    -------
    ndarray of identical shape to `arr`

    """
    func_args = func_args or []
    func_kwargs = func_kwargs or {}

    if isinstance(func_kwargs, Sequence):
        if len(func_kwargs) != len(in_list):
            raise ValueError(
                "The length of func_kwargs should be the same as the length of in_list"
            )
        func_kwargs_sequence = True
    else:
        func_kwargs_sequence = False

    if engine == "joblib":
        if not has_joblib:
            raise joblib()
        if backend is None:
            backend = "loky"
        pp = joblib.Parallel(n_jobs=n_jobs, backend=backend, **kwargs)
        dd = joblib.delayed(func)
        if func_kwargs_sequence:
            d_l = [dd(ii, *func_args, **fk) for ii, fk in zip(in_list, func_kwargs)]
        else:
            d_l = [dd(ii, *func_args, **func_kwargs) for ii in in_list]
        results = pp(tqdm(d_l, disable=not verbose))

    elif engine == "dask":
        if not has_dask:
            raise dask()
        if backend is None:
            backend = "threading"

        if n_jobs == -1:
            n_jobs = multiprocessing.cpu_count()
            n_jobs = n_jobs - 1

        def partial(func, *args, **keywords):
            def newfunc(in_arg):
                return func(in_arg, *args, **keywords)

            return newfunc

        delayed_func = dask.delayed(func)

        if func_kwargs_sequence:
            dd = [
                delayed_func(ii, *func_args, **fk)
                for ii, fk in zip(in_list, func_kwargs)
            ]
        else:
            pp = dask.delayed(partial(func, *func_args, **func_kwargs))
            dd = [pp(ii) for ii in in_list]

        if backend == "multiprocessing":
            results = dask.compute(*dd, scheduler="processes", workers=n_jobs, **kwargs)
        elif backend == "threading":
            results = dask.compute(*dd, scheduler="threads", workers=n_jobs, **kwargs)
        else:
            raise ValueError(f"{backend} is not a backend for dask")

    elif engine == "ray":
        if not has_ray:
            raise ray()

        if clean_spill:
            tmp_dir = tempfile.TemporaryDirectory()

            if not ray.is_initialized():
                ray.init(
                    _system_config={
                        "object_spilling_config": json.dumps(
                            {
                                "type": "filesystem",
                                "params": {"directory_path": tmp_dir.name},
                            }
                        )
                    }
                )

        shared_refs = None
        if shared_objects:
            shared_refs = {k: ray.put(v) for k, v in shared_objects.items()}

        func = ray.remote(func)

        def _build_kwargs(base_kw):
            if shared_refs is not None:
                sobj_kw = {f"_sobj_{k}": v for k, v in shared_refs.items()}
                return {**base_kw, **sobj_kw}
            return base_kw

        def _submit_one(item, kw):
            return func.remote(item, *func_args, **_build_kwargs(kw))

        n_chunks = len(in_list)
        if inflight_cap is not None and inflight_cap > 0:
            results = []
            pending = []
            items_kw = (
                zip(in_list, func_kwargs)
                if func_kwargs_sequence
                else ((ii, func_kwargs) for ii in in_list)
            )
            with tqdm(
                total=n_chunks, disable=not verbose, desc="Fitting (ray)"
            ) as pbar:
                for item, kw in items_kw:
                    pending.append(_submit_one(item, kw))
                    if len(pending) >= inflight_cap:
                        results.extend(ray.get(pending))
                        pbar.update(len(pending))
                        pending = []
                if pending:
                    results.extend(ray.get(pending))
                    pbar.update(len(pending))
        else:
            if func_kwargs_sequence:
                futures = [_submit_one(ii, fk) for ii, fk in zip(in_list, func_kwargs)]
            else:
                futures = [_submit_one(ii, func_kwargs) for ii in in_list]
            future_to_idx = {f: i for i, f in enumerate(futures)}
            results = [None] * n_chunks
            with tqdm(
                total=n_chunks, disable=not verbose, desc="Fitting (ray)"
            ) as pbar:
                remaining = list(futures)
                while remaining:
                    done, remaining = ray.wait(remaining, num_returns=1)
                    for f in done:
                        results[future_to_idx.pop(f)] = ray.get(f)
                        pbar.update(1)

        if clean_spill:
            shutil.rmtree(tmp_dir.name)

    elif engine == "serial":
        results = []
        with tqdm(
            total=len(in_list), disable=not verbose, desc="Fitting (serial)"
        ) as pbar:
            if func_kwargs_sequence:
                for in_element, fk in zip(in_list, func_kwargs):
                    results.append(func(in_element, *func_args, **fk))
                    pbar.update(1)
            else:
                for in_element in in_list:
                    results.append(func(in_element, *func_args, **func_kwargs))
                    pbar.update(1)
    else:
        raise ValueError(f"{engine} is not a valid engine")

    if out_shape is not None:
        return np.array(results).reshape(out_shape)
    else:
        return results
