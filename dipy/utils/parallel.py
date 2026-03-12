from collections.abc import Sequence
import multiprocessing
import shutil
import tempfile

import numpy as np
from tqdm.auto import tqdm

from dipy.testing.decorators import warning_for_keywords
from dipy.utils.optpkg import optional_package

ray, has_ray, _ = optional_package("ray")
joblib, has_joblib, _ = optional_package("joblib")
dask, has_dask, _ = optional_package("dask")


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
        once via ``ray.put()``.  The resulting ``ObjectRef`` dictionary is
        passed to each worker as the ``_shared_refs`` keyword argument.
        Workers can resolve them with ``ray.get()``.  Only used with the
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
                ray.init(object_spilling_directory=tmp_dir.name)

        shared_refs = None
        if shared_objects:
            shared_refs = {k: ray.put(v) for k, v in shared_objects.items()}

        func = ray.remote(func)

        def _build_kwargs(base_kw):
            if shared_refs is not None:
                return {**base_kw, "_shared_refs": shared_refs}
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
            with tqdm(
                total=n_chunks, disable=not verbose, desc="Fitting (ray)"
            ) as pbar:
                results = ray.get(futures)
                pbar.update(n_chunks)

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
        raise ValueError("%s is not a valid engine" % engine)

    if out_shape is not None:
        return np.array(results).reshape(out_shape)
    else:
        return results
