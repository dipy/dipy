import numpy as np
import multiprocessing
from tqdm.auto import tqdm
from dipy.utils.optpkg import optional_package

joblib, has_joblib, _ = optional_package('joblib')
dask, has_dask, _ = optional_package('dask')
ray, has_ray, _ = optional_package('ray')


def paramap(func, in_list, out_shape=None, n_jobs=-1, engine="joblib",
            backend=None, func_args=None, func_kwargs=None,
            **kwargs):
    """
    Maps a function to a list of inputs in parallel.

    Parameters
    ----------
    func : callable
        The function to apply to each item in the array. Must have the form:
        ``func(arr, idx, *args, *kwargs)`` where arr is an ndarray and idx is an
        index into that array (a tuple). The Return of `func` needs to be one
        item (e.g. float, int) per input item.
    in_list : list
       A sequence of items each of which can be an input to ``func``.
    out_shape : tuple, optional
         The shape of the output array. If not specified, the output shape will
         be `(len(in_list),)`.
    n_jobs : integer, optional
        The number of jobs to perform in parallel. -1 to use all but one cpu.
        Default: -1.
    engine : str
        {"dask", "joblib", "ray", "serial"}
        The last one is useful for debugging -- runs the code without any
        parallelization. Default: "joblib"
    backend : str, optional
        What joblib or dask backend to use. For joblib, the default is "loky".
        For dask the default is "threading".
    func_args : list, optional
        Positional arguments to `func`.
    func_kwargs : dict, optional
        Keyword arguments to `func`.
    kwargs : dict, optional
        Additional arguments to pass to either joblib.Parallel
        or dask.compute depending on the engine used.
        Default: {}

    Returns
    -------
    ndarray of identical shape to `arr`

    """

    func_args = func_args or []
    func_kwargs = func_kwargs or {}

    if engine == "joblib":
        if not has_joblib:
            raise joblib()
        if backend is None:
            backend = "loky"
        pp = joblib.Parallel(
            n_jobs=n_jobs, backend=backend,
            **kwargs)
        dd = joblib.delayed(func)
        d_l = [dd(ii, *func_args, **func_kwargs) for ii in in_list]
        results = pp(tqdm(d_l))

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
        pp = partial(func, *func_args, **func_kwargs)
        dd = [dask.delayed(pp)(ii) for ii in in_list]
        if backend == "multiprocessing":
            results = dask.compute(*dd, scheduler="processes",
                                   workers=n_jobs, **kwargs)
        elif backend == "threading":
            results = dask.compute(*dd, scheduler="threads",
                                   workers=n_jobs, **kwargs)
        else:
            raise ValueError("%s is not a backend for dask" % backend)

    if engine == "ray":
        if not has_ray:
            raise ray()

        func = ray.remote(func)
        results = ray.get([func.remote(ii, *func_args, **func_kwargs)
                           for ii in in_list])

    elif engine == "serial":
        results = []
        for in_element in in_list:
            results.append(func(in_element, *func_args, **func_kwargs))

    if out_shape is not None:
        return np.array(results).reshape(out_shape)
    else:
        return results
