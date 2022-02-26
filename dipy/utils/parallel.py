import numpy as np
import multiprocessing
from tqdm.auto import tqdm
from dipy.utils.optpkg import optional_package

joblib, has_joblib, _ = optional_package('joblib')
dask, has_dask, _ = optional_package('dask')

def parfor(func, in_list, out_shape=None, n_jobs=-1, engine="joblib",
           backend="threading", func_args=[], func_kwargs={},
           **kwargs):
    """
    Parallel for loop for numpy arrays

    Parameters
    ----------
    func : callable
        The function to apply to each item in the array. Must have the form:
        func(arr, idx, *args, *kwargs) where arr is an ndarray and idx is an
        index into that array (a tuple). The Return of `func` needs to be one
        item (e.g. float, int) per input item.
    in_list : list
       All legitimate inputs to the function to operate over.
    n_jobs : integer, optional
        The number of jobs to perform in parallel. -1 to use all cpus.
        Default: 1
    engine : str
        {"dask", "joblib", "serial"}
        The last one is useful for debugging -- runs the code without any
        parallelization.
    backend : str
        What joblib or dask backend to use. Irrelevant for other engines.
    func_args : list, optional
        Positional arguments to `func`.
    func_kwargs : list, optional
        Keyword arguments to `func`.
    kwargs : dict, optional
        Additional arguments to pass to either joblib.Parallel
        or dask.compute depending on the engine used.
        Default: {}

    Returns
    -------
    ndarray of identical shape to `arr`

    Examples
    --------

    """
    if engine == "joblib":
        p = joblib.Parallel(
            n_jobs=n_jobs, backend=backend,
            **kwargs)
        d = joblib.delayed(func)
        d_l = []
        for in_element in in_list:
            d_l.append(d(in_element, *func_args, **func_kwargs))
        results = p(tqdm(d_l))

    elif engine == "dask":
        if n_jobs == -1:
            n_jobs = multiprocessing.cpu_count()
            n_jobs = n_jobs - 1

        def partial(func, *args, **keywords):
            def newfunc(in_arg):
                return func(in_arg, *args, **keywords)
            return newfunc
        p = partial(func, *func_args, **func_kwargs)
        d = [dask.delayed(p)(i) for i in in_list]
        if backend == "multiprocessing":
            results = dask.compute(*d, scheduler="processes",
                                   workers=n_jobs, **kwargs)
        elif backend == "threading":
            results = dask.compute(*d, scheduler="threads",
                                   workers=n_jobs, **kwargs)
        else:
            raise ValueError("%s is not a backend for dask" % backend)

    elif engine == "serial":
        results = []
        for in_element in in_list:
            results.append(func(in_element, *func_args, **func_kwargs))

    if out_shape is not None:
        return np.array(results).reshape(out_shape)
    else:
        return results
