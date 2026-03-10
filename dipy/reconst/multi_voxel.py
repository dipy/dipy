"""Tools to easily make multi voxel models"""

from functools import partial
import multiprocessing

import numpy as np
from tqdm import tqdm

from dipy.core.ndindex import ndindex
from dipy.reconst.base import ReconstFit
from dipy.reconst.quick_squash import quick_squash as _squash
from dipy.utils.parallel import paramap


def _parallel_fit_worker(vox_data, fit_func, **kwargs):
    """Process a chunk of voxel data.

    When ``_batched`` is True the entire chunk is handed to *fit_func*
    in a single batched call (the ``fit`` method handles the 2-D
    input directly and returns an array of fit objects).
    Otherwise each voxel is fitted individually (the classic path).

    Parameters
    ----------
    vox_data : ndarray, shape (n_voxels, ...)
        The data to fit.
    fit_func : callable
        ``partial(single_voxel_fit, model)`` — used for both per-voxel
        and batched (``_batched=True``) paths.
    """
    _resolve_shared_refs(fit_func, kwargs)
    batched = kwargs.pop("_batched", False)
    vox_weights = kwargs.pop("weights", None)
    if batched:
        if type(vox_weights) is np.ndarray:
            return fit_func(vox_data, weights=vox_weights, **kwargs)
        return fit_func(vox_data, **kwargs)

    if type(vox_weights) is np.ndarray:
        return [
            fit_func(data, **(dict({"weights": weights}, **kwargs)))
            for data, weights in zip(vox_data, vox_weights)
        ]
    return [fit_func(data, **kwargs) for data in vox_data]


def _resolve_shared_refs(fit_func, kwargs):
    """Resolve Ray shared object references on the worker side.

    If ``_shared_refs`` is present in *kwargs*, resolve each
    ``ObjectRef`` via ``ray.get()`` and apply them to the model
    instance captured inside *fit_func* (a ``functools.partial``).

    Parameters
    ----------
    fit_func : functools.partial
        Partial wrapping a model method.  The model is in
        ``fit_func.args[0]`` (``partial(single_voxel_fit, model)``).
    kwargs : dict
        Worker keyword arguments (mutated in-place to pop
        ``_shared_refs``).
    """
    shared_refs = kwargs.pop("_shared_refs", None)
    if shared_refs is not None:
        import ray

        resolved = {k: ray.get(v) for k, v in shared_refs.items()}
        if fit_func.args:
            model = fit_func.args[0]
        elif hasattr(fit_func.func, "__self__"):
            model = fit_func.func.__self__
        else:
            raise ValueError(
                "_resolve_shared_refs: could not locate model instance in "
                "fit_func — shared objects were never applied."
            )
        for name, val in resolved.items():
            setattr(model, name, val)


def multi_voxel_fit(_func=None, *, batched=False, shared_obj=None):
    """Method decorator to turn a single voxel model fit
    definition into a multi voxel model fit definition.

    Supports two calling styles::

        @multi_voxel_fit                   # existing models — unchanged
        @multi_voxel_fit(batched=True)     # batched models (e.g. FORCE)

    When ``batched=True`` the decorated ``fit`` method must accept
    both 1-D (single voxel) and 2-D (batch) input and return a single
    fit object or a 1-D object array of fit objects respectively.  The
    decorator calls it with the whole chunk at once instead of
    iterating voxel-by-voxel.

    Parameters
    ----------
    _func : callable, optional
        When the decorator is used without parentheses (``@multi_voxel_fit``)
        Python passes the decorated function here directly.
    batched : bool, optional
        When True the fit method handles batched 2-D input itself.
    shared_obj : tuple of str, optional
        Names of model attributes to place into the Ray object store once
        and reuse across workers (avoids per-task serialization of large
        arrays).  Example: ``("_penalty_array", "_index", "simulations")``.
        Only active when ``engine="ray"``.
    """

    def decorator(single_voxel_fit):
        def new_fit(self, data, *, mask=None, **kwargs):
            """Fit method for every voxel in data"""

            # If only one voxel just return a standard fit, passing through
            # the functions key-word arguments (no mask needed).
            if data.ndim == 1:
                svf = single_voxel_fit(self, data, **kwargs)
                # If fit method does not return extra, cannot return extra
                if isinstance(svf, tuple):
                    svf, extra = svf
                    return svf, extra
                else:
                    return svf

            # Make a mask if mask is None
            if mask is None:
                mask = np.ones(data.shape[:-1], bool)
            # Check the shape of the mask if mask is not None
            elif mask.shape != data.shape[:-1]:
                raise ValueError("mask and data shape do not match")

            # Get weights from kwargs if provided
            weights = kwargs["weights"] if "weights" in kwargs else None
            weights_is_array = True if type(weights) is np.ndarray else False

            # Fit data where mask is True
            fit_array = np.empty(data.shape[:-1], dtype=object)
            return_extra = False

            # Default to serial execution:
            engine = kwargs.get("engine", "serial")

            if engine == "serial" and batched:
                # Batched serial path — pass the whole chunk to fit() at once
                data_to_fit = data[np.where(mask)]
                vox_per_chunk = kwargs.get("vox_per_chunk", 10000)
                n_vox = data_to_fit.shape[0]
                all_chunk_results = []
                bar = tqdm(
                    total=n_vox,
                    position=0,
                    disable=not kwargs.get("verbose", False),
                )
                bar.set_description("Fitting (batched serial)")
                for start in range(0, n_vox, vox_per_chunk):
                    chunk = data_to_fit[start : start + vox_per_chunk]
                    chunk_result = single_voxel_fit(self, chunk)
                    all_chunk_results.append(chunk_result)
                    bar.update(len(chunk))
                bar.close()
                tmp_fit_array = np.concatenate(all_chunk_results)
                fit_array[np.where(mask)] = tmp_fit_array
            elif engine == "serial":
                extra_list = []
                bar = tqdm(
                    total=np.sum(mask),
                    position=0,
                    disable=not kwargs.get("verbose", False),
                )
                bar.set_description(
                    "Fitting reconstruction model using serial execution"
                )
                for ijk in ndindex(data.shape[:-1]):
                    if mask[ijk]:
                        if weights_is_array:
                            kwargs["weights"] = weights[ijk]

                        svf = single_voxel_fit(self, data[ijk], **kwargs)

                        # Not all fit methods return extra, handle this here
                        if isinstance(svf, tuple):
                            fit_array[ijk], extra = svf
                            return_extra = True
                        else:
                            fit_array[ijk], extra = svf, None

                        extra_list.append(extra)

                    bar.update()
                bar.close()
            else:
                data_to_fit = data[np.where(mask)]
                if weights_is_array:
                    weights_to_fit = weights[np.where(mask)]
                n_jobs = kwargs.get("n_jobs", max(multiprocessing.cpu_count() - 1, 1))
                vox_per_chunk = kwargs.get(
                    "vox_per_chunk", np.max([data_to_fit.shape[0] // n_jobs, 1])
                )
                chunks = [
                    data_to_fit[ii : ii + vox_per_chunk]
                    for ii in range(0, data_to_fit.shape[0], vox_per_chunk)
                ]

                # Extract shared objects *before* creating the partial so
                # that ``self`` is lightweight when serialised by Ray.
                shared_objects = None
                if engine == "ray" and shared_obj:
                    shared_objects = {name: getattr(self, name) for name in shared_obj}
                    for name in shared_obj:
                        setattr(self, name, None)

                try:
                    fit_func = partial(single_voxel_fit, self)

                    # Build per-chunk kwargs
                    kwargs_chunks = []
                    for ii in range(0, data_to_fit.shape[0], vox_per_chunk):
                        kw = kwargs.copy()
                        if batched:
                            kw["_batched"] = True
                        if weights_is_array:
                            kw["weights"] = weights_to_fit[ii : ii + vox_per_chunk]
                        kwargs_chunks.append(kw)

                    parallel_kwargs = {}
                    for kk in [
                        "n_jobs",
                        "vox_per_chunk",
                        "engine",
                        "verbose",
                        "inflight_cap",
                    ]:
                        if kk in kwargs:
                            parallel_kwargs[kk] = kwargs[kk]
                    if shared_objects is not None:
                        parallel_kwargs["shared_objects"] = shared_objects

                    mvf = paramap(
                        _parallel_fit_worker,
                        chunks,
                        func_args=[fit_func],
                        func_kwargs=kwargs_chunks,
                        **parallel_kwargs,
                    )
                finally:
                    # Always restore the model even on error
                    if shared_objects is not None:
                        for name, val in shared_objects.items():
                            setattr(self, name, val)

                if batched:
                    # Each element of mvf is an array of fit objects.
                    tmp_fit_array = np.concatenate(mvf)
                    fit_array[np.where(mask)] = tmp_fit_array
                    extra_list = None
                elif isinstance(mvf[0][0], tuple):
                    tmp_fit_array = np.concatenate(
                        [[svf[0] for svf in mvf_ch] for mvf_ch in mvf]
                    )
                    tmp_extra = np.concatenate(
                        [[svf[1] for svf in mvf_ch] for mvf_ch in mvf]
                    ).tolist()
                    fit_array[np.where(mask)], extra_list = tmp_fit_array, tmp_extra
                    return_extra = True
                else:
                    tmp_fit_array = np.concatenate(mvf)
                    fit_array[np.where(mask)], extra_list = tmp_fit_array, None

            # Redefine extra to be a single dictionary
            if return_extra:
                if extra_list[0] is not None:
                    extra_mask = {
                        key: np.vstack([e[key] for e in extra_list])
                        for key in extra_list[0]
                    }
                    extra = {}
                    for key in extra_mask:
                        extra[key] = np.zeros(data.shape)
                        extra[key][mask == 1] = extra_mask[key]
                else:
                    extra = None

            # If fit method does not return extra, assume we cannot return extra
            if return_extra:
                return MultiVoxelFit(self, fit_array, mask), extra
            else:
                return MultiVoxelFit(self, fit_array, mask)

        return new_fit

    if _func is not None:
        # Decorator used without parentheses: @multi_voxel_fit
        return decorator(_func)
    # Decorator used with parentheses: @multi_voxel_fit(batched=True)
    return decorator


class MultiVoxelFit(ReconstFit):
    """Holds an array of fits and allows access to their attributes and
    methods"""

    def __init__(self, model, fit_array, mask):
        self.model = model
        self.fit_array = fit_array
        self.mask = mask

    @property
    def shape(self):
        return self.fit_array.shape

    def __getattr__(self, attr):
        result = CallableArray(self.fit_array.shape, dtype=object)
        for ijk in ndindex(result.shape):
            if self.mask[ijk]:
                result[ijk] = getattr(self.fit_array[ijk], attr)
        return _squash(result, self.mask)

    def __getitem__(self, index):
        item = self.fit_array[index]
        if isinstance(item, np.ndarray):
            return MultiVoxelFit(self.model, item, self.mask[index])
        else:
            return item

    def predict(self, *args, **kwargs):
        """
        Predict for the multi-voxel object using each single-object's
        prediction API, with S0 provided from an array.
        """
        S0 = kwargs.get("S0", np.ones(self.fit_array.shape))
        idx = ndindex(self.fit_array.shape)
        ijk = next(idx)

        def gimme_S0(S0, ijk):
            if isinstance(S0, np.ndarray):
                return S0[ijk]
            else:
                return S0

        kwargs["S0"] = gimme_S0(S0, ijk)
        # If we have a mask, we might have some Nones up front, skip those:
        while self.fit_array[ijk] is None:
            ijk = next(idx)

        if not hasattr(self.fit_array[ijk], "predict"):
            msg = "This model does not have prediction implemented yet"
            raise NotImplementedError(msg)

        first_pred = self.fit_array[ijk].predict(*args, **kwargs)
        result = np.zeros(self.fit_array.shape + (first_pred.shape[-1],))
        result[ijk] = first_pred
        for ijk in idx:
            kwargs["S0"] = gimme_S0(S0, ijk)
            # If it's masked, we predict a 0:
            if self.fit_array[ijk] is None:
                result[ijk] *= 0
            else:
                result[ijk] = self.fit_array[ijk].predict(*args, **kwargs)

        return result


class CallableArray(np.ndarray):
    """An array which can be called like a function"""

    def __call__(self, *args, **kwargs):
        result = np.empty(self.shape, dtype=object)
        for ijk in ndindex(self.shape):
            item = self[ijk]
            if item is not None:
                result[ijk] = item(*args, **kwargs)
        return _squash(result)
