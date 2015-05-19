"""Tools to easily make multi voxel models"""
import numpy as np
from numpy.lib.stride_tricks import as_strided

from dipy.parallel.main import ParallelFunction

from ..core.ndindex import ndindex
from .quick_squash import quick_squash as _squash
from .base import ReconstFit


class ParallelFit(ParallelFunction):

    def _main(self, single_voxel_data, model=None):
        """Fit method for every voxel in data"""
        fit = model.fit(single_voxel_data)
        return {"fit_array":fit}

    def _default_values(self, data, mask, model=None):
        return {"fit_array":np.array(None)}

# Function for fitting models in parallel
parallel_fit = ParallelFit()

def multi_voxel_fit(single_voxel_fit):
    b"""Wraps single_voxel_fit method to turn a model into a multi voxel model.

    Use this decorator on the fit method of your model to take advantage of the
    MultiVoxelFit and ParallelFit machinery of dipy.

    Parameters:
    -----------
    single_voxel_fit : callable
        Should have a signature like: model [self when a model method is being
        wrapped], data [single voxel data].

    Returns:
    --------
    multi_voxel_fit_function : callable

    Examples:
    ---------
    >>> import numpy as np
    >>> from dipy.reconst.base import ReconstModel, ReconstFit

    >>> class Model(ReconstModel):
    ...     @multi_voxel_fit
    ...     def fit(self, single_voxel_data):
    ...         return ReconstFit(self, single_voxel_data.sum())

    >>> model = Model(None)
    >>> data = np.random.random((2, 3, 4, 5))
    >>> fit = model.fit(data)
    >>> np.allclose(fit.data, data.sum(-1))
    True

    """
    def new_fit(model, data, mask=None):
        if data.ndim == 1:
            return single_voxel_fit(model, data)
        if mask is None:
            mask = np.ones(data.shape[:-1], bool)
        fit = parallel_fit(data, mask, model=model)
        return MultiVoxelFit(model, fit["fit_array"], mask)

    return new_fit


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
        if not hasattr(self.model, 'predict'):
            msg = "This model does not have prediction implemented yet"
            raise NotImplementedError(msg)

        S0 = kwargs.get('S0', np.ones(self.fit_array.shape))
        idx = ndindex(self.fit_array.shape)
        ijk = next(idx)

        def gimme_S0(S0, ijk):
            if isinstance(S0, np.ndarray):
                return S0[ijk]
            else:
                return S0

        kwargs['S0'] = gimme_S0(S0, ijk)
        # If we have a mask, we might have some Nones up front, skip those:
        while self.fit_array[ijk] is None:
            ijk = next(idx)

        first_pred = self.fit_array[ijk].predict(*args, **kwargs)
        result = np.zeros(self.fit_array.shape + (first_pred.shape[-1],))
        result[ijk] = first_pred
        for ijk in idx:
            kwargs['S0'] = gimme_S0(S0, ijk)
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
