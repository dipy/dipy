#!/usr/bin/python
""" Class to present model parameters as voxel-shaped array """
# 5/17/2010

#import modules
from numpy import asarray, ones
from copy import copy

class ModelArray(object):
    """A class that has a shape and can be indexed like an ndarray

    When using a model to describe many voxels, ModelArray allows the
    parameters of a model to be stored as an ndarray where the last dimension
    of the array represents the parameters, and the first n-1 dimensions
    represent the shape or arrangement of the voxels. Model array is meant to
    be sub-classed to make more specific model classes.

    """
    ### Shape Property ###
    def _getshape(self):
        """
        Gives the shape of the ModelArray

        """

        return self.model_params.shape[:-1]

    def _setshape(self, shape):
        """
        Sets the shape of the ModelArray

        """
        if type(shape) is not tuple:
            shape = (shape,)
        self.model_params.shape = shape + self.model_params.shape[-1:]

    shape = property(_getshape, _setshape, doc = "Shape of model array")

    ### Ndim Property ###
    @property
    def ndim(self):
        """Gives the number of dimensions of the ModelArray
        """
        return self.model_params.ndim - 1

    @property
    def mask(self):
        """If the model_params array has a mask, returns the mask
        """
        if hasattr(self.model_params, 'mask'):
            return self.model_params.mask
        else:
            return ones(self.shape, 'bool')

    ### Getitem Property ###
    def __getitem__(self, index):
        """
        Returns part of the model array

        """
        if type(index) is not tuple:
            index = (index,)
        if len(index) > self.ndim:
            raise IndexError('invalid index')
        for ii, slc in enumerate(index):
            if slc is Ellipsis:
                n_ellipsis = len(self.shape) - len(index) + 1
                index = index[:ii] + n_ellipsis*(slice(None),) + index[ii+1:]
                break

        new_model = copy(self)
        new_model.model_params = self.model_params[index]
        return new_model

    def _get_model_params(self):
        """Parameters of the model

        All the parameters needed for a model should be flattened into the last
        dimension of model_params. The shape of the ModelArray is determined by
        the model_params.shape[:-1].
        """
        return self._model_params

    def _set_model_params(self, params):
        """Sets model_params
        """
        self._model_params = params

    model_params = property(_get_model_params, _set_model_params)


