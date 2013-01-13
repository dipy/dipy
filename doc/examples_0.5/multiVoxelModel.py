"""The `multi_voxel_model` is a class decorator to help easily write multi voxel
models. A developer simply needs to write a description of the single voxel
case and wrap it using `multi_voxel_model` like bellow."""

import numpy as np
from dipy.core.sphere import unit_icosahedron
from dipy.reconst.multi_vox import multi_voxel_model

"""First the developer should write out the single voxel as bellow and either
wrap or decorate the Model Class"""

class SingleVoxelModel(object):
    def fit(self, data, mask=None):
        n = np.max(data)
        return SingleVoxelFit(self, n)

class SingleVoxelFit(object):
    model_attr = 1.0
    def __init__(self, model, n):
        self.model = model
        self.n = n

    def odf(self, sphere):
        return np.ones(len(sphere.phi))

    @property
    def directions(self):
        return np.zeros((self.n, 3))

MultiVoxelModel = multi_voxel_model(SingleVoxelModel)

@multi_voxel_model
class DecoratedModel(SingleVoxelModel):
    """Now to show how all this works

    To show how the single voxel case works
    ---------------------------------------
    >>> model = SingleVoxelModel()
    >>> fit = model.fit(4)
    >>> fit.model_attr
    1.0
    >>> fit.directions.shape
    (4, 3)
    >>> fit.odf(unit_icosahedron).shape
    (12,)


    Now we use the MultiVoxelModel
    ------------------------------
    >>> model = MultiVoxelModel()
    >>> data = np.arange(1, 6).reshape((2, 3, 1))
    >>> fit = model.fit(data)
    >>> fit.model_attr.shape
    (2, 3)
    >>> np.all(fit.model_attr == 1.)
    True
    >>> fit.directions.shape
    (2, 3)
    >>> fit.directions[0, 0].shape
    (1, 3)
    >>> fit.odf(unit_icosahedron).shape
    (2, 3, 12)


    Of course using using `multi_voxel_model` as a decorator or as a wrapper
    function is exactly the same
    -------------------------------------------------------------------------
    >>> model = DecoratedModel()
    >>> data = np.arange(1, 6).reshape((2, 3, 1))
    >>> fit = model.fit(data)
    >>> fit.model_attr.shape
    (2, 3)
    >>> np.all(fit.model_attr == 1.)
    True
    >>> fit.directions.shape
    (2, 3)
    >>> fit.directions[0, 0].shape
    (1, 3)
    >>> fit.odf(unit_icosahedron).shape
    (2, 3, 12)


    """
    pass

