import numpy as np

from dipy.utils.optpkg import optional_package

fury, has_fury, setup_module = optional_package('fury', min_version="0.10.0")

if has_fury:
    from fury.actor import surface as surface_actor


class SurfaceVisualizer:

    def __init__(self, surface, scene, color):
        self._vertices, self._faces = surface

        self._surface_actor = surface_actor(self._vertices, self._faces,
                                            np.full((self._vertices.shape[0],
                                                     3), color))

        scene.add(self._surface_actor)

    @property
    def actors(self):
        return [self._surface_actor]
