import numpy as np

from dipy.utils.deprecator import deprecate_with_version
from dipy.utils.optpkg import optional_package

fury, has_fury, setup_module = optional_package(
    "fury", min_version="0.10.0", max_version="1.0.0"
)

if has_fury:
    from fury.actor import surface as surface_actor


class SurfaceVisualizer:
    @deprecate_with_version(
        "horizon.visualizer.SurfaceVisualizer is deprecated and will be removed in a future version. "
        "Use Skyline instead.",
        since="1.13.0",
        until="2.0.0",
    )
    def __init__(self, surface, scene, color):
        self._vertices, self._faces = surface

        self._surface_actor = surface_actor(
            self._vertices,
            faces=self._faces,
            colors=np.full((self._vertices.shape[0], 3), color),
        )

        scene.add(self._surface_actor)

    @property
    @deprecate_with_version(
        "horizon.visualizer.SurfaceVisualizer.actors is deprecated and will be removed in a future version. "
        "Use Skyline instead.",
        since="1.13.0",
        until="2.0.0",
    )
    def actors(self):
        return [self._surface_actor]
