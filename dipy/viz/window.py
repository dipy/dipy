"""FURY window utilities plus DIPY-specific window fixes."""

from collections.abc import Iterable

from dipy.utils.optpkg import optional_package

fury, has_fury, _ = optional_package("fury", min_version="2.1.0")
if has_fury:
    from fury.lib import loop
    from fury.window import *
    from fury.window import Scene, ShowManager as FuryShowManager
else:

    class FuryShowManager:
        """Stand-in base class used when FURY is not installed."""

    Scene = loop = fury


# TODO(fury#1416): rendercanvas 3.0 requests a paint after a title-bar close
# stopped the loop, so the next window raises "Event loop is closed". FURY will
# adopt this ``loop.stop()`` reset in ``ShowManager.start``. Remove
# ``ShowManager`` and ``show`` once the minimum FURY version includes it.
class ShowManager(FuryShowManager):
    """FURY ShowManager whose window can be followed by another one.

    Notes
    -----
    After the window closes, :meth:`start` stops the shared render loop. This
    resets the stale loop state that rendercanvas 3.0 leaves behind when a
    window is closed from its title bar.
    """

    def start(self):
        """Start the event loop and reset the render loop once the window closes."""
        super().start()
        if self.window.get_closed():
            loop.stop()


def show(actors, *, window_type="default", title="FURY 2.0"):
    """Display one or more actors in a new window.

    Parameters
    ----------
    actors : Object or list of Object
        Actor or actors to display.
    window_type : {"default", "glfw", "qt", "jupyter", "offscreen"}, optional
        Canvas type of the window.
    title : str, optional
        Window title.

    Notes
    -----
    Same as ``fury.window.show``, but uses :class:`ShowManager`, so another
    window can be opened after this one is closed.
    """
    scene = Scene()
    if isinstance(actors, Iterable):
        scene.add(*actors)
    else:
        scene.add(actors)
    ShowManager(scene=scene, window_type=window_type, title=title).start()
