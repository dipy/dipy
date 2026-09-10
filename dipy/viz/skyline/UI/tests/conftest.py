"""Headless ImGui harness shared by the Skyline UI tests.

The harness runs genuine ImGui frames against a real context and feeds real
mouse events through ``ImGuiIO``, so widgets are exercised exactly as they are
in the running application.
"""

import pytest

from dipy.utils.optpkg import optional_package

_, has_imgui, _ = optional_package(
    "imgui_bundle", min_version="1.92.600", max_version="1.92.801"
)

if has_imgui:
    from imgui_bundle import imgui


class ImGuiHarness:
    """Drive real ImGui frames without a window or GPU backend."""

    def __init__(self, size=(500, 900)):
        self._context = imgui.create_context()
        self.size = size
        self.io = imgui.get_io()
        # ImGui persists window layout to ``imgui.ini`` in the working directory
        # when the context is destroyed. Tests have no layout worth keeping.
        self.io.set_ini_filename("")
        self.io.display_size = size
        self.io.delta_time = 1.0 / 60.0
        self.io.backend_flags |= imgui.BackendFlags_.renderer_has_textures
        self.last_rect = None

    def close(self):
        imgui.destroy_context(self._context)

    def draw(self, render, *, mouse=None, press=False, origin=(50, 50)):
        """Run one frame and return whatever ``render`` returned."""
        if mouse is not None:
            self.io.add_mouse_pos_event(float(mouse[0]), float(mouse[1]))
        if press:
            self.io.add_mouse_button_event(imgui.MouseButton_.left, True)

        imgui.new_frame()
        imgui.set_next_window_pos((0, 0))
        imgui.set_next_window_size(self.size)
        imgui.begin("harness", None, imgui.WindowFlags_.no_title_bar)
        imgui.set_cursor_screen_pos(origin)
        try:
            result = render()
            self.last_rect = (imgui.get_item_rect_min(), imgui.get_item_rect_max())
        finally:
            # A widget that only opens a popup - or one that raised before
            # drawing - submits nothing into the harness window, and ImGui
            # asserts unless the moved cursor is backed by an item.
            imgui.set_cursor_screen_pos(origin)
            imgui.dummy((1, 1))
            imgui.end()
            imgui.end_frame()
            imgui.render()
            if press:
                self.io.add_mouse_button_event(imgui.MouseButton_.left, False)
        return result

    def frame(self, render, *, mouse=None, press=False):
        """Run one frame without opening a harness window.

        Use this for code that opens its own top-level window, such as
        :meth:`UIWindow.render`.
        """
        if mouse is not None:
            self.io.add_mouse_pos_event(float(mouse[0]), float(mouse[1]))
        if press:
            self.io.add_mouse_button_event(imgui.MouseButton_.left, True)

        imgui.new_frame()
        try:
            result = render()
        finally:
            imgui.end_frame()
            imgui.render()
            if press:
                self.io.add_mouse_button_event(imgui.MouseButton_.left, False)
        return result

    def item_center(self):
        """Center of the last item drawn during :meth:`draw`."""
        rect_min, rect_max = self.last_rect
        return ((rect_min.x + rect_max.x) * 0.5, (rect_min.y + rect_max.y) * 0.5)

    def click(self, render, *, at=None, warmup=3, origin=(50, 50)):
        """Hover then press the left button, returning the press frame result.

        Widgets reacting to ``is_mouse_clicked``/``is_item_clicked`` respond on
        the press frame. ``at`` defaults to the center of the last item drawn by
        ``render``, which is the hit box of most Skyline widgets.
        """
        self.draw(render, origin=origin)
        position = at if at is not None else self.item_center()
        for _ in range(warmup):
            self.draw(render, mouse=position, origin=origin)
        return self.draw(render, mouse=position, press=True, origin=origin)

    def press_release(self, render, *, at, warmup=3, origin=(50, 50)):
        """Full press-and-release over ``at``, returning the release result.

        ``imgui.button`` fires on release, so the release event queued after the
        press frame is only consumed by the following frame.
        """
        self.click(render, at=at, warmup=warmup, origin=origin)
        return self.draw(render, mouse=at, origin=origin)


@pytest.fixture
def ui():
    if not has_imgui:
        pytest.skip("Requires imgui_bundle>=1.92.600,<=1.92.801")
    harness = ImGuiHarness()
    try:
        yield harness
    finally:
        harness.close()
