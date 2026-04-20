"""Skyline sidebar window: section grouping, file dialogs, and font setup."""

from collections.abc import Callable

from dipy.utils.optpkg import optional_package
from dipy.viz.skyline.UI.elements import (
    color_picker,
    colors_equal,
    loading,
    render_file_dialog,
    render_section_header,
)
from dipy.viz.skyline.UI.theme import FONT, FONT_AWESOME, THEME

imgui_bundle, has_imgui, _ = optional_package("imgui_bundle", min_version="1.92.600")
if has_imgui:
    imgui = imgui_bundle.imgui
    icons_fontawesome_6 = imgui_bundle.icons_fontawesome_6


_GROUP_ORDER = ["image", "tractography", "peak", "sh_glyph", "roi", "surface"]
_GROUP_LABELS = {
    "image": "Images",
    "tractography": "Tractograms",
    "peak": "Peaks",
    "sh_glyph": "ODFs",
    "roi": "ROIs",
    "surface": "Surfaces",
}


class UIManager:
    """Represent ``UIManager`` in Skyline."""

    def __init__(self):
        """Represent ``UIManager`` in Skyline."""
        self.windows = {}

    def add_window(self, window_name, window_instance):
        """Register ``window_instance`` under ``window_name``.

        Parameters
        ----------
        window_name : str
            Dictionary key for later lookup.
        window_instance : UIWindow
            Arbitrary window object owned by the caller.
        """
        self.windows[window_name] = window_instance


class UIWindow:
    """Represent ``UIWindow`` in Skyline.

    Parameters
    ----------
    title : str
        Title text shown in the UI.
    default_open : bool, optional
        Value for ``default open``.
    flags : int, optional
        Value for ``flags``.
    pos : tuple(int, int), optional
        Value for ``pos``.
    size : tuple(int, int), optional
        Value for ``size``.
    logo_tex_ref : TextureId, optional, optional
        Value for ``logo tex ref``.
    render_callback : callable, optional
        Callback used to request a render/update.
    file_dialog_callback : callable, optional
        Callback invoked after file selection.
    bg_color_callback : callable, optional
        Callback invoked when background color changes.
    """

    def __init__(
        self,
        title,
        *,
        default_open=True,
        flags=0,
        pos=(0, 0),
        size=(400, 400),
        logo_tex_ref=None,
        render_callback=None,
        file_dialog_callback=None,
        bg_color_callback=None,
        snapshot_callback=None,
    ):
        """Represent ``UIWindow`` in Skyline.

        Parameters
        ----------
        title : str
            Title text shown in the UI.
        default_open : bool, optional
            Value for ``default open``.
        flags : int, optional
            Value for ``flags``.
        pos : tuple(int, int), optional
            Value for ``pos``.
        size : tuple(int, int), optional
            Value for ``size``.
        logo_tex_ref : TextureId, optional, optional
            Value for ``logo tex ref``.
        render_callback : callable, optional
            Callback used to request a render/update.
        file_dialog_callback : callable, optional
            Callback invoked after file selection.
        bg_color_callback : callable, optional
            Callback invoked when background color changes.
        snapshot_callback : callable, optional
            Callback invoked when a snapshot path is selected.
        """
        self.title = title
        self.is_open = default_open
        self.flags = flags
        self.pos = pos
        self.size = size
        self._sections = {}
        self._section_open = {}
        self._group_open = {}
        self._group_visible = {}
        self._render_callback = render_callback
        self.logo_tex_ref = logo_tex_ref
        self.logo_size = (32, 32)
        self._title_text = "DIPY SKYLINE"
        self._show_loader = False
        self._loading_message = "Loading..."
        self.render_callback = render_callback
        if render_callback is None or not isinstance(render_callback, Callable):
            self.render_callback = lambda: None
        self.file_dialog_callback = file_dialog_callback
        if file_dialog_callback is None or not isinstance(
            file_dialog_callback, Callable
        ):
            self.file_dialog_callback = lambda: None
        self.bg_color_callback = bg_color_callback
        if bg_color_callback is None or not isinstance(bg_color_callback, Callable):
            self.bg_color_callback = lambda: None
        self.snapshot_callback = snapshot_callback
        if snapshot_callback is None or not isinstance(snapshot_callback, Callable):
            self.snapshot_callback = lambda: None
        self._bg_color = (0.1, 0.1, 0.1)
        self._draft_color = self._bg_color
        self._color_picker_open = False
        self._color_picker_popup_id = f"bg_color_picker_popup##{title}"
        self._is_dialog_open = False

        # Use direct imgui API to avoid font atlas rebuilds that cause
        # texture ID collisions on macOS
        io = imgui.get_io()
        font_size = 16.0
        io.fonts.clear()
        io.fonts.add_font_from_file_ttf(str(FONT), font_size)
        fa_cfg = imgui.ImFontConfig()
        fa_cfg.merge_mode = True
        fa_cfg.pixel_snap_h = True
        fa_cfg.glyph_min_advance_x = font_size
        io.fonts.add_font_from_file_ttf(str(FONT_AWESOME), font_size, fa_cfg)

        self.request_file_dialog = False

    def add(self, name, section_renderer, viz_type=None):
        """Register a visualization section callable and optional grouping type.

        Parameters
        ----------
        name : str
            Unique key, usually ``f"{path}:{display_name}"``.
        section_renderer : callable
            ``renderer(is_open, group_visible=...)`` from :class:`Visualization`.
        viz_type : str or None, optional
            One of the keys in ``_GROUP_ORDER`` used to cluster the sidebar.
        """
        self._sections[name] = (section_renderer, viz_type)
        self._section_open.setdefault(name, False)

    def remove(self, name):
        """Drop a section and its open-state bookkeeping.

        Parameters
        ----------
        name : str
            Key previously passed to :meth:`add`.
        """
        if name in self._sections:
            del self._sections[name]
        if name in self._section_open:
            del self._section_open[name]

    def render(self):
        """Draw the full sidebar for the current frame."""
        imgui.push_style_color(
            imgui.Col_.window_bg, imgui.get_color_u32(THEME["background"])
        )
        if self.pos is not None:
            imgui.set_next_window_pos(self.pos)
        if self.size is not None:
            imgui.set_next_window_size(self.size, imgui.Cond_.always)

        computed_flags = (
            self.flags
            | imgui.WindowFlags_.no_collapse
            | imgui.WindowFlags_.no_resize
            | imgui.WindowFlags_.no_title_bar
            | imgui.WindowFlags_.no_scroll_with_mouse
            | imgui.WindowFlags_.no_scrollbar
        )
        imgui.begin(self.title, None, computed_flags)
        imgui.push_id("logo")
        imgui.push_font(None, 36)

        draw_list = imgui.get_foreground_draw_list()
        spacing = 10
        available_width = imgui.get_content_region_avail().x

        org_start = imgui.get_cursor_screen_pos()
        start = (org_start.x, org_start.y + spacing)

        text_width, text_height = imgui.calc_text_size(self._title_text)

        total_width = self.logo_size[0] + spacing + text_width
        offset_x = max((available_width - total_width) * 0.5, 0.0)
        start = (start[0] + offset_x, start[1])

        draw_list.add_image(
            self.logo_tex_ref,
            start,
            (start[0] + self.logo_size[0], start[1] + self.logo_size[1]),
        )

        title_text_pos = (
            start[0] + self.logo_size[0] + spacing,
            start[1] + (self.logo_size[1] - text_height) * 0.5,
        )
        text_color = imgui.get_color_u32(THEME["text"])
        shadow_color = imgui.get_color_u32(THEME["shadow"])
        draw_list.add_text(title_text_pos, text_color, self._title_text)

        rect_min = (start[0] - spacing, start[1] + self.logo_size[1] + spacing)
        rect_max = (start[0] + total_width + spacing, rect_min[1] + 1)
        draw_list.add_rect_filled(rect_min, rect_max, shadow_color, 0, 0)
        imgui.pop_font()

        file_icon = icons_fontawesome_6.ICON_FA_FILE_CIRCLE_PLUS
        file_icon_size = imgui.calc_text_size(file_icon)
        file_icon_pos = (start[0], rect_max[1] + spacing)
        draw_list.add_text(file_icon_pos, text_color, file_icon)
        imgui.set_cursor_screen_pos(file_icon_pos)
        imgui.invisible_button("add_visualization", file_icon_size)
        if imgui.is_item_hovered():
            imgui.set_item_tooltip("Add Visualization")
        if imgui.is_item_clicked(imgui.MouseButton_.left) or self.request_file_dialog:
            if self.request_file_dialog:
                imgui.set_next_window_pos((self.size[0], 0))
            imgui.open_popup("my_popup")

        if imgui.begin_popup("my_popup"):
            imgui.text("Select Visualization Type")
            imgui.separator()
            if imgui.menu_item("3D/4D Images", "", False)[0]:
                self.request_file_dialog = False
                self._is_dialog_open = True
                render_file_dialog(
                    title="Select Image File(s)",
                    name="Image Files (*.nii *.nii.gz)",
                    extensions="*.nii *.gz",
                    callback=self._file_dialog_closed,
                    type="viz",
                )

            if imgui.menu_item("Peaks", "", False)[0]:
                self.request_file_dialog = False
                self._is_dialog_open = True
                render_file_dialog(
                    title="Select Peak File(s)",
                    name="Peak Files (*.pam5)",
                    extensions="*.pam5",
                    callback=self._file_dialog_closed,
                    type="viz",
                )

            if imgui.menu_item("ODFs", "", False)[0]:
                self.request_file_dialog = False
                self._is_dialog_open = True
                render_file_dialog(
                    title="Select Spherical Harmonics ODFs File(s)",
                    name="ODFs Files (*.pam5)",
                    extensions="*.pam5",
                    callback=self._file_dialog_closed,
                    type="shm_coeff",
                )

            if imgui.menu_item("Surfaces", "", False)[0]:
                self.request_file_dialog = False
                self._is_dialog_open = True
                render_file_dialog(
                    title="Select Surface File(s)",
                    name="Surface Files (*.pial *.gii *.gii.gz)",
                    extensions="*.pial *.gii *.gz",
                    callback=self._file_dialog_closed,
                    type="viz",
                )

            if imgui.menu_item("Tractograms", "", False)[0]:
                self.request_file_dialog = False
                self._is_dialog_open = True
                render_file_dialog(
                    title="Select Tractogram File(s)",
                    name="Tractogram Files (*.trx *.trk *.tck *.fib *.dpy *.vtp *.vtk)",
                    extensions="*.trx *.trk *.tck *.fib *.dpy *.vtp *.vtk",
                    callback=self._file_dialog_closed,
                    type="viz",
                )

            if imgui.menu_item("ROIs", "", False)[0]:
                self.request_file_dialog = False
                self._is_dialog_open = True
                render_file_dialog(
                    title="Select ROI File(s)",
                    name="ROI Files (*.nii *.nii.gz)",
                    extensions="*.nii *.gz",
                    callback=self._file_dialog_closed,
                    type="roi",
                )

            imgui.end_popup()

        imgui.same_line(0, 8)
        selected_color = (
            self._draft_color if self._color_picker_open else self._bg_color
        )
        changed, color, is_open = color_picker(
            selected_color=selected_color,
            tooltip="Change Background Color",
            popup_id=self._color_picker_popup_id,
        )
        if is_open and not self._color_picker_open:
            self._draft_color = self._bg_color
        if changed:
            self._draft_color = color
        if self._color_picker_open and not is_open:
            if not colors_equal(self._draft_color, self._bg_color):
                self._update_bg_color(self._draft_color)
            self._draft_color = self._bg_color
        self._color_picker_open = is_open
        imgui.same_line(0, 8)
        snapshot_icon = icons_fontawesome_6.ICON_FA_CAMERA
        imgui.text_colored(THEME["text"], snapshot_icon)
        if imgui.is_item_hovered():
            imgui.set_item_tooltip("Take Snapshot")
        if imgui.is_item_clicked(imgui.MouseButton_.left):
            render_file_dialog(
                title="Save Snapshot",
                name="PNG Files (*.png)",
                extensions="*.png",
                multiselect=False,
                callback=self._snapshot_dialog_closed,
                dialog_type="save",
                file_name="snapshot.png",
                type="viz",
            )

        imgui.set_cursor_screen_pos(org_start)
        imgui.dummy((available_width, self.logo_size[1] + spacing * 5 + 1))
        imgui.pop_id()

        imgui.begin_child(
            "sections",
            None,
            0,
            imgui.WindowFlags_.no_title_bar
            | imgui.WindowFlags_.no_collapse
            | imgui.WindowFlags_.no_resize,
        )
        names_to_remove = []

        # Group sections by viz_type
        grouped = {t: [] for t in _GROUP_ORDER}
        ungrouped = []
        for name, (renderer, viz_type) in self._sections.items():
            if viz_type in grouped:
                grouped[viz_type].append((name, renderer))
            else:
                ungrouped.append((name, renderer))

        # Render each type group in order
        for viz_type in _GROUP_ORDER:
            items = grouped[viz_type]
            if not items:
                continue
            group_label = _GROUP_LABELS[viz_type]
            imgui.push_id(f"group_{viz_type}")
            group_is_open = self._group_open.get(viz_type, True)
            group_is_visible = self._group_visible.get(viz_type, True)
            group_is_open, group_is_visible, _, _ = render_section_header(
                group_label,
                is_open=group_is_open,
                is_visible=group_is_visible,
                type=viz_type,
                show_close=False,
                show_info=False,
            )
            self._group_open[viz_type] = group_is_open
            self._group_visible[viz_type] = group_is_visible

            if group_is_open:
                child_flags = imgui.ChildFlags_.auto_resize_y
                if imgui.begin_child(f"group_{viz_type}_child", (0, 0), child_flags):
                    for name, renderer in items:
                        imgui.push_id(name)
                        is_open = self._section_open.get(name, False)
                        is_open, is_removed, should_enable_group = renderer(
                            is_open, group_visible=group_is_visible
                        )
                        self._section_open[name] = is_open
                        if should_enable_group:
                            self._group_visible[viz_type] = True
                            group_is_visible = True
                            # Only show the clicked item, hide others in this group
                            for other_name, (
                                other_renderer,
                                other_type,
                            ) in self._sections.items():
                                if other_type == viz_type and other_name != name:
                                    other_renderer.__self__._visible = False
                        if is_removed:
                            names_to_remove.append(name)
                        imgui.pop_id()
                imgui.end_child()
            imgui.pop_id()

        # Render ungrouped items (fallback for unrecognised types)
        for name, renderer in ungrouped:
            imgui.push_id(name)
            is_open = self._section_open.get(name, False)
            is_open, is_removed, _ = renderer(is_open)
            self._section_open[name] = is_open
            if is_removed:
                names_to_remove.append(name)
            imgui.pop_id()

        imgui.end_child()
        for name in names_to_remove:
            self.remove(name)
            self._render_callback()
        imgui.end()
        imgui.pop_style_color(1)

        if self._show_loader and not imgui.is_popup_open("LoadingOverlay"):
            imgui.open_popup("LoadingOverlay")
        loading("LoadingOverlay", self._loading_message, self._show_loader)

    @property
    def sections(self):
        """Map section id to ``(renderer_callable, viz_type)`` tuples."""
        return self._sections

    @property
    def section_open_states(self):
        """Collapsed/open flags for each registered section id."""
        return self._section_open

    def _file_dialog_closed(self, *, filenames=None, rois=None, shm_coeffs=None):
        """Forward dialog results to :attr:`file_dialog_callback` if present.

        Parameters
        ----------
        filenames : list or None, optional
            Selected visualization paths.
        rois : list or None, optional
            Selected ROI paths.
        shm_coeffs : list or None, optional
            Selected SH coefficient paths.
        """
        self._is_dialog_open = False
        if self.file_dialog_callback is not None:
            self.file_dialog_callback(
                filenames=filenames, rois=rois, shm_coeffs=shm_coeffs
            )

    def _update_bg_color(self, new_color):
        """Store the sidebar's background picker color and notify the host scene.

        Parameters
        ----------
        new_color : tuple of float
            RGB triplet in ``[0, 1]``.
        """
        self._bg_color = new_color
        if self.bg_color_callback is not None:
            self.bg_color_callback(new_color)

    def _snapshot_dialog_closed(self, *, filenames=None, rois=None, shm_coeffs=None):
        """Forward snapshot path to :attr:`snapshot_callback` when selected.

        Parameters
        ----------
        filenames : list, str, or None, optional
            Selected save target path(s) from the dialog.
        rois : list or None, optional
            Unused; kept for callback signature compatibility.
        shm_coeffs : list or None, optional
            Unused; kept for callback signature compatibility.
        """
        if self.snapshot_callback is None or filenames is None:
            return
        if isinstance(filenames, str):
            snapshot_path = filenames
        elif len(filenames) > 0:
            snapshot_path = filenames[0]
        else:
            return
        self.snapshot_callback(snapshot_path)

    def update_loader(self, *, show, message=None):
        """Toggle the modal loading overlay.

        Parameters
        ----------
        show : bool
            When True, ensure the loading popup is opened.
        message : str, optional
            User-facing status string.
        """
        self._show_loader = show
        if message is not None:
            self._loading_message = message
