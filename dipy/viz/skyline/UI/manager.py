from imgui_bundle import (
    hello_imgui,
    icons_fontawesome_6,
    imgui,
)

from dipy.viz.skyline.UI.elements import color_picker, render_file_dialog
from dipy.viz.skyline.UI.theme import ASSETS, FONT, THEME


class UIManager:
    def __init__(self):
        self.windows = {}

    def add_window(self, window_name, window_instance):
        self.windows[window_name] = window_instance


class UIWindow:
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
    ):
        self.title = title
        self.is_open = default_open
        self.flags = flags
        self.pos = pos
        self.size = size
        self._sections = {}
        self._section_open = {}
        self._render_callback = render_callback
        self.logo_tex_ref = logo_tex_ref
        self.logo_size = (48, 48)
        self._title_text = "DIPY SKYLINE"
        if render_callback is None:
            self.render_callback = lambda: None
        self.file_dialog_callback = file_dialog_callback
        self.bg_color_callback = bg_color_callback
        self._bg_color = (0.1, 0.1, 0.1)
        self._is_dialog_open = False
        hello_imgui.set_assets_folder(str(ASSETS))
        hello_imgui.load_font_ttf_with_font_awesome_icons(
            str(FONT.relative_to(ASSETS)), 18
        )

        imgui.push_style_color(
            imgui.Col_.window_bg, imgui.get_color_u32(THEME["background"])
        )
        self.request_file_dialog = False

    def add(self, name, section_renderer):
        self._sections[name] = section_renderer
        self._section_open.setdefault(name, False)

    def remove(self, name):
        if name in self._sections:
            del self._sections[name]
        if name in self._section_open:
            del self._section_open[name]

    def render(self):
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
                    callback=self.file_dialog_closed,
                    type="viz",
                )

            if imgui.menu_item("Peaks", "", False)[0]:
                self.request_file_dialog = False
                self._is_dialog_open = True
                render_file_dialog(
                    title="Select Peak File(s)",
                    name="Peak Files (*.pam5)",
                    extensions="*.pam5",
                    callback=self.file_dialog_closed,
                    type="viz",
                )

            if imgui.menu_item("ODFs", "", False)[0]:
                self.request_file_dialog = False
                self._is_dialog_open = True
                render_file_dialog(
                    title="Select Spherical Harmonics ODFs File(s)",
                    name="ODFs Files (*.pam5)",
                    extensions="*.pam5",
                    callback=self.file_dialog_closed,
                    type="shm_coeff",
                )

            if imgui.menu_item("Surfaces", "", False)[0]:
                self.request_file_dialog = False
                self._is_dialog_open = True
                render_file_dialog(
                    title="Select Surface File(s)",
                    name="Surface Files (*.pial *.gii *.gii.gz)",
                    extensions="*.pial *.gii *.gz",
                    callback=self.file_dialog_closed,
                    type="viz",
                )

            if imgui.menu_item("Tractograms", "", False)[0]:
                self.request_file_dialog = False
                self._is_dialog_open = True
                render_file_dialog(
                    title="Select Tractogram File(s)",
                    name="Tractogram Files (*.trx *.trk *.tck *.fib *.dpy *.vtp *.vtk)",
                    extensions="*.trx *.trk *.tck *.fib *.dpy *.vtp *.vtk",
                    callback=self.file_dialog_closed,
                    type="viz",
                )

            if imgui.menu_item("ROIs", "", False)[0]:
                self.request_file_dialog = False
                self._is_dialog_open = True
                render_file_dialog(
                    title="Select ROI File(s)",
                    name="ROI Files (*.nii *.nii.gz)",
                    extensions="*.nii *.gz",
                    callback=self.file_dialog_closed,
                    type="roi",
                )

            imgui.end_popup()

        imgui.same_line(0, 8)
        changed, color = color_picker(
            selected_color=self._bg_color,
            tooltip="Change Background Color",
        )
        if changed:
            self.update_bg_color(color)

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
        is_removed = [False] * len(self._sections)
        for idx, (name, renderer) in enumerate(self._sections.items()):
            imgui.push_id(name)
            is_open = self._section_open.get(name, False)
            is_open, is_removed[idx] = renderer(name, is_open)
            self._section_open[name] = is_open
            imgui.pop_id()
        imgui.end_child()
        for removed, name in zip(is_removed, list(self._sections.keys())):
            if removed:
                self.remove(name)
                self._render_callback()
        imgui.end()

    @property
    def sections(self):
        return self._sections

    @property
    def section_open_states(self):
        return self._section_open

    def file_dialog_closed(self, *, filenames=None, rois=None, shm_coeffs=None):
        self._is_dialog_open = False
        if self.file_dialog_callback is not None:
            self.file_dialog_callback(
                filenames=filenames, rois=rois, shm_coeffs=shm_coeffs
            )

    def update_bg_color(self, new_color):
        self._bg_color = new_color
        if self.bg_color_callback is not None:
            self.bg_color_callback(new_color)
