from imgui_bundle import hello_imgui, imgui

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

        hello_imgui.set_assets_folder(str(ASSETS))
        hello_imgui.load_font_ttf_with_font_awesome_icons(
            str(FONT.relative_to(ASSETS)), 18
        )

        imgui.push_style_color(
            imgui.Col_.window_bg, imgui.get_color_u32(THEME["background"])
        )

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
        draw_list.add_text(title_text_pos, text_color, self._title_text)

        rect_min = (start[0] - spacing, start[1] + self.logo_size[1] + spacing)
        rect_max = (start[0] + total_width + spacing, rect_min[1] + 1)
        draw_list.add_rect_filled(rect_min, rect_max, text_color, 0, 0)

        imgui.set_cursor_screen_pos(org_start)
        imgui.dummy((available_width, self.logo_size[1] + spacing * 3 + 1))
        imgui.pop_font()
        imgui.pop_id()
        is_removed = [False] * len(self._sections)
        for idx, (name, renderer) in enumerate(self._sections.items()):
            imgui.push_id(name)
            is_open = self._section_open.get(name, False)
            is_open, is_removed[idx] = renderer(name, is_open)
            self._section_open[name] = is_open
            imgui.pop_id()

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
