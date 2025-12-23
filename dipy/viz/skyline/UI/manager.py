from imgui_bundle import hello_imgui, imgui

from dipy.viz.skyline.UI.elements import render_section_header
from dipy.viz.skyline.UI.theme import FONT, SKYLINE_HOME, THEME


class UIManager:
    def __init__(self):
        self.windows = {}

    def add_window(self, window_name, window_instance):
        self.windows[window_name] = window_instance


class UIWindow:
    def __init__(
        self, title, *, default_open=True, flags=0, pos=(50, 50), size=(400, 400)
    ):
        self.title = title
        self.is_open = default_open
        self.flags = flags
        self.pos = pos
        self.size = size
        self._sections = {}
        self._section_open = {}
        hello_imgui.set_assets_folder(str(SKYLINE_HOME))
        hello_imgui.load_font_ttf_with_font_awesome_icons(
            f"{FONT.parent.name}/{FONT.name}", 20
        )
        imgui.push_style_color(
            imgui.Col_.window_bg, imgui.get_color_u32(THEME["background"])
        )

    def add(self, name, render_callback):
        self._sections[name] = render_callback
        self._section_open.setdefault(name, True)

    def remove(self, name):
        if name in self._sections:
            del self._sections[name]
        if name in self._section_open:
            del self._section_open[name]

    def render(self):
        if self.pos is not None:
            imgui.set_next_window_pos(
                (self.pos[0], self.pos[1]), imgui.Cond_.first_use_ever
            )
        if self.size is not None:
            imgui.set_next_window_size((self.size[0], self.size[1]))

        computed_flags = self.flags | imgui.WindowFlags_.no_collapse
        open_flag = imgui.begin(self.title, None, computed_flags)
        self.is_open = open_flag

        if open_flag:
            for name, renderer in self._sections.items():
                imgui.push_id(name)
                is_open = self._section_open.get(name, True)
                is_open = render_section_header(name, is_open=is_open)
                self._section_open[name] = is_open
                if is_open:
                    renderer()
                    imgui.separator()
                imgui.pop_id()
        imgui.end()
