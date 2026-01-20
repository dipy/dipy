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
            self.flags | imgui.WindowFlags_.no_collapse | imgui.WindowFlags_.no_resize
        )
        imgui.begin(self.title, None, computed_flags)
        imgui.push_id("logo")
        imgui.image(self.logo_tex_ref, (64, 64))
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
