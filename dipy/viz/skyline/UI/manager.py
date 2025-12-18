from imgui_bundle import imgui


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

    def add(self, name, render_callback):
        self._sections[name] = render_callback

    def remove(self, name):
        if name in self._sections:
            del self._sections[name]

    def render(self):
        if self.pos is not None:
            imgui.set_next_window_pos(
                (self.pos[0], self.pos[1]), imgui.Cond_.first_use_ever
            )
        if self.size is not None:
            imgui.set_next_window_size((self.size[0], self.size[1]))

        open_flag = imgui.begin(self.title, None, self.flags)
        self.is_open = open_flag

        if open_flag:
            for name, renderer in self._sections.items():
                flags = imgui.TreeNodeFlags_.default_open
                opened = imgui.collapsing_header(name, flags=flags)
                if opened:
                    renderer()

                    imgui.separator()
        imgui.end()
