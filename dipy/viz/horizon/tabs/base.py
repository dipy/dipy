from dipy.utils.optpkg import optional_package
from dipy.viz.gmem import GlobalHorizon

fury, has_fury, setup_module = optional_package('fury')

if has_fury:
    from fury import ui


class TabManager:
    def __init__(self, size):
        # TODO: Dynamically determine the number of tabs
        # TODO: Handle size dynamically
        num_tabs = 1
        self.tab_ui = ui.TabUI(
            position=(0, 0), size=size, nb_tabs=num_tabs,
            active_color=(1, 1, 1), inactive_color=(0.5, 0.5, 0.5),
            draggable=False
        )
        
        for i in range(num_tabs):
            # TODO: Standardize tabs
            self.tab_ui.tabs[i].title = 'Test'
            # TODO: Loop over the elements in the custom tab
            self.tab_ui.add_element(i, ring_slider, (0.3, 0.3))
