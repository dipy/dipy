from dipy.utils.optpkg import optional_package
from dipy.viz.horizon.tab import HorizonTab, build_label, color_slider

fury, has_fury, setup_module = optional_package('fury')

if has_fury:
    from fury import ui


class RoIsTab(HorizonTab):
    def __init__(self, contour_actors):
        self.__actors = contour_actors
        self.__name = 'RoIs'
        
        self.__tab_id = 0
        self.__tab_ui = None
        
        self.__panel = None
        
        self.__slider_label_opacity = build_label(text="Opacity")
        
        opacity = 1
        
        self.__slider_opacity = ui.LineSlider2D(
            min_value=.0, max_value=1., initial_value=opacity, length=140, 
            text_template="{ratio:.0%}")
        
        color_slider(self.__slider_opacity)
        
        self.__slider_opacity.on_change = self.__change_opacity
    
    def __change_opacity(self, slider):
        opacity = slider.value
        for contour in self.__actors:
            contour.GetProperty().SetOpacity(opacity)
    
    def build(self, tab_id, tab_ui, tab_ui_size):
        self.__tab_id = tab_id
        self.__tab_ui = tab_ui
        
        tab_ui_width, tab_ui_height = tab_ui_size
        panel_size = (tab_ui_width - 10, (tab_ui_height * .9) - 10)
        
        self.__panel = ui.Panel2D(
            panel_size, color=(1, 1, 1), opacity=.1, align='center')
        
        self.__panel.add_element(
            self.__slider_label_opacity, coords=(0.1, 0.3))
        self.__panel.add_element(
            self.__slider_opacity, coords=(0.42, 0.3))
        
        self.__tab_ui.add_element(
            self.__tab_id, self.__panel, (5, 5))
    
    @property
    def name(self):
        return self.__name
