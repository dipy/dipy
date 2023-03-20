from dipy.utils.optpkg import optional_package
from dipy.viz.horizon.tab import HorizonTab, build_label, color_slider

fury, has_fury, setup_module = optional_package('fury')

if has_fury:
    from fury import ui


class ROIsTab(HorizonTab):
    def __init__(self, contour_actors):
        self.__actors = contour_actors
        self.__name = 'ROIs'
        
        self.__tab_id = 0
        self.__tab_ui = None
        
        self.__slider_label_opacity = build_label(text='Opacity')
        
        opacity = 1
        
        length = 185
        lw = 3
        radius = 8
        fs = 16
        tt = '{ratio:.0%}'
        
        self.__slider_opacity = ui.LineSlider2D(
            initial_value=opacity, min_value=.0, max_value=1., length=length,
            line_width=lw, outer_radius=radius, font_size=fs, text_template=tt)
        
        color_slider(self.__slider_opacity)
        
        self.__slider_opacity.on_change = self.__change_opacity
    
    def __change_opacity(self, slider):
        opacity = slider.value
        for contour in self.__actors:
            contour.GetProperty().SetOpacity(opacity)
    
    def build(self, tab_id, tab_ui):
        self.__tab_id = tab_id
        self.__tab_ui = tab_ui
        
        self.__tab_ui.add_element(
            self.__tab_id, self.__slider_label_opacity, coords=(0.1, 0.3))
        self.__tab_ui.add_element(
            self.__tab_id, self.__slider_opacity, coords=(0.42, 0.3))
    
    @property
    def name(self):
        return self.__name
