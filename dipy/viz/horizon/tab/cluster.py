import numpy as np

from dipy.utils.optpkg import optional_package
from dipy.viz.horizon.tab import HorizonTab, build_label, color_single_slider

fury, has_fury, setup_module = optional_package('fury')

if has_fury:
    from fury import ui


class ClustersTab(HorizonTab):
    def __init__(
        self, centroid_actors, cluster_actors, threshold, sizes, lengths):
        self.__centroid_actors = centroid_actors
        self.__cluster_actors = cluster_actors
        self.__name = 'Clusters'
        
        self.__tab_id = 0
        self.__tab_ui = None
        
        length = 450
        lw = 3
        radius = 8
        fs = 16
        tt = '{value:.0f}'
        
        self.__slider_label_size = build_label(text='Size')
        
        self.__slider_size = ui.LineSlider2D(
            initial_value=np.percentile(sizes, 50), min_value=np.min(sizes),
            max_value=np.percentile(sizes, 98), length=length, line_width=lw,
            outer_radius=radius, font_size=fs, text_template=tt)
        
        color_single_slider(self.__slider_size)
        
        self.__slider_label_length = build_label(text='Length')
        
        self.__slider_length = ui.LineSlider2D(
            initial_value=np.percentile(lengths, 25),
            min_value=np.min(lengths), max_value=np.percentile(lengths, 98),
            length=length, line_width=lw, outer_radius=radius, font_size=fs,
            text_template=tt)
        
        color_single_slider(self.__slider_length)
        
        self.__slider_label_threshold = build_label(text='Threshold')
        
        self.__slider_threshold = ui.LineSlider2D(
            initial_value=threshold, min_value=5, max_value=25, length=length,
            line_width=lw, outer_radius=radius, font_size=fs, text_template=tt)
        
        color_single_slider(self.__slider_threshold)
    
    def build(self, tab_id, tab_ui):
        self.__tab_id = tab_id
        self.__tab_ui = tab_ui
        
        x_pos = .02
        
        self.__tab_ui.add_element(
            self.__tab_id, self.__slider_label_size, (x_pos, .85))
        self.__tab_ui.add_element(
            self.__tab_id, self.__slider_label_length, (x_pos, .62))
        self.__tab_ui.add_element(
            self.__tab_id, self.__slider_label_threshold, (x_pos, .38))
        
        x_pos=.12
        
        self.__tab_ui.add_element(
            self.__tab_id, self.__slider_size, (x_pos, .85))
        self.__tab_ui.add_element(
            self.__tab_id, self.__slider_length, (x_pos, .62))
        self.__tab_ui.add_element(
            self.__tab_id, self.__slider_threshold, (x_pos, .38))
    
    @property
    def name(self):
        return self.__name
