import numpy as np

from dipy.utils.optpkg import optional_package
from dipy.viz.horizon.tab import (HorizonTab, build_label, color_single_slider)

fury, has_fury, setup_module = optional_package('fury', min_version="0.10.0")

if has_fury:
    from fury import ui


class ClustersTab(HorizonTab):
    def __init__(self, clusters_visualizer, threshold):
        self.__visualizer = clusters_visualizer

        self.__centroid_actors = self.__visualizer.centroid_actors
        self.__cluster_actors = self.__visualizer.cluster_actors
        self.__name = 'Clusters'

        self.__tab_id = 0
        self.__tab_ui = None

        length = 450
        lw = 3
        radius = 8
        fs = 16
        tt = '{value:.0f}'

        self.__slider_label_size = build_label(text='Size')

        sizes = self.__visualizer.sizes
        self.__selected_size = np.percentile(sizes, 50)

        self.__slider_size = ui.LineSlider2D(
            initial_value=self.__selected_size, min_value=np.min(sizes),
            max_value=np.percentile(sizes, 98), length=length, line_width=lw,
            outer_radius=radius, font_size=fs, text_template=tt)

        color_single_slider(self.__slider_size)

        self.__slider_size.handle_events(self.__slider_size.handle.actor)
        self.__slider_size.on_left_mouse_button_released = self.__change_size

        self.__slider_label_length = build_label(text='Length')

        lengths = self.__visualizer.lengths
        self.__selected_length = np.percentile(lengths, 25)

        self.__slider_length = ui.LineSlider2D(
            initial_value=self.__selected_length, min_value=np.min(lengths),
            max_value=np.percentile(lengths, 98), length=length, line_width=lw,
            outer_radius=radius, font_size=fs, text_template=tt)

        color_single_slider(self.__slider_length)

        self.__slider_length.handle_events(self.__slider_length.handle.actor)
        self.__slider_length.on_left_mouse_button_released = (
            self.__change_length)

        self.__slider_label_threshold = build_label(text='Threshold')

        self.__selected_threshold = threshold

        self.__slider_threshold = ui.LineSlider2D(
            initial_value=self.__selected_threshold, min_value=5, max_value=25,
            length=length, line_width=lw, outer_radius=radius, font_size=fs,
            text_template=tt)

        color_single_slider(self.__slider_threshold)

        self.__slider_threshold.handle_events(
            self.__slider_threshold.handle.actor)
        self.__slider_threshold.on_left_mouse_button_released = (
            self.__change_threshold)

    def __change_length(self, istyle, obj, slider):
        self.__selected_length = int(np.rint(slider.value))
        self.__update_clusters()
        istyle.force_render()

    def __change_size(self, istyle, obj, slider):
        self.__selected_size = int(np.rint(slider.value))
        self.__update_clusters()
        istyle.force_render()

    def __change_threshold(self, istyle, obj, slider):
        value = int(np.rint(slider.value))
        if value != self.__selected_threshold:
            self.__visualizer.recluster_tractograms(value)

            sizes = self.__visualizer.sizes
            self.__selected_size = np.percentile(sizes, 50)

            lengths = self.__visualizer.lengths
            self.__selected_length = np.percentile(lengths, 25)

            self.__update_clusters()

            # Updating size slider
            self.__slider_size.min_value = np.min(sizes)
            self.__slider_size.max_value = np.percentile(sizes, 98)
            self.__slider_size.value = self.__selected_size
            self.__slider_size.update()

            # Updating length slider
            self.__slider_length.min_value = np.min(lengths)
            self.__slider_length.max_value = np.percentile(lengths, 98)
            self.__slider_length.value = self.__selected_length
            self.__slider_length.update()

            self.__selected_threshold = value
            istyle.force_render()

    def __update_clusters(self):
        for k in self.__cluster_actors:
            length_validation = (
                self.__cluster_actors[k]['length'] < self.__selected_length)
            size_validation = (
                self.__cluster_actors[k]['size'] < self.__selected_size)
            if (length_validation or size_validation):
                self.__cluster_actors[k]['actor'].SetVisibility(False)
                if k.GetVisibility():
                    k.SetVisibility(False)
            else:
                self.__cluster_actors[k]['actor'].SetVisibility(True)

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
