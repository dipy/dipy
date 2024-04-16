from pathlib import Path

from dipy.viz.horizon.tab import HorizonTab, build_checkbox, build_label, build_slider


class SurfaceTab(HorizonTab):
    def __init__(self, visualizer, tab_name, file_name):
        """Surface Tab.

        Parameters
        ----------
        visualizer : SurfaceVisualizer
        id : int
        """
        super().__init__()
        self._visualizer = visualizer
        self._tab_id = 0

        self._name = tab_name
        self._file_name = Path(file_name or tab_name).name

        self._actor_toggle = build_checkbox(
            labels=[''],
            checked_labels=[''],
            on_change=self._toggle_actors)

        self._surface_opacity_label, self._surface_opacity = build_slider(
            initial_value=1.,
            max_value=1.,
            text_template='{ratio:.0%}',
            on_change=self._change_opacity,
            label='Opacity'
        )

        self._file_label = build_label(text='Filename',)
        self._file_name_label = build_label(text=self._file_name)

        self._register_elements(self._actor_toggle,
                                self._surface_opacity_label,
                                self._surface_opacity, self._file_label,
                                self._file_name_label)

    def _change_opacity(self, slider):
        """Change opacity value according to slider changed.

        Parameters
        ----------
        slider : LineSlider2D
        """
        self._surface_opacity.selected_value = slider.value
        self._update_opacities()

    def _update_opacities(self):
        """Update opacities of visible actors based on selected values.
        """
        for actor in self.actors:
            actor.GetProperty().SetOpacity(
                self._surface_opacity.selected_value)

    def build(self, tab_id):
        """Build all the elements under the tab.

        Parameters
        ----------
        tab_id : int
            Id of the tab.
        """
        self._tab_id = tab_id

        y_pos = .85
        self._actor_toggle.position = (.02, y_pos)
        self._surface_opacity_label.position = (.05, y_pos)
        self._surface_opacity.position = (.10, y_pos)

        y_pos = 0.65
        self._file_label.position = (.05, y_pos)
        self._file_name_label.position = (.13, y_pos)

    @property
    def name(self):
        """Name of the tab.

        Returns
        -------
        str
        """
        return self._name

    @property
    def actors(self):
        """Actors controlled by this tab.
        """
        return self._visualizer.actors
