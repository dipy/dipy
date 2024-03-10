from pathlib import Path

from dipy.viz.horizon.tab import (HorizonTab, build_slider, build_checkbox,
                                  build_label)


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

        self._opacity_toggle = build_checkbox(
            labels=[''],
            checked_labels=[''],
            on_change=self._toggle_opacity)

        self._surface_opacity_label, self._surface_opacity = build_slider(
            initial_value=1.,
            max_value=1.,
            text_template='{ratio:.0%}',
            on_change=self._change_opacity,
            label='Opacity'
        )

        self._file_label = build_label(text='Filename', is_horizon_label=True)
        self._file_name_label = build_label(text=self._file_name,
                                            is_horizon_label=True)

        self.register_elements(self._opacity_toggle,
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

    def _toggle_opacity(self, checkbox):
        """Toggle opacity of the actor to 0% or 100%.

        Parameters
        ----------
        checkbox : _type_
            _description_
        """
        if '' in checkbox.checked_labels:
            self._surface_opacity.selected_value = 1
            self._surface_opacity.obj.value = 1
        else:
            self._surface_opacity.selected_value = 0
            self._surface_opacity.obj.value = 0
        self._update_opacities()

    def build(self, tab_id, _tab_ui):
        """Build all the elements under the tab.

        Parameters
        ----------
        tab_id : int
            Id of the tab.
        tab_ui : TabUI
            FURY TabUI object for tabs panel.

        Notes
        -----
        tab_ui will removed once every all tabs adapt new build architecture.
        """
        self._tab_id = tab_id

        y_pos = .85
        self._opacity_toggle.position = (.02, y_pos)
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
    def tab_id(self):
        """Id of the tab.
        """
        return self._tab_id

    @property
    def actors(self):
        """Actors controlled by this tab.
        """
        return self._visualizer.actors
