from dipy.viz.horizon.tab import HorizonTab, build_slider, build_checkbox


class SurfaceTab(HorizonTab):
    def __init__(self, visualizer, surf_id):
        """Surface Tab.

        Parameters
        ----------
        visualizer : SurfaceVisualizer
        id : int
        """
        super().__init__()
        self._visualizer = visualizer
        self._tab_id = 0

        if surf_id == 0:
            self._name = 'Surface'
        else:
            self._name = f'Surface {surf_id}'

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

        self.register_elements(self._opacity_toggle,
                               self._surface_opacity_label,
                               self._surface_opacity)

    def _change_opacity(self, slider):
        self._surface_opacity.selected_value = slider.value
        self._update_opacities()

    def _update_opacities(self):
        for actor in self.actors:
            actor.GetProperty().SetOpacity(
                self._surface_opacity.selected_value)

    def _toggle_opacity(self, checkbox):
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
