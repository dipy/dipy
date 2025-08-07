from dipy.viz.horizon.tab import HorizonTab, build_checkbox, build_slider


class ROIsTab(HorizonTab):
    def __init__(self, contour_actors):
        """Initialize interaction tab for ROIs visualization.

        Parameters
        ----------
        contour_actors : list
            list of vtkActor.
        """

        super().__init__()

        self._actors = contour_actors
        self._name = "ROIs"

        self._tab_id = 0

        self._actor_toggle = build_checkbox(
            labels=[""], checked_labels=[""], on_change=self._toggle_actors
        )

        self._opacity_slider_label, self._opacity_slider = build_slider(
            initial_value=1,
            max_value=1.0,
            text_template="{ratio:.0%}",
            on_change=self._change_opacity,
            label="Opacity",
        )

        self._register_elements(
            self._actor_toggle, self._opacity_slider_label, self._opacity_slider
        )

    def _change_opacity(self, slider):
        """Change opacity of all ROIs.

        Parameters
        ----------
        slider : LineSlider2D
        """
        opacity = slider.value
        self._opacity_slider.selected_value = slider.value
        for contour in self._actors:
            contour.GetProperty().SetOpacity(opacity)

    def _toggle_actors(self, checkbox):
        """Toggle the opacity of the ROIs actor and slider.

        Parameters
        ----------
        checkbox : CheckBox2D
            FURY checkbox UI element.
        """
        if "" in checkbox.checked_labels:
            self.show(self._opacity_slider.obj)
        else:
            self.hide(self._opacity_slider.obj)

        super()._toggle_actors(checkbox)

    def build(self, tab_id):
        """Position the elements in the tab.

        Parameters
        ----------
        tab_id : int
            Identifier for the tab. Index of the tab in TabUI.
        """

        self._tab_id = tab_id

        y_pos = 0.85
        self._actor_toggle.position = (0.02, y_pos)
        self._opacity_slider_label.position = (0.05, y_pos)
        self._opacity_slider.position = (0.12, y_pos)

    @property
    def name(self):
        """Title of the tab.

        Returns
        -------
        str
        """
        return self._name

    @property
    def actors(self):
        """Actors controlled by tab.

        Returns
        -------
        list
        """
        return self._actors
