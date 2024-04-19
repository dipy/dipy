import numpy as np

from dipy.viz.horizon.tab import HorizonTab, build_slider


class ClustersTab(HorizonTab):
    def __init__(self, clusters_visualizer, threshold):
        """Initialize Interaction tab for cluster visualization.

        Parameters
        ----------
        clusters_visualizer : ClusterVisualizer
        threshold : float
        """

        super().__init__()

        self._visualizer = clusters_visualizer

        self._name = 'Clusters'

        self._tab_id = 0

        sizes = self._visualizer.sizes
        self._size_slider_label, self._size_slider = build_slider(
            initial_value=np.percentile(sizes, 50), min_value=np.min(sizes),
            max_value=np.percentile(sizes, 98), text_template='{value:.0f}',
            label='Size', on_change=self._change_size
        )

        lengths = self._visualizer.lengths
        self._length_slider_label, self._length_slider = build_slider(
            initial_value=np.percentile(lengths, 25),
            min_value=np.min(lengths), max_value=np.percentile(lengths, 98),
            text_template='{value:.0f}', label='Length',
            on_change=self._change_length
        )

        self._threshold_slider_label, self._threshold_slider = build_slider(
            initial_value=threshold, min_value=5, max_value=25,
            text_template='{value:.0f}', label='Threshold',
            on_handle_released=self._change_threshold
        )

        self._register_elements(
            self._size_slider_label, self._size_slider,
            self._length_slider_label, self._length_slider,
            self._threshold_slider_label, self._threshold_slider
        )

    def _change_length(self, slider):
        """Change the length threshold for visibility.

        Parameters
        ----------
        slider : LineSlider2D
            FURY object for slider.
        """
        self._length_slider.selected_value = int(np.rint(slider.value))
        self._update_clusters()

    def _change_size(self, slider):
        """Change the size threshold for visibility.

        Parameters
        ----------
        slider : LineSlider2D
            FURY object for slider.
        """
        self._size_slider.selected_value = int(np.rint(slider.value))
        self._update_clusters()

    def _change_threshold(self, _istyle, _obj, slider):
        """Re-cluster the tractograms according to the new threshold set. It
        will also update the size and length slider for corresponding
        threshold.

        Parameters
        ----------
        _istyle : vtkInteractor
            Should not be used.
        _obj : vtkObject
            Should not be used.
        slider : LineSlider2D
            FURY object for slider.
        """
        value = int(np.rint(slider.value))
        if value != self._threshold_slider.selected_value:
            self._visualizer.recluster_tractograms(value)

            sizes = self._visualizer.sizes
            self._size_slider.selected_value = np.percentile(sizes, 50)

            lengths = self._visualizer.lengths
            self._length_slider.selected_value = np.percentile(lengths, 25)

            self._update_clusters()

            self._size_slider.obj.min_value = np.min(sizes)
            self._size_slider.obj.max_value = np.percentile(sizes, 98)
            self._size_slider.obj.value = self._size_slider.selected_value
            self._size_slider.obj.update()

            self._length_slider.obj.min_value = np.min(lengths)
            self._length_slider.obj.max_value = np.percentile(lengths, 98)
            self._length_slider.obj.value = self._length_slider.selected_value
            self._length_slider.obj.update()

            self._threshold_slider.selected_value = value

    def _update_clusters(self):
        """Updates the clusters according to set size and length.
        """
        for k, cluster in self.cluster_actors.items():

            length_validation = (
                cluster['length'] < self._length_slider.selected_value)

            size_validation = (
                cluster['size'] < self._size_slider.selected_value)

            if (length_validation or size_validation):
                cluster['actor'].SetVisibility(False)

                if k.GetVisibility():
                    k.SetVisibility(False)
            else:
                cluster['actor'].SetVisibility(True)

    def build(self, tab_id):
        """Position the elements in the tab.

        Parameters
        ----------
        tab_id : int
            Id of the tab.
        """
        self._tab_id = tab_id

        x_pos = .02

        self._size_slider_label.position = (x_pos, .85)
        self._length_slider_label.position = (x_pos, .62)
        self._threshold_slider_label.position = (x_pos, .38)

        x_pos = .12

        self._size_slider.position = (x_pos, .85)
        self._length_slider.position = (x_pos, .62)
        self._threshold_slider.position = (x_pos, .38)

    @property
    def name(self):
        """Title of the tab.

        Returns
        -------
        str
        """
        return self._name

    @property
    def cluster_actors(self):
        """Cluster actors of the tractograms.

        Returns
        -------
        dict
            various properties of clusters.
        """
        return self._visualizer.cluster_actors

    @property
    def centroid_actors(self):
        """Centroid actors of the tractograms.

        Returns
        -------
        dict
            various properties of centroids.
        """
        return self._visualizer.centroid_actors

    @property
    def actors(self):
        """All the actors in the visualizer.

        Returns
        -------
        list
        """
        actors = []
        for cluster_actor in self.cluster_actors.values():
            actors.append(cluster_actor['actor'])
        for centroid_actor in self.centroid_actors.values():
            actors.append(centroid_actor['actor'])
        return actors
