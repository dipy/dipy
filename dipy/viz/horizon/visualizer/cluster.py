import warnings

import numpy as np

from dipy.segment.clustering import qbx_and_merge
from dipy.tracking.streamline import Streamlines, length
from dipy.utils.optpkg import optional_package

fury, has_fury, setup_module = optional_package('fury')

if has_fury:
    from fury import actor
    from fury.lib import VTK_OBJECT, calldata_type
    from fury.shaders import add_shader_callback, shader_to_actor


class ClustersVisualizer:
    def __init__(self, show_manager, scene, threshold, enable_callbacks=True):
        
        # TODO: Avoid passing the entire show manager to the visualizer
        self.__show_man = show_manager
        self.__scene = scene
        self.__threshold = threshold
        self.__enable_callbacks = enable_callbacks
        
        self.__tractogram_clusters = {}
        self.__centroid_actors = {}
        self.__cluster_actors = {}
    
    def add_cluster_actors(self, tract_idx, streamlines, colors):
        print(f'Clustering threshold {self.__threshold}')
        clusters = qbx_and_merge(
            streamlines, [40, 30, 25, 20, self.__threshold])
        self.__tractogram_clusters[tract_idx] = clusters
        centroids = clusters.centroids
        print(f'Total number of centroids = {len(centroids)}')
        sizes = np.array([len(c) for c in clusters])
        linewidths = np.interp(
            sizes, [np.min(sizes), np.max(sizes)], [0.1, 2.])
        centroid_lengths = np.array([length(c) for c in centroids])

        print(f'Minimum number of streamlines in cluster {np.min(sizes)}')
        print(f'Maximum number of streamlines in cluster {np.max(sizes)}')

        print('Building cluster actors')
        for idx, cent in enumerate(centroids):

            centroid_actor = actor.streamtube(
                [cent], colors, linewidth=linewidths[idx], lod=False)
            self.__scene.add(centroid_actor)

            cluster_actor = actor.line(clusters[idx][:], lod=False)
            cluster_actor.GetProperty().SetRenderLinesAsTubes(1)
            cluster_actor.GetProperty().SetLineWidth(6)
            cluster_actor.GetProperty().SetOpacity(1)
            cluster_actor.VisibilityOff()
            self.__scene.add(cluster_actor)

            # Every centroid actor is paired to a cluster actor
            self.__centroid_actors[centroid_actor] = {
                'actor': cluster_actor, 'cluster': idx,
                'tractogram': tract_idx, 'size': sizes[idx],
                'length': centroid_lengths[idx], 'selected': 0,
                'expanded': 0}

            self.__cluster_actors[cluster_actor] = {
                'actor': centroid_actor, 'cluster': idx,
                'tractogram': tract_idx, 'size': sizes[idx],
                'length': centroid_lengths[idx], 'selected': 0,
                'highlighted': 0}
            
            self.__apply_shader(self.__centroid_actors[centroid_actor])
            self.__apply_shader(self.__cluster_actors[cluster_actor])
            
            if self.__enable_callbacks:
                centroid_actor.AddObserver(
                    'LeftButtonPressEvent',
                    self.__left_click_centroid_callback, 1.)
                cluster_actor.AddObserver(
                    'LeftButtonPressEvent',
                    self.__left_click_cluster_callback, 1.)
    
    def __apply_shader(self, dict_element):
        decl = \
            """
            uniform float selected;
            """

        impl = \
            """
            if (selected == 1)
            {
                fragOutput0 += vec4(.2, .2, .2, 0);
            }
            """
        shader_to_actor(dict_element['actor'], 'fragment', decl_code=decl)
        shader_to_actor(
            dict_element['actor'], 'fragment', impl_code=impl, block='light')
        
        @calldata_type(VTK_OBJECT)
        def uniform_selected_callback(caller, event, calldata=None):
            program = calldata
            if program is not None:
                program.SetUniformf('selected', dict_element['selected'])
        
        add_shader_callback(
            dict_element['actor'], uniform_selected_callback, priority=100)
    
    def __left_click_centroid_callback(self, obj, event):
        self.__centroid_actors[obj]['selected'] = (
            not self.__centroid_actors[obj]['selected'])
        self.__cluster_actors[self.__centroid_actors[obj]['actor']][
            'selected'] = self.__centroid_actors[obj]['selected']
        # TODO: Find another way to rerender
        self.__show_man.render()

    def __left_click_cluster_callback(self, obj, event):
        if self.__cluster_actors[obj]['selected']:
            self.__cluster_actors[obj]['actor'].VisibilityOn()
            ca = self.__cluster_actors[obj]['actor']
            self.__centroid_actors[ca]['selected'] = 0
            obj.VisibilityOff()
            self.__centroid_actors[ca]['expanded'] = 0
        # TODO: Find another way to rerender
        self.__show_man.render()
    
    @property
    def centroid_actors(self):
        return self.__centroid_actors
    
    @property
    def cluster_actors(self):
        return self.__cluster_actors