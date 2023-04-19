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
    def __init__(
        self, scene, tractograms, world_coords, threshold,
        color_generator=None, enable_callbacks=True):
        
        self.__scene = scene
        self.__tractograms = tractograms
        self.__color_generator = color_generator
        self.__world_coords = world_coords
        self.__threshold = threshold
        self.__enable_callbacks = enable_callbacks
        
        self.__tractogram_clusters = {}
        self.__centroid_actors = {}
        self.__cluster_actors = {}
        
        self.__add_cluster_actors()
    
    def __add_cluster_actors(self):
        for t, sft in enumerate(self.__tractograms):
            streamlines = sft.streamlines
            
            if self.__color_generator:
                colors = next(self.__color_generator)
            else:
                colors = None

            if not self.__world_coords:
                # TODO: Get affine from a StatefullTractogram
                raise ValueError(
                    'Currently native coordinates are not supported for'
                    'streamlines.')

            # NOTE: Cluster related stuff
            print(f'Clustering threshold {self.__threshold}')
            clusters = qbx_and_merge(
                streamlines, [40, 30, 25, 20, self.__threshold])
            self.__tractogram_clusters[t] = clusters
            centroids = clusters.centroids
            print(f'Total number of centroids = {len(centroids)}')
            sizes = np.array([len(c) for c in clusters])
            linewidths = np.interp(
                sizes, [sizes.min(), sizes.max()], [0.1, 2.])
            centroid_lengths = np.array([length(c) for c in centroids])

            print(f'Minimum number of streamlines in cluster {np.min(sizes)}')
            print(f'Maximum number of streamlines in cluster {np.max(sizes)}')

            print('Building cluster actors')
            for i, c in enumerate(centroids):

                centroid_actor = actor.streamtube(
                    [c], colors, linewidth=linewidths[i], lod=False)
                self.__scene.add(centroid_actor)

                cluster_actor = actor.line(clusters[i][:], lod=False)
                cluster_actor.GetProperty().SetRenderLinesAsTubes(1)
                cluster_actor.GetProperty().SetLineWidth(6)
                cluster_actor.GetProperty().SetOpacity(1)
                cluster_actor.VisibilityOff()
                self.__scene.add(cluster_actor)

                # Every centroid actor is paired to a cluster actor
                self.__centroid_actors[centroid_actor] = {
                    'cluster_actor': cluster_actor, 'cluster': i,
                    'tractogram': t, 'size': sizes[i],
                    'length': centroid_lengths[i], 'selected': 0,
                    'expanded': 0}

                self.__cluster_actors[cluster_actor] = {
                    'centroid_actor': centroid_actor, 'cluster': i,
                    'tractogram': t, 'size': sizes[i],
                    'length': centroid_lengths[i], 'selected': 0,
                    'highlighted': 0}
                
                self.__apply_shader(centroid_actor)
                self.__apply_shader(cluster_actor)
                
                if self.__enable_callbacks:
                    centroid_actor.AddObserver(
                        'LeftButtonPressEvent',
                        self.__left_click_centroid_callback, 1.)
                    cluster_actor.AddObserver(
                        'LeftButtonPressEvent',
                        self.__left_click_cluster_callback, 1.)
    
    def __apply_shader(self, act):
        decl = \
            """
            uniform float selected;
            uniform float opacityLevel;
            """

        impl = \
            """
            if (selected == 1)
            {
                fragOutput0 += vec4(0.2, 0.2, 0, opacityLevel);
            }
            """
        shader_to_actor(act, 'fragment', decl_code=decl)
        shader_to_actor(act, 'fragment', impl_code=impl, block='light')
        
        @calldata_type(VTK_OBJECT)
        def uniform_selected_callback(self, caller, event, calldata=None):
            program = calldata
            if program is not None:
                try:
                    program.SetUniformf(
                        'selected', self.__centroid_actors[act]['selected'])
                except KeyError:
                    pass
                try:
                    program.SetUniformf(
                        'selected', self.__cluster_actors[act]['selected'])
                except KeyError:
                    pass
                program.SetUniformf('opacityLevel', 1)
        
        add_shader_callback(act, uniform_selected_callback, priority=100)
    
    def __left_click_centroid_callback(self, obj, event):
        self.__centroid_actors[obj]['selected'] = (
            not self.__centroid_actors[obj]['selected'])
        self.__cluster_actors[self.__centroid_actors[obj]['cluster_actor']][
            'selected'] = self.__centroid_actors[obj]['selected']
        # TODO: Rerender
        #self.show_m.render()

    def __left_click_cluster_callback(self, obj, event):
        if self.__cluster_actors[obj]['selected']:
            self.__cluster_actors[obj]['centroid_actor'].VisibilityOn()
            ca = self.__cluster_actors[obj]['centroid_actor']
            self.__centroid_actors[ca]['selected'] = 0
            obj.VisibilityOff()
            self.__centroid_actors[ca]['expanded'] = 0
        # TODO: Rerender
        #self.show_m.render()
    
    @property
    def centroid_actors(self):
        return self.__centroid_actors
    
    @property
    def cluster_actors(self):
        return self.__cluster_actors
