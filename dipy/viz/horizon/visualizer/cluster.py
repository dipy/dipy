import warnings

import numpy as np

from dipy.segment.clustering import qbx_and_merge
from dipy.tracking.streamline import Streamlines, length
from dipy.utils.optpkg import optional_package

fury, has_fury, setup_module = optional_package('fury')

if has_fury:
    from fury import actor


class ClustersVisualizer:
    def __init__(
        self, interactor, scene, tractograms, random_colors, color_generator,
        world_coords, cluster, threshold, enable_callbacks=True):
        color_ind = 0
        
        self.__tractogram_clusters = {}
        self.__cea = {}  # holds centroid actors
        self.__cla = {}  # holds cluster actors
        
        for (t, sft) in enumerate(tractograms):
            streamlines = sft.streamlines

            if 'tracts' in random_colors:
                colors = next(color_generator)
            else:
                colors = None

            if not world_coords:
                # TODO we need to read the affine of a tractogram
                # from a StatefullTractogram
                msg = 'Currently native coordinates are not supported'
                msg += ' for streamlines'
                raise ValueError(msg)

            if cluster:
                print(' Clustering threshold {} \n'.format(threshold))
                clusters = qbx_and_merge(
                    streamlines, [40, 30, 25, 20, threshold])
                self.__tractogram_clusters[t] = clusters
                centroids = clusters.centroids
                print(' Number of centroids is {}'.format(len(centroids)))
                sizes = np.array([len(c) for c in clusters])
                linewidths = np.interp(
                    sizes, [sizes.min(), sizes.max()], [0.1, 2.])
                centroid_lengths = np.array([length(c) for c in centroids])

                print(' Minimum number of streamlines in cluster {}'
                      .format(sizes.min()))

                print(' Maximum number of streamlines in cluster {}'
                      .format(sizes.max()))

                print(' Construct cluster actors')
                for (i, c) in enumerate(centroids):

                    centroid_actor = actor.streamtube(
                        [c], colors, linewidth=linewidths[i], lod=False)
                    scene.add(centroid_actor)

                    cluster_actor = actor.line(clusters[i][:], lod=False)
                    cluster_actor.GetProperty().SetRenderLinesAsTubes(1)
                    cluster_actor.GetProperty().SetLineWidth(6)
                    cluster_actor.GetProperty().SetOpacity(1)
                    cluster_actor.VisibilityOff()

                    scene.add(cluster_actor)

                    # Every centroid actor (cea) is paired to a cluster actor
                    # (cla).

                    self.__cea[centroid_actor] = {
                        'cluster_actor': cluster_actor,
                        'cluster': i, 'tractogram': t,
                        'size': sizes[i], 'length': centroid_lengths[i],
                        'selected': 0, 'expanded': 0}

                    self.__cla[cluster_actor] = {
                        'centroid_actor': centroid_actor,
                        'cluster': i, 'tractogram': t,
                        'size': sizes[i], 'length': centroid_lengths[i],
                        'selected': 0, 'highlighted': 0}
                    #apply_shader(self, cluster_actor)
                    #apply_shader(self, centroid_actor)
            else:
                # TODO: Add BUAN support
                """
                s_colors = self.buan_colors[color_ind] if self.buan else colors
                streamline_actor = actor.line(streamlines, colors=s_colors)

                streamline_actor.GetProperty().SetEdgeVisibility(1)
                streamline_actor.GetProperty().SetRenderLinesAsTubes(1)
                streamline_actor.GetProperty().SetLineWidth(6)
                streamline_actor.GetProperty().SetOpacity(1)
                scene.add(streamline_actor)
                """
                pass

            color_ind += 1

        if not enable_callbacks:
            return
