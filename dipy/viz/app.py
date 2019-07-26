import numpy as np
from dipy.segment.clustering import qbx_and_merge
from dipy.tracking.streamline import length, Streamlines
from dipy.io.stateful_tractogram import Space, StatefulTractogram
from dipy.io.streamline import save_tractogram
from dipy.utils.optpkg import optional_package

fury, has_fury, setup_module = optional_package('fury')

if has_fury:
    from dipy.viz import actor, window, ui
    from dipy.viz import vtk
    from dipy.viz.panel import slicer_panel, build_label, _color_slider
    from dipy.viz.gmem import HORIZON


def apply_shader(hz, actor):

    gl_mapper = actor.GetMapper()

    gl_mapper.AddShaderReplacement(
        vtk.vtkShader.Vertex,
        "//VTK::ValuePass::Impl",  # replace the normal block
        False,
        "//VTK::ValuePass::Impl\n",  # we still want the default
        False)

    gl_mapper.AddShaderReplacement(
        vtk.vtkShader.Fragment,
        "//VTK::Light::Impl",
        True,
        "//VTK::Light::Impl\n"
        "if (selected == 1){\n"
        " fragOutput0 = fragOutput0 + vec4(0.2, 0.2, 0, opacity_level);\n"
        "}\n",
        False)

    gl_mapper.AddShaderReplacement(
        vtk.vtkShader.Fragment,
        "//VTK::Coincident::Dec",
        True,
        "//VTK::Coincident::Dec\n"
        "uniform float selected;\n"
        "uniform float opacity_level;\n",
        False)

    @window.vtk.calldata_type(window.vtk.VTK_OBJECT)
    def vtk_shader_callback(caller, event, calldata=None):
        program = calldata
        if program is not None:
            try:
                program.SetUniformf("selected",
                                    hz.cea[actor]['selected'])
            except KeyError:
                pass
            try:
                program.SetUniformf("selected",
                                    hz.cla[actor]['selected'])
            except KeyError:
                pass
            program.SetUniformf("opacity_level", 1)

    gl_mapper.AddObserver(window.vtk.vtkCommand.UpdateShaderEvent,
                          vtk_shader_callback)


HELP_MESSAGE = """
>> left click: select centroid
>> e: expand centroids
>> r: collapse open clusters
>> h: hide unselected centroids
>> i: invert selection
>> a: select all centroids
>> s: save in file
"""


class Horizon(object):

    def __init__(self, tractograms=[], images=[], pams=[],
                 cluster=False, cluster_thr=15.0,
                 random_colors=False, length_gt=0, length_lt=1000,
                 clusters_gt=0, clusters_lt=10000,
                 world_coords=True, interactive=True,
                 out_png='tmp.png'):
        """ Highly interactive visualization - invert the Horizon!

        Parameters
        ----------
        tractograms : sequence
            Sequence of Streamlines objects
        images : sequence of tuples
            Each tuple contains data and affine
        pams : PeakAndMetrics files
        cluster : bool
            Enable QuickBundlesX clustering
        cluster_thr : float
            Distance threshold used for clustering
        random_colors : bool
        length_gt : float
        length_lt : float
        clusters_gt : int
        clusters_lt : int
        world_coords : bool
        interactive : bool
        out_png : string

        References
        ----------
        .. [Horizon_ISMRM19] Garyfallidis E., M-A. Cote, B.Q. Chandio,
            S. Fadnavis, J. Guaje, R. Aggarwal, E. St-Onge, K.S. Juneja,
            S. Koudoro, D. Reagan, DIPY Horizon: fast, modular, unified and
            adaptive visualization, Proceedings of: International Society of
            Magnetic Resonance in Medicine (ISMRM), Montreal, Canada, 2019.
        """

        self.cluster = cluster
        self.cluster_thr = cluster_thr
        self.random_colors = random_colors
        self.length_lt = length_lt
        self.length_gt = length_gt
        self.clusters_lt = clusters_lt
        self.clusters_gt = clusters_gt
        self.world_coords = world_coords
        self.interactive = interactive
        self.prng = np.random.RandomState(27)
        self.tractograms = tractograms
        self.out_png = out_png
        self.images = images or []
        self.pams = pams or []

        self.cea = {}  # holds centroid actors
        self.cla = {}  # holds cluster actors
        self.tractogram_clusters = {}

    def build_scene(self):

        scene = window.Renderer()
        for (t, streamlines) in enumerate(self.tractograms):
            if self.random_colors:
                colors = self.prng.random_sample(3)
            else:
                colors = None

            if self.cluster:

                print(' Clustering threshold {} \n'.format(self.cluster_thr))
                clusters = qbx_and_merge(streamlines,
                                         [40, 30, 25, 20, self.cluster_thr])
                self.tractogram_clusters[t] = clusters
                centroids = clusters.centroids
                print(' Number of centroids is {}'.format(len(centroids)))
                sizes = np.array([len(c) for c in clusters])
                linewidths = np.interp(sizes,
                                       [sizes.min(), sizes.max()], [0.1, 2.])
                centroid_lengths = np.array([length(c) for c in centroids])

                print(' Minimum number of streamlines in cluster {}'
                      .format(sizes.min()))

                print(' Maximum number of streamlines in cluster {}'
                      .format(sizes.max()))

                print(' Construct cluster actors')
                for (i, c) in enumerate(centroids):

                    centroid_actor = actor.streamtube([c], colors,
                                                      linewidth=linewidths[i],
                                                      lod=False)
                    scene.add(centroid_actor)
                    HORIZON.centroid_actors.append(centroid_actor)

                    cluster_actor = actor.line(clusters[i],
                                               lod=False)
                    cluster_actor.GetProperty().SetRenderLinesAsTubes(1)
                    cluster_actor.GetProperty().SetLineWidth(6)
                    cluster_actor.GetProperty().SetOpacity(1)
                    cluster_actor.VisibilityOff()

                    scene.add(cluster_actor)
                    HORIZON.cluster_actors.append(cluster_actor)

                    # Every centroid actor (cea) is paired to a cluster actor
                    # (cla).

                    self.cea[centroid_actor] = {
                        'cluster_actor': cluster_actor,
                        'cluster': i, 'tractogram': t,
                        'size': sizes[i], 'length': centroid_lengths[i],
                        'selected': 0, 'expanded': 0}

                    self.cla[cluster_actor] = {
                        'centroid_actor': centroid_actor,
                        'cluster': i, 'tractogram': t,
                        'size': sizes[i], 'length': centroid_lengths[i],
                        'selected': 0}
                    apply_shader(self, cluster_actor)
                    apply_shader(self, centroid_actor)

            else:

                streamline_actor = actor.line(streamlines, colors=colors)
                streamline_actor.GetProperty().SetEdgeVisibility(1)
                streamline_actor.GetProperty().SetRenderLinesAsTubes(1)
                streamline_actor.GetProperty().SetLineWidth(6)
                streamline_actor.GetProperty().SetOpacity(1)
                scene.add(streamline_actor)
                HORIZON.streamline_actors.append(streamline_actor)
        return scene

    def remove_actors(self, scene):

        for ca_ in HORIZON.centroid_actors:
            scene.rm(ca_)
        for ca_ in HORIZON.cluster_actors:
            scene.rm(ca_)

    def add_actors(self, scene, tractograms, threshold):
        """ Experimental
        """
        for (t, streamlines) in enumerate(tractograms):
            if self.random_colors:
                # TODO use distinguished colormap
                colors = self.prng.random_sample(3)
            else:
                colors = None

            if self.cluster:

                print(' Clustering threshold {} \n'.format(threshold))
                clusters = qbx_and_merge(streamlines,
                                         [40, 30, 25, 20, threshold])
                self.tractogram_clusters[t] = clusters
                centroids = clusters.centroids
                print(' Number of centroids is {}'.format(len(centroids)))
                sizes = np.array([len(c) for c in clusters])
                linewidths = np.interp(sizes,
                                       [sizes.min(), sizes.max()], [0.1, 2.])
                centroid_lengths = np.array([length(c) for c in centroids])

                print(' Minimum number of streamlines in cluster {}'
                      .format(sizes.min()))

                print(' Maximum number of streamlines in cluster {}'
                      .format(sizes.max()))

                print(' Construct cluster actors')
                for (i, c) in enumerate(centroids):

                    centroid_actor = actor.streamtube([c], colors,
                                                      linewidth=linewidths[i],
                                                      lod=False)
                    scene.add(centroid_actor)
                    HORIZON.centroid_actors.append(centroid_actor)

                    cluster_actor = actor.line(clusters[i],
                                               lod=False)
                    cluster_actor.GetProperty().SetRenderLinesAsTubes(1)
                    cluster_actor.GetProperty().SetLineWidth(6)
                    cluster_actor.GetProperty().SetOpacity(1)
                    cluster_actor.VisibilityOff()

                    scene.add(cluster_actor)
                    HORIZON.cluster_actors.append(cluster_actor)

                    # Every centroid actor (cea) is paired to a cluster actor
                    # (cla).

                    self.cea[centroid_actor] = {
                        'cluster_actor': cluster_actor,
                        'cluster': i, 'tractogram': t,
                        'size': sizes[i], 'length': centroid_lengths[i],
                        'selected': 0, 'expanded': 0}

                    self.cla[cluster_actor] = {
                        'centroid_actor': centroid_actor,
                        'cluster': i, 'tractogram': t,
                        'size': sizes[i], 'length': centroid_lengths[i],
                        'selected': 0}
                    apply_shader(self, cluster_actor)
                    apply_shader(self, centroid_actor)

            else:

                streamline_actor = actor.line(streamlines, colors=colors)
                streamline_actor.GetProperty().SetEdgeVisibility(1)
                streamline_actor.GetProperty().SetRenderLinesAsTubes(1)
                streamline_actor.GetProperty().SetLineWidth(6)
                streamline_actor.GetProperty().SetOpacity(1)
                scene.add(streamline_actor)
                HORIZON.streamline_actors.append(streamline_actor)

        def left_click_centroid_callback(obj, event):

            self.cea[obj]['selected'] = not self.cea[obj]['selected']
            self.cla[self.cea[obj]['cluster_actor']]['selected'] = \
                self.cea[obj]['selected']
            show_m.render()

        def left_click_cluster_callback(obj, event):

            if self.cla[obj]['selected']:
                self.cla[obj]['centroid_actor'].VisibilityOn()
                ca = self.cla[obj]['centroid_actor']
                self.cea[ca]['selected'] = 0
                obj.VisibilityOff()
                self.cea[ca]['expanded'] = 0

            show_m.render()

        for cl in self.cla:
            cl.AddObserver('LeftButtonPressEvent', left_click_cluster_callback,
                           1.0)
            self.cla[cl]['centroid_actor'].AddObserver(
                'LeftButtonPressEvent', left_click_centroid_callback, 1.0)




    def build_show(self, scene):

        show_m = window.ShowManager(scene, size=(1200, 900),
                                    order_transparent=True,
                                    reset_camera=False)
        show_m.initialize()

        if self.cluster and self.tractograms:

            lengths = np.array(
                [self.cla[c]['length'] for c in self.cla])
            szs = [self.cla[c]['size'] for c in self.cla]
            sizes = np.array(szs)

            # global self.panel2, slider_length, slider_size
            self.panel2 = ui.Panel2D(size=(400, 400),
                                     position=(850, 520),
                                     color=(1, 1, 1),
                                     opacity=0.1,
                                     align="right")

            slider_label_threshold = build_label(text="Threshold")
            slider_threshold = ui.LineSlider2D(
                    min_value=5,
                    max_value=25,
                    initial_value=15,
                    text_template="{value:.0f}",
                    length=140, shape='square')
            _color_slider(slider_threshold)

            slider_label_length = build_label(text="Length")
            slider_length = ui.LineSlider2D(
                    min_value=lengths.min(),
                    max_value=np.percentile(lengths, 98),
                    initial_value=np.percentile(lengths, 25),
                    text_template="{value:.0f}",
                    length=140)
            _color_slider(slider_length)

            slider_label_size = build_label(text="Size")
            slider_size = ui.LineSlider2D(
                    min_value=sizes.min(),
                    max_value=np.percentile(sizes, 98),
                    initial_value=np.percentile(sizes, 50),
                    text_template="{value:.0f}",
                    length=140)
            _color_slider(slider_size)

            # global self.length_min, size_min
            self.size_min = sizes.min()
            self.length_min = lengths.min()

            def change_threshold(istyle, obj, slider):
                sv = np.round(slider.value,0)
                self.remove_actors(scene)
                self.add_actors(scene, self.tractograms, threshold=sv)

            slider_threshold.handle_events(slider_threshold.handle.actor)
            slider_threshold.on_left_mouse_button_released = change_threshold

            def hide_clusters_length(slider):
                self.length_min = np.round(slider.value)

                for k in self.cla:
                    if (self.cla[k]['length'] < self.length_min or
                            self.cla[k]['size'] < self.size_min):
                        self.cla[k]['centroid_actor'].SetVisibility(0)
                        if k.GetVisibility() == 1:
                            k.SetVisibility(0)
                    else:
                        self.cla[k]['centroid_actor'].SetVisibility(1)
                show_m.render()

            def hide_clusters_size(slider):
                self.size_min = np.round(slider.value)

                for k in self.cla:
                    if (self.cla[k]['length'] < self.length_min or
                            self.cla[k]['size'] < self.size_min):
                        self.cla[k]['centroid_actor'].SetVisibility(0)
                        if k.GetVisibility() == 1:
                            k.SetVisibility(0)
                    else:
                        self.cla[k]['centroid_actor'].SetVisibility(1)
                show_m.render()

            slider_length.on_change = hide_clusters_length

            self.panel2.add_element(slider_label_threshold, coords=(0.1, 0.133))
            self.panel2.add_element(slider_threshold, coords=(0.4, 0.133))

            self.panel2.add_element(slider_label_length, coords=(0.1, 0.333))
            self.panel2.add_element(slider_length, coords=(0.4, 0.333))

            slider_size.on_change = hide_clusters_size

            self.panel2.add_element(slider_label_size, coords=(0.1, 0.6666))
            self.panel2.add_element(slider_size, coords=(0.4, 0.6666))

            scene.add(self.panel2)

            text_block = build_label(HELP_MESSAGE, 16)  # ui.TextBlock2D()
            text_block.message = HELP_MESSAGE

            help_panel = ui.Panel2D(size=(300, 200),
                                    color=(1, 1, 1),
                                    opacity=0.1,
                                    align="left")

            help_panel.add_element(text_block, coords=(0.05, 0.1))
            scene.add(help_panel)

        if len(self.images) > 0:
            # !!Only first image loading supported for now')
            data, affine = self.images[0]
            if len(self.pams) > 0:
                pam = self.pams[0]
            else:
                pam = None
            self.panel = slicer_panel(scene, show_m.iren, data, affine,
                                      self.world_coords, pam=pam)
        else:
            data = None
            affine = None
            pam = None

        self.win_size = scene.GetSize()

        def win_callback(obj, event):
            if self.win_size != obj.GetSize():
                size_old = self.win_size
                self.win_size = obj.GetSize()
                size_change = [self.win_size[0] - size_old[0], 0]
                if data is not None:
                    self.panel.re_align(size_change)
                if self.cluster:
                    self.panel2.re_align(size_change)
                    help_panel.re_align(size_change)

        show_m.initialize()

        def left_click_centroid_callback(obj, event):

            self.cea[obj]['selected'] = not self.cea[obj]['selected']
            self.cla[self.cea[obj]['cluster_actor']]['selected'] = \
                self.cea[obj]['selected']
            show_m.render()

        def left_click_cluster_callback(obj, event):

            if self.cla[obj]['selected']:
                self.cla[obj]['centroid_actor'].VisibilityOn()
                ca = self.cla[obj]['centroid_actor']
                self.cea[ca]['selected'] = 0
                obj.VisibilityOff()
                self.cea[ca]['expanded'] = 0

            show_m.render()

        for cl in self.cla:
            cl.AddObserver('LeftButtonPressEvent', left_click_cluster_callback,
                           1.0)
            self.cla[cl]['centroid_actor'].AddObserver(
                'LeftButtonPressEvent', left_click_centroid_callback, 1.0)

        self.hide_centroids = True
        self.select_all = False

        def key_press(obj, event):
            key = obj.GetKeySym()
            if self.cluster:

                # hide on/off unselected centroids
                if key == 'h' or key == 'H':
                    if self.hide_centroids:
                        for ca in self.cea:
                            if (self.cea[ca]['length'] >= self.length_min or
                                    self.cea[ca]['size'] >= self.size_min):
                                if self.cea[ca]['selected'] == 0:
                                    ca.VisibilityOff()
                    else:
                        for ca in self.cea:
                            if (self.cea[ca]['length'] >= self.length_min and
                                    self.cea[ca]['size'] >= self.size_min):
                                if self.cea[ca]['selected'] == 0:
                                    ca.VisibilityOn()
                    self.hide_centroids = not self.hide_centroids
                    show_m.render()

                # invert selection
                if key == 'i' or key == 'I':

                    for ca in self.cea:
                        if (self.cea[ca]['length'] >= self.length_min and
                                self.cea[ca]['size'] >= self.size_min):
                            self.cea[ca]['selected'] = \
                                not self.cea[ca]['selected']
                            cas = self.cea[ca]['cluster_actor']
                            self.cla[cas]['selected'] = \
                                self.cea[ca]['selected']
                    show_m.render()

                # save current result
                if key == 's' or key == 'S':
                    saving_streamlines = Streamlines()
                    for bundle in self.cla.keys():
                        if bundle.GetVisibility():
                            t = self.cla[bundle]['tractogram']
                            c = self.cla[bundle]['cluster']
                            indices = self.tractogram_clusters[t][c]
                            saving_streamlines.extend(Streamlines(indices))
                    print('Saving result in tmp.trk')
                    sft = StatefulTractogram(saving_streamlines, 'same',
                                             Space.RASMM)
                    save_tractogram(sft, 'tmp.trk', bbox_valid_check=False)

                if key == 'y' or key == 'Y':
                    active_streamlines = Streamlines()
                    for bundle in self.cla.keys():
                        if bundle.GetVisibility():
                            t = self.cla[bundle]['tractogram']
                            c = self.cla[bundle]['cluster']
                            indices = self.tractogram_clusters[t][c]
                            active_streamlines.extend(Streamlines(indices))

                    # self.tractograms = [active_streamlines]
                    hz2 = horizon([active_streamlines],
                                  self.images, cluster=True, cluster_thr=5,
                                  random_colors=self.random_colors,
                                  length_lt=np.inf,
                                  length_gt=0, clusters_lt=np.inf,
                                  clusters_gt=0,
                                  world_coords=True,
                                  interactive=True)
                    ren2 = hz2.build_scene()
                    hz2.build_show(ren2)

                if key == 'a' or key == 'A':

                    if self.select_all is False:
                        for ca in self.cea:
                            if (self.cea[ca]['length'] >= self.length_min and
                                    self.cea[ca]['size'] >= self.size_min):
                                self.cea[ca]['selected'] = 1
                                cas = self.cea[ca]['cluster_actor']
                                self.cla[cas]['selected'] = \
                                    self.cea[ca]['selected']
                        show_m.render()
                        self.select_all = True
                    else:
                        for ca in self.cea:
                            if (self.cea[ca]['length'] >= self.length_min and
                                    self.cea[ca]['size'] >= self.size_min):
                                self.cea[ca]['selected'] = 0
                                cas = self.cea[ca]['cluster_actor']
                                self.cla[cas]['selected'] = \
                                    self.cea[ca]['selected']
                        show_m.render()
                        self.select_all = False

                if key == 'e' or key == 'E':

                    for c in self.cea:
                        if self.cea[c]['selected']:
                            if not self.cea[c]['expanded']:
                                len_ = self.cea[c]['length']
                                sz_ = self.cea[c]['size']
                                if (len_ >= self.length_min and
                                        sz_ >= self.size_min):
                                    self.cea[c]['cluster_actor']. \
                                        VisibilityOn()
                                    c.VisibilityOff()
                                    self.cea[c]['expanded'] = 1

                    show_m.render()

                if key == 'r' or key == 'R':

                    for c in self.cea:

                        if (self.cea[c]['length'] >= self.length_min and
                                self.cea[c]['size'] >= self.size_min):
                            self.cea[c]['cluster_actor'].VisibilityOff()
                            c.VisibilityOn()
                            self.cea[c]['expanded'] = 0

                show_m.render()


        HORIZON.window_timer_cnt = 0

        def timer_callback(obj, event):

            HORIZON.window_timer_cnt += 1
            cnt = HORIZON.window_timer_cnt
            # TODO possibly add automatic rotation option
            # print("Let's count up to 100 " + str(cnt))
            # show_m.scene.azimuth(0.05 * cnt)
            # show_m.render()

        scene.reset_camera()
        scene.zoom(1.5)
        scene.reset_clipping_range()

        if self.interactive:

            show_m.add_window_callback(win_callback)
            show_m.add_timer_callback(True, 200, timer_callback)
            show_m.iren.AddObserver('KeyPressEvent', key_press)
            show_m.render()
            show_m.start()

        else:

            window.record(scene, out_path=self.out_png,
                          size=(1200, 900),
                          reset_camera=False)


def horizon(tractograms=[], images=[], pams=[],
            cluster=False, cluster_thr=15.0,
            random_colors=False, length_gt=0, length_lt=1000,
            clusters_gt=0, clusters_lt=10000,
            world_coords=True, interactive=True, out_png='tmp.png'):
    """Highly interactive visualization - invert the Horizon!

    Parameters
    ----------
    tractograms : sequence
        Sequence of Streamlines objects
    images : sequence of tuples
        Each tuple contains data and affine
    pams : peaks
    cluster : bool
        Enable QuickBundlesX clustering
    cluster_thr : float
        Distance threshold used for clustering
    random_colors : bool
    length_gt : float
    length_lt : float
    clusters_gt : int
    clusters_lt : int
    world_coords : bool
    interactive : bool
    out_png : string

    References
    ----------
    .. [Horizon_ISMRM19] Garyfallidis E., M-A. Cote, B.Q. Chandio,
        S. Fadnavis, J. Guaje, R. Aggarwal, E. St-Onge, K.S. Juneja,
        S. Koudoro, D. Reagan, DIPY Horizon: fast, modular, unified and
        adaptive visualization, Proceedings of: International Society of
        Magnetic Resonance in Medicine (ISMRM), Montreal, Canada, 2019.
    """
    hz = Horizon(tractograms, images, pams, cluster, cluster_thr, random_colors,
                 length_gt, length_lt, clusters_gt, clusters_lt,
                 world_coords, interactive, out_png)

    scene = hz.build_scene()

    hz.build_show(scene)
