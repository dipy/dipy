import numpy as np
from dipy.segment.clustering import qbx_and_merge
from dipy.tracking.streamline import length, Streamlines
from dipy.utils.optpkg import optional_package
from dipy import __version__ as horizon_version
from dipy.viz.gmem import GlobalHorizon
from dipy.io.stateful_tractogram import Space, StatefulTractogram
from dipy.io.streamline import save_tractogram

fury, has_fury, setup_module = optional_package('fury')

if has_fury:
    from dipy.viz import actor, window, ui, shaders
    from dipy.viz.panel import slicer_panel, build_label, _color_slider
    from fury.colormap import distinguishable_colormap


def apply_shader(hz, act):
    """ Apply a shader to an actor (act) that access shared memory

    Parameters
    ----------
    hz : GlobalHorizon instance
    act : fury.actor visual object
    """

    frag_decl = \
        """
        uniform float selected;
        uniform float opacity_level;
        """

    frag_impl = \
        """
        if (selected == 1){
            fragOutput0 = fragOutput0 + vec4(0.2, 0.2, 0, opacity_level);
            }
        """

    shaders.shader_to_actor(act, "vertex", impl_code="\n",
                            replace_first=False,
                            replace_all=False)
    shaders.shader_to_actor(act, "fragment", decl_code=frag_decl,
                            block="coincident")
    shaders.shader_to_actor(act, "fragment", impl_code=frag_impl,
                            block="light")

    def shader_selected_callback(caller, event, calldata=None):
        program = calldata
        if program is not None:
            try:
                program.SetUniformf("selected",
                                    hz.cea[act]['selected'])
            except KeyError:
                pass
            try:
                program.SetUniformf("selected",
                                    hz.cla[act]['selected'])
            except KeyError:
                pass
            program.SetUniformf("opacity_level", 1)

    shaders.add_shader_callback(act, shader_selected_callback, priority=100)


HELP_MESSAGE = """
>> left click: select centroid
>> right click: see menu
>> e: expand centroids
>> r: collapse all clusters
>> h: hide unselected centroids
>> i: invert selection
>> a: select all centroids
>> s: save in file
"""


class Horizon(object):

    def __init__(self, tractograms=None, images=None, pams=None, cluster=False,
                 cluster_thr=15.0, random_colors=None, length_gt=0,
                 length_lt=1000, clusters_gt=0, clusters_lt=10000,
                 world_coords=True, interactive=True, out_png='tmp.png',
                 recorded_events=None, return_showm=False, bg_color=(0, 0, 0),
                 order_transparent=True, buan=False, buan_colors=None,
                 roi_images=False, roi_colors=(1, 0, 0)):
        """Interactive medical visualization - Invert the Horizon!


        Parameters
        ----------
        tractograms : sequence of StatefulTractograms
            StatefulTractograms are used for making sure that the coordinate
            systems are correct
        images : sequence of tuples
            Each tuple contains data and affine
        pams : sequence of PeakAndMetrics
            Contains peak directions and spherical harmonic coefficients
        cluster : bool
            Enable QuickBundlesX clustering
        cluster_thr : float
            Distance threshold used for clustering. Default value 15.0 for
            small animal data you may need to use something smaller such
            as 2.0. The threshold is in mm. For this parameter to be active
            ``cluster`` should be enabled.
        random_colors : string, optional
            Given multiple tractograms and/or ROIs then each tractogram and/or
            ROI will be shown with a different color. If no value is provided,
            both the tractograms and the ROIs will have a different random
            color generated from a distinguishable colormap. If the effect
            should only be applied to one of the 2 types, then use the
            options 'tracts' and 'rois' for the tractograms and the ROIs
            respectively.
        length_gt : float
            Clusters with average length greater than ``length_gt`` amount
            in mm will be shown.
        length_lt : float
            Clusters with average length less than ``length_lt`` amount in mm
            will be shown.
        clusters_gt : int
            Clusters with size greater than ``clusters_gt`` will be shown.
        clusters_lt : int
            Clusters with size less than ``clusters_lt`` will be shown.
        world_coords : bool
            Show data in their world coordinates (not native voxel coordinates)
            Default True.
        interactive : bool
            Allow user interaction. If False then Horizon goes on stealth mode
            and just saves pictures.
        out_png : string
            Filename of saved picture.
        recorded_events : string
            File path to replay recorded events
        return_showm : bool
            Return ShowManager object. Used only at Python level. Can be used
            for extending Horizon's cababilities externally and for testing
            purposes.
        bg_color : ndarray or list or tuple
            Define the background color of the scene.
            Default is black (0, 0, 0)
        order_transparent : bool
            Default True. Use depth peeling to sort transparent objects.
            If True also enables anti-aliasing.
        buan : bool, optional
            Enables BUAN framework visualization. Default is False.
        buan_colors : list, optional
            List of colors for bundles.
        roi_images : bool, optional
            Displays binary images as contours. Default is False.
        roi_colors : ndarray or list or tuple, optional
            Define the colors of the roi images. Default is red (1, 0, 0)


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
        self.tractograms = tractograms or []
        self.out_png = out_png
        self.images = images or []
        self.pams = pams or []

        self.cea = {}  # holds centroid actors
        self.cla = {}  # holds cluster actors
        self.tractogram_clusters = {}
        self.recorded_events = recorded_events
        self.show_m = None
        self.return_showm = return_showm
        self.bg_color = bg_color
        self.order_transparent = order_transparent
        self.buan = buan
        self.buan_colors = buan_colors
        self.roi_images = roi_images
        self.roi_colors = roi_colors

        if self.random_colors is not None:
            self.color_gen = distinguishable_colormap()
            if not self.random_colors:
                self.random_colors = ['tracts', 'rois']
        else:
            self.random_colors = []

    def build_scene(self):

        self.mem = GlobalHorizon()
        scene = window.Scene()
        scene.background(self.bg_color)
        self.add_cluster_actors(scene, self.tractograms,
                                self.cluster_thr,
                                enable_callbacks=False)
        return scene

    def remove_cluster_actors(self, scene):

        for ca_ in self.mem.centroid_actors:
            scene.rm(ca_)
            del self.cea[ca_]
        for ca_ in self.mem.cluster_actors:
            scene.rm(ca_)
            del self.cla[ca_]
        self.mem.centroid_actors = []
        self.mem.cluster_actors = []

    def add_cluster_actors(self, scene, tractograms,
                           threshold, enable_callbacks=True):
        """ Add streamline actors to the scene

        Parameters
        ----------
        scene : Scene
        tractograms : list
            list of tractograms
        threshold : float
            Cluster threshold
        enable_callbacks : bool
            Enable callbacks for selecting clusters
        """
        color_ind = 0
        for (t, sft) in enumerate(tractograms):
            streamlines = sft.streamlines

            if 'tracts' in self.random_colors:
                colors = next(self.color_gen)
            else:
                colors = None

            if not self.world_coords:
                # TODO we need to read the affine of a tractogram
                # from a StatefullTractogram
                msg = 'Currently native coordinates are not supported'
                msg += ' for streamlines'
                raise ValueError(msg)

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
                    self.mem.centroid_actors.append(centroid_actor)

                    cluster_actor = actor.line(clusters[i],
                                               lod=False)
                    cluster_actor.GetProperty().SetRenderLinesAsTubes(1)
                    cluster_actor.GetProperty().SetLineWidth(6)
                    cluster_actor.GetProperty().SetOpacity(1)
                    cluster_actor.VisibilityOff()

                    scene.add(cluster_actor)
                    self.mem.cluster_actors.append(cluster_actor)

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
                        'selected': 0, 'highlighted': 0}
                    apply_shader(self, cluster_actor)
                    apply_shader(self, centroid_actor)

            else:

                s_colors = self.buan_colors[color_ind] if self.buan else colors
                streamline_actor = actor.line(streamlines, colors=s_colors)

                streamline_actor.GetProperty().SetEdgeVisibility(1)
                streamline_actor.GetProperty().SetRenderLinesAsTubes(1)
                streamline_actor.GetProperty().SetLineWidth(6)
                streamline_actor.GetProperty().SetOpacity(1)
                scene.add(streamline_actor)
                self.mem.streamline_actors.append(streamline_actor)

            color_ind += 1

        if not enable_callbacks:
            return

        def left_click_centroid_callback(obj, event):
            self.cea[obj]['selected'] = not self.cea[obj]['selected']
            self.cla[self.cea[obj]['cluster_actor']]['selected'] = \
                self.cea[obj]['selected']
            self.show_m.render()

        def left_click_cluster_callback(obj, event):
            if self.cla[obj]['selected']:
                self.cla[obj]['centroid_actor'].VisibilityOn()
                ca = self.cla[obj]['centroid_actor']
                self.cea[ca]['selected'] = 0
                obj.VisibilityOff()
                self.cea[ca]['expanded'] = 0

            self.show_m.render()

        for cl in self.cla:
            cl.AddObserver('LeftButtonPressEvent', left_click_cluster_callback,
                           1.0)
            self.cla[cl]['centroid_actor'].AddObserver(
                'LeftButtonPressEvent', left_click_centroid_callback, 1.0)

    def build_show(self, scene):

        title = 'Horizon ' + horizon_version
        self.show_m = window.ShowManager(
            scene, title=title,
            size=(1200, 900),
            order_transparent=self.order_transparent,
            reset_camera=False)
        self.show_m.initialize()

        if self.cluster and self.tractograms:

            lengths = np.array(
                [self.cla[c]['length'] for c in self.cla])
            szs = [self.cla[c]['size'] for c in self.cla]
            sizes = np.array(szs)

            # global self.panel2, slider_length, slider_size
            self.panel2 = ui.Panel2D(size=(320, 200), position=(870, 520),
                                     color=(1, 1, 1), opacity=0.1,
                                     align="right")

            cluster_panel_label = build_label(text="Cluster panel", bold=True)

            slider_label_threshold = build_label(text="Threshold")
            print("Cluster threshold", self.cluster_thr)
            slider_threshold = ui.LineSlider2D(
                    min_value=5,
                    max_value=25,
                    initial_value=self.cluster_thr,
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
                sv = np.round(slider.value, 0)
                self.remove_cluster_actors(scene)
                self.add_cluster_actors(scene, self.tractograms, threshold=sv)

                # TODO need to double check if this section is still needed
                lengths = np.array(
                    [self.cla[c]['length'] for c in self.cla])
                szs = [self.cla[c]['size'] for c in self.cla]
                sizes = np.array(szs)

                slider_length.min_value = lengths.min()
                slider_length.max_value = lengths.max()
                slider_length.value = lengths.min()
                slider_length.update()

                slider_size.min_value = sizes.min()
                slider_size.max_value = sizes.max()
                slider_size.value = sizes.min()
                slider_size.update()

                self.length_min = min(lengths)
                self.size_min = min(sizes)

                self.show_m.render()

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
                self.show_m.render()

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
                self.show_m.render()

            slider_length.on_change = hide_clusters_length

            # Clustering panel
            self.panel2.add_element(slider_label_threshold, coords=(0.1, 0.15))
            self.panel2.add_element(slider_threshold, coords=(0.42, 0.15))

            self.panel2.add_element(slider_label_length, coords=(0.1, 0.4))
            self.panel2.add_element(slider_length, coords=(0.42, 0.4))

            slider_size.on_change = hide_clusters_size

            self.panel2.add_element(slider_label_size, coords=(0.1, 0.65))
            self.panel2.add_element(slider_size, coords=(0.42, 0.65))

            self.panel2.add_element(cluster_panel_label, coords=(0.05, 0.85))

            scene.add(self.panel2)

            # Information panel
            text_block = build_label(HELP_MESSAGE, 18)
            text_block.message = HELP_MESSAGE

            self.help_panel = ui.Panel2D(size=(320, 200), position=(10, 10),
                                         color=(0.8, 0.8, 1), opacity=0.2,
                                         align="left")

            self.help_panel.add_element(text_block, coords=(0.05, 0.1))
            scene.add(self.help_panel)

        if len(self.images) > 0:
            # Only first non-binary image loading supported for now
            first_img = True
            first_roi = True
            if self.roi_images:
                roi_color = self.roi_colors
                for img in self.images:
                    img_data, img_affine = img
                    dim = np.unique(img_data).shape[0]
                    if dim == 2:
                        if 'rois' in self.random_colors:
                            roi_color = next(self.color_gen)
                        roi_actor = actor.contour_from_roi(
                            img_data, affine=img_affine,
                            color=roi_color, opacity=self.mem.roi_opacity)
                        self.mem.slicer_roi_actor.append(roi_actor)
                        scene.add(roi_actor)

                        if first_roi:
                            self.panel3 = ui.Panel2D(
                                size=(320, 100), position=(870, 730),
                                color=(1, 1, 1), opacity=0.1, align="right")

                            rois_panel_label = build_label(text="ROIs panel",
                                                           bold=True)

                            slider_label_opacity = build_label(text="Opacity")
                            slider_opacity = ui.LineSlider2D(
                                min_value=0.0, max_value=1.0,
                                initial_value=self.mem.roi_opacity, length=140,
                                text_template="{ratio:.0%}")
                            _color_slider(slider_opacity)

                            def change_opacity(slider):
                                roi_opacity = slider.value
                                self.mem.roi_opacity = roi_opacity
                                for contour in self.mem.slicer_roi_actor:
                                    contour.GetProperty().SetOpacity(
                                        roi_opacity)

                            slider_opacity.on_change = change_opacity

                            self.panel3.add_element(slider_label_opacity,
                                                    coords=(0.1, 0.3))
                            self.panel3.add_element(slider_opacity,
                                                    coords=(0.42, 0.3))

                            self.panel3.add_element(rois_panel_label,
                                                    coords=(0.05, 0.7))

                            scene.add(self.panel3)

                            first_roi = False
                    else:
                        if first_img:
                            data, affine = img
                            self.vox2ras = affine

                            if len(self.pams) > 0:
                                pam = self.pams[0]
                            else:
                                pam = None
                            self.panel = slicer_panel(
                                scene, self.show_m.iren, data, affine,
                                self.world_coords, pam=pam, mem=self.mem)
                            first_img = False
            else:
                data, affine = self.images[0]
                self.vox2ras = affine

                if len(self.pams) > 0:
                    pam = self.pams[0]
                else:
                    pam = None
                self.panel = slicer_panel(scene, self.show_m.iren, data,
                                          affine, self.world_coords, pam=pam,
                                          mem=self.mem)
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
                    self.help_panel.re_align(size_change)
                if self.roi_images:
                    self.panel3.re_align(size_change)

        self.show_m.initialize()

        self.hide_centroids = True
        self.select_all = False

        def hide():
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
            self.show_m.render()

        def invert():
            for ca in self.cea:
                if (self.cea[ca]['length'] >= self.length_min and
                        self.cea[ca]['size'] >= self.size_min):
                    self.cea[ca]['selected'] = \
                        not self.cea[ca]['selected']
                    cas = self.cea[ca]['cluster_actor']
                    self.cla[cas]['selected'] = \
                        self.cea[ca]['selected']
            self.show_m.render()

        def save():
            saving_streamlines = Streamlines()
            for bundle in self.cla.keys():
                if bundle.GetVisibility():
                    t = self.cla[bundle]['tractogram']
                    c = self.cla[bundle]['cluster']
                    indices = self.tractogram_clusters[t][c]
                    saving_streamlines.extend(Streamlines(indices))
            print('Saving result in tmp.trk')

            # Using the header of the first of the tractograms
            sft_new = StatefulTractogram(saving_streamlines,
                                         self.tractograms[0],
                                         Space.RASMM)
            save_tractogram(sft_new, 'tmp.trk', bbox_valid_check=False)
            print('Saved!')

        def new_window():
            active_streamlines = Streamlines()
            for bundle in self.cla.keys():
                if bundle.GetVisibility():
                    t = self.cla[bundle]['tractogram']
                    c = self.cla[bundle]['cluster']
                    indices = self.tractogram_clusters[t][c]
                    active_streamlines.extend(Streamlines(indices))

            # Using the header of the first of the tractograms
            active_sft = StatefulTractogram(active_streamlines,
                                            self.tractograms[0],
                                            Space.RASMM)
            hz2 = Horizon([active_sft],
                          self.images, cluster=True,
                          cluster_thr=self.cluster_thr/2.,
                          random_colors=self.random_colors,
                          length_lt=np.inf,
                          length_gt=0, clusters_lt=np.inf,
                          clusters_gt=0,
                          world_coords=True,
                          interactive=True)
            ren2 = hz2.build_scene()
            hz2.build_show(ren2)

        def show_all():
            if self.select_all is False:
                for ca in self.cea:
                    if (self.cea[ca]['length'] >= self.length_min and
                            self.cea[ca]['size'] >= self.size_min):
                        self.cea[ca]['selected'] = 1
                        cas = self.cea[ca]['cluster_actor']
                        self.cla[cas]['selected'] = \
                            self.cea[ca]['selected']
                self.show_m.render()
                self.select_all = True
            else:
                for ca in self.cea:
                    if (self.cea[ca]['length'] >= self.length_min and
                            self.cea[ca]['size'] >= self.size_min):
                        self.cea[ca]['selected'] = 0
                        cas = self.cea[ca]['cluster_actor']
                        self.cla[cas]['selected'] = \
                            self.cea[ca]['selected']
                self.show_m.render()
                self.select_all = False

        def expand():
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

            self.show_m.render()

        def reset():
            for c in self.cea:

                if (self.cea[c]['length'] >= self.length_min and
                        self.cea[c]['size'] >= self.size_min):
                    self.cea[c]['cluster_actor'].VisibilityOff()
                    c.VisibilityOn()
                    self.cea[c]['expanded'] = 0

            self.show_m.render()

        def key_press(obj, event):
            key = obj.GetKeySym()
            if self.cluster:

                # hide on/off unselected centroids
                if key == 'h' or key == 'H':
                    hide()

                # invert selection
                if key == 'i' or key == 'I':
                    invert()

                # retract help panel
                if key == 'o' or key == 'O':
                    self.help_panel._set_position((-300, 0))
                    self.show_m.render()

                # save current result
                if key == 's' or key == 'S':
                    save()

                if key == 'y' or key == 'Y':
                    new_window()

                if key == 'a' or key == 'A':
                    show_all()

                if key == 'e' or key == 'E':
                    expand()

                if key == 'r' or key == 'R':
                    reset()

        options = [r'un\hide centroids', 'invert selection',
                   r'un\select all', 'expand clusters',
                   'collapse clusters', 'save streamlines',
                   'recluster']
        listbox = ui.ListBox2D(values=options, position=(10, 300),
                               size=(200, 270),
                               multiselection=False, font_size=18)

        def display_element():
            action = listbox.selected[0]
            if action == r'un\hide centroids':
                hide()
            if action == 'invert selection':
                invert()
            if action == r'un\select all':
                show_all()
            if action == 'expand clusters':
                expand()
            if action == 'collapse clusters':
                reset()
            if action == 'save streamlines':
                save()
            if action == 'recluster':
                new_window()

        listbox.on_change = display_element
        listbox.panel.opacity = 0.2
        listbox.set_visibility(0)

        self.show_m.scene.add(listbox)

        def left_click_centroid_callback(obj, event):

            self.cea[obj]['selected'] = not self.cea[obj]['selected']
            self.cla[self.cea[obj]['cluster_actor']]['selected'] = \
                self.cea[obj]['selected']
            self.show_m.render()

        def right_click_centroid_callback(obj, event):
            for lactor in listbox._get_actors():
                lactor.SetVisibility(not lactor.GetVisibility())

            listbox.scroll_bar.set_visibility(False)
            self.show_m.render()

        def left_click_cluster_callback(obj, event):

            if self.cla[obj]['selected']:
                self.cla[obj]['centroid_actor'].VisibilityOn()
                ca = self.cla[obj]['centroid_actor']
                self.cea[ca]['selected'] = 0
                obj.VisibilityOff()
                self.cea[ca]['expanded'] = 0

            self.show_m.render()

        def right_click_cluster_callback(obj, event):
            print('Cluster Area Selected')
            self.show_m.render()

        for cl in self.cla:
            cl.AddObserver('LeftButtonPressEvent',
                           left_click_cluster_callback,
                           1.0)
            cl.AddObserver('RightButtonPressEvent',
                           right_click_cluster_callback,
                           1.0)
            self.cla[cl]['centroid_actor'].AddObserver(
                'LeftButtonPressEvent', left_click_centroid_callback, 1.0)
            self.cla[cl]['centroid_actor'].AddObserver(
                'RightButtonPressEvent', right_click_centroid_callback, 1.0)

        self.mem.window_timer_cnt = 0

        def timer_callback(obj, event):

            self.mem.window_timer_cnt += 1
            # TODO possibly add automatic rotation option
            # self.show_m.scene.azimuth(0.01 * self.mem.window_timer_cnt)
            # self.show_m.render()

        scene.reset_camera()
        scene.zoom(1.5)
        scene.reset_clipping_range()

        if self.interactive:

            self.show_m.add_window_callback(win_callback)
            self.show_m.add_timer_callback(True, 200, timer_callback)
            self.show_m.iren.AddObserver('KeyPressEvent', key_press)

            if self.return_showm:
                return self.show_m

            if self.recorded_events is None:
                self.show_m.render()
                self.show_m.start()

            else:

                # set to True if event recorded file needs updating
                recording = False
                recording_filename = self.recorded_events

                if recording:
                    self.show_m.record_events_to_file(recording_filename)
                else:
                    self.show_m.play_events_from_file(recording_filename)

        else:

            window.record(scene, out_path=self.out_png,
                          size=(1200, 900),
                          reset_camera=False)


def horizon(tractograms=None, images=None, pams=None,
            cluster=False, cluster_thr=15.0,
            random_colors=None, bg_color=(0, 0, 0), order_transparent=True,
            length_gt=0, length_lt=1000, clusters_gt=0, clusters_lt=10000,
            world_coords=True, interactive=True, buan=False, buan_colors=None,
            roi_images=False, roi_colors=(1, 0, 0), out_png='tmp.png',
            recorded_events=None, return_showm=False):
    """Interactive medical visualization - Invert the Horizon!


    Parameters
    ----------
    tractograms : sequence of StatefulTractograms
            StatefulTractograms are used for making sure that the coordinate
            systems are correct
    images : sequence of tuples
        Each tuple contains data and affine
    pams : sequence of PeakAndMetrics
        Contains peak directions and spherical harmonic coefficients
    cluster : bool
        Enable QuickBundlesX clustering
    cluster_thr : float
        Distance threshold used for clustering. Default value 15.0 for
        small animal data you may need to use something smaller such
        as 2.0. The threshold is in mm. For this parameter to be active
        ``cluster`` should be enabled.
    random_colors : string
        Given multiple tractograms and/or ROIs then each tractogram and/or
        ROI will be shown with different color. If no value is provided both
        the tractograms and the ROIs will have a different random color
        generated from a distinguishable colormap. If the effect should only be
        applied to one of the 2 objects, then use the options 'tracts' and
        'rois' for the tractograms and the ROIs respectively.
    bg_color : ndarray or list or tuple
        Define the background color of the scene. Default is black (0, 0, 0)
    order_transparent : bool
        Default True. Use depth peeling to sort transparent objects.
        If True also enables anti-aliasing.
    length_gt : float
        Clusters with average length greater than ``length_gt`` amount
        in mm will be shown.
    length_lt : float
        Clusters with average length less than ``length_lt`` amount in mm
        will be shown.
    clusters_gt : int
        Clusters with size greater than ``clusters_gt`` will be shown.
    clusters_lt : int
        Clusters with size less than ``clusters_lt`` will be shown.
    world_coords : bool
        Show data in their world coordinates (not native voxel coordinates)
        Default True.
    interactive : bool
        Allow user interaction. If False then Horizon goes on stealth mode
        and just saves pictures.
    buan : bool, optional
        Enables BUAN framework visualization. Default is False.
    buan_colors : list, optional
        List of colors for bundles.
    roi_images : bool, optional
        Displays binary images as contours. Default is False.
    roi_colors : ndarray or list or tuple, optional
        Define the color of the roi images. Default is red (1, 0, 0)
    out_png : string
        Filename of saved picture.
    recorded_events : string
        File path to replay recorded events
    return_showm : bool
        Return ShowManager object. Used only at Python level. Can be used
        for extending Horizon's cababilities externally and for testing
        purposes.

    References
    ----------
    .. [Horizon_ISMRM19] Garyfallidis E., M-A. Cote, B.Q. Chandio,
        S. Fadnavis, J. Guaje, R. Aggarwal, E. St-Onge, K.S. Juneja,
        S. Koudoro, D. Reagan, DIPY Horizon: fast, modular, unified and
        adaptive visualization, Proceedings of: International Society of
        Magnetic Resonance in Medicine (ISMRM), Montreal, Canada, 2019.
    """

    hz = Horizon(tractograms, images, pams, cluster, cluster_thr,
                 random_colors, length_gt, length_lt, clusters_gt, clusters_lt,
                 world_coords, interactive, out_png, recorded_events,
                 return_showm, bg_color=bg_color,
                 order_transparent=order_transparent, buan=buan,
                 buan_colors=buan_colors, roi_images=roi_images,
                 roi_colors=roi_colors)

    scene = hz.build_scene()

    if return_showm:
        return hz.build_show(scene)
    hz.build_show(scene)
