from warnings import warn
from packaging.version import Version

import numpy as np

from dipy import __version__ as horizon_version
from dipy.io.stateful_tractogram import Space, StatefulTractogram
from dipy.io.streamline import save_tractogram
from dipy.tracking.streamline import Streamlines
from dipy.utils.optpkg import optional_package
from dipy.viz.gmem import GlobalHorizon
from dipy.viz.horizon.tab import (ClustersTab, PeaksTab, ROIsTab, SlicesTab,
                                  TabManager, build_label, SurfaceTab)
from dipy.viz.horizon.visualizer import (ClustersVisualizer, SlicesVisualizer,
                                         SurfaceVisualizer, PeaksVisualizer)
from dipy.viz.horizon.util import (check_img_dtype, check_img_shapes,
                                   unpack_image, is_binary_image,
                                   unpack_surface, check_peak_size)

fury, has_fury, setup_module = optional_package('fury', min_version="0.10.0")

if has_fury:
    from fury import __version__ as fury_version
    from fury import actor, ui, window
    from fury.colormap import distinguishable_colormap


# TODO: Re-enable >> right click: see menu
HELP_MESSAGE = """
>> left click: select centroid
>> e: expand centroids
>> r: collapse all clusters
>> h: hide unselected centroids
>> i: invert selection
>> a: select all centroids
>> s: save in file
>> y: new window
>> o: hide/show this panel
"""


class Horizon:

    def __init__(self, tractograms=None, images=None, pams=None, surfaces=None,
                 cluster=False, rgb=False, cluster_thr=15.0,
                 random_colors=None, length_gt=0, length_lt=1000,
                 clusters_gt=0, clusters_lt=10000,
                 world_coords=True, interactive=True, out_png='tmp.png',
                 recorded_events=None, return_showm=False, bg_color=(0, 0, 0),
                 order_transparent=True, buan=False, buan_colors=None,
                 roi_images=False, roi_colors=(1, 0, 0),
                 surface_colors=[(1, 0, 0)]):
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
        surfaces : sequence of tuples
            Each tuple contains vertices and faces
        cluster : bool
            Enable QuickBundlesX clustering
        rgb : bool, optional
            Enable the color image (rgb only, alpha channel will be ignored).
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
        if not has_fury:
            raise ImportError('Horizon requires FURY. Please install it '
                              'with pip install fury')
        if Version(fury_version) < Version('0.10.0'):
            ValueError('Horizon requires FURY version 0.10.0 or higher.'
                       ' Please upgrade FURY with pip install -U fury.')

        self.cluster = cluster
        self.rgb = rgb
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
        self._surfaces = surfaces or []

        self.cea = {}  # holds centroid actors
        self.cla = {}  # holds cluster actors
        self.recorded_events = recorded_events
        self.show_m = None
        self.return_showm = return_showm
        self.bg_color = bg_color
        self.order_transparent = order_transparent
        self.buan = buan
        self.buan_colors = buan_colors
        self.__roi_images = roi_images
        self.__roi_colors = roi_colors
        self._surface_colors = surface_colors

        self.color_gen = distinguishable_colormap()

        if self.random_colors is not None:
            if not self.random_colors:
                self.random_colors = ['tracts', 'rois']
        else:
            self.random_colors = []

        self.__clusters_visualizer = None
        self.__tabs = []
        self.__tab_mgr = None

        self.__help_visible = True

        # TODO: Move to another class/module
        self.__hide_centroids = True
        self.__select_all = False

        self.__win_size = (0, 0)

    # TODO: Move to another class/module
    def __expand(self):
        centroid_actors = self.__clusters_visualizer.centroid_actors
        lengths = self.__clusters_visualizer.lengths
        sizes = self.__clusters_visualizer.sizes
        min_length = np.min(lengths)
        min_size = np.min(sizes)
        for cent in centroid_actors:
            if centroid_actors[cent]['selected']:
                if not centroid_actors[cent]['expanded']:
                    len_ = centroid_actors[cent]['length']
                    sz_ = centroid_actors[cent]['size']
                    if (len_ >= min_length and sz_ >= min_size):
                        centroid_actors[cent]['actor'].VisibilityOn()
                        cent.VisibilityOff()
                        centroid_actors[cent]['expanded'] = 1
        self.show_m.render()

    # TODO: Move to another class/module
    def __hide(self):
        centroid_actors = self.__clusters_visualizer.centroid_actors
        lengths = self.__clusters_visualizer.lengths
        sizes = self.__clusters_visualizer.sizes
        min_length = np.min(lengths)
        min_size = np.min(sizes)
        for cent in centroid_actors:
            valid_length = centroid_actors[cent]['length'] >= min_length
            valid_size = centroid_actors[cent]['size'] >= min_size
            if self.__hide_centroids:
                if valid_length or valid_size:
                    if centroid_actors[cent]['selected'] == 0:
                        cent.VisibilityOff()
            else:
                if valid_length and valid_size:
                    if centroid_actors[cent]['selected'] == 0:
                        cent.VisibilityOn()
        self.__hide_centroids = not self.__hide_centroids
        self.show_m.render()

    # TODO: Move to another class/module
    def __invert(self):
        centroid_actors = self.__clusters_visualizer.centroid_actors
        cluster_actors = self.__clusters_visualizer.cluster_actors
        lengths = self.__clusters_visualizer.lengths
        sizes = self.__clusters_visualizer.sizes
        min_length = np.min(lengths)
        min_size = np.min(sizes)
        for cent in centroid_actors:
            valid_length = centroid_actors[cent]['length'] >= min_length
            valid_size = centroid_actors[cent]['size'] >= min_size
            if valid_length and valid_size:
                centroid_actors[cent]['selected'] = (
                    not centroid_actors[cent]['selected'])
                clus = centroid_actors[cent]['actor']
                cluster_actors[clus]['selected'] = (
                    centroid_actors[cent]['selected'])
        self.show_m.render()

    def __key_press_events(self, obj, event):
        key = obj.GetKeySym()
        # TODO: Move to another class/module
        if self.cluster:
            # retract help panel
            if key in ('o', 'O'):
                panel_size = self.help_panel._get_size()
                if self.__help_visible:
                    new_pos = np.array(self.__win_size) - 10
                    self.__help_visible = False
                else:
                    new_pos = np.array(self.__win_size) - panel_size - 5
                    self.__help_visible = True
                self.help_panel._set_position(new_pos)
                self.show_m.render()
            if key in ('a', 'A'):
                self.__show_all()
            if key in ('e', 'E'):
                self.__expand()
            # hide on/off unselected centroids
            if key in ('h', 'H'):
                self.__hide()
            # invert selection
            if key in ('i', 'I'):
                self.__invert()
            if key in ('r', 'R'):
                self.__reset()
            # save current result
            if key in ('s', 'S'):
                self.__save()
            if key in ('y', 'Y'):
                self.__new_window()

    # TODO: Move to another class/module
    def __new_window(self):
        cluster_actors = self.__clusters_visualizer.cluster_actors
        tractogram_clusters = self.__clusters_visualizer.tractogram_clusters
        active_streamlines = Streamlines()
        for bundle in cluster_actors.keys():
            if bundle.GetVisibility():
                t = cluster_actors[bundle]['tractogram']
                c = cluster_actors[bundle]['cluster']
                indices = tractogram_clusters[t][c]
                active_streamlines.extend(Streamlines(indices))

        # Using the header of the first of the tractograms
        active_sft = StatefulTractogram(
            active_streamlines, self.tractograms[0], Space.RASMM)
        hz2 = Horizon(
            [active_sft], self.images, cluster=True,
            cluster_thr=self.cluster_thr/2., random_colors=self.random_colors,
            length_lt=np.inf, length_gt=0, clusters_lt=np.inf,
            clusters_gt=0, world_coords=True, interactive=True)
        ren2 = hz2.build_scene()
        hz2.build_show(ren2)

    # TODO: Move to another class/module
    def __reset(self):
        centroid_actors = self.__clusters_visualizer.centroid_actors
        lengths = self.__clusters_visualizer.lengths
        sizes = self.__clusters_visualizer.sizes
        min_length = np.min(lengths)
        min_size = np.min(sizes)
        for cent in centroid_actors:
            valid_length = centroid_actors[cent]['length'] >= min_length
            valid_size = centroid_actors[cent]['size'] >= min_size
            if valid_length and valid_size:
                centroid_actors[cent]['actor'].VisibilityOff()
                cent.VisibilityOn()
                centroid_actors[cent]['expanded'] = 0
        self.show_m.render()

    # TODO: Move to another class/module
    def __save(self):
        cluster_actors = self.__clusters_visualizer.cluster_actors
        tractogram_clusters = self.__clusters_visualizer.tractogram_clusters
        saving_streamlines = Streamlines()
        for bundle in cluster_actors.keys():
            if bundle.GetVisibility():
                t = cluster_actors[bundle]['tractogram']
                c = cluster_actors[bundle]['cluster']
                indices = tractogram_clusters[t][c]
                saving_streamlines.extend(Streamlines(indices))
        print('Saving result in tmp.trk')

        # Using the header of the first of the tractograms
        sft_new = StatefulTractogram(
            saving_streamlines, self.tractograms[0], Space.RASMM)
        save_tractogram(sft_new, 'tmp.trk', bbox_valid_check=False)
        print('Saved!')

    # TODO: Move to another class/module
    def __show_all(self):
        centroid_actors = self.__clusters_visualizer.centroid_actors
        cluster_actors = self.__clusters_visualizer.cluster_actors
        lengths = self.__clusters_visualizer.lengths
        sizes = self.__clusters_visualizer.sizes
        min_length = np.min(lengths)
        min_size = np.min(sizes)
        if self.__select_all:
            for cent in centroid_actors:
                valid_length = centroid_actors[cent]['length'] >= min_length
                valid_size = centroid_actors[cent]['size'] >= min_size
                if valid_length and valid_size:
                    centroid_actors[cent]['selected'] = 0
                    clus = centroid_actors[cent]['actor']
                    cluster_actors[clus]['selected'] = (
                        centroid_actors[cent]['selected'])
            self.__select_all = False
        else:
            for cent in centroid_actors:
                valid_length = centroid_actors[cent]['length'] >= min_length
                valid_size = centroid_actors[cent]['size'] >= min_size
                if valid_length and valid_size:
                    centroid_actors[cent]['selected'] = 1
                    clus = centroid_actors[cent]['actor']
                    cluster_actors[clus]['selected'] = (
                        centroid_actors[cent]['selected'])
            self.__select_all = True
        self.show_m.render()

    def __win_callback(self, obj, event):
        if self.__win_size != obj.GetSize():
            self.__win_size = obj.GetSize()
            if len(self.__tabs) > 0:
                self.__tab_mgr.reposition(self.__win_size)
            if self.cluster:
                if self.__help_visible:
                    panel_size = self.help_panel._get_size()
                    new_pos = np.array(self.__win_size) - panel_size - 5
                else:
                    new_pos = np.array(self.__win_size) - 10
                self.help_panel._set_position(new_pos)

    def build_scene(self):
        self.mem = GlobalHorizon()
        scene = window.Scene()
        scene.background(self.bg_color)
        return scene

    def _show_force_render(self, _element):
        """
        Callback function for lower level elements to force render.
        """
        self.show_m.render()

    def build_show(self, scene):

        title = 'Horizon ' + horizon_version
        self.show_m = window.ShowManager(
            scene, title=title, size=(1920, 1080), reset_camera=False,
            order_transparent=self.order_transparent)

        if len(self.tractograms) > 0:

            if self.cluster:
                self.__clusters_visualizer = ClustersVisualizer(
                    self.show_m, scene, self.tractograms)

            color_ind = 0

            for t, sft in enumerate(self.tractograms):
                streamlines = sft.streamlines

                if 'tracts' in self.random_colors:
                    colors = next(self.color_gen)
                else:
                    colors = None

                if not self.world_coords:
                    # TODO: Get affine from a StatefullTractogram
                    raise ValueError(
                        'Currently native coordinates are not supported for '
                        'streamlines.')

                if self.cluster:
                    self.__clusters_visualizer.add_cluster_actors(
                        t, streamlines, self.cluster_thr, colors)
                else:
                    if self.buan:
                        colors = self.buan_colors[color_ind]

                    streamline_actor = actor.line(streamlines, colors=colors)
                    streamline_actor.GetProperty().SetEdgeVisibility(1)
                    streamline_actor.GetProperty().SetRenderLinesAsTubes(1)
                    streamline_actor.GetProperty().SetLineWidth(6)
                    streamline_actor.GetProperty().SetOpacity(1)
                    scene.add(streamline_actor)

                color_ind += 1

            if self.cluster:
                # Information panel
                # It will be changed once all the elements wrapped in horizon
                # elements.
                text_block = build_label(HELP_MESSAGE, 18)
                text_block.message = HELP_MESSAGE

                self.help_panel = ui.Panel2D(
                    size=(300, 200), position=(1615, 875), color=(.8, .8, 1.),
                    opacity=.2, align='left')

                self.help_panel.add_element(text_block, coords=(.02, .01))
                scene.add(self.help_panel)
                self.__tabs.append(ClustersTab(
                    self.__clusters_visualizer, self.cluster_thr))

        sync_slices = sync_vol = False
        self.images = check_img_dtype(self.images)
        if len(self.images) > 0:
            if self.__roi_images:
                roi_color = self.__roi_colors
            roi_actors = []
            img_count = 0
            sync_slices, sync_vol = check_img_shapes(self.images)
            for img in self.images:
                title = 'Image {}'.format(img_count+1)
                data, affine, fname = unpack_image(img)
                self.vox2ras = affine
                if is_binary_image(data):
                    if self.__roi_images:
                        if 'rois' in self.random_colors:
                            roi_color = next(self.color_gen)
                        roi_actor = actor.contour_from_roi(
                            data, affine=affine, color=roi_color)
                        scene.add(roi_actor)
                        roi_actors.append(roi_actor)
                    else:
                        slices_viz = SlicesVisualizer(
                            self.show_m.iren, scene, data, affine=affine,
                            world_coords=self.world_coords,
                            percentiles=[0, 100], rgb=self.rgb)
                        self.__tabs.append(SlicesTab(
                            slices_viz, title, fname, self._show_force_render))
                        img_count += 1
                else:
                    slices_viz = SlicesVisualizer(
                        self.show_m.iren, scene, data, affine=affine,
                        world_coords=self.world_coords, rgb=self.rgb)
                    self.__tabs.append(SlicesTab(
                        slices_viz, title, fname, self._show_force_render))
                    img_count += 1
            if len(roi_actors) > 0:
                self.__tabs.append(ROIsTab(roi_actors))

        sync_peaks = False
        if len(self.pams) > 0:
            if self.images:
                sync_peaks = check_peak_size(
                    self.pams, self.images[0][0].shape[:3], sync_slices)
            else:
                sync_peaks = check_peak_size(self.pams)
            for pam in self.pams:
                peak_viz = PeaksVisualizer((pam.peak_dirs, pam.affine),
                                           self.world_coords)
                scene.add(peak_viz.actors[0])
                self.__tabs.append(PeaksTab(peak_viz.actors[0]))

        if len(self._surfaces) > 0:
            for idx, surface in enumerate(self._surfaces):
                try:
                    vertices, faces, fname = unpack_surface(surface)
                except ValueError as e:
                    warn(str(e))
                    continue
                color = next(self.color_gen)
                title = 'Surface {}'.format(idx+1)
                surf_viz = SurfaceVisualizer((vertices, faces), scene, color)
                surf_tab = SurfaceTab(surf_viz, title, fname)
                self.__tabs.append(surf_tab)

        self.__win_size = scene.GetSize()

        if len(self.__tabs) > 0:
            def on_tab_changed(actors):
                for act in actors:
                    scene.rm(act)
                    scene.add(act)

            self.__tab_mgr = TabManager(
                self.__tabs, scene.GetSize(),
                on_tab_changed, sync_slices, sync_vol, sync_peaks)

            scene.add(self.__tab_mgr.tab_ui)
            self.__tab_mgr.handle_text_overflows()

        self.show_m.initialize()

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
                self.__hide()
            if action == 'invert selection':
                self.__invert()
            if action == r'un\select all':
                self.__show_all()
            if action == 'expand clusters':
                self.__expand()
            if action == 'collapse clusters':
                self.__reset()
            if action == 'save streamlines':
                self.__save()
            if action == 'recluster':
                self.__new_window()

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

            self.show_m.add_window_callback(self.__win_callback)
            self.show_m.add_timer_callback(True, 200, timer_callback)
            self.show_m.iren.AddObserver(
                'KeyPressEvent', self.__key_press_events)

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


def horizon(tractograms=None, images=None, pams=None, surfaces=None,
            cluster=False, rgb=False, cluster_thr=15.0,
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
    surfaces : sequence of tuples
        Each tuple contains vertices and faces
    cluster : bool
        Enable QuickBundlesX clustering
    rgb: bool, optional
        Enable the color image.
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

    hz = Horizon(tractograms, images, pams, surfaces, cluster, rgb,
                 cluster_thr, random_colors, length_gt, length_lt, clusters_gt,
                 clusters_lt, world_coords, interactive, out_png,
                 recorded_events, return_showm, bg_color=bg_color,
                 order_transparent=order_transparent, buan=buan,
                 buan_colors=buan_colors, roi_images=roi_images,
                 roi_colors=roi_colors)

    scene = hz.build_scene()

    if return_showm:
        return hz.build_show(scene)
    hz.build_show(scene)
