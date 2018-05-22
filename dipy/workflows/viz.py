import numpy as np
from dipy.workflows.workflow import Workflow
from dipy.io.streamline import load_trk, save_trk
from dipy.tracking.streamline import transform_streamlines, length, Streamlines
from dipy.io.image import load_nifti, save_nifti
from dipy.segment.clustering import qbx_and_merge
from dipy.viz import actor, window, ui
from dipy.viz.window import vtk
from dipy.viz.utils import get_polydata_lines


def check_range(streamline, lt, gt):
    length_s = length(streamline)
    if (length_s < gt) & (length_s > lt):
        return True
    else:
        return False


def slicer_panel(renderer, data, affine, world_coords):

    #renderer = showm.ren
    shape = data.shape
    if not world_coords:
        image_actor_z = actor.slicer(data, affine=np.eye(4))
    else:
        image_actor_z = actor.slicer(data, affine)

    slicer_opacity = 0.6
    image_actor_z.opacity(slicer_opacity)

    image_actor_x = image_actor_z.copy()
    x_midpoint = int(np.round(shape[0] / 2))
    image_actor_x.display_extent(x_midpoint,
                                 x_midpoint, 0,
                                 shape[1] - 1,
                                 0,
                                 shape[2] - 1)

    image_actor_y = image_actor_z.copy()
    y_midpoint = int(np.round(shape[1] / 2))
    image_actor_y.display_extent(0,
                                 shape[0] - 1,
                                 y_midpoint,
                                 y_midpoint,
                                 0,
                                 shape[2] - 1)

    renderer.add(image_actor_z)
    renderer.add(image_actor_x)
    renderer.add(image_actor_y)

    line_slider_z = ui.LineSlider2D(min_value=0,
                                    max_value=shape[2] - 1,
                                    initial_value=shape[2] / 2,
                                    text_template="{value:.0f}",
                                    length=140)

    line_slider_x = ui.LineSlider2D(min_value=0,
                                    max_value=shape[0] - 1,
                                    initial_value=shape[0] / 2,
                                    text_template="{value:.0f}",
                                    length=140)

    line_slider_y = ui.LineSlider2D(min_value=0,
                                    max_value=shape[1] - 1,
                                    initial_value=shape[1] / 2,
                                    text_template="{value:.0f}",
                                    length=140)

    opacity_slider = ui.LineSlider2D(min_value=0.0,
                                     max_value=1.0,
                                     initial_value=slicer_opacity,
                                     length=140)

    def change_slice_z(i_ren, obj, slider):
        z = int(np.round(slider.value))
        image_actor_z.display_extent(0, shape[0] - 1,
                                     0, shape[1] - 1, z, z)

    def change_slice_x(i_ren, obj, slider):
        x = int(np.round(slider.value))
        image_actor_x.display_extent(x, x, 0, shape[1] - 1, 0,
                                     shape[2] - 1)

    def change_slice_y(i_ren, obj, slider):
        y = int(np.round(slider.value))
        image_actor_y.display_extent(0, shape[0] - 1, y, y,
                                     0, shape[2] - 1)

    def change_opacity(i_ren, obj, slider):
        slicer_opacity = slider.value
        image_actor_z.opacity(slicer_opacity)
        image_actor_x.opacity(slicer_opacity)
        image_actor_y.opacity(slicer_opacity)

    line_slider_z.add_callback(line_slider_z.slider_disk,
                               "MouseMoveEvent",
                               change_slice_z)
    line_slider_z.add_callback(line_slider_z.slider_line,
                               "LeftButtonPressEvent",
                               change_slice_z)

    line_slider_x.add_callback(line_slider_x.slider_disk,
                               "MouseMoveEvent",
                               change_slice_x)
    line_slider_x.add_callback(line_slider_x.slider_line,
                               "LeftButtonPressEvent",
                               change_slice_x)

    line_slider_y.add_callback(line_slider_y.slider_disk,
                               "MouseMoveEvent",
                               change_slice_y)
    line_slider_y.add_callback(line_slider_y.slider_line,
                               "LeftButtonPressEvent",
                               change_slice_y)

    opacity_slider.add_callback(opacity_slider.slider_disk,
                                "MouseMoveEvent",
                                change_opacity)
    opacity_slider.add_callback(opacity_slider.slider_line,
                                "LeftButtonPressEvent",
                                change_opacity)

    def build_label(text):
        label = ui.TextBlock2D()
        label.message = text
        label.font_size = 18
        label.font_family = 'Arial'
        label.justification = 'left'
        label.bold = False
        label.italic = False
        label.shadow = False
        label.actor.GetTextProperty().SetBackgroundColor(0, 0, 0)
        label.actor.GetTextProperty().SetBackgroundOpacity(0.0)
        label.color = (1, 1, 1)

        return label

    line_slider_label_z = build_label(text="Z Slice")
    line_slider_label_x = build_label(text="X Slice")
    line_slider_label_y = build_label(text="Y Slice")
    opacity_slider_label = build_label(text="Opacity")

    panel = ui.Panel2D(center=(1030, 120),
                       size=(300, 200),
                       color=(1, 1, 1),
                       opacity=0.1,
                       align="right")

    panel.add_element(line_slider_label_x, 'relative', (0.1, 0.75))
    panel.add_element(line_slider_x, 'relative', (0.65, 0.8))
    panel.add_element(line_slider_label_y, 'relative', (0.1, 0.55))
    panel.add_element(line_slider_y, 'relative', (0.65, 0.6))
    panel.add_element(line_slider_label_z, 'relative', (0.1, 0.35))
    panel.add_element(line_slider_z, 'relative', (0.65, 0.4))
    panel.add_element(opacity_slider_label, 'relative', (0.1, 0.15))
    panel.add_element(opacity_slider, 'relative', (0.65, 0.2))

    #showm.ren.add(panel)
    renderer.add(panel)
    return panel


def horizon(tractograms, images, cluster, cluster_thr, random_colors,
            length_lt, length_gt, clusters_lt, clusters_gt):

    world_coords = True
    interactive = True
    global select_all
    select_all = False

    prng = np.random.RandomState(27) # 1838
    global centroid_actors, cluster_actors, visible_centroids, visible_clusters
    global cluster_access
    centroid_actors = {}
    cluster_actors = {}
    global tractogram_clusters, text_block
    tractogram_clusters = {}

    # cluster_actor_access = {}

    ren = window.Renderer()
    for (t, streamlines) in enumerate(tractograms):
        if random_colors:
            colors = prng.random_sample(3)
        else:
            colors = None

        """
        if not world_coords:
            # !!! Needs AFFINE from header or image
            streamlines = transform_streamlines(streamlines,
                                                np.linalg.inv(affine))
        """

        if cluster:

            text_block = ui.TextBlock2D()
            text_block.message = \
                ' >> a: show all, c: on/off centroids, s: save in file'

            ren.add(text_block.get_actor())
            print(' Clustering threshold {} \n'.format(cluster_thr))
            clusters = qbx_and_merge(streamlines,
                                     [40, 30, 25, 20, cluster_thr])
            tractogram_clusters[t] = clusters
            centroids = clusters.centroids
            print(' Number of centroids is {}'.format(len(centroids)))
            sizes = np.array([len(c) for c in clusters])
            linewidths = np.interp(sizes,
                                   [sizes.min(), sizes.max()], [0.1, 2.])

            print(' Minimum number of streamlines in cluster {}'
                  .format(sizes.min()))

            print(' Maximum number of streamlines in cluster {}'
                  .format(sizes.max()))

            print(' Construct cluster actors')
            for (i, c) in enumerate(centroids):
                if check_range(c, length_lt, length_gt):
                    if sizes[i] > clusters_lt and sizes[i] < clusters_gt:
                        act = actor.streamtube([c], colors,
                                               linewidth=linewidths[i],
                                               lod=False)

                        ren.add(act)

                        bundle = actor.line(clusters[i],
                                            lod=False)
                        bundle.GetProperty().SetRenderLinesAsTubes(1)
                        bundle.GetProperty().SetLineWidth(6)
                        bundle.GetProperty().SetOpacity(1)
                        bundle.VisibilityOff()
                        ren.add(bundle)

                        # Every centroid actor is paired to a cluster actor
                        centroid_actors[act] = {
                            'pair': bundle, 'cluster': i, 'tractogram': t}
                        cluster_actors[bundle] = {
                            'pair': act, 'cluster': i, 'tractogram': t}

        else:
            streamline_actor = actor.line(streamlines, colors=colors)
            # streamline_actor.GetProperty().SetEdgeVisibility(1)
            streamline_actor.GetProperty().SetRenderLinesAsTubes(1)
            streamline_actor.GetProperty().SetLineWidth(6)
            streamline_actor.GetProperty().SetOpacity(1)
            ren.add(streamline_actor)

    show_m = window.ShowManager(ren, size=(1200, 900))
    show_m.initialize()

    if len(images) > 0:

        # !!Only first image loading supported')
        data, affine = images[0]
        panel = slicer_panel(ren, data, affine, world_coords)
        # show_m.ren.add(panel)

    global size
    size = ren.GetSize()

    def win_callback(obj, event):
        global size
        if size != obj.GetSize():
            size_old = size
            size = obj.GetSize()
            size_change = [size[0] - size_old[0], 0]
            if data is not None:
                panel.re_align(size_change)

    show_m.initialize()

    global picked_actors
    picked_actors = {}

    def pick_callback(obj, event):

        try:
            paired_obj = cluster_actors[obj]['pair']
            obj.SetVisibility(not obj.GetVisibility())
            paired_obj.SetVisibility(not paired_obj.GetVisibility())

        except KeyError:
            pass

        try:
            paired_obj = centroid_actors[obj]['pair']
            obj.SetVisibility(not obj.GetVisibility())
            paired_obj.SetVisibility(not paired_obj.GetVisibility())

        except KeyError:
            pass


    for act in centroid_actors:

        act.AddObserver('LeftButtonPressEvent', pick_callback, 1.0)

    for cl in cluster_actors:

        cl.AddObserver('LeftButtonPressEvent', pick_callback, 1.0)


    # for prop in picked_actors.values():
    #   prop.AddObserver('LeftButtonPressEvent', pick_callback, 1.0)


    global centroid_visibility
    centroid_visibility = True

    def key_press(obj, event):
        print('Inside key_press')
        global centroid_visibility, select_all, tractogram_clusters
        key = obj.GetKeySym()
        if cluster:
            if key == 'c' or key == 'C':
                if centroid_visibility is True:
                    for ca in centroid_actors:
                        ca.VisibilityOff()
                    centroid_visibility = False
                else:
                    for ca in centroid_actors:
                        ca.VisibilityOn()
                    centroid_visibility = True
                show_m.render()
            if key == 'a' or key == 'A':
                if select_all:
                    for bundle in cluster_actors.keys():
                        bundle.VisibilityOn()
                        cluster_actors[bundle]['pair'].VisibilityOff()
                else:
                    for bundle in cluster_actors.keys():
                        bundle.VisibilityOff()
                        cluster_actors[bundle]['pair'].VisibilityOn()

                select_all = not select_all
                show_m.render()

            if key == 's' or key == 'S':
                saving_streamlines = Streamlines()
                for bundle in cluster_actors.keys():
                    if bundle.GetVisibility():
                        t = cluster_actors[bundle]['tractogram']
                        c = cluster_actors[bundle]['cluster']
                        indices = tractogram_clusters[t][c]
                        saving_streamlines.extend(Streamlines(indices))
                print('Saving result in tmp.trk')
                save_trk('tmp.trk', saving_streamlines, np.eye(4))


    ren.zoom(1.5)
    ren.reset_clipping_range()

    if interactive:

        show_m.add_window_callback(win_callback)
        show_m.iren.AddObserver('KeyPressEvent', key_press)
        # show_m.iren.AddObserver("EndPickEvent", pick_callback)
        show_m.render()
        show_m.start()

    else:

        window.record(ren, out_path='bundles_and_3_slices.png',
                      size=(1200, 900),
                      reset_camera=False)


class HorizonFlow(Workflow):

    @classmethod
    def get_short_name(cls):
        return 'horizon'

    def run(self, input_files, cluster=False, cluster_thr=15.,
            random_colors=False,
            length_lt=0, length_gt=1000,
            clusters_lt=0, clusters_gt=10**8):
        """ Advanced visualization utility

        Parameters
        ----------
        input_files : variable string
        cluster : bool
        cluster_thr : float
        random_colors : bool
        length_lt : float
        length_gt : float
        clusters_lt : int
        clusters_gt : int
        """
        verbose = True
        tractograms = []
        images = []

        for f in input_files:

            if verbose:
                print('Loading file ...')
                print(f)
                print('\n')

            if f.endswith('.trk'):

                streamlines, hdr = load_trk(f)
                tractograms.append(streamlines)

            if f.endswith('.nii.gz') or f.endswith('.nii'):

                data, affine = load_nifti(f)
                images.append((data, affine))
                if verbose:
                    print(affine)

        horizon(tractograms, images, cluster, cluster_thr,
                random_colors, length_lt, length_gt, clusters_lt,
                clusters_gt)
