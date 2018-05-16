import numpy as np
from dipy.workflows.workflow import Workflow
from dipy.io.streamline import load_trk, save_trk
from dipy.tracking.streamline import transform_streamlines, length
from dipy.io.image import load_nifti, save_nifti
from dipy.segment.clustering import qbx_and_merge
from dipy.viz import actor, window, ui
from dipy.viz.window import vtk


def check_range(streamline, lt, gt):
    length_s = length(streamline)
    if (length_s < gt) & (length_s > lt):
        return True
    else:
        return False


def old_horizon(tractograms, data, affine, cluster=False, cluster_thr=15.,
                random_colors=False,
                length_lt=0, length_gt=np.inf, clusters_lt=0,
                clusters_gt=np.inf):

    slicer_opacity = .8

    ren = window.Renderer()
    global centroid_actors
    centroid_actors = []

    # np.random.seed(42)
    prng = np.random.RandomState(1838)

    for streamlines in tractograms:

        if random_colors:
            colors = prng.random_sample(3)
        else:
            colors = None
        print(' Number of streamlines loaded {} \n'.format(len(streamlines)))

        if cluster:
            print(' Clustering threshold {} \n'.format(cluster_thr))
            clusters = qbx_and_merge(streamlines,
                                     [40, 30, 25, 20, cluster_thr])
            centroids = clusters.centroids
            print(' Number of centroids is {}'.format(len(centroids)))
            sizes = np.array([len(c) for c in clusters])
            linewidths = np.interp(sizes,
                                   [sizes.min(), sizes.max()], [0.1, 2.])
            visible_cluster_id = []
            print(' Minimum number of streamlines in cluster {}'
                  .format(sizes.min()))

            print(' Maximum number of streamlines in cluster {}'
                  .format(sizes.max()))

            for (i, c) in enumerate(centroids):
                # set_trace()
                if check_range(c, length_lt, length_gt):
                    if sizes[i] > clusters_lt and sizes[i] < clusters_gt:
                        act = actor.streamtube([c], colors,
                                               linewidth=linewidths[i],
                                               lod=False)
                        centroid_actors.append(act)
                        ren.add(act)
                        visible_cluster_id.append(i)
        else:
            ren.add(actor.line(streamlines, colors,
                               opacity=1.,
                               linewidth=4, lod_points=10 ** 5))

    class SimpleTrackBallNoBB(window.vtk.vtkInteractorStyleTrackballCamera):
        def HighlightProp(self, p):
            pass

    style = SimpleTrackBallNoBB()
    # very hackish way
    style.SetPickColor(0, 0, 0)
    # style.HighlightProp(None)
    show_m = window.ShowManager(ren, size=(1200, 900), interactor_style=style)
    show_m.initialize()

    if data is not None:
        # from dipy.core.geometry import rodrigues_axis_rotation
        # affine[:3, :3] = np.dot(affine[:3, :3], rodrigues_axis_rotation((0, 0, 1), 45))

        image_actor = actor.slicer(data, affine)
        image_actor.opacity(slicer_opacity)
        image_actor.SetInterpolate(False)
        ren.add(image_actor)

        ren.add(actor.axes((10, 10, 10)))

        def change_slice(obj, event):
            z = int(np.round(obj.get_value()))
            # image_actor.display(None, None, z)
            image_actor.display(None, None, z)

        line_slider_z = ui.LineSlider2D(min_value=0,
                                        max_value=data.shape[2] - 1,
                                        initial_value=data.shape[2] / 2,
                                        text_template="{value:.0f}",
                                        length=140)
        panel = ui.Panel2D(center=(1030, 120),
                           size=(300, 200),
                           color=(1, 1, 1),
                           opacity=0.1,
                           align="right")

        panel.add_element(line_slider_z, 'relative', (0.65, 0.4))


        """
        slider = widget.slider(show_m.iren, show_m.ren,
                               callback=change_slice,
                               min_value=0,
                               max_value=image_actor.shape[1] - 1,
                               value=image_actor.shape[1] / 2,
                               label="Move slice",
                               right_normalized_pos=(.98, 0.6),
                               size=(120, 0), label_format="%0.lf",
                               color=(1., 1., 1.),
                               selected_color=(0.86, 0.33, 1.))
        """
    global size
    size = ren.GetSize()
    # ren.background((1, 0.5, 0))
    global picked_actors
    picked_actors = {}

    def pick_callback(obj, event):
        global centroid_actors
        global picked_actors

        print('Inside pick callback')
        prop = obj.GetProp3D()

        ac = np.array(centroid_actors)
        index = np.where(ac == prop)[0]

        if len(index) > 0:
            try:
                bundle = picked_actors[prop]
                ren.rm(bundle)
                del picked_actors[prop]
            except:
                bundle = actor.line(clusters[visible_cluster_id[index]],
                                    lod=False)
                picked_actors[prop] = bundle
                ren.add(bundle)

        if prop in picked_actors.values():
            ren.rm(prop)

    def win_callback(obj, event):
        global size, panel
        if size != obj.GetSize():

            if data is not None:

                size_old = size
                size = obj.GetSize()
                size_change = [size[0] - size_old[0], 0]
                panel.re_align(size_change)
                #slider.place(ren)

            size = obj.GetSize()

    global centroid_visibility
    centroid_visibility = True

    def key_press(obj, event):
        print('Inside key_press')
        global centroid_visibility
        key = obj.GetKeySym()
        if key == 'h' or key == 'H':
            if cluster:
                if centroid_visibility is True:
                    for ca in centroid_actors:
                        ca.VisibilityOff()
                    centroid_visibility = False
                else:
                    for ca in centroid_actors:
                        ca.VisibilityOn()
                    centroid_visibility = True
                show_m.render()



    show_m.initialize()

    show_m.ren.add(panel)
    show_m.iren.AddObserver('KeyPressEvent', key_press)
    show_m.add_window_callback(win_callback)
    show_m.add_picker_callback(pick_callback)
    show_m.render()
    show_m.start()


def horizon(tractograms, data, affine, cluster, cluster_thr, random_colors,
            length_lt, length_gt, clusters_lt, clusters_gt):

    world_coords = True
    interactive = True

    prng = np.random.RandomState(27) #1838
    global centroid_actors
    centroid_actors = []

#    if not world_coords:
#        from dipy.tracking.streamline import transform_streamlines
#        streamlines = transform_streamlines(streamlines, np.linalg.inv(affine))

    ren = window.Renderer()
    for streamlines in tractograms:
        if random_colors:
            colors = prng.random_sample(3)
        else:
            colors = None

        if cluster:

            print(' Clustering threshold {} \n'.format(cluster_thr))
            clusters = qbx_and_merge(streamlines,
                                     [40, 30, 25, 20, cluster_thr])
            centroids = clusters.centroids
            print(' Number of centroids is {}'.format(len(centroids)))
            sizes = np.array([len(c) for c in clusters])
            linewidths = np.interp(sizes,
                                   [sizes.min(), sizes.max()], [0.1, 2.])
            visible_cluster_id = []
            print(' Minimum number of streamlines in cluster {}'
                  .format(sizes.min()))

            print(' Maximum number of streamlines in cluster {}'
                  .format(sizes.max()))

            for (i, c) in enumerate(centroids):
                # set_trace()
                if check_range(c, length_lt, length_gt):
                    if sizes[i] > clusters_lt and sizes[i] < clusters_gt:
                        act = actor.streamtube([c], colors,
                                               linewidth=linewidths[i],
                                               lod=False)
                        centroid_actors.append(act)
                        ren.add(act)
                        visible_cluster_id.append(i)

        else:
            streamline_actor = actor.line(streamlines, colors=colors)
            # streamline_actor.GetProperty().SetEdgeVisibility(1)
            streamline_actor.GetProperty().SetRenderLinesAsTubes(1)
            streamline_actor.GetProperty().SetLineWidth(6)
            streamline_actor.GetProperty().SetOpacity(1)
            ren.add(streamline_actor)

    if data is not None:
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

        # ren.add(stream_actor)
        ren.add(image_actor_z)
        ren.add(image_actor_x)
        ren.add(image_actor_y)

    show_m = window.ShowManager(ren, size=(1200, 900))
    show_m.initialize()

    if data is not None:

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

        show_m.ren.add(panel)

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
        print('Inside pick_callbacks')
        global centroid_actors
        global picked_actors

        prop = obj  # GetProp3D()
        prop.GetProperty().SetOpacity(0.5)

        return
        ac = np.array(centroid_actors)
        index = np.where(ac == prop)[0]

        if len(index) > 0:
            try:
                bundle = picked_actors[prop]
                ren.rm(bundle)
                del picked_actors[prop]
            except:
                bundle = actor.line(clusters[visible_cluster_id[index]],
                                    lod=False)
                picked_actors[prop] = bundle
                ren.add(bundle)

        if prop in picked_actors.values():
            ren.rm(prop)

    for act in centroid_actors:

        act.AddObserver('LeftButtonPressEvent', pick_callback, 1.0)

    global centroid_visibility
    centroid_visibility = True

    def key_press(obj, event):
        print('Inside key_press')
        global centroid_visibility
        key = obj.GetKeySym()
        if key == 'h' or key == 'H':
            if cluster:
                if centroid_visibility is True:
                    for ca in centroid_actors:
                        ca.VisibilityOff()
                    centroid_visibility = False
                else:
                    for ca in centroid_actors:
                        ca.VisibilityOn()
                    centroid_visibility = True
                show_m.render()

    """
    Finally, please set the following variable to ``True`` to interact with the
    datasets in 3D.
    """



    ren.zoom(1.5)
    ren.reset_clipping_range()



    if interactive:

        show_m.add_window_callback(win_callback)
        #show_m.iren.AddObserver('KeyPressEvent', key_press)
        #show_m.iren.AddObserver("EndPickEvent", pick_callback)
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
                if verbose:
                    print(affine)
            else:
                data = None
                affine = None

        horizon(tractograms, data, affine, cluster, cluster_thr, random_colors,
                length_lt, length_gt, clusters_lt, clusters_gt)
