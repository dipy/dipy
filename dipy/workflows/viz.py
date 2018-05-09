import numpy as np
from dipy.workflows.workflow import Workflow
from dipy.io.streamline import load_trk, save_trk
from dipy.tracking.streamline import transform_streamlines, length
from dipy.io.image import load_nifti, save_nifti
from dipy.segment.clustering import qbx_and_merge
from dipy.viz import actor, window, ui


def check_range(streamline, lt, gt):
    length_s = length(streamline)
    if (length_s < gt) & (length_s > lt):
        return True
    else:
        return False


def horizon(tractograms, data, affine, cluster=False, cluster_thr=15.,
            random_colors=False,
            length_lt=0, length_gt=np.inf, clusters_lt=0, clusters_gt=np.inf):

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
        global size
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

        horizon(tractograms, data, affine, cluster, cluster_thr, random_colors,
                length_lt, length_gt, clusters_lt, clusters_gt)
