# TODO: calculate also the view_up

from dipy.align.streamlinear import StreamlineLinearRegistration
from dipy.tracking.streamline import (set_number_of_points,
                                      select_random_set_of_streamlines,
                                      unlist_streamlines,
                                      transform_streamlines)
from dipy.viz import actor, window, utils, fvtk
from dipy.data.fetcher import fetch_bundles_2_subjects, read_bundles_2_subjects
import numpy as np


def vtk_matrix_to_numpy(matrix):
    import vtk
    """ Converts VTK matrix to numpy array.
    """
    if matrix is None:
        return None

    size = (4, 4)
    if isinstance(matrix, vtk.vtkMatrix3x3):
        size = (3, 3)

    mat = np.zeros(size)
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            mat[i, j] = matrix.GetElement(i, j)

    return mat


def numpy_to_vtk_matrix(array):
    """ Converts a numpy array to a VTK matrix.
    """
    import vtk
    if array is None:
        return None

    if array.shape == (4, 4):
        matrix = vtk.vtkMatrix4x4()
    elif array.shape == (3, 3):
        matrix = vtk.vtkMatrix3x3()
    else:
        raise ValueError("Invalid matrix shape: {0}".format(array.shape))

    for i in range(array.shape[0]):
        for j in range(array.shape[1]):
            matrix.SetElement(i, j, array[i, j])

    return matrix


def show_both_bundles(bundles, colors=None, size=(1080, 600),
                      show=False, fname=None):

    ren = window.Renderer()
    ren.background((1., 1, 1))

    positions = np.zeros(3)
    focal_points = np.zeros(3)
    for (i, bundle) in enumerate(bundles):
        color = colors[i]
        lines = actor.line(bundle, color, linewidth=1.5)
        ren.add(lines)
        position, focal_point, view_up, _, _ = utils.auto_camera(lines,
                                                                 15, 'max')
        positions += position
        focal_points += focal_point

    positions = positions/float(len(bundles))
    focal_points = focal_points/float(len(bundles))

    ren.set_camera(positions, focal_points, view_up)
    # ren.reset_clipping_range()
    # ren.reset_camera()

    if show:
        window.show(ren, size=size, reset_camera=False)

    if fname is not None:
        window.record(ren, n_frames=1, out_path=fname, size=size)


def write_movie(bundles, transforms, size=(1280, 720), video_fname='test.avi'):

    global show_m, mw, moving_actor, cnt, first_message, time, moved_actor, cnt_trans, stamp_time

    first_message = actor.text_overlay('Streamline-based Linear Registration (SLR)',
                                       position=(size[0]/2 - 300, size[1]/2),
                                       color=(0, 0, 0),
                                       font_size=32)

    ref_message = actor.text_overlay('Garyfallidis et al., Neuroimage 2015',
                                     position=(10, 10),
                                     color=(0, 0, 0),
                                     font_size=14,
                                     font_family='Times', bold=False)

    iteration_message = actor.text_overlay('SLR \nIterations',
                                           position=(10, 650),
                                           color=(0, 0, 0),
                                           font_size=14,
                                           font_family='Times', bold=True)

    description_message = actor.text_overlay('The SLR is an algorithm for linear registration \nof streamlines using streamline distances',
                                             position=(1280/2, 10),
                                             color=(0, 0, 0),
                                             font_size=20,
                                             justification='center',
                                             bold=False)

    static_bundle = bundles[0]
    moving_bundle = bundles[1]
    moved_bundle = bundles[2]

    ren = window.Renderer()
    ren.background((1., 1., 1))

    ren.add(first_message)
    ren.add(ref_message)
    ren.add(iteration_message)
    ren.add(description_message)

    static_actor = actor.line(static_bundle, fvtk.colors.red, linewidth=1.5)
    moving_actor = actor.line(moving_bundle, fvtk.colors.orange, linewidth=1.5)
    moved_actor = actor.line(moved_bundle, fvtk.colors.orange, linewidth=1.5)

    static_pts, _ = unlist_streamlines(set_number_of_points(static_bundle, 20))
    moved_pts, _ = unlist_streamlines(set_number_of_points(moved_bundle, 20))

    static_dots = fvtk.dots(static_pts, fvtk.colors.red)
    moved_dots = fvtk.dots(moved_pts, fvtk.colors.orange)

    ren.add(static_actor)
    ren.add(moving_actor)

    show_m = window.ShowManager(ren, size=size, order_transparent=False)
    show_m.initialize()

    position, focal_point, view_up, _, _ = utils.auto_camera(static_actor,
                                                             15, 'max')
    ren.reset_camera()
    view_up = (0, 0., 1)
    ren.set_camera(position, focal_point, view_up)
    ren.zoom(1.5)
    ren.reset_clipping_range()

    repeat_time = 20
    cnt = 0
    time = 0

    np.set_printoptions(3, suppress=True)

    cnt_trans = 0
    stamp_time = 0

    def apply_transformation():
        global cnt_trans
        if cnt_trans < len(transforms):
            mat = transforms[cnt_trans]
            moving_actor.SetUserMatrix(numpy_to_vtk_matrix(mat))
            iteration_message.set_message('SLR \nIteration #' + str(cnt_trans))
            cnt_trans += 1

    def timer_callback(obj, event):
        global cnt, time, cnt_trans, stamp_time

        if time == 3000:

            first_message.VisibilityOff()

        if time > 1000 and cnt % 10 == 0:
            apply_transformation()

        if cnt_trans == len(transforms) and stamp_time==0:

            print('Show description')
            msg = 'Registration finished! \n Visualizing only the points to highlight the overlap.'
            description_message.set_message(msg)
            description_message.Modified()

            stamp_time = time
            ren.rm(static_actor)
            ren.rm(moving_actor)
            ren.add(static_dots)
            ren.add(moved_dots)
            static_dots.GetProperty().SetOpacity(.5)
            moved_dots.GetProperty().SetOpacity(.5)

        if time == stamp_time + 5000:

            ren.add(static_actor)
            ren.add(moving_actor)
            ren.rm(static_dots)
            ren.rm(moved_dots)
            msg = 'Showing final registration.'
            description_message.set_message(msg)
            description_message.Modified()

        if time == 20000:

            first_message.VisibilityOn()

        ren.reset_clipping_range()
        show_m.ren.azimuth(.4)
        show_m.render()
        mw.write()

        #print('Time %d' % (time,))
        #print('Cnt %d' % (cnt,))
        #print('Len transforms %d' % (cnt_trans,))
        time = repeat_time * cnt
        cnt += 1

    mw = window.MovieWriter(video_fname, show_m.window)
    mw.start()

    show_m.add_timer_callback(True, repeat_time, timer_callback)

    show_m.render()
    show_m.start()

    del mw
    del show_m


fetch_bundles_2_subjects()

subj1 = read_bundles_2_subjects('subj_1', ['fa', 't1'],
                                ['af.left', 'cst.left', 'cst.right', 'cc_1', 'cc_2'])

subj2 = read_bundles_2_subjects('subj_2', ['fa', 't1'],
                                ['af.left', 'cst.left', 'cst.right', 'cc_1', 'cc_2'])


for bundle_type in ['af.left']:  # 'cst.right', 'cc_1']:

    bundle1 = subj1[bundle_type]
    bundle2 = subj2[bundle_type]

    sbundle1 = set_number_of_points(bundle1, 20)
    sbundle2 = set_number_of_points(bundle2, 20)

    sbundle1 = select_random_set_of_streamlines(sbundle1, 400)
    sbundle2 = select_random_set_of_streamlines(sbundle2, 400)

#    show_both_bundles([bundle1, bundle2],
#                      colors=[fvtk.colors.orange, fvtk.colors.red],
#                      show=True)

    slr = StreamlineLinearRegistration(x0='affine', evolution=True)
    slm = slr.optimize(static=sbundle1, moving=sbundle2)

    bundle2_moved = slm.transform(bundle2)

#    show_both_bundles([bundle1, bundle2_moved],
#                      colors=[fvtk.colors.orange, fvtk.colors.red],
#                      show=True)


bundles = []
bundles.append(bundle1)
bundles.append(bundle2)

transforms = []

for mat in slm.matrix_history:
    transforms.append(mat)
transforms.append(slm.matrix)

bundles.append(bundle2_moved)

from time import time as tim

t = tim()

write_movie(bundles, transforms, video_fname='slr_1.avi')

print('Done in %0.3f' % (tim() - t, ))
