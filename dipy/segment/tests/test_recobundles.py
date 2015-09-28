import numpy as np
import numpy.testing as npt
import nibabel.trackvis as tv
from dipy.tracking.streamline import (transform_streamlines,
                                      select_random_set_of_streamlines,
                                      set_number_of_points)
from copy import deepcopy
from itertools import chain
from dipy.segment.bundles import RecoBundles
from dipy.viz import fvtk
from dipy.align.bundlemin import distance_matrix_mdf


def show_bundles(static, moving, linewidth=1., tubes=False,
                 opacity=1., fname=None):

    ren = fvtk.ren()
    ren.clear()
    ren.SetBackground(1, 1, 1.)

    if tubes:
        static_actor = fvtk.streamtube(static, fvtk.colors.red,
                                       linewidth=linewidth, opacity=opacity)
        moving_actor = fvtk.streamtube(moving, fvtk.colors.green,
                                       linewidth=linewidth, opacity=opacity)

    else:
        static_actor = fvtk.line(static, fvtk.colors.red,
                                 linewidth=linewidth, opacity=opacity)
        moving_actor = fvtk.line(moving, fvtk.colors.green,
                                 linewidth=linewidth, opacity=opacity)

    fvtk.add(ren, static_actor)
    fvtk.add(ren, moving_actor)

    fvtk.add(ren, fvtk.axes(scale=(20, 20, 20)))

    fvtk.show(ren, size=(900, 900))
    if fname is not None:
        fvtk.record(ren, size=(900, 900), out_path=fname)


def show_grid(list_of_streamlines, list_of_captions, linewidth=1., opacity=1., dim=None):

    actors = []
    for streamlines in list_of_streamlines:
        actors.append(fvtk.line(streamlines, fvtk.colors.red, linewidth=linewidth, opacity=opacity))

    from dipy.viz import actor, window, utils
    from dipy.viz.interactor import InteractorStyleBundlesGrid

    new_actors = []
    for act in actors:
        new_actors.append(utils.auto_orient(act, (0, 0, -1),
                                            data_up=(0, 0, 1)))

    caption_actors = []
    for caption in list_of_captions:
        caption_actors.append(actor.text_3d(caption, justification='center'))

    ren = window.Renderer()
    ren.projection('parallel')

    ren.add(actor.grid(new_actors, caption_actors,
                       dim=dim, cell_padding=0, cell_shape='diagonal'))
    show_m = window.ShowManager(ren, size=(900, 900),
                                #interactor_style=InteractorStyleImageAndTrackballActor())
                                interactor_style=InteractorStyleBundlesGrid(new_actors))
    show_m.initialize()
    show_m.render()
    show_m.start()


def test_recognition():

    disp = True
    dname = '/home/eleftherios/Data/ISMRM_2015_challenge_bundles_RAS/'

    bundle_trk = ['CA', 'CC', 'Cingulum_left',
                  'Cingulum_right', 'CP',
                  'CST_left', 'CST_right',
                  'Fornix', 'FPT_left', 'FPT_right',
                  'ICP_left', 'ICP_right',
                  'IOFF_left', 'IOFF_right', 'MCP',
                  'OR_left', 'OR_right',
                  'POPT_left', 'POPT_right',
                  'SCP_left', 'SCP_right',
                  'SLF_left', 'SLF_right',
                  'UF_left', 'UF_right']

    fnames = [dname + bundle_name + '.trk' for bundle_name in bundle_trk]

    model_bundles_dix = {}
    model_indices_dix = {}

    cnt = 0

    for (i, fname) in enumerate(fnames):
        streams, hdr = tv.read(fname, points_space='rasmm')
        bundle = [s[0] for s in streams]
        key = bundle_trk[i].split('.trk')[0]
        model_bundles_dix[key] = bundle
        model_indices_dix[key] = cnt + np.arange(len(bundle))
        cnt = cnt + len(bundle)

    play_bundles_dix = deepcopy(model_bundles_dix)

    mat = np.eye(4)
    # mat[:3, 3] = np.array([0., 0, 0])
    mat[:3, 3] = np.array([-5, 5, 0])

    # tag = 'MCP'
    # tag = 'Fornix'
    # tag = 'Cingulum_right'
    # tag = 'CST_right'
    # tag = 'CST_left'
    tag = 'POPT_left'

    play_bundles_dix[tag] = transform_streamlines(play_bundles_dix[tag], mat)

    model_bundle = model_bundles_dix[tag]

    # make sure that you put the bundles the correct order for the
    # classification tests
    streamlines = []

    for (i, f) in enumerate(fnames):
        streamlines += play_bundles_dix[bundle_trk[i]]

    # show_bundles(model_bundle, streamlines)

    rb = RecoBundles(streamlines, mdf_thr=15)
    recognized_bundle = rb.recognize(model_bundle, mdf_thr=5,
                                     reduction_thr=20,
                                     slr=True,
                                     slr_metric='static',
                                     slr_x0='scaling',
                                     slr_bounds=[(-20, 20), (-20, 20), (-20, 20), (-45, 45), (-45, 45), (-45, 45), (0.8, 1.2), (0.8, 1.2), (0.8, 1.2)],
                                     slr_select=(400, 400),
                                     slr_method='L-BFGS-B',
                                     slr_use_centroids=False,
                                     slr_progressive=True,
                                     pruning_thr=5)

    if disp:

        print('Show model centroids and all centroids of new space')
        show_bundles(rb.model_centroids, rb.centroids)

        print('Show model bundle and neighborhood')
        show_bundles(model_bundle, rb.neighb_streamlines)

        print('Show model bundle and transformed neighborhood')
        show_bundles(model_bundle, rb.transf_streamlines)

        print('Show model bundles and pruned streamlines')
        show_bundles(model_bundle, recognized_bundle)

        mat2 = np.eye(4)
        mat2[:3, 3] = np.array([60, 0, 0])

        print('Same with a shift')
        show_bundles(transform_streamlines(model_bundle, mat2),
                     recognized_bundle)

        print('Show initial labels vs model bundle')
        show_bundles(transform_streamlines(rb.labeled_streamlines, mat2),
                     model_bundle)

    print('\a')
    print('Recognized bundle has %d streamlines' % (len(recognized_bundle),))
    print('Model bundle has %d streamlines' % (len(model_bundle),))
    print('\a')

    def investigate_space():

        moving = set_number_of_points(rb.transf_streamlines, 20)
        static = set_number_of_points(rb.model_bundle, 20)
        moving = select_random_set_of_streamlines(moving, 400)
        static = select_random_set_of_streamlines(static, 400)

        A = []

        for x in np.arange(-40, 40, 2):
            for y in np.arange(-40, 40, 2):
                print(x, y)
                tmat = np.eye(4)
                tmat[:3, 3] = np.array([x, y, 0])
                moved = transform_streamlines(moving, tmat)
                d01 = distance_matrix_mdf(static, moved)

                rows, cols = d01.shape
                bmd = 0.25 * (np.sum(np.min(d01, axis=0)) / float(cols) +
                    np.sum(np.min(d01, axis=1)) / float(rows)) ** 2

                A.append(bmd)
        A = np.array(A)
        A = A.reshape(40, 40)

        return A

    # A = investigate_space()

    # intersection = np.intersect1d(model_indices_dix['MCP'], rb.labels)
    difference = np.setdiff1d(rb.labels, model_indices_dix[tag])
    print('Difference %d' % (len(difference),))

#    figure()
#    A = np.sqrt(rb.slr_initial_matrix)
#    A = np.sort(np.sort(A, axis=0), axis=1)
#    imshow(A, vmin=0, vmax=6.5)
#    colorbar()
#    B = np.sqrt(rb.slr_final_matrix)
#    B = np.sort(np.sort(B, axis=0), axis=1)
#    figure()
#    imshow(B, vmin=0, vmax=6.5)
#    colorbar()

    from ipdb import set_trace

    set_trace()

    print('\a')
    print('Build the KDTree for this bundle')
    print('Start expansion')
    print('\a')

    rb.build_kdtree(mam_metric=None)

    dists, actual_indices, expansion_streamlines = rb.expand(300, True)

    expansion_intersection = np.intersect1d(actual_indices, rb.labels)
    print(len(expansion_intersection))
    npt.assert_equal(len(expansion_intersection), 0)

    if disp:
        show_bundles(recognized_bundle, expansion_streamlines, tubes=False)

    print('Start reduction')

    nb_reduced = 100

    dists, actual_indices, reduced_streamlines = rb.reduce(nb_reduced, True)

    if disp:
        show_bundles(recognized_bundle, reduced_streamlines, tubes=False)

    npt.assert_equal(len(np.intersect1d(actual_indices, rb.labels)),
                     len(rb.labels) - nb_reduced)

    show_grid([model_bundle, recognized_bundle,
              expansion_streamlines, reduced_streamlines],
              ['model', 'recognized', 'expanded', 'reduced'], dim=(2, 2))

    # return rb

    1/0

if __name__ == '__main__':

    rb = test_recognition()