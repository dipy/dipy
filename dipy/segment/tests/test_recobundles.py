import numpy as np
import numpy.testing as npt
import nibabel.trackvis as tv
from dipy.tracking.streamline import transform_streamlines
from copy import deepcopy
from itertools import chain
from dipy.segment.bundles import RecoBundles


def show_bundles(static, moving, linewidth=1., tubes=False,
                 opacity=1., fname=None):

    from dipy.viz import fvtk
    ren = fvtk.ren()
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

    fvtk.add(ren, fvtk.axes(scale=(2, 2, 2)))

    fvtk.show(ren, size=(900, 900))
    if fname is not None:
        fvtk.record(ren, size=(900, 900), out_path=fname)


def test_recognition():

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

    for (i, fname) in enumerate(fnames):
        streams, hdr = tv.read(fname, points_space='rasmm')
        bundle = [s[0] for s in streams]
        model_bundles_dix[bundle_trk[i].split('.trk')[0]] = bundle

    play_bundles_dix = deepcopy(model_bundles_dix)

    mat = np.eye(4)
    mat[:3, 3] = np.array([10, 0, 0])

    tag = 'MCP'
    play_bundles_dix[tag] = transform_streamlines(play_bundles_dix[tag], mat)

    model_bundle = model_bundles_dix[tag]

    # make sure that you put the bundles back in the correct order
    streamlines = list(chain(*play_bundles_dix.values()))

    # show_bundles(model_bundle, streamlines)

    rb = RecoBundles(streamlines)
    recognized_bundle = rb.recognize(model_bundle)

    np.set_printoptions(3, suppress=True)
    print(rb.transf_matrix)


    show_bundles(model_bundle, recognized_bundle)
    mat2 = np.eye(4)
    mat2[:3, 3] = np.array([60, 0, 0])

    show_bundles(transform_streamlines(model_bundle, mat2),
                 recognized_bundle)

    print('Recognized bundle %d' % (len(recognized_bundle),))
    print('Model bundle %d' % (len(model_bundle),))

if __name__ == '__main__':

    test_recognition()