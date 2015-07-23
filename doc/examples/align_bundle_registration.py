from dipy.align.streamlinear import StreamlineLinearRegistration
from dipy.tracking.streamline import set_number_of_points, select_random_set_of_streamlines
from dipy.viz import fvtk
from dipy.data.fetcher import fetch_bundles_2_subjects, read_bundles_2_subjects


def show_both_bundles(bundles, colors=None, show=False, fname=None):

    ren = fvtk.ren()
    ren.SetBackground(1., 1, 1)
    for (i, bundle) in enumerate(bundles):
        color = colors[i]
        lines = fvtk.streamtube(bundle, color, linewidth=0.3)
        lines.RotateX(-90)
        lines.RotateZ(90)
        fvtk.add(ren, lines)
    if show:
        fvtk.show(ren)
    if fname is not None:
        sleep(1)
        fvtk.record(ren, n_frames=1, out_path=fname, size=(900, 900))


fetch_bundles_2_subjects()

subj1 = read_bundles_2_subjects('subj_1', ['fa', 't1'],
                               ['af.left', 'cst.right', 'cc_1'])

subj2 = read_bundles_2_subjects('subj_2', ['fa', 't1'],
                               ['af.left', 'cst.right', 'cc_1'])


for bundle_type in ['af.left', 'cst.right', 'cc_1']:


    bundle1 = subj1[bundle_type]
    bundle2 = subj2[bundle_type]

    sbundle1 = set_number_of_points(bundle1, 20)
    sbundle2 = set_number_of_points(bundle2, 20)

    sbundle1 = select_random_set_of_streamlines(sbundle1, 400)
    sbundle2 = select_random_set_of_streamlines(sbundle2, 400)

    slr = StreamlineLinearRegistration(x0='affine')
    slm = slr.optimize(static=sbundle1, moving=sbundle2)

    sbundle2_moved = slm.transform(sbundle2)


    show_both_bundles([sbundle1, sbundle2],
                      colors=[fvtk.colors.orange, fvtk.colors.red],
                      show=True)


    show_both_bundles([sbundle1, sbundle2_moved],
                      colors=[fvtk.colors.orange, fvtk.colors.red],
                      show=True)
