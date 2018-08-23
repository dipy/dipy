import numpy as np
import nibabel as nib
from numpy.testing import (assert_equal, run_module_suite,
                           assert_array_almost_equal)
from dipy.data import get_data
from dipy.tracking.streamline import Streamlines
from dipy.align.streamlinear import whole_brain_slr, slr_with_qbx
from dipy.tracking.distances import bundles_distances_mam
from dipy.align.streamlinear import transform_streamlines
from dipy.align.streamlinear import compose_matrix44, decompose_matrix44


def test_whole_brain_slr():
    streams, hdr = nib.trackvis.read(get_data('fornix'))
    fornix = [s[0] for s in streams]

    f = Streamlines(fornix)
    f1 = f.copy()
    f2 = f.copy()

    # check translation
    f2._data += np.array([50, 0, 0])
    # old_f2 = f2.copy()

#    from dipy.viz import actor, window
#
#    ren = window.Renderer()
#    ren.add(actor.line(f1, colors=(1, 0, 0)))
#    ren.add(actor.line(f2, colors=(0, 1, 0)))
#    window.show(ren)


    moved, transform, qb_centroids1, qb_centroids2 = whole_brain_slr(
            f1, f2, x0='affine', verbose=True, rm_small_clusters=1,
            greater_than=0, less_than=np.inf,
            qbx_thr=[40, 30, 20, 15, 5, 1], progressive=False)



#    ren = window.Renderer()
#    ren.add(actor.line(f1, colors=(1, 0, 0)))
#    ren.add(actor.line(moved, colors=(0, 1, 0)))
#    #ren.add(actor.line(moved, colors=(0, 0, 1)))
#    window.show(ren)

    print("transform = ", transform)
    # we can check the quality of registration by comparing the matrices
    # MAM streamline distances before and after SLR
    D12 = bundles_distances_mam(f1, f2)
    D1M = bundles_distances_mam(f1, moved)

    d12_minsum = np.sum(np.min(D12, axis=0))
    d1m_minsum = np.sum(np.min(D1M, axis=0))

    print("distances= ", d12_minsum, " ", d1m_minsum)

    assert_equal(d1m_minsum < d12_minsum, True)

    assert_array_almost_equal(transform[:3, 3], [-50, -0, -0], 3)

    # check rotation

    mat = compose_matrix44([0, 0, 0, 15, 0, 0])

    f3 = f.copy()
    f3 = transform_streamlines(f3, mat)

    moved, transform, qb_centroids1, qb_centroids2 = slr_with_qbx(
            f1, f3, verbose=False, rm_small_clusters=0, greater_than=0,
            less_than=np.inf, qbx_thr=[40, 30, 20, 15, 5, 1],
            progressive=True)

    # we can also check the quality by looking at the decomposed transform

    assert_array_almost_equal(decompose_matrix44(transform)[3], -15, 2)

    moved, transform, qb_centroids1, qb_centroids2 = slr_with_qbx(
            f1, f3, verbose=False, rm_small_clusters=0, select_random=400,
            greater_than=0,
            less_than=np.inf, qbx_thr=[40, 30, 20, 15, 5, 1],
            progressive=False)

    # we can also check the quality by looking at the decomposed transform

    assert_array_almost_equal(decompose_matrix44(transform)[3], -15, 2)

if __name__ == '__main__':
    run_module_suite()
