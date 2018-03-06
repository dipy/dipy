import numpy as np
import nibabel as nib
from numpy.testing import assert_equal
from dipy.data import get_data
from dipy.tracking.streamline import Streamlines
from dipy.align.streamlinear import whole_brain_slr


def tmp_show(f):
    from dipy.viz import actor, window
    ren = window.Renderer()
    ren.add(actor.line(f))
    window.show(ren)

def tmp_show_two(f1, f2):
    from dipy.viz import actor, window
    ren = window.Renderer()
    ren.add(actor.line(f1, colors=(1, 0, 0)))
    ren.add(actor.line(f2, colors=(0, 1, 0)))
    window.show(ren)

 
streams, hdr = nib.trackvis.read(get_data('fornix'))
fornix = [s[0] for s in streams]

f = Streamlines(fornix)
f1 = f.copy()
f2 = f.copy()

f2._data += np.array([50, 0, 0])

moved, transform, qb_centroids1, qb_centroids2 = whole_brain_slr(
        f1, f2, verbose=True, rm_small_clusters=2, greater_than=2, 
        less_than=1, qb_thr=5, progressive=True)

