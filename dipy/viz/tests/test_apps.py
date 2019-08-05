import os
import numpy as np
import numpy.testing as npt
from dipy.tracking.streamline import Streamlines
from dipy.testing.decorators import xvfb_it, use_xvfb
from dipy.utils.optpkg import optional_package
from dipy.data import DATA_DIR

fury, has_fury, setup_module = optional_package('fury')

if has_fury:
    from dipy.viz.app import horizon

skip_it = use_xvfb == 'skip'

# we will have to skip this as creates issues with xvfb (XIO error)
@npt.dec.skipif(True)
@xvfb_it
def test_horizon_events():

    affine = np.diag([2., 1, 1, 1]).astype('f8')
    data = 255 * np.random.rand(150, 150, 150)
    images = [(data, affine)]
    from dipy.segment.tests.test_bundles import f1
    streamlines = f1.copy()
    tractograms = [streamlines]
    fname = os.path.join(DATA_DIR, 'record_01.log.gz')

    horizon(tractograms, images=images, cluster=True, cluster_thr=5,
            random_colors=False, length_lt=np.inf, length_gt=0,
            clusters_lt=np.inf, clusters_gt=0,
            world_coords=True, interactive=True, recorded_events=fname)

# see comment above
@npt.dec.skipif(True)
@xvfb_it
def test_horizon_events2():

    affine = np.diag([2., 1, 1, 1]).astype('f8')
    data = 255 * np.random.rand(150, 150, 150)
    images = [(data, affine)]
    from dipy.segment.tests.test_bundles import f1
    streamlines = f1.copy()
    tractograms = [streamlines]
    fname = os.path.join(DATA_DIR, 'record_02.log.gz')
    horizon(tractograms, images=images, cluster=False, cluster_thr=5,
            random_colors=False, length_lt=np.inf, length_gt=0,
            clusters_lt=np.inf, clusters_gt=0,
            world_coords=True, interactive=True, recorded_events=fname)


@npt.dec.skipif(skip_it or not has_fury)
@xvfb_it
def test_horizon():

    s1 = 10 * np.array([[0, 0, 0],
                        [1, 0, 0],
                        [2, 0, 0],
                        [3, 0, 0],
                        [4, 0, 0]], dtype='f8')

    s2 = 10 * np.array([[0, 0, 0],
                        [0, 1, 0],
                        [0, 2, 0],
                        [0, 3, 0],
                        [0, 4, 0]], dtype='f8')

    s3 = 10 * np.array([[0, 0, 0],
                        [1, 0.2, 0],
                        [2, 0.2, 0],
                        [3, 0.2, 0],
                        [4, 0.2, 0]], dtype='f8')

    print(s1.shape)
    print(s2.shape)
    print(s3.shape)

    streamlines = Streamlines()
    streamlines.append(s1)
    streamlines.append(s2)
    streamlines.append(s3)

    tractograms = [streamlines]
    images = None

    horizon(tractograms, images=images, cluster=True, cluster_thr=5,
            random_colors=False, length_lt=np.inf, length_gt=0,
            clusters_lt=np.inf, clusters_gt=0,
            world_coords=False, interactive=False)

    affine = np.diag([2., 1, 1, 1]).astype('f8')

    data = 255 * np.random.rand(150, 150, 150)

    images = [(data, affine)]

    horizon(tractograms, images=images, cluster=True, cluster_thr=5,
            random_colors=False, length_lt=np.inf, length_gt=0,
            clusters_lt=np.inf, clusters_gt=0,
            world_coords=True, interactive=False)

    tractograms = []
    horizon(tractograms, images=images, cluster=True, cluster_thr=5,
            random_colors=False, length_lt=np.inf, length_gt=0,
            clusters_lt=np.inf, clusters_gt=0,
            world_coords=True, interactive=False)


if __name__ == '__main__':

    test_horizon_events()
    test_horizon_events2()
    test_horizon()


