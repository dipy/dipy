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


@npt.dec.skipif(skip_it or not has_fury)
def test_horizon_events():
    affine = np.diag([2., 1, 1, 1]).astype('f8')
    data = 255 * np.random.rand(150, 150, 150)
    images = [(data, affine)]
    # images = None
    from dipy.segment.tests.test_bundles import setup_module
    setup_module()
    from dipy.segment.tests.test_bundles import f1
    streamlines = f1.copy()
    tractograms = [streamlines]

    enable = [3]

    if 1 in enable: # just close the window
        fname = os.path.join(DATA_DIR, 'record_01.log.gz')

        horizon(tractograms=tractograms, images=images, pams=None,
                cluster=True, cluster_thr=5.0,
                random_colors=False, length_gt=0, length_lt=np.inf,
                clusters_gt=0, clusters_lt=np.inf,
                world_coords=True, interactive=True, out_png='tmp.png',
                recorded_events=fname)

    if 2 in enable: # just zoom and close
        fname = os.path.join(DATA_DIR, 'record_02.log.gz')

        horizon(tractograms=tractograms, images=images, pams=None,
                cluster=True, cluster_thr=5.0,
                random_colors=False, length_gt=0, length_lt=np.inf,
                clusters_gt=0, clusters_lt=np.inf,
                world_coords=True, interactive=True, out_png='tmp.png',
                recorded_events=fname)

    if 3 in enable: # select all centroids and expand and everything else
        # save a trk at the end
        fname = os.path.join(DATA_DIR, 'record_03.log.gz')

        horizon(tractograms=tractograms, images=images, pams=None,
                cluster=True, cluster_thr=5.0,
                random_colors=False, length_gt=0, length_lt=np.inf,
                clusters_gt=0, clusters_lt=np.inf,
                world_coords=True, interactive=True, out_png='tmp.png',
                recorded_events=fname)
        # npt.assert_equal(os.path.exists('tmp.trk'), True)
        # npt.assert_equal(os.stat('tmp.trk').st_size > 0, True)
        # os.remove('tmp.trk')


@npt.dec.skipif(skip_it or not has_fury)
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

    streamlines = Streamlines()
    streamlines.append(s1)
    streamlines.append(s2)
    streamlines.append(s3)

    # only tractograms
    tractograms = [streamlines]
    images = None
    horizon(tractograms, images=images, cluster=True, cluster_thr=5,
            random_colors=False, length_lt=np.inf, length_gt=0,
            clusters_lt=np.inf, clusters_gt=0,
            world_coords=True, interactive=False)

    affine = np.diag([2., 1, 1, 1]).astype('f8')
    data = 255 * np.random.rand(150, 150, 150)
    images = [(data, affine)]

    # tractograms in native coords (not supported for now)
    with npt.assert_raises(ValueError) as ve:
        horizon(tractograms, images=images, cluster=True, cluster_thr=5,
                random_colors=False, length_lt=np.inf, length_gt=0,
                clusters_lt=np.inf, clusters_gt=0,
                world_coords=False, interactive=False)

    msg = 'Currently native coordinates are not supported for streamlines'
    npt.assert_(msg in str(ve.exception))

    # only images
    tractograms = None
    horizon(tractograms, images=images, cluster=True, cluster_thr=5,
            random_colors=False, length_lt=np.inf, length_gt=0,
            clusters_lt=np.inf, clusters_gt=0,
            world_coords=True, interactive=False)

    # no clustering tractograms and images
    horizon(tractograms, images=images, cluster=False, cluster_thr=5,
            random_colors=False, length_lt=np.inf, length_gt=0,
            clusters_lt=np.inf, clusters_gt=0,
            world_coords=True, interactive=False)


if __name__ == '__main__':

    test_horizon_events()
    test_horizon()




