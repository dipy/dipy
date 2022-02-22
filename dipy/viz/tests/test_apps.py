import os
import numpy as np
import pytest
import numpy.testing as npt
from dipy.tracking.streamline import Streamlines
from dipy.io.stateful_tractogram import Space, StatefulTractogram
from dipy.io.utils import create_nifti_header
from dipy.testing.decorators import use_xvfb
from dipy.utils.optpkg import optional_package
from dipy.data import DATA_DIR
from nibabel.tmpdirs import TemporaryDirectory

fury, has_fury, setup_module = optional_package('fury')

if has_fury:
    from dipy.viz.app import horizon
    from fury.io import load_image

skip_it = use_xvfb == 'skip'


@pytest.mark.skipif(skip_it or not has_fury, reason="Needs xvfb")
def test_horizon_events():
    # using here MNI template affine 2009a
    affine = np.array([[1., 0., 0., -98.],
                       [0., 1., 0., -134.],
                       [0., 0., 1., -72.],
                       [0., 0., 0., 1.]])

    data = 255 * np.random.rand(197, 233, 189)
    vox_size = (1., 1., 1.)

    images = [(data, affine)]
    # images = None
    from dipy.segment.tests.test_bundles import setup_module
    setup_module()
    from dipy.segment.tests.test_bundles import f1
    streamlines = f1.copy()
    streamlines._data += np.array([-98., -134., -72.])

    header = create_nifti_header(affine, data.shape, vox_size)
    sft = StatefulTractogram(streamlines, header, Space.RASMM)

    tractograms = [sft]

    # select all centroids and expand and click everything else
    # do not press the key shortcuts as vtk generates warning that
    # blocks recording
    fname = os.path.join(DATA_DIR, 'record_horizon.log.gz')

    horizon(tractograms=tractograms, images=images, pams=None,
            cluster=True, cluster_thr=5.0,
            random_colors=False, length_gt=0, length_lt=np.inf,
            clusters_gt=0, clusters_lt=np.inf,
            world_coords=True, interactive=True, out_png='tmp.png',
            recorded_events=fname)


@pytest.mark.skipif(skip_it or not has_fury, reason="Needs xvfb")
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

    affine = np.array([[1., 0., 0., -98.],
                       [0., 1., 0., -134.],
                       [0., 0., 1., -72.],
                       [0., 0., 0., 1.]])

    data = 255 * np.random.rand(197, 233, 189)
    vox_size = (1., 1., 1.)

    streamlines._data += np.array([-98., -134., -72.])

    header = create_nifti_header(affine, data.shape, vox_size)
    sft = StatefulTractogram(streamlines, header, Space.RASMM)

    # only tractograms
    tractograms = [sft]
    images = None
    horizon(tractograms, images=images, cluster=True, cluster_thr=5,
            random_colors=False, length_lt=np.inf, length_gt=0,
            clusters_lt=np.inf, clusters_gt=0,
            world_coords=True, interactive=False)

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


@pytest.mark.skipif(skip_it or not has_fury, reason="Needs xvfb")
def test_roi_images():
    np.random.seed(42)
    img1 = np.random.rand(5, 5, 5)
    img2 = np.zeros((5, 5, 5))
    img2[1, 1, 1] = 1
    img3 = np.zeros((5, 5, 5))
    img3[3, 3, 3] = 1
    images = [(img1, np.eye(4)), (img2, np.eye(4)), (img3, np.eye(4))]
    with TemporaryDirectory() as out_dir:
        tmp_fname = os.path.join(out_dir, 'tmp_x.png')

        # If roi_images=False, only the first non-binary image is loaded
        horizon(images=images, interactive=False, out_png=tmp_fname)
        npt.assert_equal(os.path.exists(tmp_fname), True)

        # If roi_images=True, all th binary images are shown as contours
        horizon(images=images, roi_images=True, interactive=False,
                out_png=tmp_fname)
        npt.assert_equal(os.path.exists(tmp_fname), True)
        ss = load_image(tmp_fname)
        npt.assert_equal(ss[650, 800, :], [147, 0, 0])
