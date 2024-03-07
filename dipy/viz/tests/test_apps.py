import os
import warnings

import numpy as np
import numpy.testing as npt
import pytest

from dipy.data import DATA_DIR
from dipy.io.stateful_tractogram import Space, StatefulTractogram
from dipy.io.utils import create_nifti_header
from dipy.testing import check_for_warnings
from dipy.testing.decorators import use_xvfb
from dipy.tracking.streamline import Streamlines
from dipy.utils.optpkg import optional_package
from dipy.testing.decorators import set_random_number_generator

fury, has_fury, setup_module = optional_package('fury', min_version="0.10.0")

if has_fury:
    from fury import window

    from dipy.viz.horizon.app import horizon

skip_it = use_xvfb == 'skip'


@pytest.mark.skipif(skip_it or not has_fury, reason="Needs xvfb")
@set_random_number_generator()
def test_horizon_events(rng):
    # using here MNI template affine 2009a
    affine = np.array([[1., 0., 0., -98.],
                       [0., 1., 0., -134.],
                       [0., 0., 1., -72.],
                       [0., 0., 0., 1.]])

    data = 255 * rng.random((197, 233, 189))
    vox_size = (1., 1., 1.)
    images = [(data, affine, '/test/filename.nii.gz')]
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
@set_random_number_generator()
def test_horizon(rng):
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

    data = 255 * rng.random((197, 233, 189))
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

    images = [(data, affine, '/test/filename.nii.gz')]
    # tractograms in native coords (not supported for now)
    with npt.assert_raises(ValueError) as ve:
        horizon(tractograms, images=images, cluster=True, cluster_thr=5,
                random_colors=False, length_lt=np.inf, length_gt=0,
                clusters_lt=np.inf, clusters_gt=0,
                world_coords=False, interactive=False)

    msg = 'Currently native coordinates are not supported for streamlines.'
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
def test_horizon_wrong_dtype_images():
    affine = np.array([[1., 0., 0., -98.],
                       [0., 1., 0., -134.],
                       [0., 0., 1., -72.],
                       [0., 0., 0., 1.]])

    data = np.random.rand(197, 233, 189).astype(np.bool_)
    images = [(data, affine)]
    with warnings.catch_warnings(record=True) as l_warns:
        horizon(images=images, interactive=False)
        check_for_warnings(l_warns, 'skipping image 1, passed image is not in '
                           + 'numerical format')


@pytest.mark.skipif(skip_it or not has_fury, reason="Needs xvfb")
@set_random_number_generator(42)
def test_roi_images(rng):
    img1 = rng.random((5, 5, 5))
    img2 = np.zeros((5, 5, 5))
    img2[2, 2, 2] = 1
    img3 = np.zeros((5, 5, 5))
    img3[0, :, :] = 1
    images = [
        (img1, np.eye(4)),
        (img2, np.eye(4), '/test/filename.nii.gz'),
        (img3, np.eye(4), '/test/filename.nii.gz')
    ]
    show_m = horizon(images=images, return_showm=True)
    analysis = window.analyze_scene(show_m.scene)
    npt.assert_equal(analysis.actors, 0)
    arr = window.snapshot(show_m.scene)
    report = window.analyze_snapshot(arr, colors=[(0, 0, 0), (255, 255, 255)])
    npt.assert_array_equal(report.colors_found, [True, True])
    show_m = horizon(images=images, roi_images=True, return_showm=True)
    analysis = window.analyze_scene(show_m.scene)
    npt.assert_equal(analysis.actors, 2)


@pytest.mark.skipif(skip_it or not has_fury, reason="Needs xvfb")
@set_random_number_generator(42)
def test_surfaces(rng):
    vertices = rng.random((100, 3))
    faces = rng.integers(0, 100, size=(100, 3))
    surfaces = [
        (vertices, faces),
        (vertices, faces, '/test/filename.pial'),
        (vertices, faces, '/test/filename.pial')
    ]
    show_m = horizon(surfaces=surfaces, return_showm=True)
    analysis = window.analyze_scene(show_m.scene)
    npt.assert_equal(analysis.actors, 3)

    vertices = rng.random((100, 4))
    faces = rng.integers(0, 100, size=(100, 3))
    surfaces = [
        (vertices, faces),
        (vertices, faces, '/test/filename.pial'),
        (vertices, faces, '/test/filename.pial')
    ]

    with warnings.catch_warnings(record=True) as l_warns:
        show_m = horizon(surfaces=surfaces, return_showm=True)
        analysis = window.analyze_scene(show_m.scene)
        npt.assert_equal(analysis.actors, 0)
        check_for_warnings(l_warns, 'Vertices do not have correct shape: ' +
                           '(100, 4)')

    vertices = rng.random((100, 3))
    faces = rng.integers(0, 100, size=(100, 4))
    surfaces = [
        (vertices, faces),
        (vertices, faces, '/test/filename.pial'),
        (vertices, faces, '/test/filename.pial')
    ]

    with warnings.catch_warnings(record=True) as l_warns:
        show_m = horizon(surfaces=surfaces, return_showm=True)
        analysis = window.analyze_scene(show_m.scene)
        npt.assert_equal(analysis.actors, 0)
        check_for_warnings(l_warns, 'Faces do not have correct shape: ' +
                           '(100, 4)')


@pytest.mark.skipif(skip_it, reason="Needs xvfb")
def test_small_horizon_import():
    from dipy.viz import horizon as Horizon
    if has_fury:
        assert Horizon == horizon
    else:
        npt.assert_raises(ImportError, Horizon)
