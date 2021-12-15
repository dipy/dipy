import pytest
import numpy as np
import numpy.testing as npt
from dipy.testing.decorators import use_xvfb
from dipy.utils.optpkg import optional_package
from dipy.data import DATA_DIR, get_sphere
from dipy.align.reslice import reslice
from nibabel.tmpdirs import TemporaryDirectory
from dipy.data import read_stanford_labels
from dipy.reconst.shm import CsaOdfModel
from dipy.data import default_sphere
from dipy.direction import peaks_from_model
from dipy.tracking import utils
from dipy.tracking.stopping_criterion import \
    ThresholdStoppingCriterion
from dipy.tracking.local_tracking import LocalTracking
from dipy.reconst.shm import sh_to_sf_matrix
from dipy.align.tests.test_streamlinear import fornix_streamlines
from dipy.tracking.streamline import (center_streamlines,
                                      transform_streamlines)
fury, has_fury, setup_module = optional_package('fury')

if has_fury:
    from fury.io import load_image
    from fury import actor, window
    from fury.colormap import line_colors

skip_it = use_xvfb == 'skip'


@pytest.mark.skipif(skip_it or not has_fury,
                    reason="Needs xvfb")
def test_slicer():
    scene = window.Scene()
    data = (255 * np.random.rand(50, 50, 50))
    affine = np.diag([1, 3, 2, 1])
    data2, affine2 = reslice(data, affine, zooms=(1, 3, 2),
                                new_zooms=(1, 1, 1))

    slicer = actor.slicer(data2, affine2, interpolation='linear')
    slicer.display(None, None, 25)

    scene.add(slicer)
    scene.reset_camera()
    scene.reset_clipping_range()

    window.show(scene, reset_camera=False)
    arr = window.snapshot(scene, offscreen=True)
    report = window.analyze_snapshot(arr, find_objects=True)
    npt.assert_equal(report.objects, 1)
    npt.assert_array_equal([1, 3, 2] * np.array(data.shape),
                            np.array(slicer.shape))


@pytest.mark.skipif(skip_it or not has_fury,
                    reason="Needs xvfb")
def test_contour_from_roi():
    hardi_img, gtab, labels_img = read_stanford_labels()
    data = np.asanyarray(hardi_img.dataobj)
    labels = np.asanyarray(labels_img.dataobj)
    affine = hardi_img.affine

    white_matter = (labels == 1) | (labels == 2)

    csa_model = CsaOdfModel(gtab, sh_order=6)
    csa_peaks = peaks_from_model(csa_model, data, default_sphere,
                                    relative_peak_threshold=.8,
                                    min_separation_angle=45,
                                    mask=white_matter)

    classifier = ThresholdStoppingCriterion(csa_peaks.gfa, .25)

    seed_mask = labels == 2
    seeds = utils.seeds_from_mask(seed_mask, density=[1, 1, 1],
                                    affine=affine)

    # Initialization of LocalTracking.
    # The computation happens in the next step.
    streamlines = LocalTracking(csa_peaks, classifier, seeds, affine,
                                step_size=2)

    # Compute streamlines and store as a list.
    streamlines = list(streamlines)

    # Prepare the display objects.
    streamlines_actor = actor.line(streamlines, line_colors(streamlines))
    seedroi_actor = actor.contour_from_roi(seed_mask, affine,
                                            [0, 1, 1], 0.5)

    # Create the 3d display.
    sc = window.Scene()
    sc2 = window.Scene()
    sc.add(streamlines_actor)
    arr3 = window.snapshot(sc, 'test_surface3.png', offscreen=True)
    report3 = window.analyze_snapshot(arr3, find_objects=True)
    sc2.add(streamlines_actor)
    sc2.add(seedroi_actor)
    arr4 = window.snapshot(sc2, 'test_surface4.png', offscreen=True)
    report4 = window.analyze_snapshot(arr4, find_objects=True)

    # assert that the seed ROI rendering is not far
    # away from the streamlines (affine error)
    npt.assert_equal(report3.objects, report4.objects)
    # window.show(sc)
    # window.show(sc2)


@pytest.mark.skipif(skip_it or not has_fury,
                    reason="Needs xvfb")
def test_bundle_maps():
    scene = window.Scene()
    bundle = fornix_streamlines()
    bundle, _ = center_streamlines(bundle)

    mat = np.array([[1, 0, 0, 100],
                    [0, 1, 0, 100],
                    [0, 0, 1, 100],
                    [0, 0, 0, 1.]])

    bundle = transform_streamlines(bundle, mat)

    # metric = np.random.rand(*(200, 200, 200))
    metric = 100 * np.ones((200, 200, 200))

    # add lower values
    metric[100, :, :] = 100 * 0.5

    # create a nice orange-red colormap
    lut = actor.colormap_lookup_table(scale_range=(0., 100.),
                                      hue_range=(0., 0.1),
                                      saturation_range=(1, 1),
                                      value_range=(1., 1))

    line = actor.line(bundle, metric, linewidth=0.1, lookup_colormap=lut)
    scene.add(line)
    scene.add(actor.scalar_bar(lut, ' '))

    report = window.analyze_scene(scene)

    npt.assert_almost_equal(report.actors, 1)
    # window.show(scene)

    scene.clear()

    nb_points = np.sum([len(b) for b in bundle])
    values = 100 * np.random.rand(nb_points)
    # values[:nb_points/2] = 0

    line = actor.streamtube(bundle, values, linewidth=0.1, lookup_colormap=lut)
    scene.add(line)
    # window.show(scene)

    report = window.analyze_scene(scene)
    npt.assert_equal(report.actors_classnames[0], 'vtkLODActor')

    scene.clear()

    colors = np.random.rand(nb_points, 3)
    # values[:nb_points/2] = 0

    line = actor.line(bundle, colors, linewidth=2)
    scene.add(line)
    # window.show(scene)

    report = window.analyze_scene(scene)
    npt.assert_equal(report.actors_classnames[0], 'vtkLODActor')
    # window.show(scene)

    arr = window.snapshot(scene)
    report2 = window.analyze_snapshot(arr)
    npt.assert_equal(report2.objects, 1)

    # try other input options for colors
    scene.clear()
    actor.line(bundle, (1., 0.5, 0))
    actor.line(bundle, np.arange(len(bundle)))
    actor.line(bundle)
    colors = [np.random.rand(*b.shape) for b in bundle]
    actor.line(bundle, colors=colors)


@pytest.mark.skipif(skip_it or not has_fury,
                    reason="Needs xvfb")
def test_odf_slicer(interactive=False):
    # Prepare our data
    sphere = get_sphere('repulsion100')
    shape = (11, 11, 11, sphere.vertices.shape[0])
    odfs = np.ones(shape)

    affine = np.array([[2.0, 0.0, 0.0, 3.0],
                       [0.0, 2.0, 0.0, 3.0],
                       [0.0, 0.0, 2.0, 1.0],
                       [0.0, 0.0, 0.0, 1.0]])
    mask = np.ones(odfs.shape[:3], bool)
    mask[:4, :4, :4] = False

    # Test that affine and mask work
    odf_actor = actor.odf_slicer(odfs, sphere=sphere, affine=affine, mask=mask,
                                 scale=.25, colormap='blues')

    k = 2
    I, J, _ = odfs.shape[:3]
    odf_actor.display_extent(0, I - 1, 0, J - 1, k, k)

    scene = window.Scene()
    scene.add(odf_actor)
    scene.reset_camera()
    scene.reset_clipping_range()

    if interactive:
        window.show(scene, reset_camera=False)

    arr = window.snapshot(scene)
    report = window.analyze_snapshot(arr, find_objects=True)
    npt.assert_equal(report.objects, 11 * 11 - 16)

    # Test that global colormap works
    odf_actor = actor.odf_slicer(odfs, sphere=sphere, mask=mask, scale=.25,
                                 colormap='blues', norm=False, global_cm=True)
    scene.clear()
    scene.add(odf_actor)
    scene.reset_camera()
    scene.reset_clipping_range()
    if interactive:
        window.show(scene)

    # Test that the most basic odf_slicer instanciation works
    odf_actor = actor.odf_slicer(odfs)
    scene.clear()
    scene.add(odf_actor)
    scene.reset_camera()
    scene.reset_clipping_range()
    if interactive:
        window.show(scene)

    # Test that odf_slicer.display works properly
    scene.clear()
    scene.add(odf_actor)
    scene.add(actor.axes((11, 11, 11)))
    for i in range(11):
        odf_actor.display(i, None, None)
        if interactive:
            window.show(scene)
    for j in range(11):
        odf_actor.display(None, j, None)
        if interactive:
            window.show(scene)

    # With mask equal to zero everything should be black
    mask = np.zeros(odfs.shape[:3])
    odf_actor = actor.odf_slicer(odfs, sphere=sphere, mask=mask,
                                 scale=.25, colormap='blues',
                                 norm=False, global_cm=True)
    scene.clear()
    scene.add(odf_actor)
    scene.reset_camera()
    scene.reset_clipping_range()
    if interactive:
        window.show(scene)

    # global_cm=True with colormap=None should raise an error
    npt.assert_raises(IOError, actor.odf_slicer, odfs, sphere=sphere,
                      mask=None, scale=.25, colormap=None, norm=False,
                      global_cm=True)

    # Dimension mismatch between sphere vertices and number
    # of SF coefficients will raise an error.
    npt.assert_raises(ValueError, actor.odf_slicer, odfs, mask=None,
                      sphere=get_sphere('repulsion200'), scale=.25)

    # colormap=None and global_cm=False results in directionally encoded colors
    odf_actor = actor.odf_slicer(odfs, sphere=sphere, mask=None,
                                 scale=.25, colormap=None,
                                 norm=False, global_cm=False)
    scene.clear()
    scene.add(odf_actor)
    scene.reset_camera()
    scene.reset_clipping_range()
    if interactive:
        window.show(scene)

    # Test that SH coefficients input works
    B = sh_to_sf_matrix(sphere, sh_order=4, return_inv=False)
    odfs = np.zeros((11, 11, 11, B.shape[0]))
    odfs[..., 0] = 1.0
    odf_actor = actor.odf_slicer(odfs, sphere=sphere, B_matrix=B)

    scene.clear()
    scene.add(odf_actor)
    scene.reset_camera()
    scene.reset_clipping_range()
    if interactive:
        window.show(scene)

    # Dimension mismatch between sphere vertices and dimension of
    # B matrix will raise an error.
    npt.assert_raises(ValueError, actor.odf_slicer, odfs, mask=None,
                      sphere=get_sphere('repulsion200'))

    # Test that constant colormap color works. Also test that sphere
    # normals are oriented correctly. Will show purple spheres with
    # a white contour.
    odf_contour = actor.odf_slicer(odfs, sphere=sphere, B_matrix=B,
                                   colormap=(255, 255, 255))
    odf_contour.GetProperty().SetAmbient(1.0)
    odf_contour.GetProperty().SetFrontfaceCulling(True)

    odf_actor = actor.odf_slicer(odfs, sphere=sphere, B_matrix=B,
                                 colormap=(255, 0, 255), scale=0.4)
    scene.clear()
    scene.add(odf_contour)
    scene.add(odf_actor)
    scene.reset_camera()
    scene.reset_clipping_range()
    if interactive:
        window.show(scene)

    # Test that we can change the sphere on an active actor
    new_sphere = get_sphere('symmetric362')
    new_B = sh_to_sf_matrix(new_sphere, sh_order=4, return_inv=False)
    odf_actor.update_sphere(new_sphere.vertices, new_sphere.faces, new_B)
    if interactive:
        window.show(scene)

    del odf_actor
    del odfs


# test_slicer()
# test_contour_from_roi()
test_bundle_maps()
test_odf_slicer()