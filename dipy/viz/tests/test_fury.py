import pytest
import numpy as np
import numpy.testing as npt
from dipy.testing.decorators import use_xvfb
from dipy.utils.optpkg import optional_package
from dipy.data import DATA_DIR
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


# test_slicer()
# test_contour_from_roi()