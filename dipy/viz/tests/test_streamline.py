import tempfile
import os
from numpy.testing import assert_raises, assert_equal
import pytest

from dipy.utils.optpkg import optional_package
from dipy.data import read_five_af_bundles, two_cingulum_bundles
from dipy.tracking.streamline import set_number_of_points, unlist_streamlines

from dipy.align.streamwarp import bundlewarp, bundlewarp_vector_filed

fury, has_fury, _ = optional_package('fury')
if has_fury:
    from dipy.viz.streamline import (show_bundles, viz_two_bundles,
                                     viz_displacement_mag, viz_vector_field)
    from dipy.viz import window

bundles = read_five_af_bundles()


@pytest.mark.skipif(not has_fury, reason='Requires FURY')
def test_output_created():
    views = ['axial', 'sagital', 'coronal']

    colors = [[0.91, 0.26, 0.35], [0.99, 0.50, 0.38], [0.99, 0.88, 0.57],
              [0.69, 0.85, 0.64], [0.51, 0.51, 0.63]]

    with tempfile.TemporaryDirectory() as temp_dir:
        for view in views:
            fname = os.path.join(temp_dir, f'test_{view}.png')
            show_bundles(bundles, False, view=view, save_as=fname)
            assert_equal(os.path.exists(fname), True)

        fname = os.path.join(temp_dir, 'test_colors.png')
        show_bundles(bundles, False, colors=colors, save_as=fname)
        assert_equal(os.path.exists(fname), True)

        # Check rendered image is not empty
        report = window.analyze_snapshot(fname, find_objects=True)
        assert_equal(report.objects > 0, True)

        cb1, cb2 = two_cingulum_bundles()

        fname = os.path.join(temp_dir, 'test_two_bundles.png')
        viz_two_bundles(cb1, cb2, fname=fname)
        assert_equal(os.path.exists(fname), True)


@pytest.mark.skipif(not has_fury, reason='Requires FURY')
def test_incorrect_view():
    assert_raises(ValueError, show_bundles, bundles, False, 'wrong_view')


@pytest.mark.skipif(not has_fury, reason='Requires FURY')
def test_bundlewarp_viz():

    cingulum_bundles = two_cingulum_bundles()

    cb1 = cingulum_bundles[0]
    cb1 = set_number_of_points(cb1, 20)

    cb2 = cingulum_bundles[1]
    cb2 = set_number_of_points(cb2, 20)

    with tempfile.TemporaryDirectory() as temp_dir:

        deformed_bundle, affine_bundle, _, _, _ = bundlewarp(cb1, cb2)

        offsets, directions, colors = bundlewarp_vector_filed(cb1, cb2)
        points_aligned, _ = unlist_streamlines(affine_bundle)

        fname = os.path.join(temp_dir, 'test_vector_field.png')
        viz_vector_field(points_aligned, directions, colors, offsets, fname)
        assert_equal(os.path.exists(fname), True)

        fname = os.path.join(temp_dir, 'test_mag_viz.png')
        viz_displacement_mag(affine_bundle, offsets, fname)
        assert_equal(os.path.exists(fname), True)
