import os
import tempfile
import warnings

from numpy.testing import assert_equal, assert_raises
import pytest

from dipy.align.streamwarp import bundlewarp, bundlewarp_vector_filed
from dipy.data import read_five_af_bundles, two_cingulum_bundles
from dipy.tracking.streamline import (
    Streamlines,
    set_number_of_points,
    unlist_streamlines,
)
from dipy.utils.optpkg import optional_package

_, have_matplotlib, _ = optional_package("matplotlib")
fury, have_fury, _ = optional_package("fury", min_version="0.10.0")
pd, have_pd, _ = optional_package("pandas")

if have_fury:
    from dipy.viz import window
    from dipy.viz.streamline import (
        sagittal_deprecation_warning_msg,
        show_bundles,
        viz_displacement_mag,
        viz_two_bundles,
        viz_vector_field,
    )

bundles = read_five_af_bundles()


@pytest.mark.skipif(
    not have_fury or not have_matplotlib, reason="Requires FURY and Matplotlib"
)
def test_output_created():
    colors = [
        [0.91, 0.26, 0.35],
        [0.99, 0.50, 0.38],
        [0.99, 0.88, 0.57],
        [0.69, 0.85, 0.64],
        [0.51, 0.51, 0.63],
    ]

    with tempfile.TemporaryDirectory() as temp_dir:
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message=sagittal_deprecation_warning_msg,
                category=DeprecationWarning,
            )

            view = "sagital"  # codespell:ignore sagital
            fname = os.path.join(temp_dir, f"test_{view}.png")
            show_bundles(bundles, interactive=False, view=view, save_as=fname)
            assert_equal(os.path.exists(fname), True)

        views = ["axial", "sagittal", "coronal"]

        for view in views:
            fname = os.path.join(temp_dir, f"test_{view}.png")
            show_bundles(bundles, interactive=False, view=view, save_as=fname)
            assert_equal(os.path.exists(fname), True)

        fname = os.path.join(temp_dir, "test_colors.png")
        show_bundles(bundles, interactive=False, colors=colors, save_as=fname)
        assert_equal(os.path.exists(fname), True)

        # Check rendered image is not empty
        report = window.analyze_snapshot(fname, find_objects=True)
        assert_equal(report.objects > 0, True)

        cb1, cb2 = two_cingulum_bundles()

        fname = os.path.join(temp_dir, "test_two_bundles.png")
        viz_two_bundles(cb1, cb2, fname=fname)
        assert_equal(os.path.exists(fname), True)


@pytest.mark.skipif(not have_fury, reason="Requires FURY")
def test_incorrect_view():
    assert_raises(
        ValueError, show_bundles, bundles, interactive=False, view="wrong_view"
    )


@pytest.mark.skipif(
    not have_fury or not have_matplotlib or not have_pd,
    reason="Requires FURY, Matplotlib and Pandas",
)
def test_bundlewarp_viz():
    with tempfile.TemporaryDirectory() as temp_dir:
        cingulum_bundles = two_cingulum_bundles()

        cb1 = cingulum_bundles[0]
        cb1 = Streamlines(set_number_of_points(cb1, nb_points=20))

        cb2 = cingulum_bundles[1]
        cb2 = Streamlines(set_number_of_points(cb2, nb_points=20))

        deformed_bundle, affine_bundle, _, _, _ = bundlewarp(cb1, cb2)

        offsets, directions, colors = bundlewarp_vector_filed(
            affine_bundle, deformed_bundle
        )
        points_aligned, _ = unlist_streamlines(affine_bundle)

        fname = os.path.join(temp_dir, "test_vector_field.png")
        viz_vector_field(points_aligned, directions, colors, offsets, fname)
        assert_equal(os.path.exists(fname), True)

        fname = os.path.join(temp_dir, "test_mag_viz.png")
        viz_displacement_mag(affine_bundle, offsets, fname)
        assert_equal(os.path.exists(fname), True)
