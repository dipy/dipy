import numpy as np
import numpy.testing as npt
import pytest

from dipy.align.imwarp import DiffeomorphicMap, SymmetricDiffeomorphicRegistration
from dipy.align.metrics import SSDMetric
from dipy.testing.decorators import set_random_number_generator
from dipy.utils.optpkg import optional_package

_, have_matplotlib, _ = optional_package("matplotlib")

if have_matplotlib:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from dipy.viz import regtools

pytestmark = pytest.mark.skipif(not have_matplotlib, reason="Requires Matplotlib")


@pytest.fixture(autouse=True)
def close_figures():
    yield
    plt.close("all")


def _identity_mapping(domain_shape, codomain_shape):
    mapping = DiffeomorphicMap(
        2,
        domain_shape,
        domain_shape=domain_shape,
        codomain_shape=codomain_shape,
    )
    mapping.allocate()
    return mapping


@pytest.fixture(scope="module")
def sdr_mapping():
    rng = np.random.default_rng(1234)
    moving = rng.random((11, 12))
    static = rng.random((13, 14))
    sdr = SymmetricDiffeomorphicRegistration(
        metric=SSDMetric(static.ndim), level_iters=[10, 5], inv_iter=5
    )
    return sdr.optimize(static, moving)


def test_draw_lattice_2d_shape_and_lines():
    lattice = regtools.draw_lattice_2d(3, 4, 10)

    npt.assert_equal(lattice.shape, (1 + 11 * 3, 1 + 11 * 4))
    npt.assert_array_equal(lattice[0, :], 0)
    npt.assert_array_equal(lattice[:, 0], 0)
    npt.assert_array_equal(lattice[11 * 3, :], 0)
    npt.assert_array_equal(lattice[:, 11 * 4], 0)
    npt.assert_array_equal(lattice[1:11, 1:11], 127)


def test_draw_lattice_2d_single_cell():
    lattice = regtools.draw_lattice_2d(1, 1, 1)

    npt.assert_equal(lattice.shape, (3, 3))
    npt.assert_array_equal(lattice, np.array([[0, 0, 0], [0, 127, 0], [0, 0, 0]]))


def test_simple_plot_writes_file(tmp_path):
    fname = tmp_path / "simple.png"

    regtools.simple_plot(str(fname), "Title", [1, 2, 3], [0.5, 1.5, 2.5], "x", "y")

    assert fname.exists()
    assert fname.stat().st_size > 0


def test_simple_plot_clears_current_figure(tmp_path):
    regtools.simple_plot(str(tmp_path / "simple.png"), "T", [0, 1], [0, 1], "x", "y")

    assert plt.gcf().axes == []


def test_overlay_images_returns_three_axes():
    img0 = np.arange(20, dtype=float).reshape(4, 5)
    img1 = img0[::-1].copy()

    fig = regtools.overlay_images(img0, img1, title0="L", title_mid="M", title1="R")

    assert len(fig.axes) == 3
    assert [ax.get_title() for ax in fig.axes] == ["L", "M", "R"]


def test_overlay_images_channels_are_separated():
    img0 = np.arange(20, dtype=float).reshape(4, 5)
    img1 = img0[::-1].copy()

    fig = regtools.overlay_images(img0, img1)

    left, overlay, right = (ax.get_images()[0].get_array() for ax in fig.axes)
    npt.assert_array_equal(left[..., 1], 0)
    npt.assert_array_equal(left[..., 2], 0)
    npt.assert_array_equal(right[..., 0], 0)
    npt.assert_array_equal(right[..., 2], 0)
    npt.assert_array_equal(overlay[..., 0], left[..., 0])
    npt.assert_array_equal(overlay[..., 1], right[..., 1])
    npt.assert_equal(left[..., 0].max(), 255)


def test_overlay_images_saves_file(tmp_path):
    fname = tmp_path / "overlay.png"
    img = np.arange(20, dtype=float).reshape(4, 5)

    regtools.overlay_images(img, img, fname=str(fname), dpi=50)

    assert fname.stat().st_size > 0


def test_plot_slices_uses_middle_slices_by_default():
    volume = np.arange(4 * 5 * 6, dtype=float).reshape(4, 5, 6)

    fig = regtools.plot_slices(volume)

    assert [ax.get_title() for ax in fig.axes] == ["Axial", "Coronal", "Sagittal"]
    axial, coronal, sagittal = (ax.get_images()[0].get_array() for ax in fig.axes)
    assert axial.shape == (5, 4)
    assert coronal.shape == (6, 4)
    assert sagittal.shape == (6, 5)


def test_plot_slices_honours_slice_indices():
    volume = np.zeros((4, 5, 6))
    volume[1, 2, 3] = 100.0

    fig = regtools.plot_slices(volume, slice_indices=(1, 2, 3))

    axial, coronal, sagittal = (ax.get_images()[0].get_array() for ax in fig.axes)
    assert axial[2, 1] == 255
    assert coronal[3, 1] == 255
    assert sagittal[3, 2] == 255


def test_plot_slices_saves_file(tmp_path):
    fname = tmp_path / "slices.png"
    volume = np.arange(4 * 5 * 6, dtype=float).reshape(4, 5, 6)

    regtools.plot_slices(volume, fname=str(fname), dpi=50)

    assert fname.stat().st_size > 0


@pytest.mark.parametrize(
    "slice_type,expected_shape", [(0, (6, 5)), (1, (6, 4)), (2, (5, 4))]
)
def test_overlay_slices_shapes_per_slice_type(slice_type, expected_shape):
    left = np.arange(4 * 5 * 6, dtype=float).reshape(4, 5, 6)
    right = left[::-1].copy()

    fig = regtools.overlay_slices(left, right, slice_type=slice_type)

    overlay = fig.axes[1].get_images()[0].get_array()
    assert overlay.shape == expected_shape + (3,)
    assert [ax.get_title() for ax in fig.axes] == ["Left", "Overlay", "Right"]


def test_overlay_slices_titles_and_blue_channel_empty():
    left = np.arange(4 * 5 * 6, dtype=float).reshape(4, 5, 6)
    right = left[::-1].copy()

    fig = regtools.overlay_slices(left, right, ltitle="static", rtitle="moving")

    assert [ax.get_title() for ax in fig.axes] == ["static", "Overlay", "moving"]
    overlay = fig.axes[1].get_images()[0].get_array()
    npt.assert_array_equal(overlay[..., 2], 0)


def test_overlay_slices_invalid_slice_type_returns_none():
    volume = np.arange(4 * 5 * 6, dtype=float).reshape(4, 5, 6)

    assert regtools.overlay_slices(volume, volume, slice_type=3) is None


def test_overlay_slices_saves_file(tmp_path):
    fname = tmp_path / "overlay_slices.png"
    volume = np.arange(4 * 5 * 6, dtype=float).reshape(4, 5, 6)

    regtools.overlay_slices(volume, volume, fname=str(fname), dpi=50)

    assert fname.stat().st_size > 0


@set_random_number_generator()
def test_plot_2d_diffeomorphic_map(rng=None):
    mv_shape = (11, 12)
    moving = rng.random(mv_shape)
    st_shape = (13, 14)
    static = rng.random(st_shape)
    sdr = SymmetricDiffeomorphicRegistration(
        metric=SSDMetric(static.ndim), level_iters=[200, 100, 50, 25], inv_iter=50
    )
    mapping = sdr.optimize(static, moving)

    ff = regtools.plot_2d_diffeomorphic_map(mapping, delta=10)
    npt.assert_equal(ff[0].shape, st_shape)
    npt.assert_equal(ff[1].shape, mv_shape)

    ff = regtools.plot_2d_diffeomorphic_map(
        mapping, delta=10, direct_grid_shape=(7, 8), inverse_grid_shape=(9, 10)
    )
    npt.assert_equal(ff[0].shape, (7, 8))
    npt.assert_equal(ff[1].shape, (9, 10))


def test_plot_2d_diffeomorphic_map_inverse_map_swaps_default_grids(sdr_mapping):
    n_figures_before = len(plt.get_fignums())

    assert sdr_mapping.is_inverse
    forward, backward = regtools.plot_2d_diffeomorphic_map(
        sdr_mapping, delta=5, show_figure=False
    )

    assert len(plt.get_fignums()) == n_figures_before
    npt.assert_equal(forward.shape, sdr_mapping.codomain_shape)
    npt.assert_equal(backward.shape, sdr_mapping.domain_shape)


def test_plot_2d_diffeomorphic_map_draws_two_titled_subplots(sdr_mapping):
    regtools.plot_2d_diffeomorphic_map(sdr_mapping, delta=5)

    titles = [ax.get_title() for ax in plt.gcf().axes]
    assert titles == ["Direct transform", "Inverse transform"]


def test_plot_2d_diffeomorphic_map_direct_map_uses_domain_as_direct_grid():
    mapping = _identity_mapping((13, 14), (11, 12))

    assert not mapping.is_inverse
    forward, backward = regtools.plot_2d_diffeomorphic_map(
        mapping, delta=5, show_figure=False
    )

    npt.assert_equal(forward.shape, mapping.domain_shape)
    npt.assert_equal(backward.shape, mapping.codomain_shape)


def test_plot_2d_diffeomorphic_map_identity_map_returns_the_drawn_lattice():
    mapping = _identity_mapping((23, 23), (23, 23))

    forward, _ = regtools.plot_2d_diffeomorphic_map(
        mapping, delta=10, show_figure=False
    )

    expected = regtools.draw_lattice_2d(2, 2, 10)[:23, :23]
    npt.assert_allclose(forward, expected, atol=1e-5)


def test_plot_2d_diffeomorphic_map_scaled_grid2world_shrinks_the_lattice():
    mapping = _identity_mapping((40, 40), (40, 40))
    scaling = np.diag([2.0, 2.0, 1.0])

    forward, _ = regtools.plot_2d_diffeomorphic_map(
        mapping,
        delta=10,
        direct_grid2world=scaling,
        inverse_grid2world=None,
        show_figure=False,
    )

    lattice = regtools.draw_lattice_2d(4, 4, 10)[:40, :40]
    npt.assert_allclose(forward[:20, :20], lattice[::2, ::2], atol=1e-5)


def test_plot_2d_diffeomorphic_map_scaled_inverse_grid2world():
    mapping = _identity_mapping((40, 40), (40, 40))
    scaling = np.diag([2.0, 2.0, 1.0])

    _, backward = regtools.plot_2d_diffeomorphic_map(
        mapping,
        delta=10,
        direct_grid2world=None,
        inverse_grid2world=scaling,
        show_figure=False,
    )

    lattice = regtools.draw_lattice_2d(4, 4, 10)[:40, :40]
    npt.assert_allclose(backward[:20, :20], lattice[::2, ::2], atol=1e-5)


def test_plot_2d_diffeomorphic_map_identity_grid2world(sdr_mapping):
    forward, backward = regtools.plot_2d_diffeomorphic_map(
        sdr_mapping,
        delta=5,
        direct_grid2world=None,
        inverse_grid2world=None,
        show_figure=False,
    )

    npt.assert_equal(forward.shape, sdr_mapping.codomain_shape)
    npt.assert_equal(backward.shape, sdr_mapping.domain_shape)


def test_plot_2d_diffeomorphic_map_saves_file(tmp_path, sdr_mapping):
    fname = tmp_path / "diffeomorphic.png"

    regtools.plot_2d_diffeomorphic_map(sdr_mapping, delta=5, fname=str(fname), dpi=50)

    assert fname.stat().st_size > 0
