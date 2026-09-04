import numpy as np
import numpy.testing as npt
import pytest

from dipy.core.geometry import vec2vec_rotmat
from dipy.core.gradients import gradient_table
from dipy.core.sphere import HemiSphere, disperse_charges
from dipy.data import get_sphere
from dipy.reconst import qti
from dipy.reconst.dki import DiffusionKurtosisModel
from dipy.reconst.dti import TensorModel
from dipy.sims.voxel import multi_tensor
from dipy.utils.optpkg import optional_package

_, have_matplotlib, _ = optional_package("matplotlib")

if have_matplotlib:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from dipy.viz import plotting

pytestmark = [
    pytest.mark.skipif(not have_matplotlib, reason="Requires Matplotlib"),
    pytest.mark.filterwarnings("ignore:FigureCanvasAgg is non-interactive"),
]


@pytest.fixture(autouse=True)
def close_figures():
    yield
    plt.close("all")


def _multishell_gtab():
    sphere = get_sphere(name="repulsion100")
    directions = sphere.vertices[:30]
    bvals = np.concatenate([np.zeros(6), np.full(30, 1000.0), np.full(30, 2000.0)])
    bvecs = np.vstack([np.zeros((6, 3)), directions, directions])
    return gradient_table(bvals, bvecs=bvecs)


@pytest.fixture(scope="module")
def crossing_volume():
    """Real two-fibre crossing signal on a 3 x 4 x 1 grid."""
    gtab = _multishell_gtab()
    mevals = np.array([[0.0017, 0.0003, 0.0003], [0.0017, 0.0003, 0.0003]])
    signal, _ = multi_tensor(
        gtab, mevals, S0=100, angles=[(0, 0), (60, 0)], fractions=[50, 50], snr=None
    )
    rng = np.random.default_rng(42)
    data = np.tile(signal, (3, 4, 1, 1))
    data = data + rng.normal(scale=0.5, size=data.shape)
    return gtab, data


@pytest.fixture(scope="module")
def dti_fit(crossing_volume):
    gtab, data = crossing_volume
    return TensorModel(gtab).fit(data)


@pytest.fixture(scope="module")
def dki_fit(crossing_volume):
    gtab, data = crossing_volume
    return DiffusionKurtosisModel(gtab).fit(data)


def _qti_gtab():
    """b0 plus two shells in linear and planar tensor encoding."""
    rng = np.random.default_rng(123)
    n_dir = 30
    hsph = HemiSphere(
        theta=np.pi * rng.random(n_dir), phi=2 * np.pi * rng.random(n_dir)
    )
    hsph, _ = disperse_charges(hsph, 100)
    directions = hsph.vertices
    bvecs = np.vstack([np.zeros(3)] + [directions] * 4)
    bvals = np.concatenate(
        (
            np.zeros(1),
            np.ones(n_dir),
            np.ones(n_dir) * 2,
            np.ones(n_dir),
            np.ones(n_dir) * 2,
        )
    )
    btens = np.array(["LTE"] * (1 + n_dir * 2) + ["PTE"] * (n_dir * 2))
    return gradient_table(bvals, bvecs=bvecs, btens=btens)


def _dispersed_dtd(n_dispersed):
    """Six equally anisotropic tensors, ``n_dispersed`` of them fanned out.

    Keeping every tensor anisotropic keeps the distribution's covariance well
    away from zero, so ``c_mu`` stays positive and ``ufa`` is real. Keeping at
    least one tensor off the fan keeps the mean tensor anisotropic, so ``fa``
    is defined. Varying ``n_dispersed`` sweeps ``fa`` without touching ``ufa``.
    """
    evals = np.array([2.0e-3, 0.2e-3, 0.2e-3])
    golden = (1 + np.sqrt(5)) / 2
    fan = np.array(
        [
            [0, 1, golden],
            [0, 1, -golden],
            [1, golden, 0],
            [1, -golden, 0],
            [golden, 0, 1],
            [golden, 0, -1],
        ]
    ) / np.linalg.norm([0, 1, golden])
    reference = np.array([1.0, 0.0, 0.0])

    dtd = np.zeros((6, 3, 3))
    for index in range(6):
        target = fan[index] if index < n_dispersed else reference
        rotation = vec2vec_rotmat(reference, target)
        dtd[index] = rotation @ np.diag(evals) @ rotation.T
    return dtd


def _qti_volume(gtab, dispersion_levels, offset, shape=(4, 5, 3)):
    """Volume whose voxels cycle through ``dispersion_levels``."""
    signals = np.stack(
        [
            qti.qti_signal(
                gtab,
                np.mean(_dispersed_dtd(level), axis=0),
                qti.dtd_covariance(_dispersed_dtd(level)),
            )
            for level in dispersion_levels
        ]
    )
    data = np.zeros((*shape, signals.shape[1]))
    for i in range(shape[0]):
        for j in range(shape[1]):
            for k in range(shape[2]):
                data[i, j, k] = signals[(i + j + k + offset) % len(signals)]
    return data


@pytest.fixture(scope="module")
def qti_fits():
    """Three real QTI fits of distinct diffusion tensor distributions.

    The three volumes differ by how the anisotropic tensors are fanned out per
    voxel rather than by added noise: an unconstrained QTI fit of noisy data
    can return a negative ``c_mu``, and ``ufa = sqrt(c_mu)`` would then be NaN.
    """
    gtab = _qti_gtab()
    levels = [1, 2, 3, 4, 5]

    ground_truth = qti.QtiModel(gtab, fit_method="WLS").fit(
        _qti_volume(gtab, levels, 0)
    )
    fit1 = qti.QtiModel(gtab, fit_method="WLS").fit(_qti_volume(gtab, levels, 1))
    fit2 = qti.QtiModel(gtab, fit_method="OLS").fit(_qti_volume(gtab, levels, 2))

    mask = np.zeros((4, 5), dtype=bool)
    mask[1:3, 1:4] = True
    return ground_truth, fit1, fit2, mask


def _displayed(ax):
    return ax.get_images()[-1].get_array()


def test_compare_maps_lays_fits_on_rows_and_maps_on_columns(dti_fit, dki_fit):
    plotting.compare_maps(
        [dti_fit, dki_fit], ["fa", "md"], transpose=False, filename=None
    )

    axes = plt.gcf().axes
    assert len(axes) == 4
    assert [ax.get_ylabel() for ax in axes] == ["Fit 1", "", "Fit 2", ""]
    assert [ax.get_title() for ax in axes] == ["fa", "md", "", ""]


def test_compare_maps_transpose_swaps_rows_and_columns(dti_fit, dki_fit):
    plotting.compare_maps(
        [dti_fit, dki_fit], ["fa", "md"], transpose=True, filename=None
    )

    axes = plt.gcf().axes
    assert [ax.get_title() for ax in axes] == ["Fit 1", "Fit 2", "", ""]
    assert [ax.get_ylabel() for ax in axes] == ["fa", "", "md", ""]


def test_compare_maps_auto_transposes_when_more_fits_than_maps(dti_fit, dki_fit):
    plotting.compare_maps([dti_fit, dki_fit], ["fa"], filename=None)

    axes = plt.gcf().axes
    assert [ax.get_title() for ax in axes] == ["Fit 1", "Fit 2"]


def test_compare_maps_keeps_fits_on_rows_when_more_maps_than_fits(dti_fit):
    plotting.compare_maps([dti_fit], ["fa", "md"], filename=None)

    axes = plt.gcf().axes
    assert [ax.get_title() for ax in axes] == ["fa", "md"]


def test_compare_maps_custom_labels(dti_fit, dki_fit):
    plotting.compare_maps(
        [dti_fit, dki_fit],
        ["fa", "md"],
        transpose=False,
        fit_labels=["DTI", "DKI"],
        map_labels=["Anisotropy", "Diffusivity"],
        filename=None,
    )

    axes = plt.gcf().axes
    assert axes[0].get_ylabel() == "DTI"
    assert axes[2].get_ylabel() == "DKI"
    assert [axes[0].get_title(), axes[1].get_title()] == [
        "Anisotropy",
        "Diffusivity",
    ]


def test_compare_maps_displays_the_transposed_squeezed_map(dti_fit):
    plotting.compare_maps([dti_fit], ["fa"], filename=None)

    displayed = _displayed(plt.gcf().axes[0])
    npt.assert_allclose(displayed, np.squeeze(dti_fit.fa).T)


def test_compare_maps_calls_callable_attributes(dki_fit):
    plotting.compare_maps([dki_fit], ["mk"], filename=None)

    displayed = _displayed(plt.gcf().axes[0])
    npt.assert_allclose(displayed, np.squeeze(dki_fit.mk()).T)


def test_compare_maps_warns_and_blanks_unknown_attributes(dti_fit):
    with pytest.warns(UserWarning, match="Could not recover attribute nonexistent"):
        plotting.compare_maps([dti_fit], ["nonexistent"], filename=None)

    npt.assert_array_equal(_displayed(plt.gcf().axes[0]), np.zeros((2, 2)))


def test_compare_maps_shared_kwargs_dict_applies_to_every_panel(dti_fit, dki_fit):
    plotting.compare_maps(
        [dti_fit, dki_fit],
        ["fa", "md"],
        transpose=False,
        fit_kwargs={"alpha": 0.5},
        map_kwargs={"vmin": 0.0},
        filename=None,
    )

    for ax in plt.gcf().axes:
        image = ax.get_images()[0]
        assert image.get_alpha() == 0.5
        assert image.get_clim()[0] == 0.0


def test_compare_maps_per_fit_and_per_map_kwargs_lists(dti_fit, dki_fit):
    plotting.compare_maps(
        [dti_fit, dki_fit],
        ["fa", "md"],
        transpose=False,
        fit_kwargs=[{"alpha": 0.25}, {"alpha": 0.75}],
        map_kwargs=[{"vmax": 1.0}, {"vmax": 2.0}],
        filename=None,
    )

    axes = plt.gcf().axes
    assert [axes[0].get_images()[0].get_alpha() for _ in range(1)] == [0.25]
    assert axes[2].get_images()[0].get_alpha() == 0.75
    assert axes[0].get_images()[0].get_clim()[1] == 1.0
    assert axes[1].get_images()[0].get_clim()[1] == 2.0


def test_compare_maps_hides_ticks_and_spines(dti_fit):
    plotting.compare_maps([dti_fit], ["fa"], filename=None)

    ax = plt.gcf().axes[0]
    assert list(ax.get_xticks()) == []
    assert list(ax.get_yticks()) == []
    assert not any(spine.get_visible() for spine in ax.spines.values())


def test_compare_maps_saves_to_file(tmp_path, dti_fit):
    fname = tmp_path / "maps.png"

    plotting.compare_maps([dti_fit], ["fa", "md"], filename=str(fname))

    assert fname.stat().st_size > 0


def test_compare_qti_maps_builds_a_grid_of_four_columns(qti_fits):
    ground_truth, fit1, fit2, mask = qti_fits

    plotting.compare_qti_maps(ground_truth, fit1, fit2, mask, slice=1)

    axes = plt.gcf().axes
    assert len(axes) == 8
    assert [axes[i].get_title() for i in range(4)] == [
        "GROUND TRUTH",
        "QTI",
        "QTI+",
        "VALUE DISTRIBUTION",
    ]
    assert axes[0].get_ylabel() == "fa"
    assert axes[4].get_ylabel() == "ufa"


def test_compare_qti_maps_shows_rotated_slices(qti_fits):
    ground_truth, fit1, fit2, mask = qti_fits

    plotting.compare_qti_maps(ground_truth, fit1, fit2, mask, slice=1)

    axes = plt.gcf().axes
    npt.assert_allclose(
        _displayed(axes[0]), np.rot90(ground_truth.fa[:, :, 1]), atol=1e-12
    )
    npt.assert_allclose(_displayed(axes[1]), np.rot90(fit1.fa[:, :, 1]), atol=1e-12)
    npt.assert_allclose(_displayed(axes[2]), np.rot90(fit2.fa[:, :, 1]), atol=1e-12)


def test_compare_qti_maps_draws_a_background_under_every_map(qti_fits):
    ground_truth, fit1, fit2, mask = qti_fits

    plotting.compare_qti_maps(ground_truth, fit1, fit2, mask, slice=1)

    axes = plt.gcf().axes
    for column in (0, 1, 2, 4, 5, 6):
        background = axes[column].get_images()[0].get_array()
        assert background.shape == ground_truth.S0_hat.shape[0:2]
        npt.assert_array_equal(background, 0)
        assert list(axes[column].get_xticks()) == []
        assert list(axes[column].get_yticks()) == []


def test_compare_qti_maps_histogram_legend_and_limits(qti_fits):
    ground_truth, fit1, fit2, mask = qti_fits

    plotting.compare_qti_maps(
        ground_truth,
        fit1,
        fit2,
        mask,
        maps=("fa", "ufa"),
        fitname=("first", "second"),
        xlimits=([0.0, 0.5], [0.2, 0.9]),
        disprange=([0.0, 0.5], [0.1, 0.6]),
        slice=1,
    )

    axes = plt.gcf().axes
    assert [text.get_text() for text in axes[3].get_legend().get_texts()] == [
        "first",
        "second",
        "GT",
    ]
    assert axes[3].get_xlim() == (0.0, 0.5)
    assert axes[7].get_xlim() == (0.2, 0.9)
    assert axes[0].get_images()[-1].get_clim() == (0.0, 0.5)
    assert axes[4].get_images()[-1].get_clim() == (0.1, 0.6)
    assert [axes[1].get_title(), axes[2].get_title()] == ["first", "second"]


def test_compare_qti_maps_histograms_use_only_masked_voxels(qti_fits):
    ground_truth, fit1, fit2, mask = qti_fits

    plotting.compare_qti_maps(ground_truth, fit1, fit2, mask, slice=1)

    hist_ax = plt.gcf().axes[3]
    bars = hist_ax.containers[0]
    masked_values = fit1.fa[mask, 1]

    assert bars[0].get_label() == "QTI"
    assert len(bars) == 40
    npt.assert_allclose(bars[0].get_x(), masked_values.min())
    npt.assert_allclose(bars[-1].get_x() + bars[-1].get_width(), masked_values.max())
    npt.assert_allclose(sum(bar.get_height() * bar.get_width() for bar in bars), 1.0)


def test_bundle_profile_plot_mean_line(tmp_path):
    x = np.arange(1, 6)
    profile = np.array([1.0, 2.0, 3.0, 2.0, 1.0])

    plotting.bundle_profile_plot(x, profile, "FA", show=True)

    ax = plt.gcf().axes[0]
    line = ax.get_lines()[0]
    npt.assert_array_equal(line.get_xdata(), x)
    npt.assert_array_equal(line.get_ydata(), profile)
    assert line.get_label() == "Mean"
    assert ax.get_xlabel() == "Segment Number"
    assert ax.get_ylabel() == "FA"
    assert ax.get_title() == "Bundle Profile"
    npt.assert_array_equal(ax.get_xticks(), x)


def test_bundle_profile_plot_custom_title():
    x = np.arange(1, 4)

    plotting.bundle_profile_plot(x, np.ones(3), "MD", title="Arcuate", show=True)

    assert plt.gcf().axes[0].get_title() == "Arcuate"


def test_bundle_profile_plot_std_band_sets_ylim():
    x = np.arange(1, 6)
    profile = np.array([1.0, 2.0, 3.0, 2.0, 1.0])
    std = np.full(5, 0.5)

    plotting.bundle_profile_plot(x, profile, "FA", std=std, show=True)

    ax = plt.gcf().axes[0]
    labels = [text.get_text() for text in ax.get_legend().get_texts()]
    assert labels == ["Mean", "Std"]
    assert ax.get_ylim() == (0.0, (profile + std).max() + 2)


def test_bundle_profile_plot_closes_figure_when_not_shown():
    n_open = len(plt.get_fignums())

    plotting.bundle_profile_plot(np.arange(3), np.zeros(3), "FA", show=False)

    assert len(plt.get_fignums()) == n_open


def test_bundle_profile_plot_keeps_figure_open_when_shown():
    n_open = len(plt.get_fignums())

    plotting.bundle_profile_plot(np.arange(3), np.zeros(3), "FA", show=True)

    assert len(plt.get_fignums()) == n_open + 1


def test_bundle_profile_plot_saves_file(tmp_path):
    fname = tmp_path / "profile.png"

    plotting.bundle_profile_plot(
        np.arange(3), np.zeros(3), "FA", save_path=str(fname), show=False
    )

    assert fname.stat().st_size > 0


def test_image_mosaic_returns_figure_and_axes():
    images = [np.arange(6.0).reshape(2, 3), np.arange(6.0).reshape(2, 3)[::-1]]

    fig, ax = plotting.image_mosaic(images, ax_kwargs=[{}, {}], filename=None)

    assert len(ax) == 2
    npt.assert_array_equal(ax[0].get_images()[0].get_array(), images[0])
    npt.assert_array_equal(ax[1].get_images()[0].get_array(), images[1])
    assert len(fig.axes) == 4


def test_image_mosaic_labels_and_kwargs():
    images = [np.zeros((2, 2)), np.ones((2, 2))]

    _, ax = plotting.image_mosaic(
        images,
        ax_labels=["left", "right"],
        ax_kwargs=[{"cmap": "gray"}, {"vmin": 0.0, "vmax": 2.0}],
        figsize=(6, 3),
        filename=None,
    )

    assert [a.get_title() for a in ax] == ["left", "right"]
    assert ax[0].get_images()[0].get_cmap().name == "gray"
    assert ax[1].get_images()[0].get_clim() == (0.0, 2.0)


def test_image_mosaic_without_labels_has_no_titles():
    images = [np.zeros((2, 2)), np.ones((2, 2))]

    _, ax = plotting.image_mosaic(images, ax_kwargs=[{}, {}], filename=None)

    assert [a.get_title() for a in ax] == ["", ""]


def test_image_mosaic_saves_file(tmp_path):
    fname = tmp_path / "mosaic.png"
    images = [np.zeros((2, 2)), np.ones((2, 2))]

    plotting.image_mosaic(images, ax_kwargs=[{}, {}], filename=str(fname))

    assert fname.stat().st_size > 0
