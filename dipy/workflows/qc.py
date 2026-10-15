"""Quality control of dipy_auto pipeline stages.

Each pipeline stage gets a QC result: a list of metrics, each carrying a
PASS / WARN / FAIL / INFO status derived from the heuristic thresholds in
``QC_THRESHOLDS``, and a list of figures encoded as base64 PNG data URIs so
the HTML pages built from them are self-contained.
"""

import base64
import contextlib
import functools
import glob
import io
import math
import os
from pathlib import Path
import traceback

import nibabel as nib
import numpy as np
from scipy import ndimage
import trx.trx_file_memmap as tmm

from dipy.align.bundlemin import distance_matrix_mdf
from dipy.core.geometry import decompose_matrix
from dipy.core.gradients import gradient_table
from dipy.io.gradients import read_bvals_bvecs
from dipy.io.streamline import load_tractogram
from dipy.reconst.dti import tensor_prediction
from dipy.segment.threshold import otsu
from dipy.tracking.streamline import set_number_of_points
from dipy.tracking.utils import length
from dipy.utils.logging import logger
from dipy.utils.optpkg import optional_package

mpl_figure, has_mpl, _ = optional_package("matplotlib.figure")
mpl_collections, _, _ = optional_package("matplotlib.collections")

STATUS_LEVELS = ("INFO", "PASS", "WARN", "FAIL", "ERROR")
B0_THRESHOLD = 50

# Gradient file parameter names across the QC'd workflows, most common first:
# dipy_denoise_patch2self takes ``bval_files``, dipy_correct_biasfield
# ``bval`` / ``bvec``.
BVAL_PARAMS = ("bvalues_files", "bval_files", "bval")
BVEC_PARAMS = ("bvectors_files", "bvec_files", "bvec")

QC_THRESHOLDS = {
    "denoise_ssim": 0.90,
    "snr_drop_tolerance": 0.05,
    "gibbs_ssim": 0.95,
    "b0_intensity_cv": 0.10,
    "mask_fov_fraction": 0.60,
    "out_of_range_fraction": 0.01,
    "dti_residual_nrmse": 0.20,
    "short_streamline_mm": 10.0,
    "short_streamline_fraction": 0.30,
    "slr_scale": (0.7, 1.3),
    "min_bundle_streamlines": 10,
}

# Upper bounds are a little above free-water diffusivity at body temperature
# (3.0e-3 mm2/s), so that partial-volume CSF does not warn on every subject.
METRIC_RANGES = {
    "fa": (0.0, 1.0),
    "md": (0.0, 3.5e-3),
    "ad": (0.0, 3.5e-3),
    "rd": (0.0, 3.5e-3),
    "mk": (0.0, 3.0),
    "ak": (0.0, 3.0),
    "rk": (0.0, 3.0),
}


# =============================================================================
# Metric helpers
# =============================================================================


def ssim_3d(*, a, b, mask=None, sigma=1.5):
    """Structural similarity between two 3D volumes.

    Gaussian-window SSIM (Wang et al., 2004) averaged over ``mask``.

    Parameters
    ----------
    a, b : ndarray
        Volumes of identical shape.
    mask : ndarray of bool, optional
        Voxels over which the SSIM map is averaged. All voxels when ``None``.
    sigma : float, optional
        Standard deviation of the Gaussian window, in voxels.

    Returns
    -------
    float
        Mean SSIM, 1 for identical volumes.
    """
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    data_range = max(float(np.ptp(a)), np.finfo(float).eps)
    c1 = (0.01 * data_range) ** 2
    c2 = (0.03 * data_range) ** 2

    mu_a = ndimage.gaussian_filter(a, sigma)
    mu_b = ndimage.gaussian_filter(b, sigma)
    var_a = ndimage.gaussian_filter(a * a, sigma) - mu_a**2
    var_b = ndimage.gaussian_filter(b * b, sigma) - mu_b**2
    cov = ndimage.gaussian_filter(a * b, sigma) - mu_a * mu_b

    ssim_map = ((2 * mu_a * mu_b + c1) * (2 * cov + c2)) / (
        (mu_a**2 + mu_b**2 + c1) * (var_a + var_b + c2)
    )
    return float(ssim_map[mask].mean() if mask is not None else ssim_map.mean())


def foreground_mask(*, volume):
    """Otsu foreground of a 3D volume.

    Parameters
    ----------
    volume : ndarray
        3D volume.

    Returns
    -------
    ndarray of bool
        Foreground voxels; empty for a constant volume, which has no
        foreground to separate.
    """
    volume = np.nan_to_num(volume)
    if not np.ptp(volume):
        return np.zeros(volume.shape, dtype=bool)
    return volume > otsu(volume)


def snr(*, volume, mask):
    """Signal-to-noise ratio as mean foreground over background std.

    Parameters
    ----------
    volume : ndarray
        3D volume.
    mask : ndarray of bool
        Foreground voxels; the complement is the background.

    Returns
    -------
    float
        SNR, ``inf`` if the background is constant and ``nan`` if either
        region is empty. Callers must not compare non-finite values.
    """
    background, foreground = volume[~mask], volume[mask]
    if not (background.size and foreground.size):
        return float("nan")
    noise = float(np.std(background))
    if noise == 0:
        return float("inf")
    return float(np.mean(foreground)) / noise


def radial_amplitude_spectrum(*, volume, n_bins=64):
    """Radially averaged Fourier amplitude spectrum of a 3D volume.

    Parameters
    ----------
    volume : ndarray
        3D volume.
    n_bins : int, optional
        Number of spatial-frequency bins between 0 and 0.5 cycles/voxel.

    Returns
    -------
    freqs : ndarray
        Bin centers, in cycles/voxel.
    amplitude : ndarray
        Mean amplitude in each bin.
    """
    amplitude = np.abs(np.fft.fftn(volume - volume.mean()))
    grids = np.meshgrid(*[np.fft.fftfreq(n) for n in volume.shape], indexing="ij")
    radius = np.sqrt(sum(g**2 for g in grids))
    edges = np.linspace(0, 0.5, n_bins + 1)
    idx = np.digitize(radius.ravel(), edges) - 1
    valid = (idx >= 0) & (idx < n_bins)
    sums = np.bincount(idx[valid], weights=amplitude.ravel()[valid], minlength=n_bins)
    counts = np.bincount(idx[valid], minlength=n_bins)
    return 0.5 * (edges[1:] + edges[:-1]), sums / np.maximum(counts, 1)


def _plain(*, value):
    """Plain Python value, non-finite floats mapped to ``None``.

    JSON has no representation for NaN and infinity, so they are dropped
    here rather than written as the non-standard ``NaN`` / ``Infinity``.

    Parameters
    ----------
    value : object
        Metric value, possibly a NumPy scalar.

    Returns
    -------
    object
        ``value`` as a built-in type, ``None`` if it is a non-finite float.
    """
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _metric(*, name, value, unit="", ok=None, rule="", fail=False):
    """Build one metric record.

    Parameters
    ----------
    name : str
        Metric label.
    value : float or int or str
        Metric value.
    unit : str, optional
        Unit shown next to the value.
    ok : bool, optional
        Threshold outcome. ``None`` marks an informative metric.
    rule : str, optional
        Human-readable threshold.
    fail : bool, optional
        If True, a failed check is a FAIL instead of a WARN.

    Returns
    -------
    dict
        Metric record.
    """
    if ok is None:
        status = "INFO"
    elif ok:
        status = "PASS"
    else:
        status = "FAIL" if fail else "WARN"
    return {
        "name": name,
        "value": _plain(value=value),
        "unit": unit,
        "status": status,
        "rule": rule,
    }


def _fmt(*, values, precision=3):
    return ", ".join(f"{v:.{precision}g}" for v in np.ravel(values))


def _map_metrics(*, name, values, bounds):
    lo, hi = bounds
    p1, median, p99 = np.percentile(values, [1, 50, 99])
    outside = float(np.mean((values < lo) | (values > hi)))
    limit = QC_THRESHOLDS["out_of_range_fraction"]
    return [
        _metric(
            name=f"{name} median [p1, p99]",
            value=f"{median:.4g} [{p1:.4g}, {p99:.4g}]",
        ),
        _metric(
            name=f"{name} voxels outside [{lo:g}, {hi:g}]",
            value=100 * outside,
            unit="%",
            ok=outside <= limit,
            rule=f"≤ {100 * limit:g} %",
        ),
    ]


def qc_result(*, metrics=(), figures=(), error=None, status=None):
    """Assemble a stage QC result.

    Parameters
    ----------
    metrics : sequence of dict, optional
        Metric records from ``_metric``.
    figures : sequence of dict, optional
        Figure records; ``None`` entries (figures skipped because matplotlib
        is missing) are dropped.
    error : str, optional
        Error message or traceback shown on the stage page.
    status : str, optional
        Overall status. The worst metric status when ``None``.

    Returns
    -------
    dict
        ``{"status", "metrics", "figures", "error"}``.
    """
    metrics = list(metrics)
    if status is None:
        status = max(
            (m["status"] for m in metrics), key=STATUS_LEVELS.index, default="INFO"
        )
    return {
        "status": status,
        "metrics": metrics,
        "figures": [f for f in figures if f],
        "error": error,
    }


# =============================================================================
# Loaders
# =============================================================================


def _first_path(*, value):
    """First existing file behind a path, a glob pattern or a list of them.

    An existing path is returned as is, so that a file whose name holds glob
    characters such as ``[`` is not taken for a pattern.
    """
    for item in value if isinstance(value, (list, tuple)) else [value]:
        if item and os.path.exists(str(item)):
            return str(item)
        matches = sorted(glob.glob(str(item)))
        if matches:
            return matches[0]
    return None


def _is_nifti(*, path):
    p = str(path)
    return p.endswith((".nii", ".nii.gz")) and os.path.exists(p)


def _load_volume(*, path, volumes=None):
    """Load a NIfTI file reoriented to RAS, as float32.

    Parameters
    ----------
    path : str
        NIfTI file, or anything accepted by ``_first_path``.
    volumes : sequence of int, optional
        Indices along the 4th axis to read. All when ``None``.

    Returns
    -------
    data : ndarray
        RAS-oriented data.
    affine : ndarray
        Matching 4x4 affine.
    """
    img = nib.load(_first_path(value=path))
    if volumes is None or img.ndim == 3:
        data = np.asarray(img.dataobj, dtype=np.float32)
    else:
        data = np.stack([img.dataobj[..., i] for i in volumes], axis=-1)
        data = data.astype(np.float32)
    ornt = nib.orientations.io_orientation(img.affine)
    affine = img.affine @ nib.orientations.inv_ornt_aff(ornt, img.shape[:3])
    return nib.orientations.apply_orientation(data, ornt), affine


def _gradient_file(*, inputs, io_config, params, io_key):
    """Stage gradient file under any of its parameter names, else ``[io]``.

    Parameters
    ----------
    inputs : dict
        Resolved stage parameters.
    io_config : dict
        Pipeline ``[io]`` section.
    params : sequence of str
        Names the stage workflows give the file parameter, tried in order.
    io_key : str
        ``[io]`` entry used when the stage sets none of ``params``.

    Returns
    -------
    str or None
        First existing file, ``None`` when there is none.
    """
    value = next((inputs[p] for p in params if inputs.get(p)), None)
    return _first_path(value=value or io_config.get(io_key) or "")


def _bvals(*, inputs, io_config):
    """b-values and b0 threshold of a stage.

    The stage's own b-values (``bvalues_files``, or ``bval_files`` /
    ``bval`` depending on the workflow) and ``b0_threshold`` win over the
    pipeline ``[io]`` b-values and ``B0_THRESHOLD``, so the QC sees the same
    gradient table as the stage did.

    Parameters
    ----------
    inputs : dict
        Resolved stage parameters.
    io_config : dict
        Pipeline ``[io]`` section.

    Returns
    -------
    bvals : ndarray or None
        b-values, ``None`` when neither the stage nor ``[io]`` has any.
    b0_threshold : float
        b-value at or below which a volume counts as a b0.
    """
    path = _gradient_file(
        inputs=inputs, io_config=io_config, params=BVAL_PARAMS, io_key="bvals"
    )
    bvals = read_bvals_bvecs(path, None)[0] if path else None
    return bvals, float(inputs.get("b0_threshold", B0_THRESHOLD))


def _gradients(*, inputs, io_config):
    """b-values, b-vectors and b0 threshold of a stage.

    See ``_bvals``; the stage b-vectors (``bvectors_files``, ``bvec_files``
    or ``bvec``) likewise win over ``[io]`` so that a stage fed
    motion-corrected b-vectors is checked against them.

    Parameters
    ----------
    inputs : dict
        Resolved stage parameters.
    io_config : dict
        Pipeline ``[io]`` section.

    Returns
    -------
    bvals : ndarray
        b-values.
    bvecs : ndarray
        b-vectors.
    b0_threshold : float
        b-value at or below which a volume counts as a b0.
    """
    bvals, bvecs = read_bvals_bvecs(
        _gradient_file(
            inputs=inputs, io_config=io_config, params=BVAL_PARAMS, io_key="bvals"
        ),
        _gradient_file(
            inputs=inputs, io_config=io_config, params=BVEC_PARAMS, io_key="bvecs"
        ),
    )
    return bvals, bvecs, float(inputs.get("b0_threshold", B0_THRESHOLD))


def _b0_indices(*, bvals, n_volumes, threshold=B0_THRESHOLD):
    if bvals is None or len(bvals) != n_volumes:
        return np.array([0])
    idx = np.flatnonzero(bvals <= threshold)
    return idx if idx.size else np.array([0])


def _mean_b0(*, path, bvals, b0_threshold=B0_THRESHOLD):
    """Mean b0 volume of a 3D/4D NIfTI file, read volume by volume."""
    img = nib.load(_first_path(value=path))
    if img.ndim == 3:
        return _load_volume(path=path)[0]
    idx = _b0_indices(bvals=bvals, n_volumes=img.shape[3], threshold=b0_threshold)
    return _load_volume(path=path, volumes=idx)[0].mean(axis=-1)


def _load_mask(*, path):
    return _load_volume(path=path)[0] > 0


@contextlib.contextmanager
def _open_streamlines(*, path):
    """Streamlines of a tractogram, in RASMM space.

    ``.trx`` files are memory-mapped rather than read, since a whole-brain
    tractogram does not fit in memory comfortably. The mapping, and the
    temporary folder a compressed ``.trx`` is extracted to, are released on
    exit, so the streamlines must not be used outside the ``with`` block.

    Parameters
    ----------
    path : str
        Tractogram file.

    Yields
    ------
    nibabel.streamlines.ArraySequence
        Streamlines.
    """
    if str(path).endswith((".trk", ".tck")):
        # A .tck carries no reference image, which load_tractogram(..., "same")
        # needs; nibabel returns both formats in RASMM without one.
        yield nib.streamlines.load(str(path)).streamlines
        return
    if not str(path).endswith(".trx"):
        yield load_tractogram(str(path), "same", bbox_valid_check=False).streamlines
        return
    trx = tmm.load(str(path))
    try:
        yield trx.streamlines
    finally:
        trx.close()


def _lengths(*, streamlines):
    return np.asarray(list(length(streamlines)), dtype=float)


def _mean_min_mdf(*, a, b, n_points=20):
    """Symmetric mean of closest-streamline MDF distances, in mm."""
    d = distance_matrix_mdf(
        set_number_of_points(a, nb_points=n_points),
        set_number_of_points(b, nb_points=n_points),
    )
    return 0.5 * float(d.min(axis=0).mean() + d.min(axis=1).mean())


# =============================================================================
# Figures
# =============================================================================


def _optional_figure(func):
    """Skip a figure, rather than fail, when it cannot be drawn.

    A figure is dropped when matplotlib is missing and when drawing raises,
    so that a plotting problem never costs the metrics of a stage.
    """

    @functools.wraps(func)
    def wrapper(**kwargs):
        if not has_mpl:
            return None
        try:
            return func(**kwargs)
        except Exception:
            logger.warning(f"QC figure '{kwargs.get('title')}' failed to render.")
            logger.debug(traceback.format_exc())
            return None

    return wrapper


def _encode(*, title, fig):
    buffer = io.BytesIO()
    fig.savefig(
        buffer,
        format="png",
        dpi=100,
        bbox_inches="tight",
        facecolor=fig.get_facecolor(),
    )
    data = base64.b64encode(buffer.getvalue()).decode("ascii")
    return {"title": title, "data_uri": f"data:image/png;base64,{data}"}


def _window(*, volume):
    """Display range and colormap; signed volumes get a diverging map.

    A volume counts as signed only when a real share of it is negative: an
    interpolated or denoised image undershoots on a few voxels, and those
    must not turn it into a difference map.
    """
    if np.mean(volume < 0) > 0.05:
        vmax = float(np.percentile(np.abs(volume), 99)) or 1.0
        return -vmax, vmax, "RdBu_r"
    nonzero = volume[volume > 0]
    if not nonzero.size:
        return 0.0, 1.0, "gray"
    vmin, vmax = np.percentile(nonzero, [2, 98])
    if vmax - vmin < 1e-6:
        vmin, vmax = 0.0, float(nonzero.max())
    return float(vmin), float(vmax), "gray"


@_optional_figure
def _mosaic(*, title, volumes, labels, overlay=None, n_cuts=7):
    """Axial cuts of several 3D volumes, one row per volume.

    Cuts are spread over the foreground extent of the first volume and taken
    at the same relative height in every row, so rows on different grids
    stay comparable.
    """
    volumes = [np.nan_to_num(v) for v in volumes]
    ref = volumes[0]
    z = np.flatnonzero(foreground_mask(volume=ref).any(axis=(0, 1)))
    lo, hi = (z[0], z[-1]) if z.size else (0, ref.shape[2] - 1)
    fractions = np.linspace(lo, hi, n_cuts + 2)[1:-1] / max(ref.shape[2] - 1, 1)

    fig = mpl_figure.Figure(
        figsize=(1.8 * n_cuts, 1.9 * len(volumes)), facecolor="black"
    )
    axes = fig.subplots(len(volumes), n_cuts, squeeze=False)
    for row, (volume, label) in enumerate(zip(volumes, labels)):
        vmin, vmax, cmap = _window(volume=volume)
        for col, frac in enumerate(fractions):
            k = int(round(frac * (volume.shape[2] - 1)))
            ax = axes[row, col]
            ax.imshow(
                volume[..., k].T,
                cmap=cmap,
                vmin=vmin,
                vmax=vmax,
                origin="lower",
                interpolation="nearest",
            )
            if overlay is not None and overlay[..., k].any():
                ax.contour(
                    overlay[..., k].T, levels=[0.5], colors="#E87722", linewidths=0.8
                )
            ax.set_axis_off()
        axes[row, 0].text(
            0.03,
            0.97,
            label,
            color="white",
            fontsize=8,
            va="top",
            transform=axes[row, 0].transAxes,
        )
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1, wspace=0.02, hspace=0.02)
    return _encode(title=title, fig=fig)


@_optional_figure
def _histograms(*, title, values, bins=100):
    fig = mpl_figure.Figure(figsize=(3.2 * len(values), 2.6))
    axes = fig.subplots(1, len(values), squeeze=False)[0]
    for ax, (name, vals) in zip(axes, values.items()):
        ax.hist(vals, bins=bins, color="#E87722")
        ax.set_title(name, fontsize=9)
        ax.tick_params(labelsize=7)
    fig.tight_layout()
    return _encode(title=title, fig=fig)


@_optional_figure
def _spectrum_figure(*, title, curves):
    fig = mpl_figure.Figure(figsize=(5, 3))
    ax = fig.subplots()
    for label, (freqs, amplitude) in curves.items():
        ax.semilogy(freqs, amplitude, label=label)
    ax.set_xlabel("spatial frequency (cycles/voxel)")
    ax.set_ylabel("mean amplitude")
    ax.legend(fontsize=8)
    fig.tight_layout()
    return _encode(title=title, fig=fig)


@_optional_figure
def _streamlines_figure(*, title, groups, max_streamlines=3000):
    """Axial, coronal and sagittal projections of streamline groups.

    A single group is colored by streamline direction, several groups get
    one color each.
    """
    total = sum(len(s) for _, s in groups)
    rng = np.random.default_rng(0)
    fig = mpl_figure.Figure(figsize=(12, 4), facecolor="black")
    axes = fig.subplots(1, 3)
    for i, (label, streamlines) in enumerate(groups):
        n_keep = max(1, int(max_streamlines * len(streamlines) / total))
        keep = rng.choice(len(streamlines), min(n_keep, len(streamlines)), False)
        subset = [streamlines[k] for k in keep]
        if len(groups) == 1:
            colors = [
                np.abs(s[-1] - s[0]) / (np.linalg.norm(s[-1] - s[0]) or 1)
                for s in subset
            ]
        else:
            colors = f"C{i % 10}"
        for ax, dims in zip(axes, ([0, 1], [0, 2], [1, 2])):
            ax.add_collection(
                mpl_collections.LineCollection(
                    [s[:, dims] for s in subset],
                    colors=colors,
                    linewidths=0.3,
                    alpha=0.7,
                    label=label,
                )
            )
    for ax in axes:
        ax.autoscale()
        ax.set_aspect("equal")
        ax.set_axis_off()
    if 1 < len(groups) <= 15:
        axes[-1].legend(fontsize=6, loc="upper right", framealpha=0.3)
    fig.tight_layout()
    return _encode(title=title, fig=fig)


@_optional_figure
def _profiles_figure(*, title, profiles, n_cols=4):
    n_rows = int(np.ceil(len(profiles) / n_cols))
    fig = mpl_figure.Figure(figsize=(3.2 * n_cols, 2.4 * n_rows))
    axes = fig.subplots(n_rows, n_cols, squeeze=False).ravel()
    for ax, (bundle, metrics) in zip(axes, sorted(profiles.items())):
        for metric, profile in sorted(metrics.items()):
            z = (profile - np.nanmean(profile)) / (np.nanstd(profile) or 1)
            ax.plot(z / (np.nanmax(np.abs(z)) or 1), label=metric, linewidth=1)
        ax.set_title(bundle, fontsize=9)
        ax.set_ylim(-1, 1)
        ax.tick_params(labelsize=7)
    for ax in axes[len(profiles) :]:
        ax.set_axis_off()
    axes[0].legend(fontsize=6)
    fig.tight_layout()
    return _encode(title=title, fig=fig)


# =============================================================================
# Stage QC functions: (*, outputs, inputs, io_config) -> (metrics, figures)
# =============================================================================


def _before_after(*, before_path, after_path, inputs, io_config, ssim_min):
    """Shared before/after comparison on the mean b0 of two DWI files."""
    bvals, b0_threshold = _bvals(inputs=inputs, io_config=io_config)
    before = _mean_b0(path=before_path, bvals=bvals, b0_threshold=b0_threshold)
    after = _mean_b0(path=after_path, bvals=bvals, b0_threshold=b0_threshold)
    mask = foreground_mask(volume=before)
    ssim = ssim_3d(a=before, b=after, mask=mask)
    metrics = [
        _metric(
            name="SSIM before/after (mean b0)",
            value=ssim,
            ok=ssim >= ssim_min,
            rule=f"≥ {ssim_min}",
        )
    ]
    figure = _mosaic(
        title="Mean b0: before, after, difference (before − after)",
        volumes=[before, after, before - after],
        labels=["before", "after", "difference"],
    )
    return before, after, mask, metrics, [figure]


def qc_denoise(*, outputs, inputs, io_config):
    """SNR gain, SSIM and residual of denoising.

    The SNR comparison is skipped (INFO) when either SNR is not finite, as
    happens on synthetic data with a constant background.

    Parameters
    ----------
    outputs : dict
        Stage outputs, ``out_<param> -> path``; reads ``out_denoised``.
    inputs : dict
        Resolved stage parameters; reads ``input_files`` and, when present,
        ``bvalues_files`` and ``b0_threshold``.
    io_config : dict
        Pipeline ``[io]`` section, the fallback for the b-values.

    Returns
    -------
    metrics : list of dict
        Metric records.
    figures : list of dict
        Figure records.
    """
    before, after, mask, metrics, figures = _before_after(
        before_path=inputs["input_files"],
        after_path=outputs["out_denoised"],
        inputs=inputs,
        io_config=io_config,
        ssim_min=QC_THRESHOLDS["denoise_ssim"],
    )
    snr_before = snr(volume=before, mask=mask)
    snr_after = snr(volume=after, mask=mask)
    tolerance = QC_THRESHOLDS["snr_drop_tolerance"]
    ok = None
    if math.isfinite(snr_before) and math.isfinite(snr_after):
        ok = snr_after >= (1 - tolerance) * snr_before
    metrics += [
        _metric(name="SNR before (mean b0)", value=snr_before),
        _metric(
            name="SNR after (mean b0)",
            value=snr_after,
            ok=ok,
            rule=f"≥ SNR before − {100 * tolerance:g} %",
        ),
    ]
    return metrics, figures


def qc_gibbs(*, outputs, inputs, io_config):
    """SSIM and difference image of Gibbs unringing.

    Parameters
    ----------
    outputs : dict
        Stage outputs, ``out_<param> -> path``; reads ``out_unring``.
    inputs : dict
        Resolved stage parameters; reads ``input_files`` and, when present,
        ``bvalues_files`` and ``b0_threshold``.
    io_config : dict
        Pipeline ``[io]`` section, the fallback for the b-values.

    Returns
    -------
    metrics : list of dict
        Metric records.
    figures : list of dict
        Figure records.
    """
    *_, metrics, figures = _before_after(
        before_path=inputs["input_files"],
        after_path=outputs["out_unring"],
        inputs=inputs,
        io_config=io_config,
        ssim_min=QC_THRESHOLDS["gibbs_ssim"],
    )
    return metrics, figures


def qc_bias_correction(*, outputs, inputs, io_config):
    """Intensity uniformity and Fourier spectrum before/after bias correction.

    Parameters
    ----------
    outputs : dict
        Stage outputs, ``out_<param> -> path``; reads ``out_corrected`` and,
        when it exists, ``out_bias_field``.
    inputs : dict
        Resolved stage parameters; reads ``input_files`` and, when present,
        ``bvalues_files`` and ``b0_threshold``.
    io_config : dict
        Pipeline ``[io]`` section, the fallback for the b-values.

    Returns
    -------
    metrics : list of dict
        Metric records.
    figures : list of dict
        Figure records.
    """
    bvals, b0_threshold = _bvals(inputs=inputs, io_config=io_config)
    before = _mean_b0(
        path=inputs["input_files"], bvals=bvals, b0_threshold=b0_threshold
    )
    after = _mean_b0(
        path=outputs["out_corrected"], bvals=bvals, b0_threshold=b0_threshold
    )
    mask = foreground_mask(volume=before)
    cv_before = float(np.std(before[mask]) / np.mean(before[mask]))
    cv_after = float(np.std(after[mask]) / np.mean(after[mask]))
    metrics = [
        _metric(name="Intensity CV before (foreground)", value=cv_before),
        _metric(
            name="Intensity CV after (foreground)",
            value=cv_after,
            ok=cv_after <= cv_before,
            rule="≤ CV before",
        ),
    ]
    volumes, labels = [before, after], ["before", "after"]
    if _is_nifti(path=outputs.get("out_bias_field", "")):
        volumes.append(_load_volume(path=outputs["out_bias_field"])[0])
        labels.append("bias field")
    figures = [
        _mosaic(
            title="Mean b0 and estimated bias field", volumes=volumes, labels=labels
        ),
        _spectrum_figure(
            title="Radial Fourier amplitude spectrum of the mean b0",
            curves={
                "before": radial_amplitude_spectrum(volume=before),
                "after": radial_amplitude_spectrum(volume=after),
            },
        ),
    ]
    return metrics, figures


def _fov_center(*, img):
    """World coordinates of the field-of-view center of a NIfTI image."""
    return (img.affine @ np.append((np.array(img.shape[:3]) - 1) / 2, 1))[:3]


def qc_reslice(*, outputs, inputs, io_config):
    """Grid change and field-of-view preservation of reslicing.

    Parameters
    ----------
    outputs : dict
        Stage outputs, ``out_<param> -> path``; reads ``out_resliced``.
    inputs : dict
        Resolved stage parameters; reads ``input_files`` and, when present,
        ``bvalues_files`` and ``b0_threshold``.
    io_config : dict
        Pipeline ``[io]`` section, the fallback for the b-values.

    Returns
    -------
    metrics : list of dict
        Metric records.
    figures : list of dict
        Figure records.
    """
    bvals, b0_threshold = _bvals(inputs=inputs, io_config=io_config)
    img_before = nib.load(_first_path(value=inputs["input_files"]))
    img_after = nib.load(_first_path(value=outputs["out_resliced"]))
    zooms_before = img_before.header.get_zooms()[:3]
    zooms_after = img_after.header.get_zooms()[:3]
    shift = float(
        np.linalg.norm(_fov_center(img=img_after) - _fov_center(img=img_before))
    )
    metrics = [
        _metric(
            name="Voxel size before → after",
            value=f"{_fmt(values=zooms_before)} → {_fmt(values=zooms_after)}",
            unit="mm",
        ),
        _metric(
            name="Shape before → after",
            value=f"{img_before.shape[:3]} → {img_after.shape[:3]}",
        ),
        _metric(
            name="Field-of-view center shift",
            value=shift,
            unit="mm",
            ok=shift <= max(zooms_before),
            rule="≤ 1 original voxel",
        ),
    ]
    figures = [
        _mosaic(
            title="Mean b0 before and after reslicing",
            volumes=[
                _mean_b0(
                    path=inputs["input_files"],
                    bvals=bvals,
                    b0_threshold=b0_threshold,
                ),
                _mean_b0(
                    path=outputs["out_resliced"],
                    bvals=bvals,
                    b0_threshold=b0_threshold,
                ),
            ],
            labels=["before", "after"],
        )
    ]
    return metrics, figures


def qc_b0_extraction(*, outputs, inputs, io_config, max_rows=8):
    """b0 indices and consistency of the b0 volumes.

    Parameters
    ----------
    outputs : dict
        Stage outputs, ``out_<param> -> path``; unused, the b0 volumes are
        read from the stage input.
    inputs : dict
        Resolved stage parameters; reads ``input_files`` and, when present,
        ``bvalues_files`` and ``b0_threshold``.
    io_config : dict
        Pipeline ``[io]`` section, the fallback for the b-values.
    max_rows : int, optional
        Maximum number of b0 volumes shown in the mosaic.

    Returns
    -------
    metrics : list of dict
        Metric records.
    figures : list of dict
        Figure records.
    """
    bvals, threshold = _bvals(inputs=inputs, io_config=io_config)
    idx = np.flatnonzero(bvals <= threshold)
    metrics = [
        _metric(name="b0 indices", value=", ".join(map(str, idx))),
        _metric(
            name="b0 volumes",
            value=int(idx.size),
            ok=idx.size > 0,
            rule="> 0",
            fail=True,
        ),
    ]
    if not idx.size:
        return metrics, []
    b0s, _ = _load_volume(path=inputs["input_files"], volumes=idx)
    if b0s.ndim == 3:
        b0s = b0s[..., None]
    mask = foreground_mask(volume=b0s.mean(axis=-1))
    means = b0s[mask].mean(axis=0)
    cv = float(np.std(means) / np.mean(means))
    metrics.append(
        _metric(
            name="CV of b0 mean intensity",
            value=cv,
            ok=cv <= QC_THRESHOLDS["b0_intensity_cv"],
            rule=f"≤ {QC_THRESHOLDS['b0_intensity_cv']}",
        )
    )
    shown = idx[:max_rows]
    figures = [
        _mosaic(
            title=f"b0 volumes (first {shown.size} of {idx.size})",
            volumes=[b0s[..., i] for i in range(shown.size)],
            labels=[f"volume {i}" for i in shown],
            n_cuts=4,
        )
    ]
    return metrics, figures


def qc_brain_mask(*, outputs, inputs, io_config):
    """Volume, connectivity and coverage of the brain mask.

    Parameters
    ----------
    outputs : dict
        Stage outputs, ``out_<param> -> path``; reads ``out_mask``.
    inputs : dict
        Resolved stage parameters; reads ``input_files`` and, when present,
        ``bvalues_files`` and ``b0_threshold``.
    io_config : dict
        Pipeline ``[io]`` section, the fallback for the b-values.

    Returns
    -------
    metrics : list of dict
        Metric records.
    figures : list of dict
        Figure records.
    """
    bvals, b0_threshold = _bvals(inputs=inputs, io_config=io_config)
    mask, affine = _load_volume(path=outputs["out_mask"])
    mask = mask > 0
    n_components = int(ndimage.label(mask)[1])
    fraction = float(mask.mean())
    voxel_volume = abs(np.linalg.det(affine[:3, :3]))
    fov_max = QC_THRESHOLDS["mask_fov_fraction"]
    metrics = [
        _metric(
            name="Mask voxels",
            value=int(mask.sum()),
            ok=bool(mask.any()),
            rule="> 0",
            fail=True,
        ),
        _metric(name="Mask volume", value=mask.sum() * voxel_volume / 1000, unit="mL"),
        _metric(
            name="Connected components",
            value=n_components,
            ok=n_components == 1,
            rule="= 1",
        ),
        _metric(
            name="Fraction of field of view",
            value=100 * fraction,
            unit="%",
            ok=fraction < fov_max,
            rule=f"< {100 * fov_max:g} %",
        ),
    ]
    figures = [
        _mosaic(
            title="Brain mask contour over the mean b0",
            volumes=[
                _mean_b0(
                    path=inputs["input_files"],
                    bvals=bvals,
                    b0_threshold=b0_threshold,
                )
            ],
            labels=["mean b0"],
            overlay=mask,
        )
    ]
    return metrics, figures


def _maps_qc(*, outputs, mask, keys):
    """Range metrics, histograms and mosaic of scalar diffusion maps."""
    metrics, values, volumes = [], {}, {}
    for key in keys:
        path = outputs.get(f"out_{key}", "")
        if not _is_nifti(path=path):
            continue
        data, _ = _load_volume(path=path)
        vals = data[mask & np.isfinite(data)]
        if not vals.size:
            continue
        values[key.upper()] = vals
        volumes[key.upper()] = np.nan_to_num(data * mask)
        metrics += _map_metrics(
            name=key.upper(), values=vals, bounds=METRIC_RANGES[key]
        )
    if not values:
        return metrics, []
    figures = [
        _histograms(title="Metric distributions inside the brain mask", values=values),
        _mosaic(
            title="Metric maps",
            volumes=list(volumes.values()),
            labels=list(volumes.keys()),
        ),
    ]
    return metrics, figures


def _signal_nrmse(*, measured, predicted, bvals, b0_threshold=B0_THRESHOLD):
    """Per-voxel NRMSE between measured and predicted signals.

    Both signals are divided by their own mean b0 so that absolute and
    S0-normalised predictions compare alike.

    Parameters
    ----------
    measured, predicted : ndarray, shape (n_voxels, n_volumes)
        Signals of the voxels to evaluate.
    bvals : ndarray
        b-values, used to find the b0 volumes.
    b0_threshold : float, optional
        b-value at or below which a volume counts as a b0.

    Returns
    -------
    ndarray, shape (n_voxels,)
        NRMSE of each voxel.
    """
    b0 = _b0_indices(bvals=bvals, n_volumes=measured.shape[-1], threshold=b0_threshold)
    measured = measured / np.maximum(measured[:, b0].mean(-1, keepdims=True), 1e-6)
    model = predicted / np.maximum(predicted[:, b0].mean(-1, keepdims=True), 1e-6)
    return np.sqrt(np.mean((model - measured) ** 2, axis=-1)) / np.maximum(
        measured.mean(axis=-1), 1e-6
    )


def _scatter(*, values, mask):
    """3D map holding ``values`` in ``mask`` and zero elsewhere."""
    volume = np.zeros(mask.shape, dtype=np.float32)
    volume[mask] = values
    return volume


def qc_dti(*, outputs, inputs, io_config):
    """DTI metric ranges and tensor fit residual.

    The residual is computed on the brain-mask voxels only, so that neither
    the predicted signal nor the tensor parameters are held for the whole
    field of view.

    Parameters
    ----------
    outputs : dict
        Stage outputs, ``out_<param> -> path``; reads ``out_fa``, ``out_md``,
        ``out_ad``, ``out_rd`` and, for the residual, ``out_eval``,
        ``out_evec`` and ``out_s0`` when they exist.
    inputs : dict
        Resolved stage parameters; reads ``mask_files``, ``input_files``
        and, when present, ``bvalues_files``, ``bvectors_files`` and
        ``b0_threshold``.
    io_config : dict
        Pipeline ``[io]`` section, the fallback for the gradients.

    Returns
    -------
    metrics : list of dict
        Metric records.
    figures : list of dict
        Figure records.
    """
    mask = _load_mask(path=inputs["mask_files"])
    metrics, figures = _maps_qc(
        outputs=outputs, mask=mask, keys=["fa", "md", "ad", "rd"]
    )

    evals_path, evecs_path = outputs.get("out_eval", ""), outputs.get("out_evec", "")
    if not (_is_nifti(path=evals_path) and _is_nifti(path=evecs_path)):
        return metrics, figures
    bvals, bvecs, b0_threshold = _gradients(inputs=inputs, io_config=io_config)
    measured = _load_volume(path=inputs["input_files"])[0][mask]
    evals = _load_volume(path=evals_path)[0][mask]
    evecs = _load_volume(path=evecs_path)[0][mask]
    if _is_nifti(path=outputs.get("out_s0", "")):
        s0 = _load_volume(path=outputs["out_s0"])[0][mask]
    else:
        b0 = _b0_indices(
            bvals=bvals, n_volumes=measured.shape[-1], threshold=b0_threshold
        )
        s0 = measured[:, b0].mean(-1)
    params = np.concatenate([evals, evecs.reshape(-1, 9)], axis=-1)
    gtab = gradient_table(bvals, bvecs=bvecs, b0_threshold=b0_threshold)
    predicted = tensor_prediction(params, gtab, s0).astype(measured.dtype)

    nrmse = _signal_nrmse(
        measured=measured, predicted=predicted, bvals=bvals, b0_threshold=b0_threshold
    )
    median = float(np.nanmedian(nrmse))
    limit = QC_THRESHOLDS["dti_residual_nrmse"]
    metrics.append(
        _metric(
            name="Tensor fit residual (median NRMSE)",
            value=median,
            ok=median <= limit,
            rule=f"≤ {limit}; if not, consider fit_method='RESTORE'",
        )
    )
    figures.append(
        _mosaic(
            title="Tensor fit residual (NRMSE)",
            volumes=[_scatter(values=nrmse, mask=mask)],
            labels=["NRMSE"],
        )
    )
    return metrics, figures


def qc_force(*, outputs, inputs, io_config):
    """FORCE metric ranges, match uncertainty and signal matching error.

    Parameters
    ----------
    outputs : dict
        Stage outputs, ``out_<param> -> path``; reads the scalar maps
        (``out_fa``, ``out_md``, ``out_rd``, ``out_mk``, ``out_ak``,
        ``out_rk``), ``out_uncertainty``, ``out_ambiguity`` and
        ``out_predicted_signal`` when they exist.
    inputs : dict
        Resolved stage parameters; reads ``mask_files``, ``input_files`` and,
        when present, ``bvalues_files`` and ``b0_threshold``.
    io_config : dict
        Pipeline ``[io]`` section, the fallback for the b-values.

    Returns
    -------
    metrics : list of dict
        Metric records.
    figures : list of dict
        Figure records.
    """
    mask = _load_mask(path=inputs["mask_files"])
    metrics, figures = _maps_qc(
        outputs=outputs, mask=mask, keys=["fa", "md", "rd", "mk", "ak", "rk"]
    )
    volumes, labels = [], []
    for key in ("uncertainty", "ambiguity"):
        path = outputs.get(f"out_{key}", "")
        if not _is_nifti(path=path):
            continue
        data, _ = _load_volume(path=path)
        metrics.append(
            _metric(name=f"Median {key}", value=float(np.nanmedian(data[mask])))
        )
        volumes.append(np.nan_to_num(data * mask))
        labels.append(key)
    if volumes:
        figures.append(
            _mosaic(
                title="Match uncertainty and ambiguity", volumes=volumes, labels=labels
            )
        )

    if _is_nifti(path=outputs.get("out_predicted_signal", "")):
        bvals, b0_threshold = _bvals(inputs=inputs, io_config=io_config)
        predicted, _ = _load_volume(path=outputs["out_predicted_signal"])
        shape, predicted = predicted.shape, predicted[mask]
        data, _ = _load_volume(path=inputs["input_files"])
        if data.shape == shape:
            nrmse = _signal_nrmse(
                measured=data[mask],
                predicted=predicted,
                bvals=bvals,
                b0_threshold=b0_threshold,
            )
            metrics.append(
                _metric(
                    name="Signal matching error (median NRMSE)",
                    value=float(np.nanmedian(nrmse)),
                )
            )
            figures.append(
                _mosaic(
                    title="Signal matching error (NRMSE)",
                    volumes=[_scatter(values=nrmse, mask=mask)],
                    labels=["NRMSE"],
                )
            )
    return metrics, figures


def qc_tracking(*, outputs, inputs, io_config):
    """Streamline count and length distribution.

    Parameters
    ----------
    outputs : dict
        Stage outputs, ``out_<param> -> path``; reads ``out_tractogram``.
    inputs : dict
        Resolved stage parameters; unused.
    io_config : dict
        Pipeline ``[io]`` section; unused.

    Returns
    -------
    metrics : list of dict
        Metric records.
    figures : list of dict
        Figure records.
    """
    with _open_streamlines(path=outputs["out_tractogram"]) as streamlines:
        metrics = [
            _metric(
                name="Streamlines",
                value=len(streamlines),
                ok=len(streamlines) > 0,
                rule="> 0",
                fail=True,
            )
        ]
        if not len(streamlines):
            return metrics, []
        lengths = _lengths(streamlines=streamlines)
        figure = _streamlines_figure(
            title="Tractogram", groups=[("tractogram", streamlines)]
        )
    short_mm = QC_THRESHOLDS["short_streamline_mm"]
    short_max = QC_THRESHOLDS["short_streamline_fraction"]
    short = float(np.mean(lengths < short_mm))
    metrics += [
        _metric(
            name="Length min / mean / median / max",
            value=_fmt(
                values=[
                    lengths.min(),
                    lengths.mean(),
                    np.median(lengths),
                    lengths.max(),
                ]
            ),
            unit="mm",
        ),
        _metric(
            name=f"Streamlines shorter than {short_mm:g} mm",
            value=100 * short,
            unit="%",
            ok=short <= short_max,
            rule=f"≤ {100 * short_max:g} %",
        ),
    ]
    figures = [
        _histograms(title="Streamline length", values={"length (mm)": lengths}),
        figure,
    ]
    return metrics, figures


def qc_register(*, outputs, inputs, io_config):
    """Affine plausibility and centroid distance of streamline registration.

    Parameters
    ----------
    outputs : dict
        Stage outputs, ``out_<param> -> path``; reads ``out_affine``,
        ``out_moved`` and, when they all exist, ``out_stat_centroids``,
        ``out_moving_centroids`` and ``out_moved_centroids``.
    inputs : dict
        Resolved stage parameters; reads ``static_files`` when present.
    io_config : dict
        Pipeline ``[io]`` section; unused.

    Returns
    -------
    metrics : list of dict
        Metric records.
    figures : list of dict
        Figure records.
    """
    affine = np.loadtxt(outputs["out_affine"])
    scale, _, angles, translate, _ = decompose_matrix(affine)
    lo, hi = QC_THRESHOLDS["slr_scale"]
    metrics = [
        _metric(name="Affine", value=np.array2string(affine, precision=3)),
        _metric(name="Translation", value=_fmt(values=translate), unit="mm"),
        _metric(name="Rotation", value=_fmt(values=np.degrees(angles)), unit="deg"),
        _metric(
            name="Scale",
            value=_fmt(values=scale),
            ok=bool(np.all((np.asarray(scale) >= lo) & (np.asarray(scale) <= hi))),
            rule=f"each in [{lo}, {hi}]",
        ),
    ]
    centroid_paths = [
        outputs.get(key, "")
        for key in ("out_stat_centroids", "out_moving_centroids", "out_moved_centroids")
    ]
    with contextlib.ExitStack() as stack:
        if all(os.path.exists(p) for p in centroid_paths):
            static, moving, moved = (
                stack.enter_context(_open_streamlines(path=p)) for p in centroid_paths
            )
            before = _mean_min_mdf(a=static, b=moving)
            after = _mean_min_mdf(a=static, b=moved)
            metrics += [
                _metric(name="Centroid distance before", value=before, unit="mm"),
                _metric(
                    name="Centroid distance after",
                    value=after,
                    unit="mm",
                    ok=after <= before,
                    rule="≤ distance before",
                ),
            ]
        moved = stack.enter_context(_open_streamlines(path=outputs["out_moved"]))
        metrics.append(_metric(name="Moved streamlines", value=len(moved)))
        groups = [("moved", moved)]
        atlas = _first_path(value=inputs.get("static_files", ""))
        if atlas:
            groups.insert(
                0, ("atlas", stack.enter_context(_open_streamlines(path=atlas)))
            )
        figures = [
            _streamlines_figure(
                title="Atlas (static) and registered (moved) tractograms",
                groups=groups,
            )
        ]
    return metrics, figures


def _bundle_name(*, path, stem, known):
    """Bundle name of a mixed-name output file, matched against atlas names."""
    base = Path(path).name.split(".")[0].removesuffix(stem).rstrip("_")
    matches = [k for k in known if f"_{k}_" in f"_{base}_"]
    return max(matches, key=len) if matches else base


def _bundles_qc(*, out_dir, stem, io_config):
    """Per-bundle streamline counts and a colored projection.

    Bundles are globbed from ``out_dir`` rather than read from the stage
    outputs, which only carry the first of the recognized bundles.

    Parameters
    ----------
    out_dir : str
        Directory holding one tractogram per recognized bundle.
    stem : str
        Trailing part of the bundle file names, before the extension.
    io_config : dict
        Pipeline ``[io]`` section, read for ``bundle_atlas_dir``.

    Returns
    -------
    metrics : list of dict
        One record per bundle.
    figures : list of dict
        Projection of all bundles, colored per bundle.
    """
    files = sorted(Path(out_dir).glob(f"*{stem}.tr[xk]"))
    atlas_dir = io_config.get("bundle_atlas_dir", "")
    known = (
        [p.name.split(".")[0] for p in Path(atlas_dir).glob("*.tr[xk]")]
        if atlas_dir
        else []
    )
    min_count = QC_THRESHOLDS["min_bundle_streamlines"]

    metrics, groups = [], []
    with contextlib.ExitStack() as stack:
        for f in files:
            name = _bundle_name(path=f, stem=stem, known=known)
            streamlines = stack.enter_context(_open_streamlines(path=f))
            mean_length = (
                _lengths(streamlines=streamlines).mean() if len(streamlines) else 0.0
            )
            metrics.append(
                _metric(
                    name=f"{name} (mean length {mean_length:.0f} mm)",
                    value=len(streamlines),
                    unit="streamlines",
                    ok=len(streamlines) >= min_count,
                    rule=f"≥ {min_count}",
                )
            )
            if len(streamlines):
                groups.append((name, streamlines))
        figures = (
            [_streamlines_figure(title="Bundles", groups=groups)] if groups else []
        )
    if not files:
        metrics.append(
            _metric(name="Bundles found", value=0, ok=False, rule="> 0", fail=True)
        )
    return metrics, figures


def qc_segment_bundles(*, outputs, inputs, io_config):
    """Streamline count of each RecoBundles bundle, in atlas space.

    Parameters
    ----------
    outputs : dict
        Stage outputs, ``out_<param> -> path``; reads ``out_dir``.
    inputs : dict
        Resolved stage parameters; unused.
    io_config : dict
        Pipeline ``[io]`` section, read for ``bundle_atlas_dir``.

    Returns
    -------
    metrics : list of dict
        Metric records.
    figures : list of dict
        Figure records.
    """
    return _bundles_qc(
        out_dir=outputs["out_dir"], stem="recognized", io_config=io_config
    )


def qc_label_bundles(*, outputs, inputs, io_config):
    """Streamline count of each bundle, in native space.

    Parameters
    ----------
    outputs : dict
        Stage outputs, ``out_<param> -> path``; reads ``out_dir``.
    inputs : dict
        Resolved stage parameters; unused.
    io_config : dict
        Pipeline ``[io]`` section, read for ``bundle_atlas_dir``.

    Returns
    -------
    metrics : list of dict
        Metric records.
    figures : list of dict
        Figure records.
    """
    return _bundles_qc(
        out_dir=outputs["out_dir"], stem="recognized_orig", io_config=io_config
    )


def qc_buan_profiles(*, outputs, inputs, io_config):
    """Completeness and plot of BUAN along-tract profiles.

    Parameters
    ----------
    outputs : dict
        Stage outputs, ``out_<param> -> path``; reads ``out_dir``.
    inputs : dict
        Resolved stage parameters; unused.
    io_config : dict
        Pipeline ``[io]`` section; unused.

    Returns
    -------
    metrics : list of dict
        Metric records.
    figures : list of dict
        Figure records.
    """
    files = sorted(Path(outputs["out_dir"]).glob("*_profile.npy"))
    profiles = {}
    for f in files:
        bundle, _, metric = f.stem.removesuffix("_profile").rpartition("_")
        profiles.setdefault(bundle, {})[metric] = np.load(f).astype(float)
    values = [p for metrics in profiles.values() for p in metrics.values()]
    nan_fraction = float(np.mean(np.isnan(np.concatenate(values)))) if values else 0.0
    metrics = [
        _metric(
            name="Profiles", value=len(files), ok=bool(files), rule="> 0", fail=True
        ),
        _metric(
            name="Profile points that are NaN",
            value=100 * nan_fraction,
            unit="%",
            ok=nan_fraction == 0,
            rule="= 0 %",
        ),
    ]
    figures = []
    if profiles:
        figures.append(
            _profiles_figure(
                title="Along-tract profiles (normalized z-score)", profiles=profiles
            )
        )
    return metrics, figures


def qc_generic(*, outputs, inputs, io_config, max_outputs=6):
    """Non-finite voxel count and mosaic of each NIfTI output.

    Parameters
    ----------
    outputs : dict
        Stage outputs, ``out_<param> -> path``; every existing NIfTI file
        but ``out_dir`` is checked.
    inputs : dict
        Resolved stage parameters; unused.
    io_config : dict
        Pipeline ``[io]`` section; unused.
    max_outputs : int, optional
        Maximum number of NIfTI outputs checked.

    Returns
    -------
    metrics : list of dict
        Metric records.
    figures : list of dict
        Figure records.
    """
    metrics, figures = [], []
    niftis = [
        (k, p) for k, p in outputs.items() if k != "out_dir" and _is_nifti(path=p)
    ]
    for key, path in niftis[:max_outputs]:
        data, _ = _load_volume(path=path)
        bad = int(np.count_nonzero(~np.isfinite(data)))
        metrics.append(
            _metric(
                name=f"{key}: non-finite voxels", value=bad, ok=bad == 0, rule="= 0"
            )
        )
        while data.ndim > 3:
            data = data[..., data.shape[-1] // 2]
        figures.append(
            _mosaic(
                title=f"{key}: {os.path.basename(path)}",
                volumes=[np.nan_to_num(data)],
                labels=[key],
            )
        )
    return metrics, figures


QC_FUNCS = {
    "dipy_denoise": qc_denoise,
    "dipy_denoise_patch2self": qc_denoise,
    "dipy_denoise_mppca": qc_denoise,
    "dipy_denoise_lpca": qc_denoise,
    "dipy_denoise_nlmeans": qc_denoise,
    "dipy_gibbs_ringing": qc_gibbs,
    "dipy_correct_biasfield": qc_bias_correction,
    "dipy_reslice": qc_reslice,
    "dipy_extract_b0": qc_b0_extraction,
    "dipy_brain_mask": qc_brain_mask,
    "dipy_median_otsu": qc_brain_mask,
    "dipy_fit_dti": qc_dti,
    "dipy_fit_force": qc_force,
    "dipy_track": qc_tracking,
    "dipy_slr": qc_register,
    "dipy_recobundles": qc_segment_bundles,
    "dipy_labelsbundles": qc_label_bundles,
    "dipy_buan_profiles": qc_buan_profiles,
}


def run_stage_qc(*, cli, outputs, inputs, io_config):
    """Run the QC matching a stage CLI, never raising.

    Parameters
    ----------
    cli : str
        Stage CLI name, used to pick the QC function. Unknown CLIs get
        ``qc_generic``.
    outputs : dict
        Stage outputs, ``out_<param> -> path``.
    inputs : dict
        Resolved stage parameters, ``param -> value``.
    io_config : dict
        Pipeline ``[io]`` section.

    Returns
    -------
    dict
        QC result from ``qc_result``. Status is ERROR if the QC itself failed.
    """
    func = QC_FUNCS.get(cli, qc_generic)
    try:
        metrics, figures = func(outputs=outputs, inputs=inputs, io_config=io_config)
    except Exception:
        logger.warning(f"QC for '{cli}' failed:\n{traceback.format_exc()}")
        return qc_result(error=traceback.format_exc(), status="ERROR")
    return qc_result(metrics=metrics, figures=figures)
