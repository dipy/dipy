import numpy as np
import numpy.testing as npt

from dipy.core.geometry import vec2vec_rotmat
from dipy.core.sphere import Sphere
from dipy.data import get_gtab_taiwan_dsi
from dipy.direction import peak_directions
from dipy.io.peaks import load_pam, save_pam
from dipy.reconst.multi_voxel import MultiVoxelFit
from dipy.reconst.odffp import (
    OdffpDictionary,
    OdffpFit,
    OdffpModel,
    odffp_peaks,
    resample_odf,
)
from dipy.reconst.odffp_matching import select_best_match
from dipy.reconst.shm import sh_to_sf
from dipy.sims.voxel import multi_tensor


def _make_dictionary(gtab, dict_size=3000, max_peaks_num=2, seed=0):
    """Create a deterministic ODF-FP dictionary for tests."""
    odf_dict = OdffpDictionary(gtab)
    odf_dict.generate(
        dict_size=dict_size,
        max_peaks_num=max_peaks_num,
        max_chunk_size=dict_size,
        rng=np.random.default_rng(seed),
    )
    return odf_dict


def _reference_match(similarity, n_fibers, penalty):
    """Naive penalized arg-max, equivalent to select_best_match."""
    coef = np.maximum(0, n_fibers - 1)
    valid = n_fibers >= 0
    if penalty > 0:
        positive = (similarity > 0) & valid[np.newaxis, :]
        score = np.full_like(similarity, -np.inf, dtype=float)
        with np.errstate(divide="ignore", invalid="ignore"):
            penalized = np.log(similarity) - 2 * penalty * coef[np.newaxis, :]
        score[positive] = penalized[positive]
        no_positive = ~np.any(positive, axis=1)
        score[no_positive] = similarity[no_positive]
    else:
        score = similarity.astype(float)
    score[:, ~valid] = -np.inf
    return np.argmax(score, axis=1)


def test_dictionary_generation_is_deterministic():
    gtab = get_gtab_taiwan_dsi()
    first = _make_dictionary(gtab, dict_size=100, seed=42)
    second = _make_dictionary(gtab, dict_size=100, seed=42)

    for name in ("odf", "peak_dirs", "micro", "ratio", "peaks_per_voxel"):
        npt.assert_array_equal(getattr(first, name), getattr(second, name))


def test_select_best_match_matches_reference():
    # The Cython matcher must reproduce the naive penalized arg-max exactly,
    # including void fingerprints flagged with a negative group.
    rng = np.random.default_rng(1)
    n_fibers = rng.integers(-1, 4, 300)
    similarity = np.ascontiguousarray(rng.random((40, 300)))
    group = np.where(n_fibers < 0, -1, np.maximum(0, n_fibers - 1))
    group = np.ascontiguousarray(group, dtype=np.intp)
    n_groups = int(group.max()) + 1

    for penalty in (0.0, 1e-3, 1e-2):
        got = select_best_match(similarity, group, penalty, n_groups)
        expected = _reference_match(similarity, n_fibers, penalty)
        npt.assert_array_equal(got, expected)


def test_select_best_match_parallel_is_deterministic():
    rng = np.random.default_rng(2)
    similarity = np.ascontiguousarray(rng.random((100, 500)))
    group = np.ascontiguousarray(rng.integers(0, 3, 500), dtype=np.intp)
    serial = select_best_match(similarity, group, 1e-3, 3, num_threads=1)
    parallel = select_best_match(similarity, group, 1e-3, 3, num_threads=4)
    npt.assert_array_equal(serial, parallel)


def test_select_best_match_handles_nonpositive_similarity():
    similarity = np.ascontiguousarray(
        [[-0.4, -0.2, 0.5], [-0.9, -0.3, -0.5]], dtype=float
    )
    group = np.ascontiguousarray([0, 1, 2], dtype=np.intp)

    got = select_best_match(similarity, group, 1e-2, 3)

    # Use the sole positive match when available; otherwise use the highest raw
    # similarity rather than returning an invalid dictionary index.
    npt.assert_array_equal(got, [2, 1])


def test_odffp_recovers_single_fiber():
    gtab = get_gtab_taiwan_dsi()
    odf_dict = _make_dictionary(gtab)
    data = multi_tensor(
        gtab,
        np.array([[0.0015, 0.0003, 0.0003]]),
        angles=[(90, 0)],
        fractions=[100],
        snr=None,
    )[0]

    fit = OdffpModel(gtab, odf_dict, penalty=1e-4).fit(data)
    npt.assert_(isinstance(fit, OdffpFit))

    # The fiber points along +x. The *main* (first) recovered peak must be that
    # orientation -- guards against the alignment reference (vertex 0) and the
    # pole disagreeing, which silently rotates every output.
    peaks = fit.peak_dirs
    valid = ~np.isnan(peaks).any(axis=1)
    npt.assert_(np.abs(peaks[0] @ [1, 0, 0]) > 0.9)
    npt.assert_(np.abs(peaks[valid] @ [1, 0, 0]).max() > 0.9)
    npt.assert_equal(fit.odf().shape[-1], len(odf_dict.sphere.vertices) // 2)


def test_odffp_isotropic_matches_free_water():
    gtab = get_gtab_taiwan_dsi()
    odf_dict = _make_dictionary(gtab)
    data = multi_tensor(
        gtab,
        np.array([[0.003, 0.003, 0.003]]),
        angles=[(0, 0)],
        fractions=[100],
        snr=None,
    )[0]

    fit = OdffpModel(gtab, odf_dict, penalty=1e-3).fit(data)
    npt.assert_equal(odf_dict.peaks_per_voxel[fit.dict_idx], 0)


def test_odffp_fit_is_faithful_to_naive_resampling():
    # The cached/multi-voxel fit must reproduce a naive per-voxel resampling
    # (via resample_odf) and exact matching -- the optimization-correctness
    # guarantee.
    gtab = get_gtab_taiwan_dsi()
    odf_dict = _make_dictionary(gtab, dict_size=1500)
    sphere = odf_dict.sphere
    mevals = np.array([[0.0015, 0.0003, 0.0003]] * 2)

    data = np.stack(
        [
            multi_tensor(gtab, mevals[:1], angles=[(90, 0)], fractions=[100], snr=None)[
                0
            ],
            multi_tensor(
                gtab,
                mevals,
                angles=[(20, 0), (90, 0)],
                fractions=[50, 50],
                snr=None,
            )[0],
            multi_tensor(
                gtab,
                mevals,
                angles=[(40, 0), (90, 0)],
                fractions=[60, 40],
                snr=None,
            )[0],
        ]
    )

    # Exact float64 matching, so the blocked/streamed match must reproduce the
    # naive full-matrix arg-max bit-for-bit.
    model = OdffpModel(gtab, odf_dict, penalty=1e-4, matching_precision="float64")
    multi_fit = model.fit(data)
    npt.assert_(isinstance(multi_fit, MultiVoxelFit))

    dict_trace, _ = model._normalize_odf(odf_dict.odf)
    n_fibers = odf_dict.peaks_per_voxel
    for v in range(len(data)):
        odf = model._odf_recon_model.fit(data[v]).odf(sphere)
        _, _, indices = peak_directions(odf, sphere)
        rotation = (
            vec2vec_rotmat(sphere.vertices[indices[0]], model._pole)
            if len(indices)
            else np.eye(3)
        )
        rotated = Sphere(xyz=np.dot(sphere.vertices, rotation))
        aligned = resample_odf(odf, sphere, rotated)
        trace, _ = model._normalize_odf(aligned)
        reference_idx = _reference_match(
            (trace @ dict_trace)[np.newaxis], n_fibers, 1e-4
        )[0]
        npt.assert_equal(multi_fit.fit_array[v].dict_idx, reference_idx)


def test_dictionary_save_load(tmp_path):
    gtab = get_gtab_taiwan_dsi()
    odf_dict = _make_dictionary(gtab, dict_size=400)
    fname = str(tmp_path / "odf_dict.npz")
    odf_dict.save(fname)

    loaded = OdffpDictionary(gtab, dict_file=fname)
    npt.assert_array_equal(loaded.odf, odf_dict.odf)
    npt.assert_array_equal(loaded.peaks_per_voxel, odf_dict.peaks_per_voxel)
    npt.assert_equal(loaded.max_peaks_num, odf_dict.max_peaks_num)


def test_odffp_peaks_stores_odf_and_roundtrips(tmp_path):
    # odffp_peaks (FORCE-style) must build a PeaksAndMetrics whose stored SH
    # coefficients reconstruct the fit ODFs, for both a volume MultiVoxelFit
    # and a single OdffpFit, and survive a PAM5 save/load.
    gtab = get_gtab_taiwan_dsi()
    odf_dict = _make_dictionary(gtab, dict_size=1500)
    model = OdffpModel(gtab, odf_dict, penalty=1e-4)
    half = len(odf_dict.sphere.vertices) // 2

    mevals = np.array([[0.0015, 0.0003, 0.0003]] * 2)
    data = np.stack(
        [
            multi_tensor(gtab, mevals[:1], angles=[(90, 0)], fractions=[100], snr=None)[
                0
            ],
            multi_tensor(
                gtab,
                mevals,
                angles=[(20, 0), (90, 0)],
                fractions=[50, 50],
                snr=None,
            )[0],
        ]
    ).reshape(2, 1, 1, -1)
    mask = np.ones((2, 1, 1), dtype=bool)

    mfit = model.fit(data, mask=mask)
    npt.assert_(isinstance(mfit, MultiVoxelFit))
    peaks = odffp_peaks(mfit)

    npt.assert_equal(peaks.shm_coeff.shape[:3], (2, 1, 1))
    npt.assert_equal(peaks.sphere.vertices.shape[0], half)
    npt.assert_equal(peaks.peak_dirs.shape[:3], (2, 1, 1))

    odf_map = np.asarray(mfit.odf())
    recon = sh_to_sf(peaks.shm_coeff, peaks.sphere, sh_order_max=8, legacy=False)
    npt.assert_allclose(recon, odf_map, atol=1e-5)

    fname = str(tmp_path / "odffp.pam5")
    save_pam(fname, peaks, affine=np.eye(4))
    loaded = load_pam(fname)
    npt.assert_array_almost_equal(loaded.affine, np.eye(4))
    recon2 = sh_to_sf(loaded.shm_coeff, loaded.sphere, sh_order_max=8, legacy=False)
    npt.assert_allclose(recon2, odf_map, atol=1e-5)

    # Single voxel: same SH width, valid peak directions.
    single = odffp_peaks(model.fit(data[0, 0, 0]))
    npt.assert_(isinstance(single, type(peaks)))
    npt.assert_equal(single.shm_coeff.shape[-1], peaks.shm_coeff.shape[-1])


def test_odffp_sh_order_max_is_configurable():
    # The match runs in the SH subspace of the chosen order; both orders must
    # reproduce the naive full-trace arg-max (exact in float64).
    gtab = get_gtab_taiwan_dsi()
    odf_dict = _make_dictionary(gtab, dict_size=1200)
    sphere = odf_dict.sphere
    data = multi_tensor(
        gtab,
        np.array([[0.0015, 0.0003, 0.0003]] * 2),
        angles=[(30, 0), (90, 0)],
        fractions=[50, 50],
        snr=None,
    )[0]
    for order, n_sh in [(4, 15), (8, 45)]:
        model = OdffpModel(
            gtab,
            odf_dict,
            penalty=1e-4,
            sh_order_max=order,
            matching_precision="float64",
        )
        npt.assert_equal(model._dict_trace.shape[1], n_sh)
        fit = model.fit(data)

        odf = model._odf_recon_model.fit(data).odf(sphere)
        _, _, indices = peak_directions(odf, sphere)
        rotation = vec2vec_rotmat(sphere.vertices[indices[0]], model._pole)
        rotated = Sphere(xyz=np.dot(sphere.vertices, rotation))
        aligned = resample_odf(odf, sphere, rotated, sh_order_max=order)
        trace, _ = model._normalize_odf(aligned)
        dict_trace, _ = model._normalize_odf(odf_dict.odf)
        ref = _reference_match(
            (trace @ dict_trace)[np.newaxis], odf_dict.peaks_per_voxel, 1e-4
        )[0]
        npt.assert_equal(fit.dict_idx, ref)


def test_antipodal_peak_rotation_maps_to_pole():
    pole = np.array([0.0, 0.0, 1.0])
    direction = -pole
    rotation = vec2vec_rotmat(direction, pole)
    npt.assert_allclose(rotation @ direction, pole, atol=1e-10)
