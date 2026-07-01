import numpy as np
import numpy.testing as npt

from dipy.core.gradients import gradient_table
from dipy.core.sphere import Sphere
from dipy.data import get_sphere
from dipy.direction import peak_directions
from dipy.io.peaks import load_pam, save_pam
from dipy.reconst.multi_voxel import MultiVoxelFit
from dipy.reconst.odffp import (
    OdffpDictionary,
    OdffpFit,
    OdffpModel,
    _rotation_to_pole,
    odffp_peaks,
)
from dipy.reconst.odffp_matching import select_best_match
from dipy.reconst.shm import sh_to_sf
from dipy.sims.voxel import multi_tensor


def _make_gtab():
    """Two-shell gradient table built from a symmetric sphere."""
    sphere = get_sphere(name="repulsion724")
    directions = sphere.vertices[: len(sphere.vertices) // 2]
    bvecs = np.vstack([[0, 0, 0], directions, directions])
    bvals = np.hstack(
        [0, np.full(len(directions), 1000.0), np.full(len(directions), 2500.0)]
    )
    return gradient_table(bvals, bvecs=bvecs)


def _make_dictionary(gtab, dict_size=3000, max_peaks_num=2, seed=0):
    np.random.seed(seed)
    odf_dict = OdffpDictionary(gtab)
    odf_dict.generate(
        dict_size=dict_size, max_peaks_num=max_peaks_num, max_chunk_size=dict_size
    )
    return odf_dict


def _single_voxel(gtab, mevals, angles, fractions):
    signal, _ = multi_tensor(gtab, mevals, angles=angles, fractions=fractions, snr=None)
    return signal


def _reference_match(similarity, n_fibers, penalty):
    """Naive penalized arg-max, equivalent to select_best_match."""
    coef = np.maximum(0, n_fibers - 1)
    if penalty > 0:
        with np.errstate(divide="ignore"):
            score = np.log(similarity) - 2 * penalty * coef[np.newaxis, :]
    else:
        score = similarity.astype(float)
    score[:, n_fibers < 0] = -np.inf
    return np.argmax(score, axis=1)


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


def test_odffp_recovers_single_fiber():
    gtab = _make_gtab()
    odf_dict = _make_dictionary(gtab)
    data = _single_voxel(gtab, np.array([[0.0015, 0.0003, 0.0003]]), [(90, 0)], [100])

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
    gtab = _make_gtab()
    odf_dict = _make_dictionary(gtab)
    data = _single_voxel(gtab, np.array([[0.003, 0.003, 0.003]]), [(0, 0)], [100])

    fit = OdffpModel(gtab, odf_dict, penalty=1e-3).fit(data)
    npt.assert_equal(odf_dict.peaks_per_voxel[fit.dict_idx], 0)


def test_odffp_fit_is_faithful_to_naive_resampling():
    # The cached/multi-voxel fit must reproduce a naive per-voxel resampling
    # (via resample_odf) and exact matching -- the optimization-correctness
    # guarantee.
    gtab = _make_gtab()
    odf_dict = _make_dictionary(gtab, dict_size=1500)
    sphere = odf_dict.sphere
    mevals = np.array([[0.0015, 0.0003, 0.0003]] * 2)

    data = np.stack(
        [
            _single_voxel(gtab, mevals[:1], [(90, 0)], [100]),
            _single_voxel(gtab, mevals, [(20, 0), (90, 0)], [50, 50]),
            _single_voxel(gtab, mevals, [(40, 0), (90, 0)], [60, 40]),
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
            _rotation_to_pole(sphere.vertices[indices[0]], pole=model._pole)
            if len(indices)
            else np.eye(3)
        )
        rotated = Sphere(xyz=np.dot(sphere.vertices, rotation))
        aligned = OdffpModel.resample_odf(odf, sphere, rotated)
        trace, _ = model._normalize_odf(aligned)
        reference_idx = _reference_match(
            (trace @ dict_trace)[np.newaxis], n_fibers, 1e-4
        )[0]
        npt.assert_equal(multi_fit.fit_array[v].dict_idx, reference_idx)


def test_dictionary_save_load(tmp_path):
    gtab = _make_gtab()
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
    gtab = _make_gtab()
    odf_dict = _make_dictionary(gtab, dict_size=1500)
    model = OdffpModel(gtab, odf_dict, penalty=1e-4)
    half = len(odf_dict.sphere.vertices) // 2

    mevals = np.array([[0.0015, 0.0003, 0.0003]] * 2)
    data = np.stack(
        [
            _single_voxel(gtab, mevals[:1], [(90, 0)], [100]),
            _single_voxel(gtab, mevals, [(20, 0), (90, 0)], [50, 50]),
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
    gtab = _make_gtab()
    odf_dict = _make_dictionary(gtab, dict_size=1200)
    sphere = odf_dict.sphere
    data = _single_voxel(
        gtab, np.array([[0.0015, 0.0003, 0.0003]] * 2), [(30, 0), (90, 0)], [50, 50]
    )
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
        rotation = _rotation_to_pole(sphere.vertices[indices[0]], pole=model._pole)
        rotated = Sphere(xyz=np.dot(sphere.vertices, rotation))
        aligned = OdffpModel.resample_odf(odf, sphere, rotated, sh_order_max=order)
        trace, _ = model._normalize_odf(aligned)
        dict_trace, _ = model._normalize_odf(odf_dict.odf)
        ref = _reference_match(
            (trace @ dict_trace)[np.newaxis], odf_dict.peaks_per_voxel, 1e-4
        )[0]
        npt.assert_equal(fit.dict_idx, ref)


def test_rotation_to_pole_is_orthogonal():
    for direction in ([1, 0, 0], [0, 1, 0], [1, 1, 1], [0.3, -0.7, 0.5]):
        direction = np.asarray(direction, float)
        direction /= np.linalg.norm(direction)
        rotation = _rotation_to_pole(direction)
        npt.assert_allclose(rotation @ rotation.T, np.eye(3), atol=1e-10)
        npt.assert_allclose(np.linalg.det(rotation), 1.0, atol=1e-10)
