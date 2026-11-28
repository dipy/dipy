"""Testing cross-validation analysis."""

import warnings

import numpy as np
import numpy.testing as npt
import pytest

import dipy.core.gradients as gt
import dipy.data as dpd
from dipy.io.image import load_nifti_data
import dipy.reconst.base as base
import dipy.reconst.cross_validation as xval
import dipy.reconst.csdeconv as csd
import dipy.reconst.dti as dti
from dipy.reconst.shm import descoteaux07_legacy_msg
import dipy.sims.voxel as sims
from dipy.testing.decorators import set_random_number_generator

# We'll set these globally:
fdata, fbval, fbvec = dpd.get_fnames(name="small_64D")


@set_random_number_generator()
def test_coeff_of_determination(rng=None):
    model = rng.standard_normal((10, 10, 10, 150))
    data = np.copy(model)
    # If the model predicts the data perfectly, the COD is all 100s:
    cod = xval.coeff_of_determination(data, model)
    npt.assert_array_equal(100, cod)


def test_dti_xval():
    data = load_nifti_data(fdata)
    gtab = gt.gradient_table(fbval, bvecs=fbvec)
    dm = dti.TensorModel(gtab, fit_method="LS")
    # The data has 102 directions, so will not divide neatly into 10 bits
    npt.assert_raises(ValueError, xval.kfold_xval, dm, data, 10)

    # In simulation with no noise, COD should be perfect:
    psphere = dpd.get_sphere(name="symmetric362")
    bvecs = np.concatenate(([[0, 0, 0]], psphere.vertices))
    bvals = np.zeros(len(bvecs)) + 1000
    bvals[0] = 0
    gtab = gt.gradient_table(bvals, bvecs=bvecs)
    mevals = np.array(([0.0015, 0.0003, 0.0001], [0.0015, 0.0003, 0.0003]))
    mevecs = [
        np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]]),
        np.array([[0, 0, 1], [0, 1, 0], [1, 0, 0]]),
    ]
    S = sims.single_tensor(gtab, S0=100, evals=mevals[0], evecs=mevecs[0], snr=None)

    dm = dti.TensorModel(gtab, fit_method="LS")
    kf_xval = xval.kfold_xval(dm, S, 2)
    cod = xval.coeff_of_determination(S, kf_xval)
    npt.assert_array_almost_equal(cod, np.ones(kf_xval.shape[:-1]) * 100)

    # Test with 2D data for use of a mask
    S = np.array([[S, S], [S, S]])
    mask = np.ones(S.shape[:-1], dtype=bool)
    mask[1, 1] = 0
    kf_xval = xval.kfold_xval(dm, S, 2, mask=mask)
    cod2d = xval.coeff_of_determination(S, kf_xval)
    npt.assert_array_almost_equal(np.round(cod2d[0, 0]), cod)


@set_random_number_generator(12345)
def test_csd_xval(rng=None):
    # First, let's see that it works with some data:
    data = load_nifti_data(fdata)[1:3, 1:3, 1:3]  # Make it *small*
    gtab = gt.gradient_table(fbval, bvecs=fbvec)
    S0 = np.mean(data[..., gtab.b0s_mask])
    response = ([0.0015, 0.0003, 0.0001], S0)

    # In simulation, it should work rather well (high COD):
    psphere = dpd.get_sphere(name="symmetric362")
    bvecs = np.concatenate(([[0, 0, 0]], psphere.vertices))
    bvals = np.zeros(len(bvecs)) + 1000
    bvals[0] = 0
    gtab = gt.gradient_table(bvals, bvecs=bvecs)
    mevals = np.array(([0.0015, 0.0003, 0.0001], [0.0015, 0.0003, 0.0003]))
    mevecs = [
        np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]]),
        np.array([[0, 0, 1], [0, 1, 0], [1, 0, 0]]),
    ]
    S0 = 100
    S = sims.single_tensor(gtab, S0=S0, evals=mevals[0], evecs=mevecs[0], snr=None)
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=descoteaux07_legacy_msg,
            category=PendingDeprecationWarning,
        )
        sm = csd.ConstrainedSphericalDeconvModel(gtab, response)
    response = ([0.0015, 0.0003, 0.0001], S0)
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=descoteaux07_legacy_msg,
            category=PendingDeprecationWarning,
        )
        kf_xval = xval.kfold_xval(sm, S, 2, response, sh_order_max=2, rng=rng)
    # Because of the regularization, COD is not going to be perfect here:
    cod = xval.coeff_of_determination(S, kf_xval)
    # We'll just test for regressions:
    csd_cod = 97  # pre-computed by hand for this random seed

    # We're going to be really lenient here:
    npt.assert_array_almost_equal(np.round(cod), csd_cod)
    # Test for sD data with more than one voxel for use of a mask:
    S = np.array([[S, S], [S, S]])
    mask = np.ones(S.shape[:-1], dtype=bool)
    mask[1, 1] = 0
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=descoteaux07_legacy_msg,
            category=PendingDeprecationWarning,
        )
        kf_xval = xval.kfold_xval(
            sm, S, 2, response, sh_order_max=2, mask=mask, rng=rng
        )

    cod = xval.coeff_of_determination(S, kf_xval)
    npt.assert_array_almost_equal(np.round(cod[0]), csd_cod)


def test_no_predict():
    # Test that if you try to do this with a model that doesn't have a `predict`
    # method, you get something reasonable.
    class NoPredictModel(base.ReconstModel):
        def __init__(self, gtab):
            base.ReconstModel.__init__(self, gtab)

        def fit(self, data, mask=None):
            return NoPredictFit(self, data, mask=mask)

    class NoPredictFit(base.ReconstFit):
        def __init__(self, model, data, mask=None):
            base.ReconstFit.__init__(self, model, data)

    gtab = gt.gradient_table(fbval, bvecs=fbvec)
    my_model = NoPredictModel(gtab)
    data = load_nifti_data(fdata)[1:3, 1:3, 1:3]  # Whatever

    npt.assert_raises(ValueError, xval.kfold_xval, my_model, data, 2)


def _density_test_gtab(bvals, directions):
    return gt.gradient_table(np.r_[0, bvals], bvecs=np.vstack(([0, 0, 0], directions)))


def test_spherical_density_shell_balance_and_seed():
    rng = np.random.default_rng(7)
    counts = [9, 7, 2]
    bvals = np.repeat([1000, 2000, 3000], counts)
    directions = rng.normal(size=(len(bvals), 3))
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)
    order = rng.permutation(len(bvals))
    bvals, directions = bvals[order], directions[order]

    def split(directions):
        return xval._spherical_density_folds(
            _density_test_gtab(bvals, directions),
            5,
            shell_tol=20,
            rng=np.random.default_rng(17),
        )

    folds = split(directions)
    npt.assert_array_equal(np.sort(np.concatenate(folds)), np.arange(len(bvals)))
    assert np.ptp([len(fold) for fold in folds]) == 1
    for shell in np.unique(bvals):
        assert np.ptp([np.count_nonzero(bvals[fold] == shell) for fold in folds]) <= 1
    signs = rng.choice([-1, 1], size=(len(bvals), 1))
    for fold, flipped in zip(folds, split(directions * signs)):
        npt.assert_array_equal(fold, flipped)


def test_spherical_density_spreads_clustered_measurements():
    counts = [4, 8, 12]
    labels = np.repeat(np.arange(3), counts)
    directions = np.eye(3)[labels]
    directions[::2] *= -1
    # Tilt one repeated axis slightly and keep that measurement eligible.
    directions[0] = [-np.sqrt(1 - 1e-6), 0.001, 0]
    gtab = _density_test_gtab(np.full(len(labels), 1000), directions)
    folds = xval._spherical_density_folds(
        gtab, 4, shell_tol=20, rng=np.random.default_rng(12)
    )

    def worst_gap(holdouts):
        return max(
            np.max(
                np.arccos(np.clip(np.abs(np.eye(3) @ directions[fold].T).max(1), 0, 1))
            )
            for fold in holdouts
        )

    for fold in folds:
        npt.assert_array_equal(np.bincount(labels[fold], minlength=3), [1, 2, 3])
    random_gaps = [
        worst_gap(np.split(np.random.default_rng(seed).permutation(len(labels)), 4))
        for seed in range(20)
    ]
    assert worst_gap(folds) < np.deg2rad(1)
    assert worst_gap(folds) < np.median(random_gaps)


def test_spherical_density_preserves_acquisition_metadata_and_order():
    bvals = np.array([1000, 0, 1001, 2000, 80, 2001, 1002])
    directions = np.eye(3)[np.arange(len(bvals)) % 3]
    kwargs = {"b0_threshold": 100, "big_delta": 0.05, "small_delta": 0.02}
    gtab = gt.gradient_table(bvals, bvecs=directions, btens="LTE", **kwargs)
    signal = np.arange(len(bvals), dtype=float)
    signal[gtab.b0s_mask] = [1, 3]
    voxel_mask = np.array([True, False])
    fitted, held = [], []

    def check_table(table):
        for key, value in kwargs.items():
            assert getattr(table, key) == value
        original = np.array([np.flatnonzero(bvals == b)[0] for b in table.bvals])
        npt.assert_array_equal(table.btens, gtab.btens[original])
        npt.assert_array_equal(table.bvals[table.b0s_mask], [0, 80])
        return original

    class RecordingModel:
        def __init__(self, gtab):
            self.gtab = gtab

        def fit(self, data, *, mask):
            original = check_table(self.gtab)
            npt.assert_array_equal(mask, voxel_mask)
            npt.assert_array_equal(data, np.stack([signal[original]] * 2))
            fitted.append(original[2:])
            return self

        def predict(self, *, gtab, S0):
            assert S0 == 1
            held.append(check_table(gtab)[2:])
            return np.broadcast_to(gtab.bvals, (2, len(gtab.bvals)))

    args = RecordingModel(gtab), np.stack([signal] * 2), 3
    with pytest.raises(ValueError, match="np.mod"):
        xval.kfold_xval(*args)

    prediction = xval.kfold_xval(
        *args,
        split_strategy="spherical_density",
        mask=voxel_mask,
        rng=np.random.default_rng(5),
    )
    expected = 2 * bvals
    expected[gtab.b0s_mask] = 2
    npt.assert_array_equal(prediction, np.stack([expected] * 2))
    dwi_indices = np.flatnonzero(~gtab.b0s_mask)
    for train, test in zip(fitted, held):
        npt.assert_array_equal(train, np.setdiff1d(dwi_indices, test))
    npt.assert_array_equal(np.sort(np.concatenate(held)), dwi_indices)


@pytest.mark.parametrize(
    ("folds", "strategy", "tol", "message"),
    [
        (2, "invalid", 20, "split_strategy"),
        (1, "spherical_density", 20, "folds"),
        (5, "spherical_density", 20, "folds"),
        (2.0, "spherical_density", 20, "folds"),
        (2, "spherical_density", -1, "shell_tol"),
    ],
)
def test_spherical_density_invalid_options(folds, strategy, tol, message):
    gtab = _density_test_gtab(np.full(3, 1000), np.eye(3))
    model = dti.TensorModel(gtab)
    with pytest.raises(ValueError, match=message):
        xval.kfold_xval(
            model, np.ones(4), folds, split_strategy=strategy, shell_tol=tol
        )


def test_spherical_density_requires_b0_and_linear_encoding():
    no_b0 = gt.gradient_table(np.full(3, 1000), bvecs=np.eye(3))
    with pytest.raises(ValueError, match="at least one b0"):
        xval.kfold_xval(
            base.ReconstModel(no_b0), np.ones(3), 2, split_strategy="spherical_density"
        )

    spherical_encoding = gt.gradient_table(
        [0, 1000, 1000], bvecs=[[0, 0, 0], [1, 0, 0], [0, 1, 0]], btens="STE"
    )
    with pytest.raises(ValueError, match="linear encoding"):
        xval.kfold_xval(
            base.ReconstModel(spherical_encoding),
            np.ones(3),
            2,
            split_strategy="spherical_density",
        )


def test_spherical_density_rejects_ambiguous_shell_tie():
    gtab = _density_test_gtab(
        [1000, 1016, 1034, 1040, 1052], np.eye(3)[[0, 1, 2, 0, 1]]
    )
    with (
        pytest.warns(UserWarning, match="overlap"),
        pytest.raises(ValueError, match="uniquely assign"),
    ):
        xval._spherical_density_folds(
            gtab, 2, shell_tol=20, rng=np.random.default_rng(0)
        )
