"""Tests for dipy.reconst.qti module"""

import numpy as np
import numpy.testing as npt

from dipy.core.gradients import gradient_table
from dipy.core.sphere import HemiSphere, disperse_charges
from dipy.reconst.dti import fractional_anisotropy
import dipy.reconst.qti as qti
from dipy.sims.voxel import vec2vec_rotmat
from dipy.testing import assert_warns
from dipy.testing.decorators import set_random_number_generator
from dipy.utils.optpkg import optional_package
from dipy.reconst.weights_method import weights_method_wls_m_est

cp, have_cvxpy, _ = optional_package("cvxpy", min_version="1.4.1")


def test_from_3x3_to_6x1():
    """Test conversion to Voigt notation."""
    V = np.arange(1, 7)[:, np.newaxis].astype(float)
    T = np.array(
        (
            [1, 4.24264069, 3.53553391],
            [4.24264069, 2, 2.82842712],
            [3.53553391, 2.82842712, 3],
        )
    )
    npt.assert_array_almost_equal(qti.from_3x3_to_6x1(T), V)
    npt.assert_array_almost_equal(qti.from_3x3_to_6x1(qti.from_6x1_to_3x3(V)), V)
    npt.assert_raises(ValueError, qti.from_3x3_to_6x1, T[0:1])
    assert_warns(Warning, qti.from_3x3_to_6x1, T + np.arange(3))


def test_from_6x1_to_3x3():
    """Test conversion from Voigt notation."""
    V = np.arange(1, 7)[:, np.newaxis].astype(float)
    T = np.array(
        (
            [1, 4.24264069, 3.53553391],
            [4.24264069, 2, 2.82842712],
            [3.53553391, 2.82842712, 3],
        )
    )
    npt.assert_array_almost_equal(qti.from_6x1_to_3x3(V), T)
    npt.assert_array_almost_equal(qti.from_6x1_to_3x3(qti.from_3x3_to_6x1(T)), T)
    npt.assert_raises(ValueError, qti.from_6x1_to_3x3, T)


def test_from_6x6_to_21x1():
    """Test conversion to Voigt notation."""
    V = np.arange(1, 22)[:, np.newaxis].astype(float)
    T = np.array(
        (
            [1, 4.24264069, 3.53553391, 4.94974747, 5.65685425, 6.36396103],
            [4.24264069, 2, 2.82842712, 7.07106781, 7.77817459, 8.48528137],
            [3.53553391, 2.82842712, 3, 9.19238816, 9.89949494, 10.60660172],
            [4.94974747, 7.07106781, 9.19238816, 16, 13.43502884, 14.8492424],
            [5.65685425, 7.77817459, 9.89949494, 13.43502884, 17, 14.14213562],
            [6.36396103, 8.48528137, 10.60660172, 14.8492424, 14.14213562, 18],
        )
    )
    npt.assert_array_almost_equal(qti.from_6x6_to_21x1(T), V)
    npt.assert_array_almost_equal(qti.from_6x6_to_21x1(qti.from_21x1_to_6x6(V)), V)
    npt.assert_raises(ValueError, qti.from_6x6_to_21x1, T[0:1])
    assert_warns(Warning, qti.from_6x6_to_21x1, T + np.arange(6))


def test_from_21x1_to_6x6():
    """Test conversion from Voigt notation."""
    V = np.arange(1, 22)[:, np.newaxis].astype(float)
    T = np.array(
        (
            [1, 4.24264069, 3.53553391, 4.94974747, 5.65685425, 6.36396103],
            [4.24264069, 2, 2.82842712, 7.07106781, 7.77817459, 8.48528137],
            [3.53553391, 2.82842712, 3, 9.19238816, 9.89949494, 10.60660172],
            [4.94974747, 7.07106781, 9.19238816, 16, 13.43502884, 14.8492424],
            [5.65685425, 7.77817459, 9.89949494, 13.43502884, 17, 14.14213562],
            [6.36396103, 8.48528137, 10.60660172, 14.8492424, 14.14213562, 18],
        )
    )
    npt.assert_array_almost_equal(qti.from_21x1_to_6x6(V), T)
    npt.assert_array_almost_equal(qti.from_21x1_to_6x6(qti.from_6x6_to_21x1(T)), T)
    npt.assert_raises(ValueError, qti.from_21x1_to_6x6, T)


def test_cvxpy_1x6_to_3x3():
    """Test conversion from Voigt notation."""
    if have_cvxpy:
        V = np.arange(1, 7)[:, np.newaxis].astype(float)
        T = np.array(
            (
                [1, 4.24264069, 3.53553391],
                [4.24264069, 2, 2.82842712],
                [3.53553391, 2.82842712, 3],
            )
        )
        npt.assert_array_almost_equal(qti.cvxpy_1x6_to_3x3(V).value, T)
        npt.assert_array_almost_equal(
            qti.cvxpy_1x6_to_3x3(qti.from_3x3_to_6x1(T)).value, T
        )


def test_cvxpy_1x21_to_6x6():
    """Test conversion from Voigt notation."""
    if have_cvxpy:
        V = np.arange(1, 22)[:, np.newaxis].astype(float)
        T = np.array(
            (
                [1, 4.24264069, 3.53553391, 4.94974747, 5.65685425, 6.36396103],
                [4.24264069, 2, 2.82842712, 7.07106781, 7.77817459, 8.48528137],
                [3.53553391, 2.82842712, 3, 9.19238816, 9.89949494, 10.60660172],
                [4.94974747, 7.07106781, 9.19238816, 16, 13.43502884, 14.8492424],
                [5.65685425, 7.77817459, 9.89949494, 13.43502884, 17, 14.14213562],
                [6.36396103, 8.48528137, 10.60660172, 14.8492424, 14.14213562, 18],
            )
        )
        npt.assert_array_almost_equal(qti.cvxpy_1x21_to_6x6(V).value, T)
        npt.assert_array_almost_equal(
            qti.cvxpy_1x21_to_6x6(qti.from_6x6_to_21x1(T)).value, T
        )


def test_helper_tensors():
    """Test the helper tensors."""
    npt.assert_array_equal(qti.e_iso, np.eye(3) / 3)
    npt.assert_array_equal(qti.E_iso, np.eye(6) / 3)
    npt.assert_array_equal(
        qti.E_bulk,
        np.matmul(qti.from_3x3_to_6x1(qti.e_iso), qti.from_3x3_to_6x1(qti.e_iso).T),
    )
    npt.assert_array_equal(qti.E_shear, qti.E_iso - qti.E_bulk)
    npt.assert_array_equal(qti.E_tsym, qti.E_bulk + 0.4 * qti.E_shear)


def _anisotropic_DTD():
    """Return a distribution of six fully anisotropic diffusion tensors whose
    directions are uniformly distributed around the surface of a sphere."""
    evals = np.array([1, 0, 0])
    phi = (1 + np.sqrt(5)) / 2
    directions = np.array(
        [
            [0, 1, phi],
            [0, 1, -phi],
            [1, phi, 0],
            [1, -phi, 0],
            [phi, 0, 1],
            [phi, 0, -1],
        ]
    ) / np.linalg.norm([0, 1, phi])
    DTD = np.zeros((6, 3, 3))
    for i in range(6):
        R = vec2vec_rotmat(np.array([1, 0, 0]), directions[i])
        DTD[i] = np.matmul(R, np.matmul(np.eye(3) * evals, R.T))
    return DTD


def _isotropic_DTD():
    """Return a distribution of six isotropic diffusion tensors with varying
    sizes."""
    evals = np.linspace(0.1, 3, 6)
    DTD = np.array([np.eye(3) * i for i in evals])
    return DTD


def test_dtd_covariance():
    """Test diffusion tensor distribution covariance calculation."""

    # Input validation
    npt.assert_raises(ValueError, qti.dtd_covariance, np.arange(2))
    npt.assert_raises(ValueError, qti.dtd_covariance, np.zeros((1, 1, 1)))

    # Covariance of isotropic tensors (Figure 1 in Westin's paper)
    DTD = _isotropic_DTD()
    C = np.zeros((6, 6))
    C[0:3, 0:3] = 0.98116667
    npt.assert_almost_equal(qti.dtd_covariance(DTD), C)

    # Covariance of anisotropic tensors (Figure 1 in Westin's paper)
    DTD = _anisotropic_DTD()
    C = np.eye(6) * 2 / 15
    C[0:3, 0:3] = np.array(
        [
            [4 / 45, -2 / 45, -2 / 45],
            [-2 / 45, 4 / 45, -2 / 45],
            [-2 / 45, -2 / 45, 4 / 45],
        ]
    )
    npt.assert_almost_equal(qti.dtd_covariance(DTD), C)


def test_qti_signal():
    """Test QTI signal generation."""

    # Input validation
    bvals = np.ones(6)
    phi = (1 + np.sqrt(5)) / 2
    bvecs = np.array(
        [
            [0, 1, phi],
            [0, 1, -phi],
            [1, phi, 0],
            [1, -phi, 0],
            [phi, 0, 1],
            [phi, 0, -1],
        ]
    ) / np.linalg.norm([0, 1, phi])
    gtab = gradient_table(bvals, bvecs=bvecs)
    npt.assert_raises(ValueError, qti.qti_signal, gtab, np.eye(3), np.eye(6))
    gtab = gradient_table(bvals, bvecs=bvecs, btens="LTE")
    npt.assert_raises(ValueError, qti.qti_signal, gtab, np.eye(2), np.eye(6))
    npt.assert_raises(ValueError, qti.qti_signal, gtab, np.eye(3), np.eye(5))
    npt.assert_raises(
        ValueError, qti.qti_signal, gtab, np.stack((np.eye(3), np.eye(3))), np.eye(5)
    )
    npt.assert_raises(
        ValueError, qti.qti_signal, gtab, np.eye(3)[np.newaxis, :], np.eye(6)
    )
    npt.assert_raises(
        ValueError, qti.qti_signal, gtab, np.eye(3), np.eye(6), S0=np.ones(2)
    )
    qti.qti_signal(
        gradient_table(bvals, bvecs=bvecs, btens="LTE"),
        np.zeros((5, 6)),
        np.zeros((5, 21)),
    )

    # Isotropic diffusion and no 2nd order effects
    D = np.eye(3)
    C = np.zeros((6, 6))
    npt.assert_almost_equal(
        qti.qti_signal(gradient_table(bvals, bvecs=bvecs, btens="LTE"), D, C),
        np.ones(6) * np.exp(-1),
    )
    npt.assert_almost_equal(
        qti.qti_signal(gradient_table(bvals, bvecs=bvecs, btens="LTE"), D, C),
        qti.qti_signal(gradient_table(bvals, bvecs=bvecs, btens="PTE"), D, C),
    )
    npt.assert_almost_equal(
        qti.qti_signal(gradient_table(bvals, bvecs=bvecs, btens="LTE"), D, C),
        qti.qti_signal(gradient_table(bvals, bvecs=bvecs, btens="STE"), D, C),
    )

    # Anisotropic sticks aligned with the bvecs
    DTD = _anisotropic_DTD()
    D = np.mean(DTD, axis=0)
    C = qti.dtd_covariance(DTD)
    npt.assert_almost_equal(
        qti.qti_signal(gradient_table(bvals, bvecs=bvecs, btens="LTE"), D, C),
        np.ones(6) * 0.7490954,
    )
    npt.assert_almost_equal(
        qti.qti_signal(gradient_table(bvals, bvecs=bvecs, btens="PTE"), D, C),
        np.ones(6) * 0.72453716,
    )
    npt.assert_almost_equal(
        qti.qti_signal(gradient_table(bvals, bvecs=bvecs, btens="STE"), D, C),
        np.ones(6) * 0.71653131,
    )


def test_design_matrix():
    """Test QTI design matrix calculation."""
    btens = np.array([np.eye(3, 3) for i in range(3)])
    btens[0, 1, 1] = 0
    btens[0, 2, 2] = 0
    btens[1, 0, 0] = 0
    X = qti.design_matrix(btens)
    npt.assert_almost_equal(
        X,
        np.array(
            [
                [1.0, 1.0, 1.0],
                [-1.0, -0.0, -1.0],
                [-0.0, -1.0, -1.0],
                [-0.0, -1.0, -1.0],
                [-0.0, -0.0, -0.0],
                [-0.0, -0.0, -0.0],
                [-0.0, -0.0, -0.0],
                [0.5, 0.0, 0.5],
                [0.0, 0.5, 0.5],
                [0.0, 0.5, 0.5],
                [0.0, 0.70710678, 0.70710678],
                [0.0, 0.0, 0.70710678],
                [0.0, 0.0, 0.70710678],
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
            ]
        ).T,
    )


def _qti_gtab(rng):
    """Return a gradient table with b0, 2 shells, 30 directions, and linear and
    planar tensor encoding for fitting QTI."""
    n_dir = 30
    hsph_initial = HemiSphere(
        theta=np.pi * rng.random(n_dir), phi=2 * np.pi * rng.random(n_dir)
    )
    hsph_updated, _ = disperse_charges(hsph_initial, 100)
    directions = hsph_updated.vertices
    bvecs = np.vstack([np.zeros(3)] + [directions for _ in range(4)])
    bvals = np.concatenate(
        (
            np.zeros(1),
            np.ones(n_dir),
            np.ones(n_dir) * 2,
            np.ones(n_dir),
            np.ones(n_dir) * 2,
        )
    )
    btens = np.array(
        ["LTE" for i in range(1 + n_dir * 2)] + ["PTE" for i in range(n_dir * 2)]
    )
    gtab = gradient_table(bvals, bvecs=bvecs, btens=btens)
    return gtab


@set_random_number_generator(123)
def test_ls_sdp_fits(rng):
    """Test ordinary and weighted least squares and semidefinite programming
    QTI fits by comparing the estimated parameters to the ground-truth values."""
    gtab = _qti_gtab(rng)
    X = qti.design_matrix(gtab.btens)
    DTDs = [
        _anisotropic_DTD(),
        _isotropic_DTD(),
        np.concatenate((_anisotropic_DTD(), _isotropic_DTD())),
    ]
    for DTD in DTDs:
        D = np.mean(DTD, axis=0)
        C = qti.dtd_covariance(DTD)
        params = np.concatenate(
            (
                np.log(1)[np.newaxis, np.newaxis],
                qti.from_3x3_to_6x1(D),
                qti.from_6x6_to_21x1(C),
            )
        ).T
        data = qti.qti_signal(gtab, D, C)[np.newaxis, :]
        mask = np.ones(1).astype(bool)
        npt.assert_almost_equal(qti._ols_fit(X, data[mask])[0], params)
        npt.assert_almost_equal(qti._wls_fit(X, data[mask])[0], params)
        # stack -> 2 voxels (same number of images)
        data = np.vstack((data, data))
        mask = np.ones(2).astype(bool)
        params = np.vstack((params, params))
        npt.assert_almost_equal(qti._ols_fit(X, data[mask], step=1)[0], params)
        npt.assert_almost_equal(qti._wls_fit(X, data[mask], step=1)[0], params)

        if have_cvxpy:
            npt.assert_almost_equal(
                qti._sdpdc_fit(X, data[mask], cvxpy_solver="SCS")[0], params, decimal=1
            )

    # check if leverages are returned when requested
    # _, extra = qti._ols_fit(X, data[mask], step=1)[0]  # NOTE: ols can't return leverages
    _, extra =  qti._wls_fit(X, data[mask], step=1, return_leverages=True)
    npt.assert_equal("leverages" in extra, True)
    _, extra = qti._sdpdc_fit(X, data[mask], cvxpy_solver="SCS", return_leverages=True)
    npt.assert_equal("leverages" in extra, True)
    
    # test of WLS given explicit weights=signal^2 is same as calling without weights
    npt.assert_almost_equal(
        qti._wls_fit(X, data[mask], step=1)[0],
        qti._wls_fit(X, data[mask], step=1, weights=data[mask]**2)[0]
    )
    npt.assert_almost_equal(
        qti._sdpdc_fit(X, data[mask], cvxpy_solver="SCS")[0],
        qti._sdpdc_fit(X, data[mask], cvxpy_solver="SCS", weights=data[mask]**2)[0]
    )

    # test robust QTI - robust fitting without noise makes no sense, so hard to test
    # FIXME: the 'robust' extra is not coming out the same shape as the data prior to masking
    # robust fitting without noise in signal isn't really sensible... how to make sensible tests?
    data_corrupt = 10*data.copy()  # we could make a copy,copy (stack) array (and gtab) and corrupt only one thing...
    noise = 0.01 * np.random.normal(size=data_corrupt.shape)  # NOTE: noise must be reasonably size wrt signal, which has S=1
    data_corrupt = data_corrupt + noise  # error, or weights irrelevant
    data_corrupt[..., -1] *= 5  # corrupt a signal - can't be so severe that we fail to fit
    robust_signals = np.ones(data_corrupt.shape).astype(bool)
    robust_signals[..., -1] = False  # corrupted signal, lets see if we can detect it later
    # non-robust fitting should fail
    # ------------------------------
    # fit with WLS, show fitted params are different
    qtimodel = qti.QtiModel(gtab, fit_method="WLS")
    qtifit = qtimodel.fit(data_corrupt)
    npt.assert_raises(AssertionError, npt.assert_almost_equal, qtifit.params, params)
    # fit with SDPdc (constraints), show fitted params are different
    qtimodel = qti.QtiModel(gtab, fit_method="SDPdc")
    qtifit = qtimodel.fit(data_corrupt)
    npt.assert_raises(AssertionError, npt.assert_almost_equal, qtifit.params, params)
    # keywords needed to trigger iterative_fit
    # ----------------------------------------
    def wm(*args):
        # super generous cut-off, should find only the real outlier
        # even with cutoff 5, sometimes we still find outliers even if not making any...
        # never saw this before, is a bit strange
        # might be related to MIN_POSITIVE_SIGNAL effects
        # could also be number of paramers compared to observations (i.e. gtab design)
        return weights_method_wls_m_est(
            *args, m_est="gm", cutoff=10
        )
    kwargs = {"weights_method": wm, "num_iter": 10}
    # fit with WLS with weights_method (triggers iterative_fit)
    qtimodel_r = qti.QtiModel(gtab, fit_method="WLS", **kwargs)
    qtifit_r = qtimodel_r.fit(data_corrupt)
    # NOTE: something about the use of a mask, abnd the shape of saved extra[robust], is the issue
    npt.assert_almost_equal(qtimodel_r.extra["robust"], robust_signals)
    #npt.assert_almost_equal(qtifit_r.params, params)  # robust fit failing to be equal to non-robust... but is that inevitable if we remove a signal?
    # fit with SDPdc (constraints) with weights method (triggers iterative_fit)
    #qtimodel_r = qti.QtiModel(gtab, fit_method="SDPdc", **kwargs)
    #qtifit_r = qtimodel_r.fit(data_corrupt, mask=mask)
    #npt.assert_almost_equal(qtifit_r.params, params)

    # TODO: need to test the iterative_fit function in other ways


@set_random_number_generator(123)
def test_qti_model(rng):
    """Test the QTI model class."""

    # Input validation
    gtab = gradient_table(np.ones(1), bvecs=np.array([[1, 0, 0]]))
    npt.assert_raises(ValueError, qti.QtiModel, gtab)
    gtab = gradient_table(np.ones(1), bvecs=np.array([[1, 0, 0]]), btens="LTE")
    assert_warns(UserWarning, qti.QtiModel, gtab)
    npt.assert_raises(ValueError, qti.QtiModel, _qti_gtab(rng), fit_method="non-linear")

    # Design matrix calculation
    gtab = _qti_gtab(rng)
    qtimodel = qti.QtiModel(gtab)
    npt.assert_almost_equal(qtimodel.X, qti.design_matrix(gtab.btens))


@set_random_number_generator(4321)
def test_qti_fit(rng):
    """Test the QTI fit class."""

    # Generate a diffusion tensor distribution
    DTD = np.concatenate(
        (
            _isotropic_DTD(),
            _anisotropic_DTD(),
            np.array([[[3, 0, 0], [0, 0, 0], [0, 0, 0]]]),
        )
    )

    # Calculate the ground-truth parameter values
    S0 = 1000
    D = np.mean(DTD, axis=0)
    C = qti.dtd_covariance(DTD)
    params = np.concatenate(
        (
            np.log(S0)[np.newaxis, np.newaxis],
            qti.from_3x3_to_6x1(D),
            qti.from_6x6_to_21x1(C),
        )
    ).T
    evals, evecs = np.linalg.eig(DTD)
    avg_eval_var = np.mean(np.var(evals, axis=1))
    md = np.mean(evals)
    fa = fractional_anisotropy(np.linalg.eig(D)[0])
    v_md = np.var(np.mean(evals, axis=1))
    v_shear = avg_eval_var - np.var(np.linalg.eig(D)[0])
    v_iso = v_md + v_shear
    d_sq = qti.from_3x3_to_6x1(D) @ qti.from_3x3_to_6x1(D).T
    mean_d_sq = np.mean(
        np.matmul(
            qti.from_3x3_to_6x1(DTD), np.swapaxes(qti.from_3x3_to_6x1(DTD), -2, -1)
        ),
        axis=0,
    )
    c_md = v_md / np.mean(np.mean(evals, axis=1) ** 2)
    c_m = fa**2
    c_mu = 1.5 * avg_eval_var / np.mean(evals**2)
    ufa = np.sqrt(c_mu)
    c_c = c_m / c_mu
    k_bulk = (
        3
        * np.matmul(
            np.swapaxes(qti.from_6x6_to_21x1(C), -1, -2),
            qti.from_6x6_to_21x1(qti.E_bulk),
        )
        / np.matmul(
            np.swapaxes(qti.from_6x6_to_21x1(d_sq), -1, -2),
            qti.from_6x6_to_21x1(qti.E_bulk),
        )
    )[0, 0]
    k_shear = (
        6
        / 5
        * np.matmul(
            np.swapaxes(qti.from_6x6_to_21x1(C), -1, -2),
            qti.from_6x6_to_21x1(qti.E_shear),
        )
        / np.matmul(
            np.swapaxes(qti.from_6x6_to_21x1(d_sq), -1, -2),
            qti.from_6x6_to_21x1(qti.E_bulk),
        )
    )[0, 0]
    mk = k_bulk + k_shear
    k_mu = (
        6
        / 5
        * np.matmul(
            np.swapaxes(qti.from_6x6_to_21x1(mean_d_sq), -1, -2),
            qti.from_6x6_to_21x1(qti.E_shear),
        )
        / np.matmul(
            np.swapaxes(qti.from_6x6_to_21x1(d_sq), -1, -2),
            qti.from_6x6_to_21x1(qti.E_bulk),
        )
    )[0, 0]

    # Fit QTI
    gtab = _qti_gtab(rng)

    if have_cvxpy:
        for fit_method in ["OLS", "WLS", "SDPdc"]:
            qtimodel = qti.QtiModel(gtab, fit_method=fit_method)
            data = qtimodel.predict(params)
            npt.assert_raises(ValueError, qtimodel.fit, data, mask=np.ones(2))
            npt.assert_raises(ValueError, qtimodel.fit, data, mask=np.ones(data.shape))
            for mask in [None, np.ones(data.shape[0:-1])]:
                qtifit = qtimodel.fit(data, mask=mask)
                npt.assert_raises(
                    ValueError,
                    qtifit.predict,
                    gradient_table(np.zeros(3), bvecs=np.zeros((3, 3))),
                )
                npt.assert_almost_equal(qtifit.predict(gtab), data, decimal=1)
                npt.assert_almost_equal(qtifit.S0_hat, S0, decimal=2)
                npt.assert_almost_equal(qtifit.md, md, decimal=2)
                npt.assert_almost_equal(qtifit.v_md, v_md, decimal=2)
                npt.assert_almost_equal(qtifit.v_shear, v_shear, decimal=2)
                npt.assert_almost_equal(qtifit.v_iso, v_iso, decimal=2)
                npt.assert_almost_equal(qtifit.c_md, c_md, decimal=2)
                npt.assert_almost_equal(qtifit.c_mu, c_mu, decimal=2)
                npt.assert_almost_equal(qtifit.ufa, ufa, decimal=2)
                npt.assert_almost_equal(qtifit.c_m, c_m, decimal=2)
                npt.assert_almost_equal(qtifit.fa, fa, decimal=2)
                npt.assert_almost_equal(qtifit.c_c, c_c, decimal=2)
                npt.assert_almost_equal(qtifit.mk, mk, decimal=2)
                npt.assert_almost_equal(qtifit.k_bulk, k_bulk, decimal=2)
                npt.assert_almost_equal(qtifit.k_shear, k_shear, decimal=2)
                npt.assert_almost_equal(qtifit.k_mu, k_mu, decimal=2)
    else:
        for fit_method in ["OLS", "WLS"]:
            qtimodel = qti.QtiModel(gtab, fit_method=fit_method)
            data = qtimodel.predict(params)
            npt.assert_raises(ValueError, qtimodel.fit, data, mask=np.ones(2))
            npt.assert_raises(ValueError, qtimodel.fit, data, mask=np.ones(data.shape))
            for mask in [None, np.ones(data.shape[0:-1])]:
                qtifit = qtimodel.fit(data, mask=mask)
                npt.assert_raises(
                    ValueError,
                    qtifit.predict,
                    gradient_table(np.zeros(3), bvecs=np.zeros((3, 3))),
                )
                npt.assert_almost_equal(qtifit.predict(gtab), data)
                npt.assert_almost_equal(qtifit.S0_hat, S0)
                npt.assert_almost_equal(qtifit.md, md)
                npt.assert_almost_equal(qtifit.v_md, v_md)
                npt.assert_almost_equal(qtifit.v_shear, v_shear)
                npt.assert_almost_equal(qtifit.v_iso, v_iso)
                npt.assert_almost_equal(qtifit.c_md, c_md)
                npt.assert_almost_equal(qtifit.c_mu, c_mu)
                npt.assert_almost_equal(qtifit.ufa, ufa)
                npt.assert_almost_equal(qtifit.c_m, c_m)
                npt.assert_almost_equal(qtifit.fa, fa)
                npt.assert_almost_equal(qtifit.c_c, c_c)
                npt.assert_almost_equal(qtifit.mk, mk)
                npt.assert_almost_equal(qtifit.k_bulk, k_bulk)
                npt.assert_almost_equal(qtifit.k_shear, k_shear)
                npt.assert_almost_equal(qtifit.k_mu, k_mu)
