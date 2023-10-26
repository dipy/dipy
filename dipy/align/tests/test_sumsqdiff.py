import numpy as np
from dipy.align import floating
from dipy.align import sumsqdiff as ssd
from numpy.testing import (assert_equal,
                           assert_almost_equal,
                           assert_array_almost_equal,
                           assert_allclose)
from dipy.testing.decorators import set_random_number_generator


def iterate_residual_field_ssd_2d(delta_field, sigmasq_field, grad, target,
                                  lambda_param, dfield):
    r"""
    This implementation is for testing purposes only. The problem
    with Gauss-Seidel iterations is that it depends on the order
    in which we iterate over the variables, so it is necessary to
    replicate the implementation under test.
    """
    nrows, ncols = delta_field.shape
    if target is None:
        b = np.zeros_like(grad)
        b[..., 0] = delta_field * grad[..., 0]
        b[..., 1] = delta_field * grad[..., 1]
    else:
        b = target

    y = np.zeros(2)
    for r in range(nrows):
        for c in range(ncols):
            sigmasq = sigmasq_field[r, c] if sigmasq_field is not None else 1
            # This has to be done inside the nested loops because
            # some d[...] may have been previously modified
            nn = 0
            y[:] = 0
            for (dRow, dCol) in [(-1, 0), (0, 1), (1, 0), (0, -1)]:
                dr = r + dRow
                if dr < 0 or dr >= nrows:
                    continue
                dc = c + dCol
                if dc < 0 or dc >= ncols:
                    continue
                nn += 1
                y += dfield[dr, dc]

            if np.isinf(sigmasq):
                dfield[r, c] = y / nn
            else:
                tau = sigmasq * lambda_param * nn
                A = np.outer(grad[r, c], grad[r, c]) + tau * np.eye(2)
                det = np.linalg.det(A)
                if det < 1e-9:
                    nrm2 = np.sum(grad[r, c]**2)
                    if nrm2 < 1e-9:
                        dfield[r, c, :] = 0
                    else:
                        dfield[r, c] = b[r, c] / nrm2
                else:
                    y = b[r, c] + sigmasq * lambda_param * y
                    dfield[r, c] = np.linalg.solve(A, y)


def iterate_residual_field_ssd_3d(delta_field, sigmasq_field, grad, target,
                                  lambda_param, dfield):
    r"""
    This implementation is for testing purposes only. The problem
    with Gauss-Seidel iterations is that it depends on the order
    in which we iterate over the variables, so it is necessary to
    replicate the implementation under test.
    """
    nslices, nrows, ncols = delta_field.shape
    if target is None:
        b = np.zeros_like(grad)
        for i in range(3):
            b[..., i] = delta_field * grad[..., i]
    else:
        b = target

    y = np.ndarray((3,))
    for s in range(nslices):
        for r in range(nrows):
            for c in range(ncols):
                g = grad[s, r, c]
                # delta = delta_field[s, r, c]
                sigmasq = sigmasq_field[
                    s, r, c] if sigmasq_field is not None else 1
                nn = 0
                y[:] = 0
                for dSlice, dRow, dCol in [(-1, 0, 0), (0, -1, 0), (0, 0, 1),
                                           (0, 1, 0), (0, 0, -1), (1, 0, 0)]:
                    ds = s + dSlice
                    if ds < 0 or ds >= nslices:
                        continue
                    dr = r + dRow
                    if dr < 0 or dr >= nrows:
                        continue
                    dc = c + dCol
                    if dc < 0 or dc >= ncols:
                        continue
                    nn += 1
                    y += dfield[ds, dr, dc]
                if np.isinf(sigmasq):
                    dfield[s, r, c] = y / nn
                elif sigmasq < 1e-9:
                    nrm2 = np.sum(g**2)
                    if nrm2 < 1e-9:
                        dfield[s, r, c, :] = 0
                    else:
                        dfield[s, r, c, :] = b[s, r, c] / nrm2
                else:
                    tau = sigmasq * lambda_param * nn
                    y = b[s, r, c] + sigmasq * lambda_param * y
                    G = np.outer(g, g) + tau * np.eye(3)
                    try:
                        dfield[s, r, c] = np.linalg.solve(G, y)
                    except np.linalg.linalg.LinAlgError:
                        nrm2 = np.sum(g**2)
                        if nrm2 < 1e-9:
                            dfield[s, r, c, :] = 0
                        else:
                            dfield[s, r, c] = b[s, r, c] / nrm2


@set_random_number_generator(5512751)
def test_compute_residual_displacement_field_ssd_2d(rng):
    # Select arbitrary images' shape (same shape for both images)
    sh = (20, 10)

    # Select arbitrary centers
    c_f = np.asarray(sh) / 2
    c_g = c_f + 0.5

    # Compute the identity vector field I(x) = x in R^2
    x_0 = np.asarray(range(sh[0]))
    x_1 = np.asarray(range(sh[1]))
    X = np.ndarray(sh + (2,), dtype=np.float64)
    O = np.ones(sh)
    X[..., 0] = x_0[:, None] * O
    X[..., 1] = x_1[None, :] * O

    # Compute the gradient fields of F and G
    grad_F = X - c_f
    grad_G = X - c_g

    Fnoise = rng.random(
        np.size(grad_F)).reshape(
        grad_F.shape) * grad_F.max() * 0.1
    Fnoise = Fnoise.astype(floating)
    grad_F += Fnoise

    Gnoise = rng.random(
        np.size(grad_G)).reshape(
        grad_G.shape) * grad_G.max() * 0.1
    Gnoise = Gnoise.astype(floating)
    grad_G += Gnoise

    # The squared norm of grad_G
    sq_norm_grad_G = np.sum(grad_G**2, -1)

    # Compute F and G
    F = 0.5 * np.sum(grad_F**2, -1)
    G = 0.5 * sq_norm_grad_G

    Fnoise = rng.random(np.size(F)).reshape(F.shape) * F.max() * 0.1
    Fnoise = Fnoise.astype(floating)
    F += Fnoise

    Gnoise = rng.random(np.size(G)).reshape(G.shape) * G.max() * 0.1
    Gnoise = Gnoise.astype(floating)
    G += Gnoise

    delta_field = np.array(F - G, dtype=floating)

    sigma_field = \
        rng.standard_normal(delta_field.size).reshape(delta_field.shape)
    sigma_field = sigma_field.astype(floating)

    # Select some pixels to force sigma_field = infinite
    inf_sigma = rng.integers(0, 2, sh[0] * sh[1])
    inf_sigma = inf_sigma.reshape(sh)
    sigma_field[inf_sigma == 1] = np.inf

    # Select an initial displacement field
    d = rng.standard_normal(grad_G.size).reshape(grad_G.shape).astype(floating)
    lambda_param = 1.5

    # Implementation under test
    iut = ssd.compute_residual_displacement_field_ssd_2d

    # In the first iteration we test the case target=None
    # In the second iteration, target is not None
    target = None
    rtol = 1e-9
    atol = 1e-4
    for it in range(2):
        # Sum of differences with the neighbors
        s = np.zeros_like(d, dtype=np.float64)
        s[:, :-1] += d[:, :-1] - d[:, 1:]  # right
        s[:, 1:] += d[:, 1:] - d[:, :-1]  # left
        s[:-1, :] += d[:-1, :] - d[1:, :]  # down
        s[1:, :] += d[1:, :] - d[:-1, :]  # up
        s *= lambda_param

        # Dot product of displacement and gradient
        dp = d[..., 0] * grad_G[..., 0] + \
            d[..., 1] * grad_G[..., 1]
        dp = dp.astype(np.float64)

        # Compute expected residual
        if target is None:
            expected = np.zeros_like(grad_G)
            expected[..., 0] = delta_field * grad_G[..., 0]
            expected[..., 1] = delta_field * grad_G[..., 1]
        else:
            expected = target.copy().astype(np.float64)

        # Expected residuals when sigma != infinity
        expected[inf_sigma == 0, 0] -= grad_G[inf_sigma == 0, 0] * \
            dp[inf_sigma == 0] + sigma_field[inf_sigma == 0] * s[inf_sigma == 0, 0]
        expected[inf_sigma == 0, 1] -= grad_G[inf_sigma == 0, 1] * \
            dp[inf_sigma == 0] + sigma_field[inf_sigma == 0] * s[inf_sigma == 0, 1]
        # Expected residuals when sigma == infinity
        expected[inf_sigma == 1] = -1.0 * s[inf_sigma == 1]

        # Test residual field computation starting with residual = None
        actual = iut(delta_field, sigma_field, grad_G.astype(floating),
                     target, lambda_param, d, None)
        assert_allclose(actual, expected, rtol=rtol, atol=atol)
        # destroy previous result
        actual = np.ndarray(actual.shape, dtype=floating)

        # Test residual field computation starting with residual is not None
        iut(delta_field, sigma_field, grad_G.astype(floating),
            target, lambda_param, d, actual)
        assert_allclose(actual, expected, rtol=rtol, atol=atol)

        # Set target for next iteration
        target = actual

        # Test Gauss-Seidel step with residual=None and residual=target
        for residual in [None, target]:
            expected = d.copy()
            iterate_residual_field_ssd_2d(
                delta_field,
                sigma_field,
                grad_G.astype(floating),
                residual,
                lambda_param,
                expected)

            actual = d.copy()
            ssd.iterate_residual_displacement_field_ssd_2d(
                delta_field,
                sigma_field,
                grad_G.astype(floating),
                residual,
                lambda_param,
                actual)

            assert_allclose(actual, expected, rtol=rtol, atol=atol)


@set_random_number_generator(5512751)
def test_compute_residual_displacement_field_ssd_3d(rng):
    # Select arbitrary images' shape (same shape for both images)
    sh = (20, 15, 10)

    # Select arbitrary centers
    c_f = np.asarray(sh) / 2
    c_g = c_f + 0.5

    # Compute the identity vector field I(x) = x in R^2
    x_0 = np.asarray(range(sh[0]))
    x_1 = np.asarray(range(sh[1]))
    x_2 = np.asarray(range(sh[2]))
    X = np.ndarray(sh + (3,), dtype=np.float64)
    O = np.ones(sh)
    X[..., 0] = x_0[:, None, None] * O
    X[..., 1] = x_1[None, :, None] * O
    X[..., 2] = x_2[None, None, :] * O

    # Compute the gradient fields of F and G
    grad_F = X - c_f
    grad_G = X - c_g

    Fnoise = rng.random(
        np.size(grad_F)).reshape(
        grad_F.shape) * grad_F.max() * 0.1
    Fnoise = Fnoise.astype(floating)
    grad_F += Fnoise

    Gnoise = rng.random(
        np.size(grad_G)).reshape(
        grad_G.shape) * grad_G.max() * 0.1
    Gnoise = Gnoise.astype(floating)
    grad_G += Gnoise

    # The squared norm of grad_G
    sq_norm_grad_G = np.sum(grad_G**2, -1)

    # Compute F and G
    F = 0.5 * np.sum(grad_F**2, -1)
    G = 0.5 * sq_norm_grad_G

    Fnoise = rng.random(np.size(F)).reshape(F.shape) * F.max() * 0.1
    Fnoise = Fnoise.astype(floating)
    F += Fnoise

    Gnoise = rng.random(np.size(G)).reshape(G.shape) * G.max() * 0.1
    Gnoise = Gnoise.astype(floating)
    G += Gnoise

    delta_field = np.array(F - G, dtype=floating)

    sigma_field = rng.random(delta_field.size).reshape(delta_field.shape)
    sigma_field = sigma_field.astype(floating)

    # Select some pixels to force sigma_field = infinite
    inf_sigma = rng.integers(0, 2, sh[0] * sh[1] * sh[2])
    inf_sigma = inf_sigma.reshape(sh)
    sigma_field[inf_sigma == 1] = np.inf

    # Select an initial displacement field
    d = rng.random(grad_G.size).reshape(grad_G.shape).astype(floating)
    lambda_param = 1.5

    # Implementation under test
    iut = ssd.compute_residual_displacement_field_ssd_3d

    # In the first iteration we test the case target=None
    # In the second iteration, target is not None
    target = None
    rtol = 1e-9
    atol = 1e-4
    for it in range(2):
        # Sum of differences with the neighbors
        s = np.zeros_like(d, dtype=np.float64)
        s[:, :, :-1] += d[:, :, :-1] - d[:, :, 1:]  # right
        s[:, :, 1:] += d[:, :, 1:] - d[:, :, :-1]  # left
        s[:, :-1, :] += d[:, :-1, :] - d[:, 1:, :]  # down
        s[:, 1:, :] += d[:, 1:, :] - d[:, :-1, :]  # up
        s[:-1, :, :] += d[:-1, :, :] - d[1:, :, :]  # below
        s[1:, :, :] += d[1:, :, :] - d[:-1, :, :]  # above
        s *= lambda_param

        # Dot product of displacement and gradient
        dp = d[..., 0] * grad_G[..., 0] + \
            d[..., 1] * grad_G[..., 1] + \
            d[..., 2] * grad_G[..., 2]

        # Compute expected residual
        if target is None:
            expected = np.zeros_like(grad_G)
            for i in range(3):
                expected[..., i] = delta_field * grad_G[..., i]
        else:
            expected = target.copy().astype(np.float64)

        # Expected residuals when sigma != infinity
        for i in range(3):
            expected[inf_sigma == 0, i] -= grad_G[inf_sigma == 0, i] * \
                dp[inf_sigma == 0] + sigma_field[inf_sigma == 0] * s[inf_sigma == 0, i]
        # Expected residuals when sigma == infinity
        expected[inf_sigma == 1] = -1.0 * s[inf_sigma == 1]

        # Test residual field computation starting with residual = None
        actual = iut(delta_field, sigma_field, grad_G.astype(floating),
                     target, lambda_param, d, None)
        assert_allclose(actual, expected, rtol=rtol, atol=atol)
        # destroy previous result
        actual = np.ndarray(actual.shape, dtype=floating)

        # Test residual field computation starting with residual is not None
        iut(delta_field, sigma_field, grad_G.astype(floating),
            target, lambda_param, d, actual)
        assert_allclose(actual, expected, rtol=rtol, atol=atol)

        # Set target for next iteration
        target = actual

        # Test Gauss-Seidel step with residual=None and residual=target
        for residual in [None, target]:
            expected = d.copy()
            iterate_residual_field_ssd_3d(
                delta_field,
                sigma_field,
                grad_G.astype(floating),
                residual,
                lambda_param,
                expected)

            actual = d.copy()
            ssd.iterate_residual_displacement_field_ssd_3d(
                delta_field,
                sigma_field,
                grad_G.astype(floating),
                residual,
                lambda_param,
                actual)

            # the numpy linear solver may differ from our custom implementation
            # we need to increase the tolerance a bit
            assert_allclose(actual, expected, rtol=rtol, atol=atol * 10)


def test_solve_2d_symmetric_positive_definite():
    # Select some arbitrary right-hand sides
    bs = [np.array([1.1, 2.2]),
          np.array([1e-2, 3e-3]),
          np.array([1e2, 1e3]),
          np.array([1e-5, 1e5])]

    # Select arbitrary symmetric positive-definite matrices
    As = []

    # Identity
    identity = np.array([1.0, 0.0, 1.0])
    As.append(identity)

    # Small determinant
    small_det = np.array([1e-3, 1e-4, 1e-3])
    As.append(small_det)

    # Large determinant
    large_det = np.array([1e6, 1e4, 1e6])
    As.append(large_det)

    for A in As:
        AA = np.array([[A[0], A[1]], [A[1], A[2]]])
        det = np.linalg.det(AA)
        for b in bs:
            expected = np.linalg.solve(AA, b)
            actual = ssd.solve_2d_symmetric_positive_definite(A, b, det)
            assert_allclose(expected, actual, rtol=1e-9, atol=1e-9)


def test_solve_3d_symmetric_positive_definite():
    # Select some arbitrary right-hand sides
    bs = [np.array([1.1, 2.2, 3.3]),
          np.array([1e-2, 3e-3, 2e-2]),
          np.array([1e2, 1e3, 5e-2]),
          np.array([1e-5, 1e5, 1.0])]

    # Select arbitrary taus
    taus = [0.0, 1.0, 1e-4, 1e5]

    # Select arbitrary matrices
    gs = []

    # diagonal
    diag = np.array([0.0, 0.0, 0.0])
    gs.append(diag)

    # canonical basis
    gs.append(np.array([1.0, 0.0, 0.0]))
    gs.append(np.array([0.0, 1.0, 0.0]))
    gs.append(np.array([0.0, 0.0, 1.0]))

    # other
    gs.append(np.array([1.0, 0.5, 0.0]))
    gs.append(np.array([0.0, 0.2, 0.1]))
    gs.append(np.array([0.3, 0.0, 0.9]))

    for g in gs:
        A = g[:, None] * g[None, :]
        for tau in taus:
            AA = A + tau * np.eye(3)
            for b in bs:
                actual, is_singular = ssd.solve_3d_symmetric_positive_definite(
                    g, b, tau)
                if tau == 0.0:
                    assert_equal(is_singular, 1)
                else:
                    expected = np.linalg.solve(AA, b)
                    assert_allclose(expected, actual, rtol=1e-9, atol=1e-9)


def test_compute_energy_ssd_2d():
    sh = (32, 32)

    # Select arbitrary centers
    c_f = np.asarray(sh) / 2
    c_g = c_f + 0.5

    # Compute the identity vector field I(x) = x in R^2
    x_0 = np.asarray(range(sh[0]))
    x_1 = np.asarray(range(sh[1]))
    X = np.ndarray(sh + (2,), dtype=np.float64)
    O = np.ones(sh)
    X[..., 0] = x_0[:, None] * O
    X[..., 1] = x_1[None, :] * O

    # Compute the gradient fields of F and G
    grad_F = X - c_f
    grad_G = X - c_g

    # Compute F and G
    F = 0.5 * np.sum(grad_F**2, -1)
    G = 0.5 * np.sum(grad_G**2, -1)

    # Note: this should include the energy corresponding to the
    # regularization term, but it is discarded in ANTS (they just
    # consider the data term, which is not the objective function
    # being optimized). This test case should be updated after
    # further investigation
    expected = ((F - G)**2).sum()
    actual = ssd.compute_energy_ssd_2d(np.array(F - G, dtype=floating))
    assert_almost_equal(expected, actual)


def test_compute_energy_ssd_3d():
    sh = (32, 32, 32)

    # Select arbitrary centers
    c_f = np.asarray(sh) / 2
    c_g = c_f + 0.5

    # Compute the identity vector field I(x) = x in R^2
    x_0 = np.asarray(range(sh[0]))
    x_1 = np.asarray(range(sh[1]))
    x_2 = np.asarray(range(sh[2]))
    X = np.ndarray(sh + (3,), dtype=np.float64)
    O = np.ones(sh)
    X[..., 0] = x_0[:, None, None] * O
    X[..., 1] = x_1[None, :, None] * O
    X[..., 2] = x_2[None, None, :] * O

    # Compute the gradient fields of F and G
    grad_F = X - c_f
    grad_G = X - c_g

    # Compute F and G
    F = 0.5 * np.sum(grad_F**2, -1)
    G = 0.5 * np.sum(grad_G**2, -1)

    # Note: this should include the energy corresponding to the
    # regularization term, but it is discarded in ANTS (they just
    # consider the data term, which is not the objective function
    # being optimized). This test case should be updated after
    # further investigating
    expected = ((F - G)**2).sum()
    actual = ssd.compute_energy_ssd_3d(np.array(F - G, dtype=floating))
    assert_almost_equal(expected, actual)


@set_random_number_generator(1137271)
def test_compute_ssd_demons_step_2d(rng):
    r"""
    Compares the output of the demons step in 2d against an analytical
    step. The fixed image is given by $F(x) = \frac{1}{2}||x - c_f||^2$, the
    moving image is given by $G(x) = \frac{1}{2}||x - c_g||^2$,
    $x, c_f, c_g \in R^{2}$

    References
    ----------
    [Vercauteren09] Vercauteren, T., Pennec, X., Perchant, A., & Ayache, N.
                    (2009). Diffeomorphic demons: efficient non-parametric
                    image registration. NeuroImage, 45(1 Suppl), S61-72.
                    doi:10.1016/j.neuroimage.2008.10.040
    """
    # Select arbitrary images' shape (same shape for both images)
    sh = (20, 10)

    # Select arbitrary centers
    c_f = np.asarray(sh) / 2
    c_g = c_f + 0.5

    # Compute the identity vector field I(x) = x in R^2
    x_0 = np.asarray(range(sh[0]))
    x_1 = np.asarray(range(sh[1]))
    X = np.ndarray(sh + (2,), dtype=np.float64)
    O = np.ones(sh)
    X[..., 0] = x_0[:, None] * O
    X[..., 1] = x_1[None, :] * O

    # Compute the gradient fields of F and G
    grad_F = X - c_f
    grad_G = X - c_g

    Fnoise = rng.random(
        np.size(grad_F)).reshape(
        grad_F.shape) * grad_F.max() * 0.1
    Fnoise = Fnoise.astype(floating)
    grad_F += Fnoise

    Gnoise = rng.random(
        np.size(grad_G)).reshape(
        grad_G.shape) * grad_G.max() * 0.1
    Gnoise = Gnoise.astype(floating)
    grad_G += Gnoise

    # The squared norm of grad_G to be used later
    sq_norm_grad_G = np.sum(grad_G**2, -1)

    # Compute F and G
    F = 0.5 * np.sum(grad_F**2, -1)
    G = 0.5 * sq_norm_grad_G

    Fnoise = rng.random(np.size(F)).reshape(F.shape) * F.max() * 0.1
    Fnoise = Fnoise.astype(floating)
    F += Fnoise

    Gnoise = rng.random(np.size(G)).reshape(G.shape) * G.max() * 0.1
    Gnoise = Gnoise.astype(floating)
    G += Gnoise

    delta_field = np.array(G - F, dtype=floating)

    # Select some pixels to force gradient = 0 and F=G
    random_labels = rng.integers(0, 2, sh[0] * sh[1])
    random_labels = random_labels.reshape(sh)

    F[random_labels == 0] = G[random_labels == 0]
    delta_field[random_labels == 0] = 0
    grad_G[random_labels == 0, ...] = 0
    sq_norm_grad_G[random_labels == 0, ...] = 0

    # Set arbitrary values for $\sigma_i$ (eq. 4 in [Vercauteren09])
    # The original Demons algorithm used simply |F(x) - G(x)| as an
    # estimator, so let's use it as well
    sigma_i_sq = (F - G)**2

    # Now select arbitrary parameters for $\sigma_x$ (eq 4 in [Vercauteren09])
    for sigma_x_sq in [0.01, 1.5, 4.2]:
        # Directly compute the demons step according to eq. 4 in
        # [Vercauteren09]
        num = (sigma_x_sq * (F - G))[random_labels == 1]
        den = (sigma_x_sq * sq_norm_grad_G + sigma_i_sq)[random_labels == 1]
        # This is $J^{P}$ in eq. 4 [Vercauteren09]
        expected = (-1 * np.array(grad_G))
        expected[random_labels == 1, 0] *= num / den
        expected[random_labels == 1, 1] *= num / den
        expected[random_labels == 0, ...] = 0

        # Now compute it using the implementation under test
        actual = np.empty_like(expected, dtype=floating)

        ssd.compute_ssd_demons_step_2d(delta_field,
                                       np.array(grad_G, dtype=floating),
                                       sigma_x_sq,
                                       actual)

        assert_array_almost_equal(actual, expected)


@set_random_number_generator(1137271)
def test_compute_ssd_demons_step_3d(rng):
    r"""
    Compares the output of the demons step in 3d against an analytical
    step. The fixed image is given by $F(x) = \frac{1}{2}||x - c_f||^2$, the
    moving image is given by $G(x) = \frac{1}{2}||x - c_g||^2$,
    $x, c_f, c_g \in R^{3}$

    References
    ----------
    [Vercauteren09] Vercauteren, T., Pennec, X., Perchant, A., & Ayache, N.
                    (2009). Diffeomorphic demons: efficient non-parametric
                    image registration. NeuroImage, 45(1 Suppl), S61-72.
                    doi:10.1016/j.neuroimage.2008.10.040
    """

    # Select arbitrary images' shape (same shape for both images)
    sh = (20, 15, 10)

    # Select arbitrary centers
    c_f = np.asarray(sh) / 2
    c_g = c_f + 0.5

    # Compute the identity vector field I(x) = x in R^2
    x_0 = np.asarray(range(sh[0]))
    x_1 = np.asarray(range(sh[1]))
    x_2 = np.asarray(range(sh[2]))
    X = np.ndarray(sh + (3,), dtype=np.float64)
    O = np.ones(sh)
    X[..., 0] = x_0[:, None, None] * O
    X[..., 1] = x_1[None, :, None] * O
    X[..., 2] = x_2[None, None, :] * O

    # Compute the gradient fields of F and G
    grad_F = X - c_f
    grad_G = X - c_g

    Fnoise = rng.random(
        np.size(grad_F)).reshape(
        grad_F.shape) * grad_F.max() * 0.1
    Fnoise = Fnoise.astype(floating)
    grad_F += Fnoise

    Gnoise = rng.random(
        np.size(grad_G)).reshape(
        grad_G.shape) * grad_G.max() * 0.1
    Gnoise = Gnoise.astype(floating)
    grad_G += Gnoise

    # The squared norm of grad_G to be used later
    sq_norm_grad_G = np.sum(grad_G**2, -1)

    # Compute F and G
    F = 0.5 * np.sum(grad_F**2, -1)
    G = 0.5 * sq_norm_grad_G

    Fnoise = rng.random(np.size(F)).reshape(F.shape) * F.max() * 0.1
    Fnoise = Fnoise.astype(floating)
    F += Fnoise

    Gnoise = rng.random(np.size(G)).reshape(G.shape) * G.max() * 0.1
    Gnoise = Gnoise.astype(floating)
    G += Gnoise

    delta_field = np.array(G - F, dtype=floating)

    # Select some pixels to force gradient = 0 and F=G
    random_labels = rng.integers(0, 2, sh[0] * sh[1] * sh[2])
    random_labels = random_labels.reshape(sh)

    F[random_labels == 0] = G[random_labels == 0]
    delta_field[random_labels == 0] = 0
    grad_G[random_labels == 0, ...] = 0
    sq_norm_grad_G[random_labels == 0, ...] = 0

    # Set arbitrary values for $\sigma_i$ (eq. 4 in [Vercauteren09])
    # The original Demons algorithm used simply |F(x) - G(x)| as an
    # estimator, so let's use it as well
    sigma_i_sq = (F - G)**2

    # Now select arbitrary parameters for $\sigma_x$ (eq 4 in [Vercauteren09])
    for sigma_x_sq in [0.01, 1.5, 4.2]:
        # Directly compute the demons step according to eq. 4 in
        # [Vercauteren09]
        num = (sigma_x_sq * (F - G))[random_labels == 1]
        den = (sigma_x_sq * sq_norm_grad_G + sigma_i_sq)[random_labels == 1]
        # This is $J^{P}$ in eq. 4 [Vercauteren09]
        expected = (-1 * np.array(grad_G))
        expected[random_labels == 1, 0] *= num / den
        expected[random_labels == 1, 1] *= num / den
        expected[random_labels == 1, 2] *= num / den
        expected[random_labels == 0, ...] = 0

        # Now compute it using the implementation under test
        actual = np.empty_like(expected, dtype=floating)

        ssd.compute_ssd_demons_step_3d(delta_field,
                                       np.array(grad_G, dtype=floating),
                                       sigma_x_sq,
                                       actual)

        assert_array_almost_equal(actual, expected)
