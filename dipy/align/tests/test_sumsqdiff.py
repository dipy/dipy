import numpy as np
from dipy.align import floating
from numpy.testing import (assert_equal,
                           assert_almost_equal, 
                           assert_array_equal,
                           assert_array_almost_equal,
                           assert_raises,
                           assert_allclose)
import dipy.align.sumsqdiff as ssd

def test_solve_2d_symmetric_positive_definite():
    # Select some arbitrary right-hand sides
    bs = [np.array([1.1, 2.2]),
          np.array([1e-2, 3e-3]),
          np.array([1e2, 1e3]),
          np.array([1e-5, 1e5])]

    # Select arbitrary symmetric positive-definite matrices
    As = []

    # Identity
    As.append(np.array([1.0, 0.0, 1.0]))

    # Small determinant
    As.append(np.array([1e-3, 1e-4, 1e-3]))
    
    # Large determinant
    As.append(np.array([1e6, 1e4, 1e6]))

    actual = np.zeros(2, dtype=np.float64)

    for A in As:
        AA = np.array([[A[0], A[1]],[A[1], A[2]]])
        det = np.linalg.det(AA)
        for b in bs:
            expected = np.linalg.solve(AA, b)
            ssd.solve_2d_symmetric_positive_definite(A, b, det, actual)
            assert_allclose(expected, actual, rtol = 1e-9, atol = 1e-9)


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
    gs.append(np.array([0.0, 0.0, 0.0]))

    # canonical basis
    gs.append(np.array([1.0, 0.0, 0.0]))
    gs.append(np.array([0.0, 1.0, 0.0]))
    gs.append(np.array([0.0, 0.0, 1.0]))
    
    # other
    gs.append(np.array([1.0, 0.5, 0.0]))
    gs.append(np.array([0.0, 0.2, 0.1]))
    gs.append(np.array([0.3, 0.0, 0.9]))

    actual = np.zeros(3)
    for g in gs:
        A = g[:,None]*g[None,:]
        for tau in taus:
            AA = A + tau * np.eye(3)
            for b in bs:
                is_singular = ssd.solve_3d_symmetric_positive_definite(g, b, tau, actual)

                if tau == 0.0:
                    assert_equal(is_singular, 1)
                else:
                    expected = np.linalg.solve(AA, b)
                    assert_allclose(expected, actual, rtol = 1e-9, atol = 1e-9)


def test_compute_energy_ssd_2d():
    sh = (32, 32)
    
    #Select arbitrary centers
    c_f = np.asarray(sh)/2
    c_g = c_f + 0.5

    #Compute the identity vector field I(x) = x in R^2
    x_0 = np.asarray(range(sh[0]))
    x_1 = np.asarray(range(sh[1]))
    X = np.ndarray(sh + (2,), dtype = np.float64)
    O = np.ones(sh)
    X[...,0]= x_0[:, None] * O
    X[...,1]= x_1[None, :] * O

    #Compute the gradient fields of F and G
    grad_F = X - c_f
    grad_G = X - c_g 

    #Compute F and G
    F = 0.5*np.sum(grad_F**2,-1)
    G = 0.5*np.sum(grad_G**2,-1)

    # Note: this should include the energy corresponding to the 
    # regularization term, but it is discarded in ANTS (they just
    # consider the data term, which is not the objective function
    # being optimized). This test case should be updated after
    # further investigation
    expected = ((F - G)**2).sum()
    actual = ssd.compute_energy_ssd_2d(np.array(F-G, dtype = floating))
    assert_almost_equal(expected, actual)
    

def test_compute_energy_ssd_3d():
    sh = (32, 32, 32)
    
    #Select arbitrary centers
    c_f = np.asarray(sh)/2
    c_g = c_f + 0.5

    #Compute the identity vector field I(x) = x in R^2
    x_0 = np.asarray(range(sh[0]))
    x_1 = np.asarray(range(sh[1]))
    x_2 = np.asarray(range(sh[2]))
    X = np.ndarray(sh + (3,), dtype = np.float64)
    O = np.ones(sh)
    X[...,0]= x_0[:, None, None] * O
    X[...,1]= x_1[None, :, None] * O
    X[...,2]= x_2[None, None, :] * O

    #Compute the gradient fields of F and G
    grad_F = X - c_f
    grad_G = X - c_g 

    #Compute F and G
    F = 0.5*np.sum(grad_F**2,-1)
    G = 0.5*np.sum(grad_G**2,-1)

    # Note: this should include the energy corresponding to the 
    # regularization term, but it is discarded in ANTS (they just
    # consider the data term, which is not the objective function
    # being optimized). This test case should be updated after
    # further investigating
    expected = ((F - G)**2).sum()
    actual = ssd.compute_energy_ssd_3d(np.array(F-G, dtype = floating))
    assert_almost_equal(expected, actual)


def test_compute_ssd_demons_step_2d():
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
    #Select arbitrary images' shape (same shape for both images)
    sh = (20, 10)
    
    #Select arbitrary centers
    c_f = np.asarray(sh)/2
    c_g = c_f + 0.5

    #Compute the identity vector field I(x) = x in R^2
    x_0 = np.asarray(range(sh[0]))
    x_1 = np.asarray(range(sh[1]))
    X = np.ndarray(sh + (2,), dtype = np.float64)
    O = np.ones(sh)
    X[...,0]= x_0[:, None] * O
    X[...,1]= x_1[None, :] * O

    #Now select an arbitrary parameter for $\sigma_x$ (eq 4 in [Vercauteren09])
    sigma_x_sq = 1.5

    #Compute the gradient fields of F and G
    np.random.seed(1137271)

    grad_F = X - c_f
    grad_G = X - c_g 

    Fnoise = np.random.ranf(np.size(grad_F)).reshape(grad_F.shape) * grad_F.max() * 0.1
    Fnoise = Fnoise.astype(floating)
    grad_F += Fnoise

    Gnoise = np.random.ranf(np.size(grad_G)).reshape(grad_G.shape) * grad_G.max() * 0.1
    Gnoise = Gnoise.astype(floating)
    grad_G += Gnoise

    #The squared norm of grad_G to be used later
    sq_norm_grad_G = np.sum(grad_G**2,-1) 

    #Compute F and G
    F = 0.5*np.sum(grad_F**2,-1)
    G = 0.5*sq_norm_grad_G

    Fnoise = np.random.ranf(np.size(F)).reshape(F.shape) * F.max() * 0.1
    Fnoise = Fnoise.astype(floating)
    F += Fnoise

    Gnoise = np.random.ranf(np.size(G)).reshape(G.shape) * G.max() * 0.1
    Gnoise = Gnoise.astype(floating)
    G += Gnoise

    delta_field =  np.array(G - F, dtype = floating)

    #Select some pixels to force gradient = 0 and F=G
    random_labels = np.random.randint(0, 2, sh[0]*sh[1])
    random_labels = random_labels.reshape(sh)

    F[random_labels == 0] = G[random_labels == 0]
    delta_field[random_labels == 0] = 0
    grad_G[random_labels == 0, ...] = 0
    sq_norm_grad_G[random_labels == 0, ...] = 0
    
    #Set arbitrary values for $\sigma_i$ (eq. 4 in [Vercauteren09])
    #The original Demons algorithm used simply |F(x) - G(x)| as an
    #estimator, so let's use it as well
    sigma_i_sq = (F - G)**2
     
    #Directly compute the demons step according to eq. 4 in [Vercauteren09]
    num = (sigma_x_sq * (F - G))[random_labels == 1]
    den = (sigma_x_sq * sq_norm_grad_G + sigma_i_sq)[random_labels == 1]
    expected = (-1 * np.array(grad_G)) #This is $J^{P}$ in eq. 4 [Vercauteren09]
    expected[random_labels == 1, 0] *= num / den
    expected[random_labels == 1, 1] *= num / den
    expected[random_labels == 0, ...] = 0

    #Now compute it using the implementation under test
    actual = np.empty_like(expected, dtype=floating)
    
    ssd.compute_ssd_demons_step_2d(delta_field,
                                   np.array(grad_G, dtype=floating),
                                   sigma_x_sq,
                                   actual)
    
    assert_array_almost_equal(actual, expected)


def test_compute_ssd_demons_step_3d():
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

    #Select arbitrary images' shape (same shape for both images)
    sh = (20, 15, 10)
    
    #Select arbitrary centers
    c_f = np.asarray(sh)/2
    c_g = c_f + 0.5

    #Compute the identity vector field I(x) = x in R^2
    x_0 = np.asarray(range(sh[0]))
    x_1 = np.asarray(range(sh[1]))
    x_2 = np.asarray(range(sh[2]))
    X = np.ndarray(sh + (3,), dtype = np.float64)
    O = np.ones(sh)
    X[...,0]= x_0[:, None, None] * O
    X[...,1]= x_1[None, :, None] * O
    X[...,2]= x_2[None, None, :] * O

    #Now select an arbitrary parameter for $\sigma_x$ (eq 4 in [Vercauteren09])
    sigma_x_sq = 1.5

    #Compute the gradient fields of F and G
    np.random.seed(1137271)

    grad_F = X - c_f
    grad_G = X - c_g 

    Fnoise = np.random.ranf(np.size(grad_F)).reshape(grad_F.shape) * grad_F.max() * 0.1
    Fnoise = Fnoise.astype(floating)
    grad_F += Fnoise

    Gnoise = np.random.ranf(np.size(grad_G)).reshape(grad_G.shape) * grad_G.max() * 0.1
    Gnoise = Gnoise.astype(floating)
    grad_G += Gnoise

    #The squared norm of grad_G to be used later
    sq_norm_grad_G = np.sum(grad_G**2,-1) 

    #Compute F and G
    F = 0.5*np.sum(grad_F**2,-1)
    G = 0.5*sq_norm_grad_G

    Fnoise = np.random.ranf(np.size(F)).reshape(F.shape) * F.max() * 0.1
    Fnoise = Fnoise.astype(floating)
    F += Fnoise

    Gnoise = np.random.ranf(np.size(G)).reshape(G.shape) * G.max() * 0.1
    Gnoise = Gnoise.astype(floating)
    G += Gnoise

    delta_field =  np.array(G - F, dtype = floating)

    #Select some pixels to force gradient = 0 and F=G
    random_labels = np.random.randint(0, 2, sh[0]*sh[1]*sh[2])
    random_labels = random_labels.reshape(sh)

    F[random_labels == 0] = G[random_labels == 0]
    delta_field[random_labels == 0] = 0
    grad_G[random_labels == 0, ...] = 0
    sq_norm_grad_G[random_labels == 0, ...] = 0

    #Set arbitrary values for $\sigma_i$ (eq. 4 in [Vercauteren09])
    #The original Demons algorithm used simply |F(x) - G(x)| as an
    #estimator, so let's use it as well
    sigma_i_sq = (F - G)**2
     
    #Directly compute the demons step according to eq. 4 in [Vercauteren09]
    num = (sigma_x_sq * (F - G))[random_labels == 1]
    den = (sigma_x_sq * sq_norm_grad_G + sigma_i_sq)[random_labels == 1]
    expected = (-1 * np.array(grad_G)) #This is $J^{P}$ in eq. 4 [Vercauteren09]
    expected[random_labels == 1, 0] *= num / den
    expected[random_labels == 1, 1] *= num / den
    expected[random_labels == 1, 2] *= num / den
    expected[random_labels == 0, ...] = 0

    #Now compute it using the implementation under test
    actual = np.empty_like(expected, dtype=floating)

    ssd.compute_ssd_demons_step_3d(delta_field,
                                   np.array(grad_G, dtype = floating),
                                   sigma_x_sq,
                                   actual)

    assert_array_almost_equal(actual, expected)


if __name__=='__main__':
    test_compute_energy_ssd_2d()
    test_compute_energy_ssd_3d()
    test_compute_ssd_demons_step_2d()
    test_compute_ssd_demons_step_3d()
