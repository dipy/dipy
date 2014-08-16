import numpy as np
from dipy.align import floating
from numpy.testing import (assert_equal,
                           assert_almost_equal, 
                           assert_array_equal,
                           assert_array_almost_equal,
                           assert_raises)
import dipy.align.sumsqdiff as ssd


def test_compute_energy_SSD2D():
    pass


def test_compute_energy_SSD3D():
    pass


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

    #Compute the gradient fields of F and G
    grad_F = X - c_f
    grad_G = X - c_g 

    #The squared norm of grad_G to be used later
    sq_norm_grad_G = np.sum(grad_G**2,-1) 

    #Compute F and G
    F = 0.5*np.sum(grad_F**2,-1)
    G = 0.5*sq_norm_grad_G

    #Now select an arbitrary parameter for $\sigma_x$ (eq 4 in [Vercauteren09])
    sigma_x_sq = 1.5

    #Set arbitrary values for $\sigma_i$ (eq. 4 in [Vercauteren09])
    #The original Demons algorithm used simply |F(x) - G(x)| as an
    #estimator, so let's use it as well
    sigma_i_sq = (F - G)**2
     
    #Directly compute the demons step according to eq. 4 in [Vercauteren09]
    num = sigma_x_sq * (F - G) 
    den = sigma_x_sq * sq_norm_grad_G + sigma_i_sq
    expected = -1 * np.array(grad_G) #This is $J^{P}$ in eq. 4 [Vercauteren09]
    expected[...,0] *= num / den
    expected[...,1] *= num / den

    #Now compute it using the implementation under test
    delta_field =  np.array(G - F, dtype = floating)
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

    #Compute the gradient fields of F and G
    grad_F = X - c_f
    grad_G = X - c_g 

    #The squared norm of grad_G to be used later
    sq_norm_grad_G = np.sum(grad_G**2,-1) 

    #Compute F and G
    F = 0.5*np.sum(grad_F**2,-1)
    G = 0.5*sq_norm_grad_G

    #Now select an arbitrary parameter for $\sigma_x$ (eq 4 in [Vercauteren09])
    sigma_x_sq = 1.5

    #Set arbitrary values for $\sigma_i$ (eq. 4 in [Vercauteren09])
    #The original Demons algorithm used simply |F(x) - G(x)| as an
    #estimator, so let's use it as well
    sigma_i_sq = (F - G)**2
     
    #Directly compute the demons step according to eq. 4 in [Vercauteren09]
    num = sigma_x_sq * (F - G) 
    den = sigma_x_sq * sq_norm_grad_G + sigma_i_sq
    expected = -1 * np.array(grad_G) #This is $J^{P}$ in eq. 4 [Vercauteren09]
    expected[...,0] *= num / den
    expected[...,1] *= num / den
    expected[...,2] *= num / den

    #Now compute it using the implementation under test
    delta_field =  np.array(G - F, dtype = floating)
    actual = np.empty_like(expected, dtype=floating)

    ssd.compute_ssd_demons_step_3d(delta_field,
                                   np.array(grad_G, dtype = floating),
                                   sigma_x_sq,
                                   actual)

    assert_array_almost_equal(actual, expected)


if __name__=='__main__':
    #test_compute_energy_SSD2D()
    #test_compute_energy_SSD3D()
    #test_compute_ssd_demons_step_2d()
    test_compute_ssd_demons_step_3d()
