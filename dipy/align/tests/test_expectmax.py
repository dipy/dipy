import numpy as np
from dipy.align import floating
import dipy.align.expectmax as em
from numpy.testing import (assert_equal,
                           assert_almost_equal, 
                           assert_array_equal,
                           assert_array_almost_equal)




def test_compute_em_demons_step_2d():
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
    import dipy.align.expectmax as em
    from numpy.testing import (assert_equal,
                           assert_almost_equal, 
                           assert_array_equal,
                           assert_array_almost_equal)

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
    delta_field =  G - F
    actual = np.empty_like(expected, dtype=floating)
    em.compute_em_demons_step_2d(np.array(delta_field, dtype=floating),
                                 np.array(sigma_i_sq, dtype=floating),
                                 np.array(grad_G, dtype=floating),
                                 sigma_x_sq,
                                 actual)
    
    assert_array_almost_equal(actual, expected)


def test_compute_em_demons_step_3d():
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
    delta_field =  G - F
    actual = np.empty_like(expected, dtype=floating)
    em.compute_em_demons_step_3d(np.array(delta_field, dtype=floating),
                                 np.array(sigma_i_sq, dtype=floating),
                                 np.array(grad_G, dtype=floating),
                                 sigma_x_sq,
                                 actual)
    
    assert_array_almost_equal(actual, expected)

def test_quantize_positive_image():
    num_levels = 11 # an arbitrary number of quantization levels
    img_shape = (15, 20) # arbitrary test image shape (must contain at least 3 elements)
    min_positive = 0.1
    max_positive = 1.0
    epsilon = 1e-8

    delta = (max_positive - min_positive + epsilon) / (num_levels - 1)
    true_levels = np.zeros((num_levels,), dtype = np.float32)
    # put the intensities at the centers of the bins
    true_levels[1:] = np.linspace(min_positive+delta*0.5, max_positive-delta*0.5, num_levels-1)
    true_quantization = np.empty(img_shape, dtype = np.int32) # generate a target quantization image
    random_labels = np.random.randint(0, num_levels, np.size(true_quantization))
    
    # make sure there is at least one element equal to 0, 1 and num_levels-1
    random_labels[0] = 0
    random_labels[1] = 1
    random_labels[2] = num_levels-1
    true_quantization[...] = random_labels.reshape(img_shape)

    noise_amplitude = np.min([delta / 4.0, min_positive / 4.0]) # make sure additive nose doesn't change the quantization result
    noise = np.random.ranf(np.size(true_quantization)).reshape(img_shape) * noise_amplitude
    noise = noise.astype(floating)
    input_image = np.ndarray(img_shape, dtype = floating)
    input_image[...] = true_levels[true_quantization] + noise # assign intensities plus noise
    input_image[true_quantization == 0] = 0 # preserve original zeros
    input_image[true_quantization == 1] = min_positive # preserve min positive value
    input_image[true_quantization == num_levels-1] = max_positive # preserve max positive value

    out, levels, hist = em.quantize_positive_image(input_image, num_levels)
    levels = np.asarray(levels)
    assert_array_equal(out, true_quantization)
    assert_array_almost_equal(levels, true_levels)
    for i in range(num_levels):
        current_bin = np.asarray(true_quantization == i).sum()
        assert_equal(hist[i], current_bin)

def test_quantize_positive_volume():
    num_levels = 11 # an arbitrary number of quantization levels
    img_shape = (5, 10, 15) # arbitrary test image shape (must contain at least 3 elements)
    min_positive = 0.1
    max_positive = 1.0
    epsilon = 1e-8

    delta = (max_positive - min_positive + epsilon) / (num_levels - 1)
    true_levels = np.zeros((num_levels,), dtype = np.float32)
    # put the intensities at the centers of the bins
    true_levels[1:] = np.linspace(min_positive+delta*0.5, max_positive-delta*0.5, num_levels-1)
    true_quantization = np.empty(img_shape, dtype = np.int32) # generate a target quantization image
    random_labels = np.random.randint(0, num_levels, np.size(true_quantization))
    
    # make sure there is at least one element equal to 0, 1 and num_levels-1
    random_labels[0] = 0
    random_labels[1] = 1
    random_labels[2] = num_levels-1
    true_quantization[...] = random_labels.reshape(img_shape)

    noise_amplitude = np.min([delta / 4.0, min_positive / 4.0]) # make sure additive nose doesn't change the quantization result
    noise = np.random.ranf(np.size(true_quantization)).reshape(img_shape) * noise_amplitude
    noise = noise.astype(floating)
    input_image = np.ndarray(img_shape, dtype = floating)
    input_image[...] = true_levels[true_quantization] + noise # assign intensities plus noise
    input_image[true_quantization == 0] = 0 # preserve original zeros
    input_image[true_quantization == 1] = min_positive # preserve min positive value
    input_image[true_quantization == num_levels-1] = max_positive # preserve max positive value

    out, levels, hist = em.quantize_positive_volume(input_image, num_levels)
    levels = np.asarray(levels)
    assert_array_equal(out, true_quantization)
    assert_array_almost_equal(levels, true_levels)
    for i in range(num_levels):
        current_bin = np.asarray(true_quantization == i).sum()
        assert_equal(hist[i], current_bin)


def test_compute_masked_image_class_stats():
    shape = (32, 32)
    
    #Create random labels
    labels = np.ndarray(shape, dtype=np.int32)
    labels[...] = np.random.randint(0, 10, np.size(labels)).reshape(shape)

    #Create random values
    values = np.random.randn(shape[0], shape[1]).astype(floating)
    values *= labels
    values += labels

    expected_means = [values[labels == i].mean() for i in range(10)] 
    expected_vars = [values[labels == i].var() for i in range(10)] 

    mask = np.ones(shape, dtype = np.int32)
    means, vars = em.compute_masked_image_class_stats(mask, values, 10, labels)
    assert_almost_equal(means, expected_means)
    assert_almost_equal(vars, expected_vars)

def test_compute_masked_volume_class_stats():
    shape = (32, 32, 32)
    
    #Create random labels
    labels = np.ndarray(shape, dtype=np.int32)
    labels[...] = np.random.randint(0, 10, np.size(labels)).reshape(shape)

    #Create random values
    values = np.random.randn(shape[0], shape[1], shape[2]).astype(floating)
    values *= labels
    values += labels

    expected_means = [values[labels == i].mean() for i in range(10)] 
    expected_vars = [values[labels == i].var() for i in range(10)] 

    mask = np.ones(shape, dtype = np.int32)
    means, vars = em.compute_masked_volume_class_stats(mask, values, 10, labels)
    assert_almost_equal(means, expected_means)
    assert_almost_equal(vars, expected_vars)


if __name__=='__main__':
    test_compute_em_demons_step_2d()
    test_compute_em_demons_step_3d()
    test_quantize_positive_image()
    test_quantize_positive_volume()
    test_compute_masked_image_class_stats()
    test_compute_masked_volume_class_stats()

