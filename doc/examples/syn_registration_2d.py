"""
==========================================
Symmetric Diffeomorphic Registration in 2D
==========================================
This example explains how to register 2D images using the Symmetric Normalization 
(SyN) algorithm proposed by Avants et al. [1] (also implemented in
the ANTS software [2])

We will perform the classic Circle-To-C experiment for diffeomorphic registration
"""

import numpy as np
import matplotlib.pyplot as plt
from dipy.data import get_data
from dipy.align.imwarp import SymmetricDiffeomorphicRegistration
from dipy.align.metrics import SSDMetric, CCMetric, EMMetric
import dipy.align.imwarp as imwarp


fname_moving = get_data('reg_o')
fname_static = get_data('reg_c')

moving = plt.imread(fname_moving)
static = plt.imread(fname_static)

"""
We need to use 2D scalar images, so let's take only the first channel of
the RGB images (in this case the three channels are equal)
"""

moving = np.array(moving[:, :, 0])
static = np.array(static[:, :, 0])

moving = (moving - moving.min()) / (moving.max() - moving.min())
static = (static - static.min()) / (static.max() - static.min())

"""
To visually check the overlap of the static image with the transformed moving
image, we can plot them on top of each other with different channels to see
where the differences are located
"""

def overlay_images(img0, img1, title0='', title_mid='', title1='', fname=None):
    #Normalize the input images to [0,255]
    img0 = 255*((img0 - img0.min()) / (img0.max() - img0.min()))
    img1 = 255*((img1 - img1.min()) / (img1.max() - img1.min()))
    img0_red=np.zeros(shape=(img0.shape) + (3,), dtype=np.uint8)
    img1_green=np.zeros(shape=(img0.shape) + (3,), dtype=np.uint8)
    overlay=np.zeros(shape=(img0.shape) + (3,), dtype=np.uint8)
    img0_red[..., 0] = img0
    img1_green[..., 1] = img1
    overlay[..., 0]=img0
    overlay[..., 1]=img1
    plt.figure()
    plt.subplot(1, 3, 1).set_axis_off()
    plt.imshow(img0_red)
    plt.title(title0)
    plt.subplot(1, 3, 2).set_axis_off()
    plt.imshow(overlay)
    plt.title(title_mid)
    plt.subplot(1, 3, 3).set_axis_off()
    plt.imshow(img1_green)
    plt.title(title1)
    if fname is not None:
      from time import sleep
      sleep(1)
      plt.savefig(fname, bbox_inches='tight')

overlay_images(static, moving, 'Static', 'Overlay', 'Moving', 'input_images.png')

"""
.. figure:: input_images.png
   :align: center

**Input images**.
"""

"""
We want to find an invertible map that transforms the moving image (circle)
into the static image (the C letter)

The first decision we need to make is what similarity metric is appropriate
for our problem. In this example we are using two binary images, so the Sum
of Squared Differences (SSD) is a good choice. We create a metric specifying
2 as the dimension of our images' domain
"""

metric = SSDMetric(dim = 2, step_type = 'gauss_newton') 

"""
Now we define an instance of the optimizer of the metric. The SyN algorithm uses
a multi-resolution approach by building a Gaussian Pyramid. We instruct the
optimizer to perform at most [n_0, n_1, ..., n_k] iterations at each level of
the pyramid. The 0-th level corresponds to the finest resolution.  
"""

opt_iter = [25, 50, 100, 200]

optimizer = SymmetricDiffeomorphicRegistration(metric, opt_iter, inv_iter = 40)

"""
Now we execute the optimization, which returns a DiffeomorphicMap object,
that can be used to register images back and forth between the static and moving
domains
"""

mapping = optimizer.optimize(static, moving)

"""
It is a good idea to visualize the resulting deformation map to make sure the
result is reasonable (at least, visually) 
"""

def draw_lattice_2d(nrows, ncols, delta):
    lattice=np.ndarray((1 + (delta + 1) * nrows, 1 + (delta + 1) * ncols), dtype = np.float64)
    lattice[...] = 127
    for i in range(nrows + 1):
        lattice[i*(delta + 1), :] = 0
    for j in range(ncols + 1):
        lattice[:, j * (delta + 1)] = 0
    return lattice

def plot_2d_diffeomorphic_map(mapping, delta = 10, fname = None):
    #Create a grid on the moving domain
    nrows_moving = mapping.forward.shape[0]
    ncols_moving = mapping.forward.shape[1]
    X1,X0 = np.mgrid[0:nrows_moving, 0:ncols_moving]
    lattice_moving=draw_lattice_2d((nrows_moving + delta) / (delta + 1), 
                                 (ncols_moving + delta) / (delta + 1), delta)
    lattice_moving=lattice_moving[0:nrows_moving, 0:ncols_moving]
    #Warp in the forward direction (since the lattice is in the moving domain)
    warped_forward = mapping.transform(lattice_moving, 'lin')

    #Create a grid on the static domain
    nrows_static = mapping.backward.shape[0]
    ncols_static = mapping.backward.shape[1]
    X1,X0 = np.mgrid[0:nrows_static, 0:ncols_static]
    lattice_static = draw_lattice_2d((nrows_static + delta) / (delta + 1), 
                                     (ncols_static + delta) / (delta + 1), delta)
    lattice_static=lattice_static[0:nrows_static, 0:ncols_static]
    #Warp in the backward direction (since the lattice is in the static domain)
    warped_backward = mapping.transform_inverse(lattice_static, 'lin')

    #Now plot the grids
    plt.figure()
    plt.subplot(1, 3, 1).set_axis_off()
    plt.imshow(warped_forward, cmap = plt.cm.gray)
    plt.title('Direct transform')
    plt.subplot(1, 3, 2).set_axis_off()
    plt.imshow(lattice_moving, cmap = plt.cm.gray)
    plt.title('Original grid')
    plt.subplot(1, 3, 3).set_axis_off()
    plt.imshow(warped_backward, cmap = plt.cm.gray)
    plt.title('Inverse transform')
    if fname is not None:
      from time import sleep
      sleep(1)
      plt.savefig(fname, bbox_inches = 'tight')

plot_2d_diffeomorphic_map(mapping, 10, 'diffeomorphic_map.png')

"""
.. figure:: diffeomorphic_map.png
   :align: center

**Deformed lattice under the resulting diffeomorphic map**.
"""

"""
Now let's warp the moving image and see if it gets similar to the static image
"""

warped_moving = mapping.transform(moving, 'lin')
overlay_images(static, warped_moving, 'Static','Overlay','Warped moving',
    'direct_warp_result.png')

"""
.. figure:: direct_warp_result.png
    :align: center

**Moving image transformed under the (direct) transformation in green
on top of the static image (in red)**.
"""

"""
And we can also apply the inverse mapping to verify that the warped static image
is similar to the moving image 
"""

warped_static = mapping.transform_inverse(static, 'lin')
overlay_images(warped_static, moving,'Warped static','Overlay','Moving', 
    'inverse_warp_result.png')

"""
.. figure:: inverse_warp_result.png
    :align: center

**Static image transformed under the (inverse) transformation in red
on top of the moving image (in green)**.
"""

"""
Now let's register a couple of slices from a B0 image using the Cross
Correlation metric. Also, let's inspect the evolution of the registration.
To do this we will define a function that will be called by the optimizer
at each stage of the optimization process. We will draw the current warped
images after finishing each resolution.
"""

def callback_CC(optimizer, status):
    #Status indicates at which stage of the optimization we currently are
    #For now, we will only react at the end of each resolution of the scale
    #space
    if status == imwarp.RegistrationStages.SCALE_END:
        #get the current images from the metric
        wmoving = optimizer.metric.moving_image
        wstatic = optimizer.metric.static_image
        #draw the images on top of each other with different colors
        overlay_images(wmoving, wstatic, 'Warped moving', 'Overlay', 'Warped static')

"""
Now we are ready to configure and run the registration. First load the data
"""

from dipy.data.fetcher import fetch_syn_data, read_syn_data
from dipy.segment.mask import median_otsu

fetch_syn_data()

t1, b0 = read_syn_data()
data = np.array(b0.get_data(), dtype = np.float64)

"""
We first remove the skull from the B0 volume
"""

b0_mask, mask = median_otsu(data, 4, 4)

"""
And select two slices to try the 2D registration
"""

static = b0_mask[:, :, 40]
moving = b0_mask[:, :, 38]

"""
After loading the data, we instantiate the Cross Correlation metric. The metric
receives three parameters: the dimension of the input images, the standard 
deviation of the Gaussian Kernel to be used to regularize the gradient and the
radius of the window to be used for evaluating the local normalized cross
correlation.
"""

sigma_diff = 3.0
radius = 4
metric = CCMetric(2, sigma_diff, radius)

"""
Let's use a scale space of 3 levels
"""

opt_iter = [25, 50, 100]
optimizer = SymmetricDiffeomorphicRegistration(metric, opt_iter)
optimizer.callback = callback_CC

"""
And execute the optimization
"""

mapping = optimizer.optimize(static, moving)

warped = mapping.transform(moving)

'''
We can see the effect of the warping by switching between the images before and
after registration
'''

overlay_images(static, moving, 'Static', 'Overlay', 'Moving',
               't1_slices_input.png')

"""
.. figure:: t1_slices_input.png
    :align: center

**Input images.**.
"""

overlay_images(static, warped, 'Static', 'Overlay', 'Warped moving',
               't1_slices_res.png')

"""
.. figure:: t1_slices_res.png
    :align: center

**Moving image transformed under the (direct) transformation in green
on top of the static image (in red)**.
"""

'''
And we can apply the inverse warping too
'''

inv_warped = mapping.transform_inverse(static)
overlay_images(inv_warped, moving, 'Warped static', 'Overlay', 'moving',
               't1_slices_res2.png')

"""
.. figure:: t1_slices_res2.png
    :align: center

**Static image transformed under the (inverse) transformation in red
on top of the moving image (in green)**.
"""

'''
Finally, let's see the deformation
'''

plot_2d_diffeomorphic_map(mapping, 5, 'diffeomorphic_map_b0s.png')

"""
.. figure:: diffeomorphic_map_b0s.png
   :align: center

**Deformed lattice under the resulting diffeomorphic map**.
"""

"""
[1] Avants, B. B., Epstein, C. L., Grossman, M., & Gee, J. C. (2009).
    Symmetric Diffeomorphic Image Registration with Cross- Correlation: 
    Evaluating Automated Labeling of Elderly and Neurodegenerative 
    Brain, 12(1), 26-41.

[2] Avants, B. B., Tustison, N., & Song, G. (2011). Advanced 
    Normalization Tools ( ANTS ), 1-35.
"""