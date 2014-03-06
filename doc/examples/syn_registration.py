"""
====================================
Symmetric Diffeomorphic Registration
====================================
This examples explain how to register 2D images and 3D volumes using
the Symmetric Normalization (SyN) algorithm proposed by Avants et al.
[citation needed] (also implemented in the ANTS [citation needed] software)

The first example shows how to register two 2D images. We will use the classic
Circle-To-C experiment for diffeomorphic registration
"""
import numpy as np
from dipy.align import floating
import dipy.align.imwarp as imwarp
import dipy.align.metrics as metrics 
from dipy.data import get_data
import matplotlib.pyplot as plt
import dipy.align.registration_common as rcommon


fname_moving = get_data('reg_o')
fname_static = get_data('reg_c')

moving = plt.imread(fname_moving)
static = plt.imread(fname_static)

"""
Symmetric diffeomorphisms are represented as displacement fields (one 
displacement vector for each voxel), which may consume significant amount 
of memory. It is generaly a good idea to use 32-bit precision to save memory.
The 'floating' type is defined in the init file of the align module
"""

moving = np.array(moving[:, :, 0], dtype = floating)
static = np.array(static[:, :, 0], dtype = floating)

"""
It is a good practice to normalize the input images to the same dynamic
range (unless there is a specific reason not to)
"""

moving = (moving-moving.min())/(moving.max() - moving.min())
static = (static-static.min())/(static.max() - static.min())

"""
The first decision we need to make is what similarity metric is appropriate
for our problem. In this example we are using two binary images, so the Sum
of Squared Differences (SSD) is a good choice. We create a metric specifying
2 as the dimension of our images' domain
"""

metric = metrics.SSDMetric(dim = 2) 

"""
Now we define an instance of the optimizer of the metric. The SyN algorithm uses
a multi-resolution approach by building a Gaussian Pyramid. We instruct the
optimizer to perform at most [n_0, n_1, ..., n_k] iterations at each level of
the pyramid. The 0-th level corresponds to the finest resolution.  
"""

opt_iter = [20, 100, 100, 100]
optimizer = imwarp.SymmetricDiffeomorphicRegistration(metric, opt_iter)

"""
Now we execute the optimization, which returns a DiffeomorphicMap object,
that can be used to register images back and forth between the static and moving
domains
"""

mapping = optimizer.optimize(static, moving, None)

"""
It is a good idea to visualize the resulting deformation map to make sure the
result is reasonable (at least, visually) 
"""

def drawLattice2D(nrows, ncols, delta):
    lattice=np.ndarray((1+(delta+1)*nrows, 1+(delta+1)*ncols), dtype=floating)
    lattice[...]=127
    for i in range(nrows+1):
        lattice[i*(delta+1), :]=0
    for j in range(ncols+1):
        lattice[:, j*(delta+1)]=0
    return lattice

def plot_2d_diffeomorphic_map(mapping, delta=10, fname = None):
    #Create a grid on the moving domain
    nrows_moving = mapping.forward.shape[0]
    ncols_moving = mapping.forward.shape[1]
    X1,X0=np.mgrid[0:nrows_moving, 0:ncols_moving]
    lattice_moving=drawLattice2D((nrows_moving+delta)/(delta+1), (ncols_moving+delta)/(delta+1), delta)
    lattice_moving=lattice_moving[0:nrows_moving, 0:ncols_moving]
    #Warp in the forward direction (since the lattice is in the moving domain)
    warped_forward = mapping.transform(lattice_moving,'tri')

    #Create a grid on the static domain
    nrows_static = mapping.backward.shape[0]
    ncols_static = mapping.backward.shape[1]
    X1,X0=np.mgrid[0:nrows_static, 0:ncols_static]
    lattice_static=drawLattice2D((nrows_static+delta)/(delta+1), (ncols_static+delta)/(delta+1), delta)
    lattice_static=lattice_static[0:nrows_static, 0:ncols_static]
    #Warp in the backward direction (since the lattice is in the static domain)
    warped_backward = mapping.transform_inverse(lattice_static,'tri')

    #Now plot the grids
    plt.figure()
    plt.subplot(1, 3, 1)
    plt.imshow(warped_forward, cmap=plt.cm.gray)
    plt.title('Direct transform')
    plt.subplot(1, 3, 2)
    plt.imshow(lattice_moving, cmap=plt.cm.gray)
    plt.title('Original grid')
    plt.subplot(1, 3, 3)
    plt.imshow(warped_backward, cmap=plt.cm.gray)
    plt.title('Inverse transform')
    if fname is not None:
      from time import sleep
      sleep(1)
      savefig(fname)

plot_2d_diffeomorphic_map(mapping, 10, 'diffeomorphic_map.png')

"""
.. figure:: diffeomorphic_map.png
:align: center

**Defformed lattice under the resulting diffeomorhic map**.
"""
