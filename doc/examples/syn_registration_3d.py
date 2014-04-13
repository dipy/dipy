"""
==========================================
Symmetric Diffeomorphic Registration in 3D
==========================================
This example explains how to register 3D volumes using the Symmetric Normalization 
(SyN) algorithm proposed by Avants et al. [citation needed] (also implemented in
the ANTS software [citation needed])

We'll register two 3D volumes from different modalities (FA computed from
diffusion MRI and T1) using SyN with the Cross Correlation (CC) metric.
"""

import numpy as np
import matplotlib.pyplot as plt
import nibabel as nib
from dipy.align.imwarp import SymmetricDiffeomorphicRegistration
from dipy.align.imwarp import DiffeomorphicMap
from dipy.align.metrics import CCMetric
import os.path

"""
Let's fetch two b0 volumes, the first one will be the standard Sherbrooke DWI
data
"""

from dipy.data import fetch_stanford_hardi, read_stanford_hardi
fetch_stanford_hardi()
nib_stanford, gtab_stanford = read_stanford_hardi()
stanford_b0 = np.squeeze(nib_stanford.get_data())[..., 0]

"""
The second one will be the b0 we used for the 2D registration tutorial
"""

from dipy.data.fetcher import fetch_syn_data, read_syn_data
from dipy.segment.mask import median_otsu
fetch_syn_data()
nib_syn_t1, nib_syn_b0 = read_syn_data()
syn_b0 = np.array(nib_syn_b0.get_data())

"""
We first remove the skull from the b0's
"""

stanford_b0_masked, stanford_b0_mask = median_otsu(stanford_b0, 4, 4)
syn_b0_masked, syn_b0_mask = median_otsu(syn_b0, 4, 4)

static = stanford_b0_masked
static_affine = nib_stanford.get_affine()
moving = syn_b0_masked
moving_affine = nib_syn_b0.get_affine()

"""
Suppose we have already done a linear registration to roughly align the two
images
"""

pre_align = np.array([[1.02783543e+00, -4.83019053e-02, -6.07735639e-02, -2.57654118e+00],
                      [4.34051706e-03, 9.41918267e-01, -2.66525861e-01, 3.23579799e+01],
                      [5.34288908e-02, 2.90262026e-01, 9.80820307e-01, -1.46216651e+01],
                      [0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 1.00000000e+00]])

"""
Let's first re-sample the moving image on the static grid
"""

import dipy.align.vector_fields as vfu

transform = np.linalg.inv(moving_affine).dot(pre_align.dot(static_affine))
resampled = vfu.warp_volume_affine(moving.astype(np.float32), 
                                   np.asarray(static.shape, dtype=np.int32), 
                                   transform)
resampled = np.asarray(resampled)

def plot_middle_slices(V, fname=None):
    V = np.asarray(V, dtype = np.float64)
    sh=V.shape
    V = 255 * (V - V.min()) / (V.max() - V.min())
    axial = np.asarray(V[sh[0]//2, :, :]).astype(np.uint8).T
    coronal = np.asarray(V[:, sh[1]//2, :]).astype(np.uint8).T
    sagital = np.asarray(V[:, :, sh[2]//2]).astype(np.uint8).T

    plt.figure()
    plt.subplot(1, 3, 1).set_axis_off()
    plt.imshow(axial, cmap = plt.cm.gray, origin='lower')
    plt.title('Axial')
    plt.subplot(1, 3, 2).set_axis_off()
    plt.imshow(coronal, cmap = plt.cm.gray, origin='lower')
    plt.title('Coronal')
    plt.subplot(1, 3, 3).set_axis_off()
    plt.imshow(sagital, cmap = plt.cm.gray, origin='lower')
    plt.title('Sagittal')
    if fname is not None:
        from time import sleep
        sleep(1)
        plt.savefig(fname, bbox_inches='tight')


def overlay_middle_slices_coronal(L, R, ltitle='Left', rtitle='Right', fname=None):
    L = np.asarray(L, dtype = np.float64)
    R = np.asarray(R, dtype = np.float64)
    L = 255 * (L - L.min()) / (L.max() - L.min())
    R = 255 * (R - R.min()) / (R.max() - R.min())
    sh = L.shape
    colorImage = np.zeros(shape = (sh[2], sh[0], 3), dtype = np.uint8)
    ll = np.asarray(L[:, sh[1]//2, :]).astype(np.uint8).T
    rr = np.asarray(R[:, sh[1]//2, :]).astype(np.uint8).T
    colorImage[..., 0] = ll * (ll > ll[0, 0])
    colorImage[..., 1] = rr * (rr > rr[0, 0])

    plt.figure()
    plt.subplot(1, 3, 1).set_axis_off()
    plt.imshow(ll, cmap = plt.cm.gray, origin = 'lower')
    plt.title(ltitle)
    plt.subplot(1, 3, 2).set_axis_off()
    plt.imshow(colorImage, origin = 'lower')
    plt.title('Overlay')
    plt.subplot(1, 3, 3).set_axis_off()
    plt.imshow(rr, cmap = plt.cm.gray, origin = 'lower')
    plt.title(rtitle)
    if fname is not None:
        from time import sleep
        sleep(1)
        plt.savefig(fname, bbox_inches = 'tight')

overlay_middle_slices_coronal(static, resampled, 'Static', 'Moving', 'input_3d.png')

"""
.. figure:: input_3d.png
    :align: center

**Static image in red on top of the pre-aligned moving image (in green)**.
"""

"""
We want to find an invertible map that transforms the moving image into the
static image. Let's use the Cross Correlation metric, since it works well
for mono-modal and some multi-modal registration tasks.
"""

metric = CCMetric(3)

"""
Now we define an instance of the optimizer of the metric. The SyN algorithm uses
a multi-resolution approach by building a Gaussian Pyramid. We instruct the
optimizer to perform at most [n_0, n_1, ..., n_k] iterations at each level of
the pyramid. The 0-th level corresponds to the finest resolution.  
"""

opt_iter = [5, 10, 10]
registration_optimizer = SymmetricDiffeomorphicRegistration(metric, opt_iter)

"""
Execute the optimization, which returns a DiffeomorphicMap object,
that can be used to register images back and forth between the static and moving
domains
"""

mapping = registration_optimizer.optimize(static, moving, 
                                          static_affine, moving_affine, pre_align)

"""
Now let's warp the moving image and see if it gets similar to the static image
"""

warped_moving = mapping.transform(moving)

"""
To visually check the overlap of the static image with the transformed moving
image, we can plot them on top of each other with different channels to see
where the differences are located
"""

overlay_middle_slices_coronal(static, warped_moving, 'Static', 'Warped moving', 'warped_moving.png')

"""
.. figure:: warped_moving.png
    :align: center

**Moving image transformed under the (direct) transformation in green
on top of the static image (in red)**.
"""

"""
And we can also apply the inverse mapping to verify that the warped static image
is similar to the moving image 
"""

warped_static = mapping.transform_inverse(static)

overlay_middle_slices_coronal(warped_static, moving, 'Warped static', 'Moving', 'warped_static.png')

"""
.. figure:: warped_static.png
    :align: center

**Static image transformed under the (inverse) transformation in red
on top of the moving image (in green). Note that the moving image has lower 
resolution**.
"""
