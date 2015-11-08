"""
==========================================
Symmetric Diffeomorphic Registration in 3D
==========================================
This example explains how to register 3D volumes using the Symmetric Normalization 
(SyN) algorithm proposed by Avants et al. [Avants09]_ (also implemented in
the ANTS software [Avants11]_)

We will register two 3D volumes from the same modality using SyN with the Cross
Correlation (CC) metric.
"""

import numpy as np
import nibabel as nib
from dipy.align.imwarp import SymmetricDiffeomorphicRegistration
from dipy.align.imwarp import DiffeomorphicMap
from dipy.align.metrics import CCMetric
import os.path
from dipy.viz import regtools

"""
Let's fetch two b0 volumes, the first one will be the b0 from the Stanford
HARDI dataset 
"""

from dipy.data import fetch_stanford_hardi, read_stanford_hardi
fetch_stanford_hardi()
nib_stanford, gtab_stanford = read_stanford_hardi()
stanford_b0 = np.squeeze(nib_stanford.get_data())[..., 0]

"""
The second one will be the same b0 we used for the 2D registration tutorial
"""

from dipy.data.fetcher import fetch_syn_data, read_syn_data
fetch_syn_data()
nib_syn_t1, nib_syn_b0 = read_syn_data()
syn_b0 = np.array(nib_syn_b0.get_data())

"""
We first remove the skull from the b0's
"""

from dipy.segment.mask import median_otsu
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
As we did in the 2D example, we would like to visualize (some slices of) the two
volumes by overlapping them over two channels of a color image. To do that we
need them to be sampled on the same grid, so let's first re-sample the moving
image on the static grid. We create an AffineMap to transform the moving image
towards the static image
"""

from dipy.align.imaffine import AffineMap
affine_map = AffineMap(pre_align,
                       static.shape, static_affine,
                       moving.shape, moving_affine)

resampled = affine_map.transform(moving)

"""
plot the overlapped middle slices of the volumes
"""

regtools.overlay_slices(static, resampled, None, 1, 'Static', 'Moving', 'input_3d.png')

"""
.. figure:: input_3d.png
   :align: center

   **Static image in red on top of the pre-aligned moving image (in green)**.
"""

"""
We want to find an invertible map that transforms the moving image into the
static image. We will use the Cross Correlation metric
"""

metric = CCMetric(3)

"""
Now we define an instance of the registration class. The SyN algorithm uses
a multi-resolution approach by building a Gaussian Pyramid. We instruct the
registration object to perform at most [n_0, n_1, ..., n_k] iterations at
each level of the pyramid. The 0-th level corresponds to the finest resolution.  
"""

level_iters = [10, 10, 5]
sdr = SymmetricDiffeomorphicRegistration(metric, level_iters)

"""
Execute the optimization, which returns a DiffeomorphicMap object,
that can be used to register images back and forth between the static and moving
domains. We provide the pre-aligning matrix that brings the moving image closer
to the static image
"""

mapping = sdr.optimize(static, moving, static_affine, moving_affine, pre_align)

"""
Now let's warp the moving image and see if it gets similar to the static image
"""

warped_moving = mapping.transform(moving)

"""
We plot the overlapped middle slices
"""

regtools.overlay_slices(static, warped_moving, None, 1, 'Static', 'Warped moving', 'warped_moving.png')

"""
.. figure:: warped_moving.png
   :align: center

   **Moving image transformed under the (direct) transformation in green on top of the static image (in red)**.
"""

"""
And we can also apply the inverse mapping to verify that the warped static image
is similar to the moving image 
"""

warped_static = mapping.transform_inverse(static)
regtools.overlay_slices(warped_static, moving, None, 1, 'Warped static', 'Moving', 'warped_static.png')

"""
.. figure:: warped_static.png
   :align: center

   **Static image transformed under the (inverse) transformation in red on top of the moving image (in green). Note that the moving image has lower resolution**.

.. [Avants09] Avants, B. B., Epstein, C. L., Grossman, M., & Gee, J. C. (2009). Symmetric Diffeomorphic Image Registration with Cross- Correlation: Evaluating Automated Labeling of Elderly and Neurodegenerative Brain, 12(1), 26-41.
.. [Avants11] Avants, B. B., Tustison, N., & Song, G. (2011). Advanced Normalization Tools ( ANTS ), 1-35.

.. include:: ../links_names.inc

"""
