"""
==========================================
Affine Registration in 3D
==========================================
This example explains how to compute an affine transformation to register two 3D 
volumes by maximization of their Mutual Information [Mattes03]_. The optimization
strategy is similar to that implemented in ANTS [Avants11]_.
"""

import numpy as np
import nibabel as nib
import os.path
from dipy.viz import regtools
from dipy.data import fetch_stanford_hardi, read_stanford_hardi
from dipy.data.fetcher import fetch_syn_data, read_syn_data
from dipy.segment.mask import median_otsu
from dipy.align.imaffine import (aff_centers_of_mass, 
                                 aff_warp,
                                 MattesMIMetric,
                                 AffineRegistration)
from dipy.align.transforms import regtransforms

"""
Let's fetch two b0 volumes, the static image will be the b0 from the Stanford
HARDI dataset 
"""

fetch_stanford_hardi()
nib_stanford, gtab_stanford = read_stanford_hardi()
static = np.squeeze(nib_stanford.get_data())[..., 0]
static_grid2space = nib_stanford.get_affine()

"""
Now the moving image
"""

fetch_syn_data()
nib_syn_t1, nib_syn_b0 = read_syn_data()
moving = np.array(nib_syn_b0.get_data())
moving_grid2space = nib_syn_b0.get_affine()

"""
We can obtain a very rough (and fast) registration by just aligning the centers of mass
of the two images
"""

c_of_mass = aff_centers_of_mass(static, static_grid2space, moving, moving_grid2space)

"""
We can now warp the moving image and draw in on top of the static image, registration
is not likely to be good, but at least they will occupy roughly the same space
"""

warped = aff_warp(static, static_grid2space, moving, moving_grid2space, c_of_mass)
regtools.overlay_slices(static, warped, None, 0, "Static", "Warped", "warped_com.png")

"""
.. figure:: warped_com.png
   :align: center

   **Registration result by simply aligning the centers of mass of the images**.
"""

"""
This was just a translation of the moving image towards the static image, now we will
refine it by looking for an affine transform. We first create the similarity metric
(Mutual Information) to be used. We need to specify the number of bins to be used to
discretize the joint and marginal probability distribution functions (PDF), a typical
value is 32. We also need to specify the percentage (an integer in (0, 100])of voxels
to be used for computing the PDFs, the most accurate registration will be obtained by
using all of the voxels, but it is also the most time-consuming choice. Here we will
use full sampling by passing None instead
"""

nbins = 32
sampling_pc = None
metric = MattesMIMetric(nbins, sampling_pc)

"""
To avoid getting stuck at local optima, and to accelerate convergence, we use a
multi-resolution strategy (similar to ANTS [Avants11]_) by building a Gaussian Pyramid.
To have as much flexibility as possible, the user can specify how this Gaussian Pyramid
is built. First of all, we need to specify how many resolutions we want to use. This is
indirectly specified by just providing a list of the number of iterations we want to
perform at each resolution. Here we will just specify 3 resolutions and a large number 
of iterations, 10000 at the coarsest resolution, 1000 at the medium resolution and 100
at the finest. We also provide a tolerance for the optimization method. This are the 
default settings
"""

level_iters = [10000, 1000, 100]
opt_tol = 1e-5

"""
To compute the Gaussian pyramid, the original image is first smoothed at each level 
of the pyramid using a Gaussian kernel with the requested sigma. A good initial choice 
is [3.0, 1.0, 0.0], this is the default
"""

sigmas = [3.0, 1.0, 0.0]

"""
Now we specify the sub-sampling factors. A good configuration is [4, 2, 1], which means
that, if the original image shape was (nx, ny, nz) voxels, then the shape of the coarsest
image will be about (nx//4, ny//4, nz//4), the shape in the middle resolution will be
about (nx//2, ny//2, nz//2) and the image at the finest scale has the same size as the
original image. This set of factors is the default
"""

factors = [4, 2, 1]

"""
Now we go ahead and instantiate the registration class with the configuration we just
prepared
"""

affreg = AffineRegistration(metric=metric,
                            level_iters=level_iters,
                            opt_tol=opt_tol,
                            factors = factors,
                            sigmas = sigmas)
                                
"""
Using AffineRegistration we can register our images in as many stages as we want, providing
previous results as initialization for the next (the same logic as in ANTS). The reason why
it is useful is that registration is a non-convex optimization problem (it may have more
than one local optima), which means that it is very important to initialize as close to the
solution as possible. For example, lets start with our rough transformation aligning
the centers of mass of our images, and then refine it in three stages. First look for an 
optimal translation
"""

transform = regtransforms[('TRANSLATION', 3)]
x0 = None
prealign = c_of_mass
trans = affreg.optimize(static, moving, transform, x0,
                        static_grid2space, moving_grid2space,
                        prealign=prealign)

"""
If we look at the result, we can see that this translation is much better than simply
aligning the centers of mass
"""

warped = aff_warp(static, static_grid2space, moving, moving_grid2space, trans)
regtools.overlay_slices(static, warped, None, 0, "Static", "Warped", "warped_trans.png")

"""
.. figure:: warped_trans.png
   :align: center

   **Registration result by just translating the moving image, using Mutual Information**.
"""

"""
Now lets refine with a rigid transform (this may even modify our previously found
optimal translation)
"""

transform = regtransforms[('RIGID', 3)]
x0 = None
prealign = trans
rigid = affreg.optimize(static, moving, transform, x0,
                        static_grid2space, moving_grid2space,
                        prealign=prealign)

"""
This produces a slight rotation, and the images are now better aligned
"""

warped = aff_warp(static, static_grid2space, moving, moving_grid2space, rigid)
regtools.overlay_slices(static, warped, None, 0, "Static", "Warped", "warped_rigid.png")

"""
.. figure:: warped_rigid.png
   :align: center

   **Registration result with a rigid transform, using Mutual Information**.
"""

"""
Finally, lets refine with a full affine transform (translation, rotation, scale and 
shear), it is safer to fit more degrees of freadom now, since we must be very close 
to the optimal transform
"""

transform = regtransforms[('AFFINE', 3)]
x0 = None
prealign = rigid
affine = affreg.optimize(static, moving, transform, x0,
                         static_grid2space, moving_grid2space,
                         prealign=prealign)

"""
This results in a slight shear and scale
"""

warped = aff_warp(static, static_grid2space, moving, moving_grid2space, affine)
regtools.overlay_slices(static, warped, None, 0, "Static", "Warped", "warped_affine.png")

"""
.. figure:: warped_affine.png
   :align: center

   **Registration result with an affine transform, using Mutual Information**.
"""

"""
The equivalent ANTS command to perform all above operations is

antsRegistration -d 3 -r [ static_name, moving_name, 1 ] \
                      -m mattes[ static_name, moving_name, 1 , 32, 1] \
                      -t translation[ 0.1 ] \
                      -c [ 10000x111110x11110,1.e-8,20 ] \
                      -s 4x2x1vox \
                      -f 3x2x1 -l 1 \
                      -m mattes[ static_name, moving_name, 1 , 32, 1] \
                      -t rigid[ 0.1 ] \
                      -c [ 10000x111110x11110,1.e-8,20 ] \
                      -s 4x2x1vox \
                      -f 3x2x1 -l 1 \
                      -m mattes[ static_name, moving_name, 1 , 32, 1] \
                      -t affine[ 0.1 ] \
                      -c [ 10000x111110x11110,1.e-8,20 ] \
                      -s 4x2x1vox \
                      -f 3x2x1 -l 1 \
                      -o [out_name]
                      
which, after converting to RAS coordinate system (ANTS operates on LPS) yields the 
following transform
"""

ants_align = np.array([[1.02783543e+00, -4.83019053e-02, -6.07735639e-02, -2.57654118e+00],
                       [4.34051706e-03, 9.41918267e-01, -2.66525861e-01, 3.23579799e+01],
                       [5.34288908e-02, 2.90262026e-01, 9.80820307e-01, -1.46216651e+01],
                       [0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 1.00000000e+00]])

"""
After a visual inspection of the results, we can see that they are very similar
"""

ants_warped = aff_warp(static, static_grid2space, moving, moving_grid2space, ants_align)
regtools.overlay_slices(warped, ants_warped, None, 0, 'Dipy', 'ANTS', 'dipy_ants_0.png')

"""
.. figure:: dipy_ants_0.png
   :align: center

   **Aligned image using Dipy (in red) on top of the aligned image using ANTS (in green)**.

.. [Mattes03] Mattes, D., Haynor, D. R., Vesselle, H., Lewellen, T. K., & Eubank, W. (2003). PET-CT image registration in the chest using free-form deformations. IEEE Transactions on Medical Imaging, 22(1), 120-8.
.. [Avants11] Avants, B. B., Tustison, N., & Song, G. (2011). Advanced Normalization Tools ( ANTS ), 1-35.

.. include:: ../links_names.inc

"""
