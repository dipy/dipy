"""
=======================
Slice-wise Registration
=======================

This example demonstrates how to do registration of 2D images.

In cardiac MRI, it is common to have short-axis slices of the left ventricle.
These 2D slices contain multiple acquisitions that need registering together,
but this registration must be performed on a slice-by-slice basis.

"""

import numpy as np
import matplotlib.pyplot as plt
from dipy.viz import regtools
from dipy.data import get_fnames
from dipy.io.image import load_nifti
from dipy.align.imaffine import (MutualInformationMetric,
                                 AffineRegistration)
from dipy.align.transforms import RigidTransform2D
from dipy.align import (translation, rigid,
                        register_series, register_dwi_series)
from dipy.io.gradients import read_bvals_bvecs
from dipy.core.gradients import gradient_table

"""
Let's fetch an MRI dataset.
"""

hardi_fname, hardi_bval_fname, hardi_bvec_fname = get_fnames('stanford_hardi')

data, affine = load_nifti(hardi_fname)

bvals, bvecs = read_bvals_bvecs(hardi_bval_fname, hardi_bvec_fname)
gtab = gradient_table(bvals, bvecs)

"""
This dataset is a series of 3D volumes. But for demonstration purposes,
we will pretend that we have 3 separate slices of 2D images instead.
"""

data = data[:, :, [25, 30, 35], :]

fig, ax = plt.subplots(1, 3)
for idx in range(3):
    ax[idx].imshow(data[:, :, idx, 0].T, origin='lower', cmap='gray')
plt.savefig("slices.png", bbox_inches='tight')

"""
.. figure:: slices.png
   :align: center

   Three different slices.
"""

"""
Here we will show how to register these 2D slices.

Firstly, we will setup our registration:
"""

nbins = 32
sampling_prop = None
metric = MutualInformationMetric(nbins, sampling_prop)

# small number of iterations for this example
level_iters = [1000, 100, 10]
sigmas = [3.0, 1.0, 0.0]
factors = [4, 2, 1]

affreg = AffineRegistration(metric=metric,
                            level_iters=level_iters,
                            sigmas=sigmas,
                            factors=factors)

"""
Then we can register a pair of 2D images as follows. We will choose a single
slice, and two images from that slice. Note that the arrays supplied to
the optimize function are 2D.
"""

static_img = data[:, :, 0, 0]
moving_img = data[:, :, 0, 1]
affine2d = np.eye(3)  # registration in pixel coordinates

transform = RigidTransform2D()
transl = affreg.optimize(static_img, moving_img, transform, None,
                         affine2d, affine2d)

regist_img = transl.transform(moving_img)

"""
We can plot these images as follows by supplying an extra dimension:
"""

regtools.overlay_slices(static_img[None, ...], regist_img[None, ...], None, 0,
                        "Static", "Moving", "before_reg_slice.png")

regtools.overlay_slices(static_img[None, ...], regist_img[None, ...], None, 0,
                        "Static", "Registered", "after_reg_slice.png")

"""
.. figure:: before_reg_slice.png
   :align: center

   Two slices before registration.

.. figure:: after_reg_slice.png
   :align: center

   Two slices after registration.
"""

"""
We can also use a pipeline for doing this registration, allowing us to specify
multiple transforms to be applied one after another, as shown in other
registration examples.

The public API allows us to use register these images using a single
function call, which is convenient. Here, we will register 5 images
from the second slice to the first image in the series.

If we only provide a single slice of data as the first argument, we must still
ensure that the array is 4D, for example `data[[1]]` or `data[1][None, ...]`.
The same applies to the second argument, if supplying an image instead of an
index as above.

"""

pipeline = [translation, rigid]

xformed, affines_all = register_series(data[..., 0:5], 0, pipeline,
                                       series_affine=affine, ref_affine=affine,
                                       slice_index=1)

"""
We'll plot the first image of the specified slice against the mean of the
registered images, just so we can see the result:
"""

regtools.overlay_slices(data[:, :, [1], 0], xformed.mean(-1), None, 2,
                        "Static", "Moving", "series_reg_1.png")


"""
.. figure:: series_reg_1.png
   :align: center

   Series registration.
"""

"""
Note that the returned arrays are also 4D, with the first dimension of size 1.

We can register an entire DWI series by first registering all b0 images,
followed by registering the remaining images to the average of these
registered b0 images.

The default b0 image to use for the first stage is 0, but if supplying another
value then note that this argument must index the b0 images (so the value
ranges from 0 to num_of_b0_images - 1).
"""

# 'affine' will not be used to map from pixel grid to world coordinates
x_nifti, _ = register_dwi_series(data, gtab, affine=affine,
                                 b0_ref=0, pipeline=pipeline,
                                 slice_index=1)

"""
This function returns a NIFTI image, so we extract an array before plotting.
"""

xformed = x_nifti.get_fdata()
regtools.overlay_slices(data[:, :, [1], 0], xformed.mean(-1), None, 2,
                        "Static", "Moving", "series_reg_2.png")

"""
.. figure:: series_reg_2.png
   :align: center

   Series registration.
"""
