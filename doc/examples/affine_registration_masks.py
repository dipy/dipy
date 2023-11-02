"""
==============================
Affine Registration with Masks
==============================

This example explains how to compute a transformation to register two 3D
volumes by maximization of their Mutual Information [Mattes03]_. The
optimization strategy is similar to that implemented in ANTS [Avants11]_.

We will use masks to define which pixels are used in the Mutual Information.
Masking can also be done for registration of 2D images rather than 3D volumes.

Masking for registration is useful in a variety of circumstances. For example,
in cardiac MRI, where it is usually used to specify a region of interest on a
2D static image, e.g., the left ventricle in a short axis slice. This
prioritizes registering the region of interest over other structures that move
with respect to the heart.

"""

from os.path import join as pjoin
import numpy as np
import matplotlib.pyplot as plt
from dipy.viz import regtools
from dipy.data import fetch_stanford_hardi
from dipy.io.image import load_nifti
from dipy.align.imaffine import (AffineMap,
                                 MutualInformationMetric,
                                 AffineRegistration)
from dipy.align.transforms import (TranslationTransform3D,
                                   RigidTransform3D)

from dipy.align import (affine_registration, translation,
                        rigid, register_series)


###############################################################################
# Let's fetch a single b0 volume from the Stanford HARDI dataset.

files, folder = fetch_stanford_hardi()
static_data, static_affine, static_img = load_nifti(
                                            pjoin(folder, 'HARDI150.nii.gz'),
                                            return_img=True)
static = np.squeeze(static_data)[..., 0]

# pad array to help with this example
pad_by = 10
static = np.pad(static, [(pad_by, pad_by), (pad_by, pad_by), (0, 0)],
                mode='constant', constant_values=0)

static_grid2world = static_affine

###############################################################################
# Let's create a moving image by transforming the static image.

affmat = np.eye(4)
affmat[0, -1] = 4
affmat[1, -1] = 12
theta = 0.1
c, s = np.cos(theta), np.sin(theta)
affmat[0:2, 0:2] = np.array([[c, -s], [s, c]])
affine_map = AffineMap(affmat,
                       static.shape, static_grid2world,
                       static.shape, static_grid2world)
moving = affine_map.transform(static)
moving_affine = static_affine.copy()
moving_grid2world = static_grid2world.copy()

regtools.overlay_slices(static, moving, None, 2,
                        "Static", "Moving", "deregistered.png")

###############################################################################
# .. rst-class:: centered small fst-italic fw-semibold
#
# Same images but misaligned.
#
#
# Let's make some registration settings.

nbins = 32
sampling_prop = None
metric = MutualInformationMetric(nbins, sampling_prop)

# small number of iterations for this example
level_iters = [100, 10]
sigmas = [1.0, 0.0]
factors = [2, 1]

affreg = AffineRegistration(metric=metric,
                            level_iters=level_iters,
                            sigmas=sigmas,
                            factors=factors)

###############################################################################
# Now let's register these volumes together without any masking. For the
# purposes of this example, we will not provide an initial transformation
# based on centre of mass, but this would work fine with masks.
#
# Note that use of masks is not currently implemented for sparse sampling.

transform = TranslationTransform3D()
transl = affreg.optimize(static, moving, transform, None,
                         static_grid2world, moving_grid2world,
                         starting_affine=None,
                         static_mask=None, moving_mask=None)
transform = RigidTransform3D()
transl = affreg.optimize(static, moving, transform, None,
                         static_grid2world, moving_grid2world,
                         starting_affine=transl.affine,
                         static_mask=None, moving_mask=None)
transformed = transl.transform(moving)

transformed = transl.transform(moving)
regtools.overlay_slices(static, transformed, None, 2,
                        "Static", "Transformed", "transformed.png")


###############################################################################
# .. rst-class:: centered small fst-italic fw-semibold
#
# Registration result.
#
#
#
# We can also use a pipeline to achieve the same thing. For convenience in this
# tutorial, we will define a function that runs the pipeline and makes a figure.


def reg_func(figname, static_mask=None, moving_mask=None):
    """Convenience function for registration using a pipeline.
       Uses variables in global scope, except for static_mask and moving_mask.
    """

    pipeline = [translation, rigid]

    xformed_img, reg_affine = affine_registration(
        moving,
        static,
        moving_affine=moving_affine,
        static_affine=static_affine,
        nbins=32,
        metric='MI',
        pipeline=pipeline,
        level_iters=level_iters,
        sigmas=sigmas,
        factors=factors,
        static_mask=static_mask,
        moving_mask=moving_mask)

    regtools.overlay_slices(static, xformed_img, None, 2,
                            "Static", "Transformed", figname)


###############################################################################
# Now we can run this function and hopefully get the same result.

reg_func("transformed_pipeline.png")

###############################################################################
# .. rst-class:: centered small fst-italic fw-semibold
#
# Registration result using pipeline.
#
#
# Now let's modify the images in order to test masking. We will place three
# squares in the corners of both images, but in slightly different locations.
#
# We will make masks that cover these regions but with an extra border of
# pixels. This is because the masks need transforming and resampling during
# optimization, and we want to make sure that we are definitely covering the
# troublesome features.

sz = 15
pd = 10

# modify images
val = static.max()/2.0

static[-sz-pd:-pd, -sz-pd:-pd, :] = val
static[pd:sz+pd, -sz-pd:-pd, :] = val
static[-sz-pd:-pd, pd:sz+pd, :] = val

moving[pd:sz+pd, pd:sz+pd, :] = val
moving[pd:sz+pd, -sz-pd:-pd, :] = val
moving[-sz-pd:-pd, pd:sz+pd, :] = val

# create masks
squares_st = np.zeros_like(static).astype(np.int32)
squares_mv = np.zeros_like(static).astype(np.int32)

squares_st[-sz-1-pd:-pd, -sz-1-pd:-pd, :] = 1
squares_st[pd:sz+1+pd, -sz-1-pd:-pd, :] = 1
squares_st[-sz-1-pd:-pd, pd:sz+1+pd, :] = 1

squares_mv[pd:sz+1+pd, pd:sz+1+pd, :] = 1
squares_mv[pd:sz+1+pd, -sz-1-pd:-pd, :] = 1
squares_mv[-sz-1-pd:-pd, pd:sz+1+pd, :] = 1


regtools.overlay_slices(static, moving, None, 2,
                        "Static", "Moving", "deregistered_squares.png")

###############################################################################
# .. rst-class:: centered small fst-italic fw-semibold
#
# Same images but misaligned, with white squares in the corners.

static_mask = np.abs(squares_st - 1)
moving_mask = np.abs(squares_mv - 1)

fig, ax = plt.subplots(1, 2)
ax[0].imshow(static_mask[:, :, 1].T, cmap="gray", origin="lower")
ax[0].set_title("static image mask")
ax[1].imshow(moving_mask[:, :, 1].T, cmap="gray", origin="lower")
ax[1].set_title("moving image mask")
plt.savefig("masked_static.png", bbox_inches='tight')


###############################################################################
# .. rst-class:: centered small fst-italic fw-semibold
#
# The masks.
#
#
#
# Let's try to register these new images without a mask.

reg_func("transformed_squares.png")

###############################################################################
# .. rst-class:: centered small fst-italic fw-semibold
#
# Registration fails to align the images because the squares pin the images.
#
#
#
# Now we will attempt to register the images using the masks that we defined.
#
# First, use a mask on the static image. Only pixels where the mask is non-zero
# in the static image will contribute to Mutual Information.

reg_func("transformed_squares_mask.png", static_mask=static_mask)

###############################################################################
# .. rst-class:: centered small fst-italic fw-semibold
#
# Registration result using a static mask.
#
#
# We can also attempt the same thing use a moving image mask.

reg_func("transformed_squares_mask_2.png", moving_mask=moving_mask)

###############################################################################
# .. rst-class:: centered small fst-italic fw-semibold
#
# Registration result using a moving mask.
#
#
# And finally, we can use both masks at the same time.

reg_func("transformed_squares_mask_3.png",
         static_mask=static_mask, moving_mask=moving_mask)


###############################################################################
# .. rst-class:: centered small fst-italic fw-semibold
#
# Registration result using both a static mask and a moving mask.
#
#
#
# In most use cases, it is likely that only a static mask will be required,
# e.g., to register a series of images to a single static image.
#
# Let's make a series of volumes to demonstrate this idea, and register the
# series to the first image in the series using a static mask:

series = np.stack([static, moving, moving], axis=-1)

pipeline = [translation, rigid]
xformed, _ = register_series(series, 0, pipeline,
                             series_affine=moving_affine,
                             static_mask=static_mask)

regtools.overlay_slices(np.squeeze(xformed[..., 0]),
                        np.squeeze(xformed[..., -2]),
                        None, 2, "Static", "Moving 1", "series_mask_1.png")

regtools.overlay_slices(np.squeeze(xformed[..., 0]),
                        np.squeeze(xformed[..., -1]),
                        None, 2, "Static", "Moving 2", "series_mask_2.png")

###############################################################################
# .. rst-class:: centered small fst-italic fw-semibold
#
# Registration of series using a static mask.
#
#
#
# In all of the examples above, different masking choices achieved essentially
# the same result, but in general the results may differ depending on
# differences between the static and moving images.
#
#
# References
# ----------
#
# .. [Mattes03] Mattes, D., Haynor, D. R., Vesselle, H., Lewellen, T. K.,
#               Eubank, W. (2003). PET-CT image registration in the chest using
#               free-form deformations. IEEE Transactions on Medical Imaging,
#               22(1), 120-8.
# .. [Avants11] Avants, B. B., Tustison, N., & Song, G. (2011). Advanced
#               Normalization Tools (ANTS), 1-35.
