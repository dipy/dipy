"""
================================================
Applying image-based deformations to streamlines
================================================

This example shows how to register streamlines into a template space by
applying non-rigid deformations.

At times we will be interested in bringing a set of streamlines into some
common, reference space to compute statistics out of the registered
streamlines. For a discussion on the effects of spatial normalization
approaches on tractography the work by Green et al. [Greene17]_ can be read.

For brevity, we will include in this example only streamlines going through
the corpus callosum connecting left to right superior frontal cortex. The
process of tracking and finding these streamlines is fully demonstrated in
the :ref:`streamline_tools` example. If this example has been run, we can read
the streamlines from file. Otherwise, we'll run that example first, by
importing it. This provides us with all of the variables that were created in
that example.

In order to get the deformation field, we will first use two b0 volumes. Both
moving and static images are assumed to be in RAS. The first one will be the
b0 from the Stanford HARDI dataset:

"""

import numpy as np
import nibabel as nib
import os.path as op

if not op.exists('lr-superiorfrontal.trk'):
    from streamline_tools import *
    vox_size = hardi_img.header.get_zooms()[0]
else:
    from dipy.core.gradients import gradient_table
    from dipy.data import get_fnames
    from dipy.io.gradients import read_bvals_bvecs
    from dipy.io.image import load_nifti_data, load_nifti, save_nifti

    hardi_fname, hardi_bval_fname, hardi_bvec_fname = get_fnames('stanford_hardi')

    data, affine, hardi_img = load_nifti(hardi_fname, return_img=True)
    vox_size = hardi_img.header.get_zooms()[0]
    bvals, bvecs = read_bvals_bvecs(hardi_bval_fname, hardi_bvec_fname)
    gtab = gradient_table(bvals, bvecs)

"""
The second one will be the T2-contrast MNI template image, which we'll need to
reslice to 2x2x2 mm isotropic voxel resolution to match the HARDI data.

"""

from dipy.data.fetcher import (fetch_mni_template, read_mni_template)
from dipy.align.reslice import reslice

fetch_mni_template()
img_t2_mni = read_mni_template("a", contrast="T2")

new_zooms = (2., 2., 2.)
data2, affine2 = reslice(np.asarray(img_t2_mni.dataobj), img_t2_mni.affine,
                         img_t2_mni.header.get_zooms(), new_zooms)
img_t2_mni = nib.Nifti1Image(data2, affine=affine2)

"""
We filter the diffusion data from the Stanford HARDI dataset to find the b0
images.

"""

b0_idx_stanford = np.where(gtab.b0s_mask)[0]
b0_data_stanford = data[..., b0_idx_stanford]

"""
We then remove the skull from them:

"""

from dipy.segment.mask import median_otsu

b0_masked_stanford, _ = median_otsu(b0_data_stanford,
                vol_idx=list(range(b0_data_stanford.shape[-1])),
                median_radius=4, numpass=4)

"""
And go on to compute the Stanford HARDI dataset mean b0 image.

"""

mean_b0_masked_stanford = np.mean(b0_masked_stanford, axis=3,
                                  dtype=data.dtype)


"""
We will register the mean b0 to the MNI T2 image template non-rigidly to
obtain the deformation field that will be applied to the streamlines. This is
just one of the strategies that can be used to obtain an appropriate
deformation field. Other strategies include computing an FA template map as
the static image, and registering the FA map of the moving image to it. This
may may eventually lead to results with improved accuracy, since a T2-contrast
template image as the target for normalization does not provide optimal tissue
contrast for maximal SyN performance.

We will first perform an affine registration to roughly align the two volumes:

"""

from dipy.align.imaffine import (MutualInformationMetric, AffineRegistration,
                                 transform_origins)
from dipy.align.transforms import (TranslationTransform3D, RigidTransform3D,
                                   AffineTransform3D)

static = np.asarray(img_t2_mni.dataobj)
static_affine = img_t2_mni.affine
moving = mean_b0_masked_stanford
moving_affine = hardi_img.affine

"""
We estimate an affine that maps the origin of the moving image to that of the
static image. We can then use this later to account for the offsets of each
image.

"""

affine_map = transform_origins(static, static_affine, moving, moving_affine)

"""
We specify the mismatch metric:

"""

nbins = 32
sampling_prop = None
metric = MutualInformationMetric(nbins, sampling_prop)

"""
As well as the optimization strategy:

"""

level_iters = [10, 10, 5]
sigmas = [3.0, 1.0, 0.0]
factors = [4, 2, 1]
affine_reg = AffineRegistration(metric=metric, level_iters=level_iters,
                                sigmas=sigmas, factors=factors)
transform = TranslationTransform3D()

params0 = None
translation = affine_reg.optimize(static, moving, transform, params0,
                                  static_affine, moving_affine)
transformed = translation.transform(moving)
transform = RigidTransform3D()

rigid_map = affine_reg.optimize(static, moving, transform, params0,
                                static_affine, moving_affine,
                                starting_affine=translation.affine)
transformed = rigid_map.transform(moving)
transform = AffineTransform3D()

"""
We bump up the iterations to get a more exact fit:

"""

affine_reg.level_iters = [1000, 1000, 100]
highres_map = affine_reg.optimize(static, moving, transform, params0,
                                  static_affine, moving_affine,
                                  starting_affine=rigid_map.affine)
transformed = highres_map.transform(moving)


"""
We now perform the non-rigid deformation using the Symmetric Diffeomorphic
Registration (SyN) Algorithm proposed by Avants et al. [Avants09]_ (also
implemented in the ANTs software [Avants11]_):

"""

from dipy.align.imwarp import SymmetricDiffeomorphicRegistration
from dipy.align.metrics import CCMetric

metric = CCMetric(3)
level_iters = [10, 10, 5]
sdr = SymmetricDiffeomorphicRegistration(metric, level_iters)

mapping = sdr.optimize(static, moving, static_affine, moving_affine,
                       highres_map.affine)
warped_moving = mapping.transform(moving)

"""
We show the registration result with:

"""

from dipy.viz import regtools

regtools.overlay_slices(static, warped_moving, None, 0, 'Static', 'Moving',
                        'transformed_sagittal.png')
regtools.overlay_slices(static, warped_moving, None, 1, 'Static', 'Moving',
                        'transformed_coronal.png')
regtools.overlay_slices(static, warped_moving, None, 2, 'Static', 'Moving',
                        'transformed_axial.png')

"""
.. figure:: transformed_sagittal.png
   :align: center
.. figure:: transformed_coronal.png
   :align: center
.. figure:: transformed_axial.png
   :align: center

   Deformable registration result.

"""

"""
We read the streamlines from file in voxel space:

"""

from dipy.io.streamline import load_tractogram

sft = load_tractogram('lr-superiorfrontal.trk', 'same')


"""
We then apply the obtained deformation field to the streamlines. Note that the
process can be sensitive to image orientation and voxel resolution. Thus, we
first apply the non-rigid warping and simultaneously apply a computed rigid
affine transformation whose extents must be corrected to account for the
different voxel grids of the moving and static images.

"""

from dipy.tracking.streamline import deform_streamlines

# Create an isocentered affine
target_isocenter = np.diag(np.array([-vox_size, vox_size, vox_size, 1]))

# Take the off-origin affine capturing the extent contrast between the mean b0
# image and the template
origin_affine = affine_map.affine.copy()

"""
In order to align the FOV of the template and the mirror image of the
streamlines, we first need to flip the sign on the x-offset and y-offset so
that we get the mirror image of the forward deformation field.

We need to use the information about the origin offsets (i.e. between the
static and moving images) that we obtained using :meth:`transform_origins`:

"""

origin_affine[0][3] = -origin_affine[0][3]
origin_affine[1][3] = -origin_affine[1][3]

"""
:meth:`transform_origins` returns this affine transformation with (1, 1, 1)
zooms and not (2, 2, 2), which means that the offsets need to be scaled by 2.
Thus, we scale z by the voxel size:

"""

origin_affine[2][3] = origin_affine[2][3]/vox_size

"""
But when scaling the z-offset, we are also implicitly scaling the y-offset as
well (by 1/2).Thus we need to correct for this by only scaling the y by the
square of the voxel size (1/4, and not 1/2):

"""

origin_affine[1][3] = origin_affine[1][3]/vox_size**2

# Apply the deformation and correct for the extents
mni_streamlines = deform_streamlines(
    sft.streamlines, deform_field=mapping.get_forward_field(),
    stream_to_current_grid=target_isocenter,
    current_grid_to_world=origin_affine, stream_to_ref_grid=target_isocenter,
    ref_grid_to_world=np.eye(4))

"""
We display the original streamlines and the registered streamlines:

"""

from dipy.viz import has_fury


def show_template_bundles(bundles, show=True, fname=None):

    scene = window.Scene()
    template_actor = actor.slicer(static)
    scene.add(template_actor)

    lines_actor = actor.streamtube(bundles, window.colors.orange,
                                   linewidth=0.3)
    scene.add(lines_actor)

    if show:
        window.show(scene)
    if fname is not None:
        window.record(scene, n_frames=1, out_path=fname, size=(900, 900))


if has_fury:

    from fury import actor, window

    show_template_bundles(mni_streamlines, show=False,
                          fname='streamlines_DSN_MNI.png')

    """
    .. figure:: streamlines_DSN_MNI.png
       :align: center

       Streamlines before and after registration.

    The corpus callosum bundles have been deformed to adapt to the MNI
    template space.

    """

"""
Finally, we save the registered streamlines:

"""

from dipy.io.stateful_tractogram import Space, StatefulTractogram
from dipy.io.streamline import save_tractogram

sft = StatefulTractogram(mni_streamlines, img_t2_mni, Space.RASMM)

save_tractogram(sft, 'mni-lr-superiorfrontal.trk', bbox_valid_check=False)


"""
References
----------

.. [Avants09] Avants, B. B., Epstein, C. L., Grossman, M., & Gee, J. C. (2009).
   Symmetric Diffeomorphic Image Registration with Cross-Correlation:
   Evaluating Automated Labeling of Elderly and Neurodegenerative Brain, 12(1),
   26-41.

.. [Avants11] Avants, B. B., Tustison, N., & Song, G. (2011). Advanced
   Normalization Tools (ANTS), 1-35.

.. [Greene17] Greene, C., Cieslak, M., & Grafton, S. T. (2017). Effect of
   different spatial normalization approaches on tractography and structural
   brain networks. Network Neuroscience, 1-19.

.. include:: ../links_names.inc

"""
