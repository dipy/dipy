"""
=======================
Streamline Registration
=======================

This example shows how to register streamlines into a template space by
applying non-rigid deformations.

At times we will be interested in bringing a set of streamlines into some
common, reference space to compute statistics out of the registered
streamlines.

For brevity, we will include in this example only streamlines going through
the corpus callosum connecting left to right superior frontal
cortex. The process of tracking and finding these streamlines is fully
demonstrated in the :ref:`streamline_tools` example. If this example has been
run, we can read the streamlines from file. Otherwise, we'll run that example
first, by importing it. This provides us with all of the variables that were
created in that example.

In order to get the deformation field, we will first use two b0 volumes. The
first one will be the b0 from the Stanford HARDI dataset:

"""

import os.path as op

if not op.exists('lr-superiorfrontal.trk'):
    from streamline_tools import *
else:
    from dipy.data import read_stanford_hardi
    hardi_img, gtab = read_stanford_hardi()
    data = hardi_img.get_data()

"""
The second one will be the T2-contrast MNI template image.

"""

from dipy.data.fetcher import (fetch_mni_template, read_mni_template)

fetch_mni_template()
img_t2_mni = read_mni_template("a", contrast = "T2")

"""
We filter the diffusion data from the Stanford HARDI dataset to find the b0
images.

"""

import numpy as np

b0_idx_stanford = np.where(gtab.b0s_mask)[0]
b0_data_stanford = data[..., b0_idx_stanford]

"""
We then remove the skull from the b0's.

"""

from dipy.segment.mask import median_otsu

b0_masked_stanford, _ = median_otsu(b0_data_stanford,
                                    vol_idx=[i for i in range(b0_data_stanford.shape[-1])],
                                    median_radius=4, numpass=4)

"""
And go on to compute the Stanford HARDI dataset mean b0 image.

"""

mean_b0_masked_stanford = np.mean(b0_masked_stanford, axis=3,
                                  dtype=data.dtype)

"""
We will register the mean b0 to the MNI T2 image template non-rigidly to
obtain the deformation field that will be applied to the streamlines. We will
first perform an affine registration to roughly align the two volumes:

"""

from dipy.align.imaffine import (AffineMap, MutualInformationMetric,
                                 AffineRegistration)
from dipy.align.transforms import (TranslationTransform3D, RigidTransform3D,
                                   AffineTransform3D)

static = img_t2_mni.get_data()
static_affine = img_t2_mni.affine
moving = mean_b0_masked_stanford
moving_affine = hardi_img.affine

identity = np.eye(4)
affine_map = AffineMap(identity, static.shape, static_affine, moving.shape,
                       moving_affine)
resampled = affine_map.transform(moving)

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
affine_map = affine_reg.optimize(static, moving, transform, params0,
                                 static_affine, moving_affine,
                                 starting_affine=rigid_map.affine)
transformed = affine_map.transform(moving)


"""
We now perform the non-rigid deformation using the Symmetric Diffeomorphic
Registration (SyN) Algorithm:

"""

from dipy.align.imwarp import SymmetricDiffeomorphicRegistration
from dipy.align.metrics import CCMetric

metric = CCMetric(3)
level_iters = [10, 10, 5]
sdr = SymmetricDiffeomorphicRegistration(metric, level_iters)

mapping = sdr.optimize(static, moving, static_affine, moving_affine,
                       affine_map.affine)
warped_moving = mapping.transform(moving)
warped_static = mapping.transform_inverse(moving)

"""
We show the registration result with:

"""
from dipy.viz import regtools

regtools.overlay_slices(static, warped_moving, None, 0, "Static", "Moving",
                        "transformed_sagittal.png")
regtools.overlay_slices(static, warped_moving, None, 1, "Static", "Moving",
                        "transformed_coronal.png")
regtools.overlay_slices(static, warped_moving, None, 2, "Static", "Moving",
                        "transformed_axial.png")

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

from dipy.io.streamline import load_trk

streamlines, hdr = load_trk('lr-superiorfrontal.trk')


"""
We then apply the obtained deformation field to the streamlines. We first
apply the computed affine transformation, and then the non-rigid warping:

"""

from dipy.tracking.streamline import deform_streamlines

from dipy.tracking.utils import move_streamlines

rotation, scale = np.linalg.qr(affine_map.affine)
streams = move_streamlines(streamlines, rotation)
scale[0:3, 0:3] = np.dot(scale[0:3, 0:3], np.diag(1. / hdr['voxel_sizes']))
scale[0:3, 3] = abs(scale[0:3, 3])
affine_streamlines = move_streamlines(streamlines, scale)

warped_streamlines = \
    deform_streamlines(affine_streamlines,
                       deform_field=mapping.get_forward_field(),
                       stream_to_current_grid=moving_affine,
                       current_grid_to_world=mapping.codomain_grid2world,
                       stream_to_ref_grid=static_affine,
                       ref_grid_to_world=mapping.domain_grid2world)


"""
We display the original streamlines and the registered streamlines:

"""

from dipy.viz import window, actor, have_fury

from time import sleep

def show_both_bundles(bundles, colors=None, show=True, fname=None):

    renderer = window.Renderer()
    for (i, bundle) in enumerate(bundles):
        color = colors[i]
        lines_actor = actor.streamtube(bundle, color, linewidth=0.3)
        lines_actor.RotateX(-90)
        lines_actor.RotateZ(90)
        renderer.add(lines_actor)
    if show:
        window.show(renderer)
    if fname is not None:
        sleep(1)
        window.record(renderer, n_frames=1, out_path=fname, size=(900, 900))

if have_fury:
    """
    .. figure:: streamline_registration.png
       :align: center

       Streamlines before and after registration.

    As it can be seen, the corpus callosum bundles have been deformed to adapt
    to the MNI template space.

    """

    show_both_bundles([streamlines, warped_streamlines],
                      colors=[window.colors.orange, window.colors.red],
                      show=False,
                      fname='streamline_registration.png')

"""
Finally, we save the registered streamlines:

"""

from dipy.io.streamline import save_trk

save_trk("warped-lr-superiorfrontal.trk", warped_streamlines,
         affine_map.affine)
