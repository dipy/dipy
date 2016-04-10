"""
====================
Tracking Quick Start
====================

This example shows how to perform fiber tracking using Dipy.

We will use Constrained Spherical Deconvolution (CSD) [Tournier07]_ for local
reconstructions and then generate deterministic streamlines using the fiber
directions (peaks) from CSD and fractional anisotropic (FA) as a
stopping criterion.

Let's load the necessary modules.
"""

import numpy as np

from dipy.reconst.dti import TensorModel, fractional_anisotropy
from dipy.reconst.csdeconv import (ConstrainedSphericalDeconvModel,
                                   auto_response)
from dipy.direction import peaks_from_model
from dipy.tracking.eudx import EuDX
from dipy.data import fetch_stanford_hardi, read_stanford_hardi, get_sphere
from dipy.segment.mask import median_otsu
from dipy.viz import fvtk
from dipy.viz.colormap import line_colors

"""
Load one of the available datasets with 150 gradients on the sphere and 10 b0s
"""

fetch_stanford_hardi()
img, gtab = read_stanford_hardi()

data = img.get_data()

"""
Create a brain mask. This dataset is a bit difficult to segment with the
default ``median_otsu`` parameters (see :ref:`example_brain_extraction_dwi`)
therefore we use here a bit more advanced options.
"""

maskdata, mask = median_otsu(data, 3, 1, False,
                             vol_idx=range(10, 50), dilate=2)

"""
For the constrained spherical deconvolution we need to estimate the response
function (see :ref:`example_reconst_csd`) and create a model.
"""

response, ratio = auto_response(gtab, data, roi_radius=10, fa_thr=0.7)

csd_model = ConstrainedSphericalDeconvModel(gtab, response)

"""
Next, we use ``peaks_from_model`` to fit the data and calculated the fiber
directions in all voxels.
"""

sphere = get_sphere('symmetric724')

csd_peaks = peaks_from_model(model=csd_model,
                             data=data,
                             sphere=sphere,
                             mask=mask,
                             relative_peak_threshold=.5,
                             min_separation_angle=25,
                             parallel=True)

"""
For the tracking part, we will use the fiber directions from the ``csd_model``
but stop tracking in areas where fractional anisotropy (FA) is low (< 0.1).
To derive the FA, used here as a stopping criterion, we would need to fit a
tensor model first. Here, we fit the Tensor using weighted least squares (WLS).
"""

tensor_model = TensorModel(gtab, fit_method='WLS')
tensor_fit = tensor_model.fit(data, mask)

FA = fractional_anisotropy(tensor_fit.evals)

"""
In order for the stopping values to be used with our tracking algorithm we need
to have the same dimensions as the ``csd_peaks.peak_values``. For this reason,
we can assign the same FA value to every peak direction in the same voxel in
the following way.
"""

stopping_values = np.zeros(csd_peaks.peak_values.shape)
stopping_values[:] = FA[..., None]

"""
For quality assurance we can also visualize a slice from the direction field
which we will use as the basis to perform the tracking.
"""

ren = fvtk.ren()

slice_no = data.shape[2] / 2

fvtk.add(ren, fvtk.peaks(csd_peaks.peak_dirs[:, :, slice_no:slice_no + 1],
                         stopping_values[:, :, slice_no:slice_no + 1]))

print('Saving illustration as csd_direction_field.png')
fvtk.record(ren, out_path='csd_direction_field.png', size=(900, 900))

"""
.. figure:: csd_direction_field.png
 :align: center

 **Direction Field (peaks)**

``EuDX`` [Garyfallidis12]_ is a fast algorithm that we use here to generate
streamlines. If the parameter ``seeds`` is a positive integer it will generate
that number of randomly placed seeds everywhere in the volume. Alternatively,
you can specify the exact seed points using an array (N, 3) where N is the
number of seed points. For simplicity, here we will use the first option
(random seeds). ``a_low`` is the threshold of the fist parameter
(``stopping_values``) which means that there will that tracking will stop in
regions with FA < 0.1.
"""

streamline_generator = EuDX(stopping_values,
                            csd_peaks.peak_indices,
                            seeds=10**4,
                            odf_vertices=sphere.vertices,
                            a_low=0.1)

streamlines = [streamline for streamline in streamline_generator]

"""
We can visualize the streamlines using ``fvtk.line`` or ``fvtk.streamtube``.
"""

fvtk.clear(ren)

fvtk.add(ren, fvtk.line(streamlines, line_colors(streamlines)))

print('Saving illustration as csd_streamlines_eudx.png')
fvtk.record(ren, out_path='csd_streamlines_eudx.png', size=(900, 900))

"""
.. figure:: csd_streamlines_eudx.png
 :align: center

 **CSD-based streamlines using EuDX**

We used above ``fvtk.record`` because we want to create a figure for the
tutorial but you can visualize the same objects in 3D using
``fvtk.show(ren)``.

To learn more about this process you could start playing with the number of
seed points or, even better, specify seeds to be in specific regions of interest
in the brain.

``fvtk`` gives some minimal interactivity however you can save the resulting
streamlines in a Trackvis (.trk) format and load them for example with the
Fibernavigator_ or another tool for medical visualization.

Finally, let's save the streamlines as a (.trk) file and FA as a Nifti image.
"""

import nibabel as nib

hdr = nib.trackvis.empty_header()
hdr['voxel_size'] = img.get_header().get_zooms()[:3]
hdr['voxel_order'] = 'LAS'
hdr['dim'] = FA.shape[:3]

csd_streamlines_trk = ((sl, None, None) for sl in streamlines)

csd_sl_fname = 'csd_streamline.trk'

nib.trackvis.write(csd_sl_fname, csd_streamlines_trk, hdr, points_space='voxel')

nib.save(nib.Nifti1Image(FA, img.get_affine()), 'FA_map.nii.gz')

"""

In Windows if you get a runtime error about frozen executable please start
your script by adding your code above in a ``main`` function and use:

if __name__ == '__main__':
    import multiprocessing
    multiprocessing.freeze_support()
    main()

.. [Garyfallidis12] Garyfallidis E., "Towards an accurate brain tractography", PhD thesis, University of Cambridge, 2012.
.. [Tournier07] J-D. Tournier, F. Calamante and A. Connelly, "Robust determination of the fibre orientation distribution in diffusion MRI: Non-negativity constrained super-resolved spherical deconvolution", Neuroimage, vol. 35, no. 4, pp. 1459-1472, 2007.

.. NOTE::
    Dipy has a new and very modular fiber tracking machinery. Our new machinery
    for fiber tracking is featured in the example :ref:`example_tracking_quick_start`.


.. include:: ../links_names.inc

"""
