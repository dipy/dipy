"""

.. _reconst-mcsd:

=======================================================
Reconstruction with Multi-Shell Multi-Tissue Constrained Spherical
Deconvolution
=======================================================

This example shows how to use Multi-Shell Multi-Tissue Constrained Spherical
Deconvolution (MSMT-CSD) introduced by Tournier et al. [Jeurissen2014]_. This
tutorial goes through the steps involved in implementing the method.

This method provides improved White Matter(WM), Grey Matter (GM), and
Cerebrospinal fluid (CSF) volume fraction maps, which is otherwise
overestimated in the standard CSD (SSST-CSD). This is done by using b-value
dependencies of the different tissue types to estimate ODFs. This method thus
extends the SSST-CSD introduced in [Tournier2007]_.

The reconstruction of the fiber orientation distribution function
(fODF) in MSMT-CSD involves the following steps:
    1. Generate a mask using Median Otsu (optional step)
    2. Denoise the data using MP-PCA (optional step)
    3. Generate  Anisotropic Powermap (if T1 unavailable)
    4. Tissue Classification (needs to be at least two classes of tissues)
    5. Estimation of the fiber response function
    6. Use the response function to reconstruct the fODF

First, we import all the modules we need from dipy as follows:
"""

import numpy as np
import dipy.reconst.shm as shm
import dipy.direction.peaks as dp
import dipy.reconst.dti as dti

from dipy.denoise.localpca import mppca
from dipy.data import (fetch_cfin_multib, read_cfin_dwi)
from dipy.core.gradients import gradient_table
from dipy.segment.mask import median_otsu
from dipy.reconst.csdeconv import auto_response
from dipy.segment.tissue import TissueClassifierHMRF
from dipy.sims.voxel import multi_shell_fiber_response
from dipy.reconst.mcsd import MultiShellDeconvModel
from dipy.viz import window, actor

from dipy.data import get_sphere
sphere = get_sphere('symmetric724')

"""
For this example, we use fetch to download a multi-shell dataset which was
kindly provided by Hansen and Jespersen (more details about the data are
provided in their paper [Hansen2016]_). The total size of the downloaded data
is 192 MBytes, however you only need to fetch it once.
"""

fetch_cfin_multib()
img, gtab = read_cfin_dwi()
data = img.get_data()
affine = img.affine

"""
For the sake of simplicity, we only select two non-zero b-values for this
example.
"""

bvals = gtab.bvals
bvecs = gtab.bvecs

sel_b = np.logical_or(np.logical_or(bvals == 0, bvals == 1000), bvals == 2000)
data = data[..., sel_b]

"""
The gradient table is also selected to have the selected b-values (0, 1000 and
2000)
"""

gtab = gradient_table(bvals[sel_b], bvecs[sel_b])

"""
We make use of the ``median_otsu`` method to generate the mask for the data as
follows:
"""

b0_mask, mask = median_otsu(data, median_radius=2, numpass=1, vol_idx=[0, 1])


print(data.shape)

"""
As one can see from its shape, the selected data contains a total of 67
volumes of images corresponding to all the diffusion gradient directions
of the selected b-values and call the ``mppca`` as follows:
"""
denoised_arr = mppca(data, mask=mask, patch_radius=2)

"""
"""
model = shm.QballModel(gtab, 8)

peaks = dp.peaks_from_model(model=model, data=denoised_arr,
                            relative_peak_threshold=.5,
                            min_separation_angle=25,
                            sphere=sphere, mask=mask)

ap = shm.anisotropic_power(peaks.shm_coeff)

tenmodel = dti.TensorModel(gtab)
tenfit = tenmodel.fit(denoised_arr)

nclass = 3
beta = 0.1

FA = tenfit.fa
MD = tenfit.md

hmrf = TissueClassifierHMRF()
initial_segmentation, final_segmentation, PVE = hmrf.classify(ap, nclass,
                                                              beta)


csf = PVE[..., 0]
cgm = PVE[..., 1]


indices_csf = np.where(((FA < 0.2) & (csf > 0.95)))
indices_cgm = np.where(((FA < 0.2) & (cgm > 0.95)))

selected_csf = np.zeros(FA.shape, dtype='bool')
selected_cgm = np.zeros(FA.shape, dtype='bool')

selected_csf[indices_csf] = True
selected_cgm[indices_cgm] = True

csf_md = np.mean(tenfit.md[selected_csf])
cgm_md = np.mean(tenfit.md[selected_cgm])


response, ratio = auto_response(gtab, denoised_arr, roi_radius=10, fa_thr=0.7)
evals_d = response[0]

response_mcsd = multi_shell_fiber_response(sh_order=8, bvals=bvals,
                                           evals=evals_d, csf_md=csf_md,
                                           gm_md=cgm_md)

mcsd_model = MultiShellDeconvModel(gtab, response_mcsd)

mcsd_fit = mcsd_model.fit(denoised_arr[:, :, 10:10+1])
mcsd_odf = mcsd_fit.odf(sphere)
fodf_spheres = actor.odf_slicer(mcsd_odf, sphere=sphere, scale=0.01,
                                norm=False, colormap='plasma')
interactive = False
ren = window.Renderer()
ren.add(fodf_spheres)

print('Saving illustration as mcsd_peaks.png')
window.record(ren, out_path='mcsd_peaks.png', size=(600, 600))

if interactive:
    window.show(ren)

"""
.. figure:: mcsd_peaks.png
   :align: center

   CSD Peaks and ODFs.

References
----------

.. [Tournier2007] J-D. Tournier, F. Calamante and A. Connelly, "Robust
   determination of the fibre orientation distribution in diffusion MRI:
   Non-negativity constrained super-resolved spherical deconvolution",
   Neuroimage, vol. 35, no. 4, pp. 1459-1472, 2007.

.. [Tax2014] C.M.W. Tax, B. Jeurissen, S.B. Vos, M.A. Viergever, A. Leemans,
   "Recursive calibration of the fiber response function for spherical
   deconvolution of diffusion MRI data", Neuroimage, vol. 86, pp. 67-80, 2014.

.. include:: ../links_names.inc

"""