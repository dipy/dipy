"""

.. _reconst-mcsd:

==================================================================
Reconstruction with Multi-Shell Multi-Tissue Constrained Spherical
Deconvolution
==================================================================

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
Now we will use the denoised array (``denoised_arr``) obtained from ``mppca``
in the rest of the steps in the tutorial.

As for the next step, we generate the anisotropic powermap introduced by
[Dell'Acqua2014]_. To do so, we make use of the Qball Model as follows:
"""

qball_model = shm.QballModel(gtab, 8)

"""
We generate the peaks from the ``qball_model`` as follows:
"""

peaks = dp.peaks_from_model(model=qball_model, data=denoised_arr,
                            relative_peak_threshold=.5,
                            min_separation_angle=25,
                            sphere=sphere, mask=mask)

ap = shm.anisotropic_power(peaks.shm_coeff)

print(ap.shape)

"""
As we can see from the shape of the Anisotropic Power Map, it is 3D and can be
used for tissue classification using Hidden Markov Random Fields (HMRF). The
HMRF needs the specification of the number of classes. For the case of MSMT-CSD
the ``nclass`` parameter needs to be ``>=2``. In our case, we set it to 3:
namely corticospinal fluid (CSF), white matter (WM) and gray matter (GM).
"""

nclass = 3

"""
Then, the smoothness factor of the segmentation. Good performance is achieved
with values between 0 and 0.5.
"""

beta = 0.2

"""
 We then call the ``TissueClassifierHMRF`` with the parameters specified as
 above:
"""

hmrf = TissueClassifierHMRF()
initial_segmentation, final_segmentation, PVE = hmrf.classify(ap, nclass, beta)
print(PVE.shape)

"""
Now that we hae the segmentation step, we would like to classify the tissues
into White Matter (WM), Grey Matter (GM) and corticospinal fluid (CSF). We do
so using the Fractional Anisotropy (FA) and Mean Diffusivity (MD) metrics
obtained from the Diffusion Tensor Imaging Model (DTI) fit as follows:
"""

# Construct the  DTI model
tenmodel = dti.TensorModel(gtab)

# fit the denoised data with DTI model
tenfit = tenmodel.fit(denoised_arr)

# obtain the FA and MD metrics
FA = tenfit.fa
MD = tenfit.md

"""

"""

csf = PVE[..., 0]
cgm = PVE[..., 1]


indices_csf = np.where(((FA < 0.2) & (csf > 0.95)))
indices_cgm = np.where(((FA < 0.2) & (cgm > 0.95)))

selected_csf = np.zeros(FA.shape, dtype='bool')
selected_cgm = np.zeros(FA.shape, dtype='bool')

selected_csf[indices_csf] = True
selected_cgm[indices_cgm] = True

csf_md = np.mean(MD[selected_csf])
cgm_md = np.mean(MD[selected_cgm])


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

"""