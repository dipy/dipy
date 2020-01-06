"""

.. _reconst-mcsd:

================================================
Reconstruction with Multi-Shell Multi-Tissue CSD
================================================

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
    4. Fit DTI model to the data
    5. Tissue Classification (needs to be at least two classes of tissues)
    6. Estimation of the fiber response function
    7. Use the response function to reconstruct the fODF

First, we import all the modules we need from dipy as follows:
"""

import numpy as np
import dipy.reconst.shm as shm
import dipy.direction.peaks as dp
import dipy.reconst.dti as dti
import matplotlib.pyplot as plt

from dipy.denoise.localpca import mppca
from dipy.core.gradients import gradient_table
from dipy.io.gradients import read_bvals_bvecs
from dipy.io.image import load_nifti
from dipy.segment.mask import median_otsu
from dipy.reconst.csdeconv import auto_response
from dipy.segment.tissue import TissueClassifierHMRF
from dipy.sims.voxel import multi_shell_fiber_response
from dipy.reconst.mcsd import MultiShellDeconvModel
from dipy.viz import window, actor

from dipy.data import get_sphere, get_fnames
sphere = get_sphere('symmetric724')

"""
For this example, we use fetch to download a multi-shell dataset which was
kindly provided by Hansen and Jespersen (more details about the data are
provided in their paper [Hansen2016]_). The total size of the downloaded data
is 192 MBytes, however you only need to fetch it once.
"""

fraw, fbval, fbvec, t1_fname = get_fnames('cfin_multib')

data, affine = load_nifti(fraw)
bvals, bvecs = read_bvals_bvecs(fbval, fbvec)
gtab = gradient_table(bvals, bvecs)

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
[DellAcqua2014]_. To do so, we make use of the Q-ball Model as follows:
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

plt.matshow(np.rot90(ap[:, :, 10]), cmap=plt.cm.bone)
plt.savefig("anisotropic_power_map.png")
plt.close()

"""
.. figure:: anisotropic_power_map.png
   :align: center

   Anisotropic Power Map (Axial Slice)
"""

print(ap.shape)

"""
The above figure is a visualization of the axial slice of the Anisotropic
Power Map. It can be treated as a pseudo-T1 for classification purposes
using the Hidden Markov Random Fields (HMRF) classifier, if the T1 image is not available.

As we can see from the shape of the Anisotropic Power Map, it is 3D and can be
used for tissue classification using HMRF. The
HMRF needs the specification of the number of classes. For the case of MSMT-CSD
the ``nclass`` parameter needs to be ``>=2``. In our case, we set it to 3:
namely corticospinal fluid (csf), white matter (wm) and gray matter (gm).
"""

nclass = 3

"""
Then, the smoothness factor of the segmentation. Good performance is achieved
with values between 0 and 0.5.
"""


beta = 0.1


"""
We then call the ``TissueClassifierHMRF`` with the parameters specified as
above:
"""

hmrf = TissueClassifierHMRF()
initial_segmentation, final_segmentation, PVE = hmrf.classify(ap, nclass, beta)
print(PVE.shape)

"""
Now that we have the segmentation step, we would like to classify the tissues
into ``wm``, ``gm`` and ``csf`` We do
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
Now that we have the FA and the MD obtained from DTI, we use it to distinguish
between the ``wm``, ``gm`` and ``csf``. As we can see
from the shape of the PVE, the last dimension refers to the classification. We
will now index them as: 0 -> ``csf``, 1 -> ``gm`` and 2 -> ``wm`` as per their
FA values and the confidence of prediction obtained from
``TissueClassifierHMRF``.
"""

csf = PVE[..., 0]
gm = PVE[..., 1]
wm = PVE[..., 2]

indices_csf = np.where(((FA < 0.2) & (csf > 0.95)))
indices_gm = np.where(((FA < 0.2) & (gm > 0.95)))

selected_csf = np.zeros(FA.shape, dtype='bool')
selected_gm = np.zeros(FA.shape, dtype='bool')

selected_csf[indices_csf] = True
selected_gm[indices_gm] = True

csf_md = np.mean(MD[selected_csf])
gm_md = np.mean(MD[selected_gm])

"""
The ``auto_response`` function will calculate FA for an ROI of radius equal to
``roi_radius`` in the center of the volume and return the response function
estimated in that region for the voxels with FA higher than 0.7.
"""

response, ratio = auto_response(gtab, denoised_arr, roi_radius=10, fa_thr=0.7)
evals_d = response[0]

"""
We will now use the evals obtained from the ``auto_response`` to generate the
``multi_shell_fiber_response`` rquired by the MSMT-CSD model. Note that we
mead diffusivities of ``csf`` and ``gm`` as inputs to generate th response.
"""

response_mcsd = multi_shell_fiber_response(sh_order=8, bvals=bvals,
                                           evals=evals_d, csf_md=csf_md,
                                           gm_md=gm_md)

"""
Now we build the MSMT-CSD model with the ``response_mcsd`` as input. We then
call the ``fit`` function to fit one slice of the 3D data and visualize it.
"""

mcsd_model = MultiShellDeconvModel(gtab, response_mcsd)
mcsd_fit = mcsd_model.fit(denoised_arr[:, :, 10:11])

"""
From the fit obtained in the previous step, we generate the ODFs which can be
visualized as follows:
"""

mcsd_odf = mcsd_fit.odf(sphere)
fodf_spheres = actor.odf_slicer(mcsd_odf, sphere=sphere, scale=0.01,
                                norm=False, colormap='plasma')

interactive = False
ren = window.Renderer()
ren.add(fodf_spheres)
ren.reset_camera_tight()

print('Saving illustration as msdodf.png')
window.record(ren, out_path='msdodf.png', size=(600, 600))

if interactive:
    window.show(ren)

"""
.. figure:: msdodf.png
   :align: center

   MSMT-CSD Peaks and ODFs.

References
----------

.. [Jeurissen2014] B. Jeurissen, et al., "Multi-tissue constrained spherical
                    deconvolution for improved analysis of multi-shell
                    diffusion MRI data." NeuroImage 103 (2014): 411-426.

.. [Tournier2007] J-D. Tournier, F. Calamante and A. Connelly, "Robust
                    determination of the fibre orientation distribution in
                    diffusion MRI: Non-negativity constrained super-resolved
                    spherical deconvolution", Neuroimage, vol. 35, no. 4,
                    pp. 1459-1472, (2007).

.. [Hansen2016] B. Hansen and SN. Jespersen, " Data for evaluation of fast
                    kurtosis strategies, b-value optimization and exploration
                    of diffusion MRI contrast", Scientific Data 3: 160072
                    doi:10.1038/sdata.2016.72, (2016)

.. [DellAcqua2014] F. Dell'Acqua, et. al., "Anisotropic Power Maps: A
                    diffusion contrast to reveal low anisotropy tissues from
                    HARDI data", Proceedings of International Society for
                    Magnetic Resonance in Medicine. Milan, Italy, (2014).

.. include:: ../links_names.inc

"""
