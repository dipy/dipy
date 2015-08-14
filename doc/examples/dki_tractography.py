"""
==============================================================
Exploring the 3D information of DKI for DKI based Tractography
==============================================================

In this example we show how to analyze the 3D information of the tensors
estimated from diffusion kurtosis imaging (DKI) and how DKI can be used for
tractography. This example is based on the work done by [Raf2015]_.

First we import all relevant modules:
"""

import numpy as np
import nibabel as nib
import dipy.reconst.dki as dki
from dipy.data import read_cenir_multib
from dipy.sims.voxel import multi_tensor_dki
from dipy.segment.mask import median_otsu
from dipy.data import get_sphere
from dipy.core.geometry import sphere2cart
from dipy.viz import fvtk

"""
This example will be based on the same multi-shell data used in the previous
DKI usage example :ref:`example_reconst_dki`. This data was collected with
similar acquisition parameters used on the Human Connectome Project (HCP),
however we only use the data's b-values up to 2000 $s.mm^{-2}$ to decrease the
influence of the diffusion signal taylor approximation componets not taken into
account by the diffusion kurtosis model. 
"""

bvals = [200, 400, 1000, 2000]

gtab, img = read_cenir_multib(bvals)

data = img.get_data()

affine = img.get_affine()

"""
Having the acquition paramenters of the loaded data, we can define the
diffusion kurtosis model as:
"""

dkimodel = dki.DiffusionKurtosisModel(gtab)

"""
We first illustrate the 3D information provided by DKI from simulations. For
this, based on the same GradientTable of the loaded data, we generate the
signal of five voxels containing two fibers populations with increasing
intersection angle (for more details on DKI simulates see
:ref:`example_simulate_dki`):
"""

# Defining the parameters that we mantain constant across the five voxels
mevals = np.array([[0.00099, 0, 0], [0.00226, 0.00087, 0.00087],
                   [0.00099, 0, 0], [0.00226, 0.00087, 0.00087]])
fie = 0.49
fractions = [fie*50, (1 - fie)*50, fie*50, (1 - fie)*50]

# `five_voxel_angles` is a 5x4 matrix that contains the information of the
# fiber directions for the 5 simulated voxels each formed by 4 compartments
five_voxel_angles = np.array([[(90, 0), (90, 0), (90, 0), (90, 0)],
                              [(90, 0), (90, 0), (90, 30), (90, 30)],
                              [(90, 0), (90, 0), (90, 45), (90, 45)],
                              [(90, 0), (90, 0), (90, 70), (90, 70)],
                              [(90, 0), (90, 0), (90, 90), (90, 90)]])

# simulate the signals for all 5 voxels
signal_dki = np.zeros((len(five_voxel_angles), len(gtab.bvecs)))
for v in range(len(five_voxel_angles)):
    angles = five_voxel_angles[v]
    signal_dki[v], dt, kt = multi_tensor_dki(gtab, mevals, S0=200,
                                             angles=angles,
                                             fractions=fractions, snr=None)

""" Now we fit the signal of the simulated voxels using the fit method of the
DiffusinKurtosisModel: """

dkifit = dkimodel.fit(signal_dki)

""" To visualize the 3D information provide by the diffusion and kurtosis
tensors, we compute and plot the directional diffusivity and kurtosis values
on 724 directions evenly sampled on a sphere. """

sphere = get_sphere('symmetric724')

# Compute the apparent diffusivity and kurtosis
ADC = dkifit.adc(sphere)
AKC = dkifit.akc(sphere)

# Prepare grafical representation
ren = fvtk.ren()
tensors = np.vstack([AKC, ADC])
tensors = tensors.reshape((5, 2, 1, len(sphere.vertices)), order='F')
tensors_actor = fvtk.sphere_funcs(tensors, sphere)
fvtk.add(ren, tensors_actor)

"""
For the graphical presentation of the tensors estimated from DKI, we add the
ground truth direction of the fibers. This is done using the following lines of
code:
"""

# convert fiber directions ground truth angles to Cartesian coordinates
gt_dir = np.zeros((5, 1, 2, 3))
for d in range(5):
    a = five_voxel_angles[d]
    gt_dir[d, 0, 0] = sphere2cart(1, np.deg2rad(a[0, 0]), np.deg2rad(a[0, 1]))
    gt_dir[d, 0, 1] = sphere2cart(1, np.deg2rad(a[2, 0]), np.deg2rad(a[2, 1]))

# Duplicate directions ground truth to plot the direction reference in both
# directional diffusivity and kurtosis values
gt_dir_2copies = np.vstack([gt_dir, gt_dir])
gt_dir_2copies = gt_dir_2copies.reshape((5, 2, 1, 2, 3), order='F')

gt_peaks = fvtk.peaks(gt_dir_2copies, 1.05 * np.ones(gt_dir_2copies.shape))
fvtk.add(ren, gt_peaks)

""" Now we are ready to save and show the figure containing the tensor
geometries """

fvtk.record(ren, out_path='geometry_of_dki_tensors.png', size=(1200, 1200))

fvtk.show(ren, title='Geometry of DKI tensors', size=(500, 500))

""" 
.. figure:: geometry_of_DKI_tensors.png
   :align: center
   ** Geometrical representation of the diffusion tensor (upper panels) and the
   kurtosis tensor (lower panels) as a function of the crossing intersection
   angle between two simulated white matter fibers. Fiber ground truth are
   plotted in red **.

From the figure, we can see that the diffusion tensor does not provide any
information of crossing fibers. Nevertheless, the kurtosis tensor shows
to be sensitive to the direction of both crossing fibers for intersection
angles larger than $30^{o}$. Particulary, in these cases, kurtosis maxima are
present prependicularly to both fibers (for more information see [Raf2015]_).

By combining the information of both diffusion and kurtosis tensors, we can
also estimate the DKI based orientation distribution function (DKI-ODF)
(see [Jen2014]_ and [Raf2015]_). The DKI-ODF function can be computed in Dipy
from the DiffusionKurtosisFit object:
"""

ODF = dkifit.dki_odf(sphere)

""" We plot below the DKI-ODF in the analogous way done for the directional
diffusivity and kurtosis values:
"""

ren = fvtk.ren()
ODF = ODF.reshape((5, 1, 1, len(sphere.vertices)), order='F')
ODF_actor = fvtk.sphere_funcs(ODF, sphere)
fvtk.add(ren, ODF_actor)
gt_dir = gt_dir.reshape((5, 1, 1, 2, 3), order='F')
gt_peaks = fvtk.peaks(gt_dir, 1.05 * np.ones(gt_dir.shape))
fvtk.add(ren, gt_peaks)

fvtk.record(ren, out_path='DKI_ODF_geometry.png', size=(1200, 1200))

fvtk.show(ren, title='DKI-ODF geometry', size=(500, 500))

""" 
.. figure:: DKI_ODF_geometry.png
   :align: center
   ** Geometrical representation of the DKI based orientation distribution
   function (ODF) estimate as a function of the crossing intersection angle
   between two simulated white matter fibers. Fiber ground truth are plotted in
   red **.

We can see from figure that DKI-ODF peaks are near to the fiber directions
ground truth. In this way, we show that when the ground truth fibers directions
are not known (the case of real brain data), these can estimated by finding
the DKI-ODF maxima. Below we illustrate how this is done the HCP-like brain
dataset.

As mention in :ref:`example_reconst_dki`, diffusion kurtosis imaging requires
that some pre-processing is done on real brain datasets to reduce the impact of
signal artefacts. To avoid reprocessing data, here we load the data previously
denoised in :ref:`example_reconst_dki`
"""

img = nib.load('denoised_cenir_multib.nii.gz') 

den = img.get_data()

""" Now, we mask the data to avoid processing unnecessary background voxels:  
"""

maskdata, mask = median_otsu(den, 4, 2, False, vol_idx=[0, 1], dilate=1)

""" To fit the diffusion kurtosis model, we just need to call again the
function fit of the DiffusinKurtosisModel. For illustration, we first show the
DKI based ODF reconstructions in a small portion of the data. """

data_portion = maskdata[30:70, 50:51, 35:65]

dkifit = dkimodel.fit(data_portion)

"""
Having fitted the diffusion kurtosis model, we are ready to compute and plot
the DKI based ODFs
"""

dkiodf = dkifit.dki_odf(sphere)

ren = fvtk.ren()
odf_spheres = fvtk.sphere_funcs(dkiodf, sphere)
odf_spheres.RotateX(-90)
fvtk.add(ren, odf_spheres)
fvtk.record(ren, out_path='dki_odfs.png')
fvtk.show(ren)

"""
References:

.. [Jen2014] Jensen, J.H., Helpern, J.A., Tabesh, A., (2014). Leading
             non-Gaussian corrections for diffusion orientation distribution
             function. NMR Biomed. 27, 202-211.
             http://dx.doi.org/10.1002/nbm.3053.
.. [RNH2015] Neto Henriques R, Correia MM, Nunes RG, Ferreira HA (2015).
             Exploring the 3D geometry of the diffusion kurtosis tensor -
             Impact on the development of robust tractography procedures and
             novel biomarkers, NeuroImage 111: 85-99

.. include:: ../links_names.inc
"""