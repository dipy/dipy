"""
==============================================================
Deterministic Tracking with the Sparse Fascicle Model and EuDX
==============================================================

In this example, we will use the Sparse Fascicle Model [Rokem2014]_, together
with the EuDX [Garyfallidis12]_ deterministic tracking algorithm.

First, we import the modules we will use in this example:
"""

import numpy as np
import nibabel as nib
import dipy.reconst.sfm as sfm
import dipy.data as dpd
from dipy.data import fetch_stanford_hardi
import dipy.reconst.peaks as dpp
from dipy.tracking.eudx import EuDX
from dipy.segment.mask import median_otsu
from dipy.viz import fvtk
from dipy.viz.colormap import line_colors


"""
We will use the default sphere (362 vertices, symmetrically distributed on the
surface of the sphere):
"""

sphere = dpd.get_sphere()

"""
The Stanford HARDI dataset (150 directions, single b-value of 2000 s/mm$^2$) is
read into memory

"""

fetch_stanford_hardi()
from dipy.data import read_stanford_hardi
img, gtab = read_stanford_hardi()
data = img.get_data()

"""
Tracking requires a per-voxel model. Here, the model is the Sparse Fascicle
Model, described in [Rokem2014]_. This model reconstructs the diffusion signal
as a combination of the signals from different fascicles. This model can be
written as:

.. math::

    y = X\beta

Where $y$ is the signal and $\beta$ are weights on different points in the
sphere. The columns of the design matrix, $X$ are the signals in each point in
the measurement that would be predicted if there was a fascicle oriented in the
direction represented by that column. Typically, the signal used for this
kernel will be a prolate tensor with axial diffusivity 3-5 times higher than
its radial diffusivity. The exact numbers can also be estimated from examining
parts of the brain in which there is known to be only one fascicle (e.g. in
corpus callosum).

Sparsity constraints on the fiber ODF ($\beta$) are set through the Elastic Net
algorihtm [Zou2005]_.

Elastic Net optimizes the following cost function:

.. math::

    \sum_{i=1}^{n}{(y_i - \hat{y}_i)^2} + \alpha (\lambda \sum_{j=1}^{m}{w_j} +
(1-\lambda) \sum_{j=1}^{m}{w^2_j}

where $\hat{y}$ is the signal predicted for a particular setting of $\beta$,
such that the left part of this expression is the squared loss function;
$\alpha$ is a parameter that sets the balance between the squared loss on
the data, and the regularization constraints. The regularization parameter
$\lambda$ sets the `l1_ratio`, which controls the balance between L1-sparsity
(low sum of weights), and low L2-sparsity (low sum-of-squares of the weights).

We start by considering a small volume of data.
"""

data_small = data[20:50, 55:85, 38:39]

"""
We initialize an SFM model object:
"""

sf_model = sfm.SparseFascicleModel(gtab, sphere=sphere,
                                   l1_ratio=0.5, alpha=0.001)


"""
Fitting the model to this small volume of data, we calculate the ODF of this
model on the sphere, and plot it.
"""
sf_fit = sf_model.fit(data_small)
sf_odf = sf_fit.odf(sphere)

fodf_spheres = fvtk.sphere_funcs(sf_odf, sphere, scale=1.5, norm=True)

ren = fvtk.ren()
fvtk.add(ren, fodf_spheres)

print('Saving illustration as sf_odfs.png')
fvtk.record(ren, out_path='sf_odfs.png', size=(1000, 1000))

"""
We can extract the peaks from the ODF, and plot these as well
"""
sf_peaks = dpp.peaks_from_model(sf_model,
                                data_small,
                                sphere,
                                relative_peak_threshold=.5,
                                min_separation_angle=25,
                                return_sh=False)


fvtk.clear(ren)
fodf_peaks = fvtk.peaks(sf_peaks.peak_dirs, sf_peaks.peak_values, scale=1.5)
fvtk.add(ren, fodf_peaks)
#fvtk.show(ren)

print('Saving illustration as sf_peaks.png')
fvtk.record(ren, out_path='sf_peaks.png', size=(1000, 1000))

"""
Finally, we plot both the peaks and the ODFs, overlayed:
"""

fodf_spheres.GetProperty().SetOpacity(0.4)
fvtk.add(ren, fodf_spheres)

print('Saving illustration as sf_both.png')
fvtk.record(ren, out_path='sf_both.png', size=(1000, 1000))


"""
Next, we will use the EuDX algorithm, to perfom deterministic tracking on the
entire brain. To reduce the computational load, we focus on the parts of the
data that can be segmented using the median Otsu filter in the b=0 volumes:

"""

maskdata, mask = median_otsu(data, 3, 1, True,
                             vol_idx=range(10), dilate=2)

"""
The peaks of the ODFs are extracted using the `peaks_from_model` function

"""

pnm = dpp.peaks_from_model(sf_model, maskdata, sphere,
                           mask=mask,
                           relative_peak_threshold=.5,
                           min_separation_angle=25,
                           return_sh=False)


"""
We will use the generalized fractional anisotropy (GFA) as a stopping
criterion, and seed in 10k points
"""

eu = EuDX(pnm.gfa,
          pnm.peak_indices[..., 0],
          seeds=10000,
          odf_vertices=sphere.vertices,
          a_low=0.2)


sfm_streamlines =[streamline for streamline in eu]

"""
We visualize the resulting tracks, using fvtk:
"""

r = fvtk.ren()
fvtk.add(r, fvtk.line(sfm_streamlines, line_colors(sfm_streamlines)))
print('Saving illustration as sfm_tracks.png')
fvtk.record(r, n_frames=1, out_path='sfm_tracks.png', size=(1000, 1000))

"""
References
----------

.. [Rokem2014] Ariel Rokem, Jason D. Yeatman, Franco Pestilli, Kendrick
   N. Kay, Aviv Mezer, Stefan van der Walt, Brian A. Wandell
   (2014). Evaluating the accuracy of diffusion MRI models in white
   matter. http://arxiv.org/abs/1411.0721

.. [Zou2005] Zou H, Hastie T (2005). Regularization and variable
   selection via the elastic net. J R Stat Soc B:301-320

"""
