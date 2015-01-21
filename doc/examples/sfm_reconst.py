"""
.. _sfm-reconst:

==============================================
Reconstruction with the Sparse Fascicle Model
==============================================

In this example, we will use the Sparse Fascicle Model [Rokem2014]_, to
reconstruct the fiber orientation distribution function (fODF) in every voxel.

First, we import the modules we will use in this example:
"""

import dipy.reconst.sfm as sfm
import dipy.data as dpd
import dipy.reconst.peaks as dpp
from dipy.viz import fvtk

"""
For the purpose of this example, we will use the Stanford HARDI dataset (150
directions, single b-value of 2000 s/mm$^2$) that can be automatically
downloaded. If you have not yet downloaded this data-set in one of the other
examples, you will need to be connected to the internet the first time you run
this example. The data will be stored for subsequent runs, and for use with
other examples.

"""

from dipy.data import read_stanford_hardi
img, gtab = read_stanford_hardi()
data = img.get_data()

"""
Reconstruction of the fiber ODF in each voxel guides subsequent tracking
steps. Here, the model is the Sparse Fascicle Model, described in
[Rokem2014]_. This model reconstructs the diffusion signal as a combination of
the signals from different fascicles. This model can be written as:

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

    \sum_{i=1}^{n}{(y_i - \hat{y}_i)^2} + \alpha (\lambda \sum_{j=1}^{m}{w_j}+(1-\lambda) \sum_{j=1}^{m}{w^2_j}

where $\hat{y}$ is the signal predicted for a particular setting of $\beta$,
such that the left part of this expression is the squared loss function;
$\alpha$ is a parameter that sets the balance between the squared loss on
the data, and the regularization constraints. The regularization parameter
$\lambda$ sets the `l1_ratio`, which controls the balance between L1-sparsity
(low sum of weights), and low L2-sparsity (low sum-of-squares of the weights).

Just like constrained spherical deconvolution (see :ref:`reconst-csd`), the SFM
requires the definition of a response function. We'll take advantage of the
automated algorithm in the :mod:`csdeconv` module to find this response
function:

"""

from dipy.reconst.csdeconv import auto_response
response, ratio = auto_response(gtab, data, roi_radius=10, fa_thr=0.7)

"""
The ``response`` return value contains two entries. The first is an array with
the eigenvalues of the response function and the second is the average S0 for
this response.

It is a very good practice to always validate the result of auto_response. For,
this purpose we can print it and have a look at its values.
"""

print(response)

"""
(array([ 0.0014,  0.00029,  0.00029]), 416.206)

We initialize an SFM model object, using these values. We will use the default
sphere (362  vertices, symmetrically distributed on the surface of the sphere),
as a set of putative fascicle directions that are considered in the model
"""

sphere = dpd.get_sphere()
sf_model = sfm.SparseFascicleModel(gtab, sphere=sphere,
                                   l1_ratio=0.5, alpha=0.001,
                                   response=response[0])

"""
For the purpose of the example, we will consider a small volume of data
containing parts of the corpus callosum and of the centrum semiovale
"""

data_small = data[20:50, 55:85, 38:39]

"""
Fitting the model to this small volume of data, we calculate the ODF of this
model on the sphere, and plot it.
"""

sf_fit = sf_model.fit(data_small)
sf_odf = sf_fit.odf(sphere)

fodf_spheres = fvtk.sphere_funcs(sf_odf, sphere, scale=1.3, norm=True)

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
fodf_peaks = fvtk.peaks(sf_peaks.peak_dirs, sf_peaks.peak_values, scale=1.3)
fvtk.add(ren, fodf_peaks)

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
.. figure:: sf_both.png
   :align: center

   **SFM Peaks and ODFs**.

To see how to use this information in tracking, proceed to :ref:`sfm-track`.

References
----------

.. [Rokem2014] Ariel Rokem, Jason D. Yeatman, Franco Pestilli, Kendrick
   N. Kay, Aviv Mezer, Stefan van der Walt, Brian A. Wandell
   (2014). Evaluating the accuracy of diffusion MRI models in white
   matter. http://arxiv.org/abs/1411.0721

.. [Zou2005] Zou H, Hastie T (2005). Regularization and variable
   selection via the elastic net. J R Stat Soc B:301-320

"""
