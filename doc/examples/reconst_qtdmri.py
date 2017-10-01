# -*- coding: utf-8 -*-
"""
================================================================
Estimating diffusion time dependent q-space indices using qt-dMRI
================================================================
Effective representation of the four-dimensional diffusion MRI signal -- 
varying over three-dimensional q-space and diffusion time -- is a sought-after
and still unsolved challenge in diffusion MRI (dMRI). We propose a functional
basis approach that is specifically designed to represent the dMRI signal in
this qtau-space [Fick2017]_.  Following recent terminology, we refer to our
qtau-functional basis as ``q$\tau$-dMRI''. We use GraphNet regularization --
imposing both signal smoothness and sparsity -- to drastically reduce the
number of diffusion-weighted images (DWIs) that is needed to represent the dMRI
signal in the qtau-space. As the main contribution, q$\tau$-dMRI provides the
framework to -- without making biophysical assumptions -- represent the
q$\tau$-space signal and estimate time-dependent q-space indices
(q$\tau$-indices), providing a new means for studying diffusion in nervous
tissue. qtau-dMRI is the first of its kind in being specifically designed to
provide open interpretation of the qtau-diffusion signal.

q$\tau$-dMRI can be seen as a time-dependent extension of the MAP-MRI
functional basis [Ozarslan2013]_, and all the previously proposed q-space
can be estimated for any diffusion time. These include rotationally
invariant quantities such as the Mean Squared Displacement (MSD), Q-space
Inverse Variance (QIV) and Return-To-Origin Probability (RTOP). Also
directional indices such as the Return To the Axis Probability (RTAP) and
Return To the Plane Probability (RTPP) are available, as well as the
Orientation Distribution Function (ODF).

In this example we illustrate how to use the qtau-dMRI to estimate
time-dependent q-space indices from a qtau-acquisition of a mouse.

First import the necessary modules:
"""

from dipy.data.fetcher import (fetch_qtdMRI_test_retest_2subjects,
                               read_qtdMRI_test_retest_2subjects)
from dipy.reconst import qtdmri
import matplotlib.pyplot as plt
import numpy as np
#from mpl_toolkits.axes_grid1 import make_axes_locatable

"""
Download and read the data for this tutorial.

qt-dMRI requires data with multiple gradient directions, gradient strength and
diffusion times. We will use the test acquisition of one of the mice that was
used in the test-retest study by [Fick2017]_.
"""

fetch_qtdMRI_test_retest_2subjects()
data, cc_masks, gtabs = read_qtdMRI_test_retest_2subjects()

"""
data contains the voxel data and gtab contains a GradientTable
object (gradient information e.g. b-values). For example, to show the b-values
it is possible to write print(gtab.bvals).

For the values of the q-space
indices to make sense it is necessary to explicitly state the big_delta and
small_delta parameters in the gradient table.
"""
plt.figure()
qtdmri.visualise_gradient_table_G_Delta_rainbow(gtabs[0])
plt.savefig('qt-dMRI_acquisition_scheme.png')

"""
.. figure:: qt-dMRI_acquisition_scheme.png
   :align: center
"""

"""
- show mask over FA overlay.
- fit qt-dMRI
- estimate qt-space indices
- show test-retest reproducibility
"""

"""
.. [Fick2017]_ Fick, Rutger HJ, et al. "Non-Parametric GraphNet-Regularized
            Representation of dMRI in Space and Time", Medical Image Analysis,
            2017.
"""
