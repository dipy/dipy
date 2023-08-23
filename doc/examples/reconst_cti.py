"""
========================================================================
Reconstruction of the diffusion signal with the correlation tensor model
========================================================================

The Correlation Tensor Model (CTI) is an advanced extension of the Diffusion Kurtosis Model (see :ref:sphx_glr_examples_built_reconstruction_reconst_dki.py). [ not sure]
In addition to the DKI, CTI introduces a more complete representation of the underlying microstructural complexity. Specifically, the CTI differentiates between different sources of kurtosis by distinguishing between isotropic, anisotropic, and microscopic contributions. This refined characterization provides richer insights into tissue microstructure and can reveal features that may be obscured or conflated in simpler models.
Also, similar to DTI, assumes that the diffusion properties within each Gaussian compartment can be described by a diffusion tensor. However, unlike DTI, CTI allows for the presence of multiple diffusion tensors within a voxel, representing different Gaussian compartments.

In DTI, we’ve diffusion tensor which measure the average diffusion properties within each voxel. in case of DKI we’ve diffusion tensor as well as kurtosis tensor. kurtosis tensor measures the dialation of diffusion properties from the gaussaian behaviour. But in case of CTI we’ve diffusion tensor, kurtosis tensor as well as covariance tensor. The covariance tensor describes the correlation between diffusion measurements at different b-values, offering insights into the relationship between diffusion processes at different scales.

CTI is also very suitable for situations which involve crossing fibers. As unlike DKI, it provides additional information in those cases by providing information about microscopic anisotropy.

[ give maths for formulas ] 
The CorrelationTensorModel expresses the diffusion-weighted signal in the same way as the DiffusionKurtosisModel. 

However it differentiates from the DiffusionKurtosis Model by calculating the different sources of kurtosis. The isotropic source is calculated as:  

.. math::
	\[K_{iso} = 3 \cdot \frac{V({\overline{D}^c})}{\overline{D}^2}\]
        
        where: \(K_{iso}\) is the isotropic kurtosis,
            \(V({\overline{D}^c})\) represents the variance of the diffusion tensor raised to the power c, 
            \(\overline{D}\) is the mean of the diffusion tensor.
            
The anisotropic source is calculated as: 

:math:: 
            
            \[K_{aniso} = \frac{6}{5} \cdot \frac{\langle V_{\lambda}(D_c) \rangle}{\overline{D}^2}\]
            
        where: \(K_{aniso}\) is the anisotropic kurtosis, 
            \(\langle V_{\lambda}(D_c) \rangle\) represents the mean of the variance of eigenvalues of the diffusion tensor,
            \(\overline{D}\) is the mean of the diffusion tensor.
            
The microscopic source is calculated as: 

:math:: 
                \[\Psi = \frac{2}{5} \cdot \frac{D_{11}^2 + D_{22}^2 + D_{33}^2 + 2D_{12}^2 + 2D_{13}^2 + 2D_{23}^2{\overline{D}^2} - \frac{6}{5} \]
                \[{\overline{W}} = \frac{1}{5} \cdot (W_{1111} + W_{2222} + W_{3333} + 2W_{1122} + 2W_{1133} + 2W_{2233})\]
            
            where \(\Psi\) is a variable representing a part of the total excess kurtosis,
            \(D_{ij}\) are elements of the diffusion tensor,
            \(\overline{D}\) is the mean of the diffusion tensor.
            \{\overline{W}} is the mean kurtosis,
            \(W_{ijkl}\) are elements of the kurtosis tensor.
            
and 

:math:: 
	K_{\text{intra}} = K_{\text{T}} - K_{\text{aniso}} - K_{\text{iso}}
	
	where K_{\text{intra}} refers to intra-compartmental kurtosis, 
	K_{\text{T}} refers to total kurtosis, 
	K_{\text{aniso}} refers to anisotropic source of kurtosis, 
	K_{\text{iso}} refers to isotropic source of kurtosis

In the following example we show how to fit the correlation tensor model on a real life dataset and how to estimate correlation tensor based statistics. 

First, we'll import all relevant modules. 
""" 
import matplotlib.pyplot as plt
import numpy as np
import math

from dipy.reconst.tests.test_qti import _anisotropic_DTD, _isotropic_DTD
from dipy.core.sphere import Sphere, HemiSphere
from dipy.reconst.tests.test_qti import _anisotropic_DTD, _isotropic_DTD
import dipy.reconst.qti as qti
from dipy.reconst.qti import (from_3x3_to_6x1)
from dipy.reconst.dti import (decompose_tensor, mean_diffusivity)
import dipy.reconst.cti as cti
from dipy.io import read_bvals_bvecs
from dipy.core.gradients import gradient_table
from dipy.io.image import load_nifti
"""
CTI analysis requires specific gradient information from the data, often acquired from different gradient tables. In this example, we manually load b-values and b-vectors from two separate sets of files, 'bvals1.bval' and 'bvec1.bvec', and 'temp_data/bvals2.bval' and 'temp_data/bvec2.bvec'. 
The users should ensure that the data is formatted correctly for the CTI analysis they are performing.
""" 
data, affine = load_nifti('temp_data/RB_invivo_cti_data_f3.nii')
bvals1, bvecs1 = read_bvals_bvecs('temp_data/bvals1.bval',
                                  'temp_data/bvec1.bvec')
bvals2, bvecs2 = read_bvals_bvecs('temp_data/bvals2.bval',
                                  'temp_data/bvec2.bvec')
                                  """ 
In this example, the function `load_nifti` is used to load the CTI data from the file 'RB_invivo_cti_data_f3.nii' and returns the data as a nibabel Nifti1Image object along with the affine transformation. The b-values and b-vectors for two different gradient tables are loaded from 'bvals1.bval' and 'bvec1.bvec', and 'bvals2.bval' and 'bvec2.bvec' respectively using the `read_bvals_bvecs` function. These are then  further converted into gradient tables `gtab1` and `gtab2` which are essential for the CTI analysis being conducted.
"""
gtab1 = gradient_table(bvals1, bvecs1)
gtab2 = gradient_table(bvals2, bvecs2)
"""
Before fitting the data, we perform some data pre-processing. We first compute a brain mask to avoid unnecessary calculations on the background of the image.
"""
maskdata, mask = median_otsu(data, vol_idx=[0, 1], median_radius=4, numpass=2,
                             autocrop=False, dilate=1)
"""
Now that we have loaded and created a mask for the data we can go forward with CTI fitting. For this, the CTI model is first defined for the data’s GradientTable object by instantiating the CorrelationTensorModel object in the following way:
"""
ctimodel = cti.CorrelationTensorModel(gtab1, gtab2)
"""
To fit the data using the defined model object, we call the fit function of this object.
"""
ctifit = ctimodel.fit(data, mask = mask)
"""
The fit method for the CTI model produces a CorrelationTensorFit object, which contains the attributes of both the DKI and DTI models. Given that CTI is a layered model built upon the DKI, which itself extends the DTI model, the CorrelationTensorFit instance captures a comprehensive set of parameters and attributes from these underlying models.

For instance, since the DKI model inherently estimates the diffusion tensor parameters, all standard DTI statistics, such as fractional anisotropy (FA), mean diffusivity (MD), axial diffusivity (AD), and radial diffusivity (RD), can be directly extracted from the CorrelationTensorFit instance. Additionally, the CTI offers further insights by evaluating correlation tensors, offering a richer perspective on the tissue's microstructural environment.
Below we draw a feature map of the 3 different sources of kurtosis which can exclusively be calculated from the CTI model.
"""
kiso_map = ctifit.K_iso 
kaniso_map = ctifit.K_aniso 
kmicro_map = ctifit.K_micro

slice_idx = 0
fig, axarr = plt.subplots(1, 3, figsize=(15,5))

axarr[0].imshow(kiso_map[:, :, slice_idx], cmap='gray', origin='lower', vmin = 0, vmax = 1)
axarr[0].set_title('Kiso Map')

axarr[1].imshow(kaniso_map[:, :, slice_idx], cmap='gray', origin='lower', vmin = 0, vmax = 1)
axarr[1].set_title('Kaniso Map')

axarr[2].imshow(kmicro_map[:, :, slice_idx], cmap='gray', origin='lower', vmin = 0, vmax = 1)
axarr[2].set_title('Kmicro Map')

plt.tight_layout()
plt.show()


























