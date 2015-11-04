"""
==============================================
Contextual enhancement
==============================================

This demo presents an example of crossing-preserving contextual enhancement of FOD/ODF fields, implementing the contextual PDE framework of [Portegies2015_PLoSOne]_ for processing HARDI data. The aim is to enhance the alignment of elongated structures in the data such that crossing/junctions are maintained while reducing noise and small incoherent structures. This is achieved via a hypo-elliptic 2nd order PDE in the domain of coupled positions and orientations :math:`\mathbb{R}^3 \rtimes S^2`. This domain carries a non-flat geometrical differential structure that allows including a notion of alignment between neighboring points. 

Let :math:`({\bf y},{\bf n}) \in \mathbb{R}^3\rtimes S^2` where :math:`{\bf y} \in \mathbb{R}^{3}` denotes the spatial part, and :math:`{\bf n} \in S^2` the angular part. Let :math:`W:\mathbb{R}^3\rtimes S^2\times \mathbb{R}^{+} \to \mathbb{R}` be the function representing the evolution of FOD/ODF field. Then, the contextual PDE with evolution time :math:`t\geq 0` is given by:

.. math::

    \begin{cases}
	\frac{\partial}{\partial t} W({\bf y},{\bf n},t) &= ((D^{33}({\bf n} \cdot 
	        \nabla)^2 + D^{44} \Delta_{S^2})W)({\bf y},{\bf n},t)
	\\ W({\bf y},{\bf n},0) &= U({\bf y},{\bf n})
	\end{cases},

where:

* :math:`D^{33}>0` is  the coefficient for the spatial smoothing (which goes only in the direction of :math:`n`);

* :math:`D^{44}>0` is the coefficient for the angular smoothing (here :math:`\Delta_{S^2}` denotes the Laplace-Beltrami operator on the sphere :math:`S^2`);

* :math:`U:\mathbb{R}^3\rtimes S^2 \to \mathbb{R}` is the initial condition given by the noisy FOD/ODF’s field.

This equation is solved via a shift-twist convolution (denoted by :math:`\ast_{\mathbb{R}^3\rtimes S^2}`) with its corresponding kernel :math:`P_t:\mathbb{R}^3\rtimes S^2 \to \mathbb{R}^+`:

.. math::

	W(\vec{y},\vec{n},t) = (P_t \ast_{\mathbb{R}^3 \rtimes S^2} U)(\vec{y},\vec{n}) = \int_{\mathbb{R}^3} \int_{S^2} P_t (R^T_{\vec{n}^\prime}(\vec{y}-\vec{y}^\prime), R^T_{\vec{n}^\prime} \vec{n} ) U(\vec{y}^\prime, \vec{n}^\prime) \hspace{0.2em} d\sigma (\vec{n}^\prime) d\vec{y}^\prime.

Here, :math:`R_{\bf n}` is any 3D rotation that maps the vector :math:`(0,0,1)` onto :math:`{\bf n}`.

Note that the shift-twist convolution differs from a Euclidean convolution and takes into account the non-flat structure of the space  :math:`\mathbb{R}^3\rtimes S^2`. 

The kernel :math:`P_t` has a stochastic interpretation. It can be seen as the limiting distribution obtained by accumulating random walks of particles in the position/orientation domain, where in each step the particles can (randomly) move forward/backward along their current orientation, and (randomly) change their orientation.  This is an extension to the 3D case of the process for contour enhancement of 2D images. 

.. figure:: ../../../stochastic_process.png
   :align: center

   The random motion of particles (a) and it's corresponding probability map (b) in 2D. The 3D kernel is shown on the right. Adapted from [Portegies2015b]_.

"""

"""
The enhancement is evaluated on the Stanford HARDI dataset (160 orientations, b=3000). Rician noise is added to degrade the image quality and mimick a lower quality dataset. Constrained spherical deconvolution is used to model the fiber orientations.

"""

import numpy as np
from dipy.data import fetch_stanford_hardi, read_stanford_hardi
from dipy.sims.voxel import add_noise
from dipy.core.gradients import gradient_table

# Read data
fetch_stanford_hardi()
img, gtab = read_stanford_hardi()
data = img.get_data()

# Add Rician noise
np.random.seed = 1234
data_noisy = add_noise(data, 3, 500 , noise_type='rician')

# Select a small part of it
data_small = data[25:40, 65:80, 35:42]
data_noisy_small = data_noisy[25:40, 65:80, 35:42]

"""
Perform a model fitting, in this case Constrained Spherical Deconvolution is used.
"""

# Perform CSD on the original data
from dipy.reconst.csdeconv import auto_response
from dipy.reconst.csdeconv import ConstrainedSphericalDeconvModel
response, ratio = auto_response(gtab, data, roi_radius=10, fa_thr=0.7)
csd_model_orig = ConstrainedSphericalDeconvModel(gtab, response)
csd_fit_orig = csd_model_orig.fit(data_small)
csd_shm_orig = csd_fit_orig.shm_coeff

# Perform CSD on the original data + noise
response, ratio = auto_response(gtab, data_noisy, roi_radius=10, fa_thr=0.7)
csd_model_noisy = ConstrainedSphericalDeconvModel(gtab, response)
csd_fit_noisy = csd_model_noisy.fit(data_noisy_small)
csd_shm_noisy = csd_fit_noisy.shm_coeff

"""
A lookup-table is created, containing rotated versions of the kernel :math:`P_t` sampled over a discrete set of orientations. In order to ensure rotationally invariant processing, the discrete orientations are required to be equally distributed over a sphere. By default, a sphere with 100 directions is used.
"""

from dipy.denoise.enhancement_kernel import EnhancementKernel
from dipy.denoise.shift_twist_convolution import convolve

# Create lookup table
D33 = 1.0
D44 = 0.02
t = 1
k = EnhancementKernel(D33, D44, t)

"""
Visualize the kernel
"""

from dipy.viz import fvtk
from dipy.data import get_sphere
from dipy.reconst.shm import sf_to_sh, sh_to_sf
ren = fvtk.ren()
sphere = get_sphere('symmetric724')
lutkernel = sh_to_sf(sf_to_sh(np.swapaxes(k.get_lookup_table()[0],0,3)[3,:,:,:], k.get_sphere(), sh_order=8), sphere, sh_order=8)
model_kernel = fvtk.sphere_funcs(lutkernel*10, sphere, norm=False, radial_scale=True)
fvtk.add(ren, model_kernel)
fvtk.camera(ren, pos=(-25, 0, 0), focal=(-1, -1, -1), viewup=(0, 1, 0))
fvtk.record(ren, out_path='kernel.png', size=(512,512), scale=2)

"""
.. figure:: kernel.png
   :align: center

   Visualization of the contour enhancement kernel.
"""

""" 
Shift-twist convolution is applied on the noisy data
"""

# Perform convolution
csd_shm_enh = convolve(csd_shm_noisy, k, sh_order=8)


"""
The Spherical Deconvolution Transform is applied to sharpen the ODF field. 
"""

# Sharpen via the Spherical Deconvolution Transform
from dipy.reconst.csdeconv import odf_sh_to_sharp
csd_shm_enh_sharp = odf_sh_to_sharp(csd_shm_enh, sphere,  sh_order=8, lambda_=0.1)

# Convert raw and enhanced data to discrete form
csd_sf_orig = sh_to_sf(csd_shm_orig, sphere, sh_order=8)
csd_sf_noisy = sh_to_sf(csd_shm_noisy, sphere, sh_order=8)
csd_sf_enh = sh_to_sf(csd_shm_enh, sphere, sh_order=8)
csd_sf_enh_sharp = sh_to_sf(csd_shm_enh_sharp, sphere, sh_order=8)

# Normalize the sharpened ODFs
csd_sf_enh_sharp = csd_sf_enh_sharp * np.amax(csd_sf_orig)/np.amax(csd_sf_enh_sharp)

""" 
The end results are visualized. It can be observed that the end result after diffusion and sharpening is closer to the original noiseless dataset.
"""

ren = fvtk.ren()

# original ODF field
fodf_spheres_org = fvtk.sphere_funcs(csd_sf_orig_slice, sphere, scale=2, norm=False, radial_scale=True)
fodf_spheres_org.SetPosition(0, 35, 0)
fvtk.add(ren, fodf_spheres_org)

# ODF field with added noise
fodf_spheres = fvtk.sphere_funcs(csd_sf_noisy_slice, sphere, scale=2, norm=False, radial_scale=True)
fodf_spheres.SetPosition(0, 0, 0)
fvtk.add(ren, fodf_spheres)

# Enhancement of noisy ODF field
fodf_spheres_enh = fvtk.sphere_funcs(csd_sf_enh_slice, sphere, scale=2, norm=False, radial_scale=True)
fodf_spheres_enh.SetPosition(35, 0, 0)
fvtk.add(ren, fodf_spheres_enh)

# Additional sharpening
fodf_spheres_enh_sharp = fvtk.sphere_funcs(csd_sf_enh_sharp_slice*1.5, sphere, scale=2, norm=False, radial_scale=True)
fodf_spheres_enh_sharp.SetPosition(35, 35, 0)
fvtk.add(ren, fodf_spheres_enh_sharp)

#fvtk.show(ren, size=(600, 600))
fvtk.record(ren, out_path='enhancements.png', scale=2, size=(512, 512))

"""

.. figure:: enhancements.png
   :align: center

   The results after enhancements. Top-left: original noiseless data. Bottom-left: original data with added Rician noise (SNR=2). Bottom-left: After enhancement of noisy data. Top-right: After enhancement and sharpening of noisy data.

References
~~~~~~~~~~~~~~~~~~~~~~

.. [DuitsAndFranken2011] Duits, R. and Franken, E. (2011) Morphological and
                      Linear Scale Spaces for Fiber Enhancement in DWI-MRI.
                      J Math Imaging Vis, 46(3):326-368.
.. [Portegies2015] J. Portegies, G. Sanguinetti, S. Meesters, and R. Duits. (2015)
                 New Approximation of a Scale Space Kernel on SE(3) and
                 Applications in Neuroimaging. Fifth International
                 Conference on Scale Space and Variational Methods in
                 Computer Vision
.. [Portegies2015b] J. Portegies, R. Fick, G. Sanguinetti, S. Meesters, G.Girard,
                 and R. Duits. (2015) Improving Fiber Alignment in HARDI by 
                 Combining Contextual PDE flow with Constrained Spherical 
                 Deconvolution. PLoS One.

"""