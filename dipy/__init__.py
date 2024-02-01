"""
Diffusion Imaging in Python
============================

For more information, please visit https://dipy.org

Subpackages
-----------
::

 align         -- Registration, streamline alignment, volume resampling
 core          -- Spheres, gradient tables
 core.geometry -- Spherical geometry, coordinate and vector manipulation
 core.meshes   -- Point distributions on the sphere
 data          -- Small testing datasets
 denoise       -- Denoising algorithms
 direction     -- Manage peaks and tracking
 io            -- Loading/saving of dpy datasets
 nn            -- Neural networks algorithms
 reconst       -- Signal reconstruction modules (tensor, spherical harmonics,
                  diffusion spectrum, etc.)
 segment       -- Tractography segmentation
 sims          -- MRI phantom signal simulation
 stats         -- Tractometry
 tracking      -- Tractography, metrics for streamlines
 viz           -- Visualization and GUIs
 workflows      -- Predefined Command line for common tasks

Utilities
---------
::

 test          -- Run unittests
 __version__   -- Dipy version

"""

from dipy.version import version as __version__  # noqa
import lazy_loader as lazy

__getattr__, __lazy_dir__, _ = lazy.attach_stub(__name__, __file__)
