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
import sys

from dipy.version import version as __version__

# Plumb in version etc info stuff
from .pkg_info import get_pkg_info as _get_pkg_info


def get_info():
    from os.path import dirname
    return _get_pkg_info(dirname(__file__))


del sys

submodules = [
    'align',
    'core',
    'data',
    'denoise',
    'direction',
    'io',
    'nn',
    'reconst',
    'segment',
    'sims',
    'stats',
    'tracking',
    'utils',
    'viz',
    'workflows',
    'tests',
    'testing'
]

__all__ = submodules + ['__version__', 'setup_test', 'get_info']
