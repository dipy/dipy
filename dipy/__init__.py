"""
Diffusion Imaging in Python
============================

For more information, please visit http://dipy.org

Subpackages
-----------
::

 align         -- Registration, streamline alignment, volume resampling
 boots         -- Bootstrapping algorithms
 core          -- Spheres, gradient tables
 core.geometry -- Spherical geometry, coordinate and vector manipulation
 core.meshes   -- Point distributions on the sphere
 data          -- Small testing datasets
 denoise       -- Denoising algorithms
 direction     -- Manage peaks and tracking
 io            -- Loading/saving of dpy datasets
 reconst       -- Signal reconstruction modules (tensor, spherical harmonics,
                  diffusion spectrum, etc.)
 segment       -- Tractography segmentation
 sims          -- MRI phantom signal simulation
 tracking      -- Tractography, metrics for streamlines
 viz           -- Visualization and GUIs

Utilities
---------
::

 test          -- Run unittests
 __version__   -- Dipy version

"""
import sys

from .info import __version__
from .testing import setup_test

# Test callable
from numpy.testing import Tester
test = Tester().test
bench = Tester().bench
del Tester

# Plumb in version etc info stuff
from .pkg_info import get_pkg_info as _get_pkg_info
def get_info():
    from os.path import dirname
    return _get_pkg_info(dirname(__file__))
del sys
