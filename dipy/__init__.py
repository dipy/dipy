"""
DiPy: Analysis of MR diffusion imaging in Python
================================================

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
 external      -- Interfaces to external tools such as FSL
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

from .info import __version__

# Test callable
from numpy.testing import Tester
test = Tester().test
del Tester

# Plumb in version etc info stuff
import os
from .pkg_info import get_pkg_info as _get_pkg_info
get_info = lambda : _get_pkg_info(os.path.dirname(__file__))
del os
