"""pytest initialization."""

import numpy as np
import scipy
from distutils.version import LooseVersion

""" Set numpy print options to "legacy" for new versions of numpy

If imported into a file, pytest will run this before any doctests.

References
-----------
https://github.com/numpy/numpy/commit/710e0327687b9f7653e5ac02d222ba62c657a718
https://github.com/numpy/numpy/commit/734b907fc2f7af6e40ec989ca49ee6d87e21c495
https://github.com/nipy/nibabel/pull/556
"""

if LooseVersion(np.__version__) >= LooseVersion('1.14'):
    np.set_printoptions(legacy='1.13')

# Temporary fix until scipy release in October 2018
# must be removed after that
# print the first occurrence of matching warnings for each location
# (module + line number) where the warning is issued
if LooseVersion(np.__version__) >= LooseVersion('1.15') and \
        LooseVersion(scipy.version.short_version) <= '1.1.0':
    import warnings
    warnings.simplefilter("default")
