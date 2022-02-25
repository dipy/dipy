"""pytest initialization."""
import numpy as np
from packaging.version import Version
import warnings

""" Set numpy print options to "legacy" for new versions of numpy
 If imported into a file, pytest will run this before any doctests.

References
----------
https://github.com/numpy/numpy/commit/710e0327687b9f7653e5ac02d222ba62c657a718
https://github.com/numpy/numpy/commit/734b907fc2f7af6e40ec989ca49ee6d87e21c495
https://github.com/nipy/nibabel/pull/556
"""
if Version(np.__version__) >= Version('1.14'):
    np.set_printoptions(legacy='1.13')

warnings.simplefilter(action="default", category=FutureWarning)
warnings.simplefilter("always", category=UserWarning)
# List of files that pytest should ignore
collect_ignore = ["testing/decorators.py", ]
