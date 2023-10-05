""" This file contains defines parameters for DIPY that we use to fill
settings in setup.py, the DIPY top-level docstring, and for building the
docs.  In setup.py in particular, we exec this file, so it cannot import dipy
"""

# DIPY version information.  An empty _version_extra corresponds to a
# full release.  '.dev' as a _version_extra string means this is a development
# version
_version_major = 1
_version_minor = 8
_version_micro = 0
_version_extra = 'dev0'
# _version_extra = ''

# Format expected by setup.py and doc/source/conf.py: string of form "X.Y.Z"
__version__ = f"{_version_major}.{_version_minor}.{_version_micro}{_version_extra}"
