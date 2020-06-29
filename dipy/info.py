""" This file contains defines parameters for DIPY that we use to fill
settings in setup.py, the DIPY top-level docstring, and for building the
docs.  In setup.py in particular, we exec this file, so it cannot import dipy
"""

# DIPY version information.  An empty _version_extra corresponds to a
# full release.  '.dev' as a _version_extra string means this is a development
# version
_version_major = 1
_version_minor = 2
_version_micro = 0
_version_extra = 'dev'
# _version_extra = ''

# Format expected by setup.py and doc/source/conf.py: string of form "X.Y.Z"
__version__ = "%s.%s.%s%s" % (_version_major,
                              _version_minor,
                              _version_micro,
                              _version_extra)

CLASSIFIERS = ["Development Status :: 3 - Alpha",
               "Environment :: Console",
               "Intended Audience :: Science/Research",
               "License :: OSI Approved :: BSD License",
               "Operating System :: OS Independent",
               "Programming Language :: Python",
               "Topic :: Scientific/Engineering"]

description  = 'Diffusion MRI utilities in python'

# Note: this long_description is actually a copy/paste from the top-level
# README.rst, so that it shows up nicely on PyPI.  So please remember to edit
# it only in one place and sync it correctly.
long_description = """
======
 DIPY
======

DIPY is a python toolbox for analysis of MR diffusion imaging.

DIPY is for research only; please do not use results from DIPY for
clinical decisions.

Website
=======

Current information can always be found from the DIPY website - https://dipy.org

Mailing Lists
=============

Please see the developer's list at
http://mail.scipy.org/mailman/listinfo/nipy-devel

Code
====

You can find our sources and single-click downloads:

* `Main repository`_ on Github.
* Documentation_ for all releases and current development tree.
* Download as a tar/zip file the `current trunk`_.

.. _main repository: http://github.com/dipy/dipy
.. _Documentation: http://dipy.org
.. _current trunk: https://github.com/dipy/dipy/archive/master.zip

License
=======

DIPY is licensed under the terms of the BSD license.
Please see the LICENSE file in the dipy distribution.

DIPY uses other libraries also licensed under the BSD or the
MIT licenses.
"""

# versions for dependencies
# Check these versions against .travis.yml and requirements.txt
CYTHON_MIN_VERSION = '0.29'
NUMPY_MIN_VERSION = '1.12.0'
SCIPY_MIN_VERSION = '1.0'
NIBABEL_MIN_VERSION = '3.0.0'
H5PY_MIN_VERSION = '2.5.0'
PACKAGING_MIN_VERSION = '19.0'

# Main setup parameters
NAME                = 'dipy'
MAINTAINER          = "Eleftherios Garyfallidis"
MAINTAINER_EMAIL    = "neuroimaging@python.org"
DESCRIPTION         = description
LONG_DESCRIPTION    = long_description
URL                 = "http://dipy.org"
DOWNLOAD_URL        = "http://github.com/dipy/dipy/archives/master"
LICENSE             = "BSD license"
CLASSIFIERS         = CLASSIFIERS
AUTHOR              = "dipy developers"
AUTHOR_EMAIL        = "neuroimaging@python.org"
PLATFORMS           = "OS Independent"
MAJOR               = _version_major
MINOR               = _version_minor
MICRO               = _version_micro
ISRELEASE           = _version_extra == ''
VERSION             = __version__
PROVIDES            = ["dipy"]
REQUIRES            = ["numpy (>=%s)" % NUMPY_MIN_VERSION,
                       "scipy (>=%s)" % SCIPY_MIN_VERSION,
                       "nibabel (>=%s)" % NIBABEL_MIN_VERSION,
                       "h5py (>=%s)" % H5PY_MIN_VERSION,
                       "packaging (>=%s)" % PACKAGING_MIN_VERSION]
