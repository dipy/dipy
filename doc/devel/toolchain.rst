.. _toolchain-roadmap:

🛠️ Toolchain Roadmap
====================

The DIPY_ library relies on a set of primary dependencies, most notably Python,
NumPy and Nibabel_ to function effectively. Additionally, a broader array of
libraries and tools is necessary for both building the library and constructing
its documentation.

It's important to note that these tools and libraries are continuously evolving.
This document aims to outline how DIPY_ will manage its usage of these dynamic
dependencies over time.

DIPY_ strives to maintain compatibility across multiple releases of its
dependent libraries and tools. Requiring users to switch to other components
with every release would significantly reduce the value of DIPY_. Nevertheless,
retaining compatibility with older toolsets and libraries sets boundaries on
incorporating newer functionalities and capabilities.

DIPY_ adopts a moderately conservative approach by ensuring compatibility with
several major releases of Python and NumPy across major platforms. However,
this stance may introduce further constraints, as illustrated in the C Compilers
section.

- First and foremost, DIPY_ is a Python project, hence it requires a Python environment.
- Compilers for C code is needed, as well as for Cython.
- The Python environment needs ``NumPy`` and ``Nibabel`` package to be installed.
- Testing requires the ``pytest`` Python packages.
- Building the documentation requires Sphinx packages along with ``sphinx_design``,
  ``sphinxcontrib-bibtex``, ``sphinx-gallery``, PyData/GRG theme.

Building DIPY
--------------

Python Versions
^^^^^^^^^^^^^^^

DIPY is compatible with several versions of Python.  When dropping support for
older Python versions, DIPY takes guidance from `Scientific Python SPECS`_.
Python 2.7 support was dropped starting from DIPY 1.0.0

================  =======================================================================
 Date             Pythons supported
================  =======================================================================
 2018              Py2.7, Py3.4+ (DIPY 0.16.x is the last release to support Python 2.7)
 2019              Py3.5+
 mid-2020          Py3.6+
 2022              Py3.7+
 mid-2023          Py3.8+
 2024              Py3.9+
 2025              Py3.10+
================  =======================================================================

NumPy
^^^^^

While DIPY_ relies on NumPy, its releases are not bound to specific NumPy
releases. Instead, DIPY_ strives for compatibility with at least the four
preceding releases of NumPy. This approach necessitates writing DIPY_ using
features common to all these versions, rather than relying solely on the
latest NumPy features [1]_.

The table provided illustrates the compatible NumPy versions for each major
Python release.

=================  ========================    =======================
 DIPY_ version      Python versions             NumPy versions
=================  ========================    =======================
 0.16.0             2.7, >=3.4, <=3.7           >=1.8.2, <= 1.16.x
 1.0.0              >=3.5, <=3.7                >=1.13.3, <= 1.16.x
 1.1.0/1            >=3.5, <=3.8                >=1.13.3, <= 1.17.3
 1.2.0              >=3.6, <=3.8                >=1.14.5, <= 1.17.3
 1.3.0              >=3.6, <=3.8                >=1.14.5, <1.17.3
 1.4.0/1            >=3.6, <=3.9                >=1.14.5, <1.20.x
 1.5.0              >=3.7, <=3.10               >=1.16.x, <1.23.0
 1.6.0              >=3.7, <=3.11               >=1.16.x, <1.24.0
 1.7.0              >=3.8, <=3.11               >=1.19.5, <1.24.0
 1.8.0              >=3.8, <=3.12               >=1.19.5, <1.26.0
 1.9.0              >=3.9, <3.12                >=1.21.6, <1.27.0
 1.10.0             >=3.10, <3.13                >=1.21.6, <2.2.0
 1.11.0             >=3.10, <3.14                >=1.21.6, <2.2.0
=================  ========================    =======================

.. note::

    The table above is not exhaustive. In specific cases, such as a
    particular architecture, these requirements could vary.


Compilers
^^^^^^^^^

Building DIPY_ requires compilers for C as well as the python transpilers
Cython.

To maintain compatibility with a large number of platforms & setups, especially
where using the official wheels (or other distribution channels like Anaconda
or conda-forge) is not possible, DIPY_ tries to keep compatibility with older
compilers, on platforms that have not yet reached their official end-of-life.

As explained in more detail below, the current minimal compiler versions are:

==========  ===========================  ===============================  ============================
 Compiler    Default Platform (tested)    Secondary Platform (untested)    Minimal Version
==========  ===========================  ===============================  ============================
 GCC         Linux                        AIX, Alpine Linux, OSX           GCC 8.x
 LLVM        OSX                          Linux, FreeBSD, Windows          LLVM 10.x
 MSVC        Windows                      -                                Visual Studio 2019 (vc142)
==========  ===========================  ===============================  ============================

Note that the lower bound for LLVM is not enforced. Older versions should
work, but no version below LLVM 12 is tested regularly during development.
Please file an issue if you encounter a problem during compilation.

Official Builds
~~~~~~~~~~~~~~~

Currently, DIPY_ wheels are being built as follows:

================    ==============================   ==============================   =============================
 Platform            CI Base Images [2]_ [3]_ [4]_    Compilers                        Comment
================    ==============================   ==============================   =============================
Linux x86            ``ubuntu-22.04``                 GCC 10.2.1                       ``cibuildwheel``
Linux arm            ``docker-builder-arm64``         GCC 11.3.0                       ``cibuildwheel``
OSX x86              ``macOS-11``                     clang-13                         ``cibuildwheel``
OSX arm              ``macos-monterey-xcode:14``      clang-13.1.6                     ``cibuildwheel``
Windows              ``windows-2019``                 Microsoft Visual 2019 (14.20)    ``cibuildwheel``
================    ==============================   ==============================   =============================


Testing and Benchmarking
--------------------------

Testing and benchmarking require recent versions of:

=========================  ========  ====================================
 Tool                      Version    URL
=========================  ========  ====================================
pytest                     Recent     https://docs.pytest.org/en/latest/
asv (airspeed velocity)    Recent     https://asv.readthedocs.io/
=========================  ========  ====================================


Building the Documentation
--------------------------

====================  =================================================
 Tool                 Version
====================  =================================================
Sphinx                Whatever recent versions work. >= 6.0.
GRG Sphinx theme      Whatever recent versions work. >= 0.4.0.
Sphinx-Design         Whatever recent versions work. >= 0.5.0.
numpydoc              Whatever recent versions work. >= 1.8.0.
Sphinx-Gallery        Whatever recent versions work. >= 0.15.0.
Sphinxcontrib-bibtex  Whatever recent versions work. >= 2.6.1
====================  =================================================

.. note::

    Developer Note: The documentation build might require additional libraries
    based on the examples provided. Specifically, the versions of "numpy" and
    "nibabel" needed have implications for the Python docstring examples.

    It's essential for these examples to execute seamlessly in both the
    documentation build environment and any supported versions of "numpy/nibabel"
    that users might utilize with this DIPY_ release.


Packaging
---------

A Recent version of:

=============  ========  ===============================================
 Tool          Version    URL
=============  ========  ===============================================
meson-python   Recent     https://meson-python.readthedocs.io/en/latest/
wheel          Recent     https://pythonwheels.com
cibuildwheel   Recent     https://cibuildwheel.readthedocs.io/en/stable/
=============  ========  ===============================================

:ref:`release-guide` contain information on making and distributing a
DIPY_ release.

References
----------

.. [1] https://numpy.org/doc/stable/release.html
.. [2] https://github.com/actions/runner-images
.. [3] https://cirrus-ci.org/guide/docker-builder-vm/#under-the-hood
.. [4] https://github.com/orgs/cirruslabs/packages?tab=packages&q=macos


.. include:: ../links_names.inc
