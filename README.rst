.. image:: doc/_static/images/logos/dipy-logo.png
  :height: 180px
  :target: http://dipy.org
  :alt: DIPY - Diffusion Imaging in Python

|

.. image:: https://github.com/dipy/dipy/actions/workflows/test.yml/badge.svg?branch=master
  :target: https://github.com/dipy/dipy/actions/workflows/test.yml

.. image:: https://github.com/dipy/dipy/actions/workflows/build_docs.yml/badge.svg?branch=master
  :target: https://github.com/dipy/dipy/actions/workflows/build_docs.yml

.. image:: https://codecov.io/gh/dipy/dipy/branch/master/graph/badge.svg
  :target: https://codecov.io/gh/dipy/dipy

.. image:: https://img.shields.io/pypi/v/dipy.svg
  :target: https://pypi.python.org/pypi/dipy

.. image:: https://anaconda.org/conda-forge/dipy/badges/platforms.svg
  :target: https://anaconda.org/conda-forge/dipy

.. image:: https://anaconda.org/conda-forge/dipy/badges/downloads.svg
  :target: https://anaconda.org/conda-forge/dipy

.. image:: https://img.shields.io/badge/License-BSD%203--Clause-blue.svg
  :target: https://github.com/dipy/dipy/blob/master/LICENSE


DIPY [DIPYREF]_ is a python library for the analysis of MR diffusion imaging.

DIPY is for research only; please contact admins@dipy.org if you plan to deploy
in clinical settings.

Website
=======

Current information can always be found from the DIPY website - http://dipy.org

Mailing Lists
=============

Please see the DIPY community list at
https://mail.python.org/mailman3/lists/dipy.python.org/

Please see the users' forum at
https://github.com/dipy/dipy/discussions

Please join the gitter chatroom `here <https://gitter.im/dipy/dipy>`_.

Code
====

You can find our sources and single-click downloads:

* `Main repository`_ on Github.
* Documentation_ for all releases and current development tree.
* Download as a tar/zip file the `current trunk`_.

.. _main repository: http://github.com/dipy/dipy
.. _Documentation: http://dipy.org
.. _current trunk: http://github.com/dipy/dipy/archives/master


Installing DIPY
===============

DIPY can be installed using `pip`::

    pip install dipy

or using `conda`::

    conda install -c conda-forge dipy

For detailed installation instructions, including instructions for installing
from source, please read our `installation documentation <https://docs.dipy.org/stable/user_guide/installation.html>`_.

Python versions and dependencies
--------------------------------

DIPY follows the `Scientific Python`_ `SPEC 0 — Minimum Supported Versions`_
recommendation as closely as possible, including the supported Python and
dependencies versions.

Further information can be found in `Toolchain Roadmap <https://docs.dipy.org/stable/devel/toolchain.html>`_.

License
=======

DIPY is licensed under the terms of the BSD license.
Please see the `LICENSE file <https://github.com/dipy/dipy/blob/master/LICENSE>`_.

Contributing
============

We welcome contributions from the community. Please read our `Contributing guidelines <https://github.com/dipy/dipy/blob/master/.github/CONTRIBUTING.md>`_.

Reference
=========

.. [DIPYREF] E. Garyfallidis, M. Brett, B. Amirbekian, A. Rokem,
    S. Van Der Walt, M. Descoteaux, I. Nimmo-Smith and DIPY contributors,
    "DIPY, a library for the analysis of diffusion MRI data",
    Frontiers in Neuroinformatics, vol. 8, p. 8, Frontiers, 2014.


.. _`Scientific Python`: https://scientific-python.org/
.. _`SPEC 0 — Minimum Supported Versions`: https://scientific-python.org/specs/spec-0000/
