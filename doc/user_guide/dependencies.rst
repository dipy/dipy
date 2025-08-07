.. _dependencies:

================================
Python versions and dependencies
================================

DIPY follows the `Scientific Python`_ `SPEC 0 — Minimum Supported Versions`_
recommendation as closely as possible, including the supported Python and
dependencies versions.

Further information can be found in :ref:`toolchain-roadmap`.

Dependencies
------------

Depends on a few standard libraries: python_ (the core language), numpy_ (for
numerical computation), scipy_ (for more specific mathematical operations),
cython_ (for extra speed), nibabel_ (for file formats; we require version 2.4
or higher), h5py_ (for handling large datasets), tqdm_ and trx-python_.
Optionally, it can use fury_ (for visualisation), matplotlib_ (for
scientific plotting), and ipython_ (for interaction with the code and its
results). cvxpy_, scikit-learn_, statsmodels_, pandas_ and tensorflow_ are
required for some modules.

.. include:: ../links_names.inc
