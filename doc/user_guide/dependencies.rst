.. _dependencies:

================================
Python versions and dependencies
================================

DIPY follows the `Scientific Python`_ `SPEC 0 — Minimum Supported Versions`_
recommendation as closely as possible. SPEC 0 defines the minimum supported versions 
of Python and core dependencies based on their release dates.

Further information can be found in :ref:`toolchain-roadmap`.

Dependencies
------------

Core dependencies (required)
------------------------------

DIPY depends on the following third-party Python packages:

* numpy - numerical computation and array operations
* scipy - mathematical and scientific operations
* cython - performance-critical compiled extensions
* nibabel_ - reading and writing neuroimaging file formats
* h5py_ - handling large datasets in HDF5 formats
* tqdm_ - progress bars for long-running operations
* trx-python_ - tractography file format support

Optional dependencies
----------------------

The following packages are optional but enable additional features:

* fury_ - 3D visualisation of imaging data
* matplotlib_ - scientific plotting and 2D visualisation
* ipython_ - interactive computing and code exploration

Module-specific dependencies
-----------------------------

The following packages are required only for specific DIPY modules:

* cvxpy_ - convex optimization (required for some reconstruction models)
* scikit-learn_ - machine learning utilities
* statsmodels_ - statistical analysis
* pandas - data manipulation and analysis
* tensorflow_ - deep learning models

.. include:: ../links_names.inc
