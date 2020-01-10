.. _release1.1:

====================================
 Release notes for DIPY version 1.1
====================================

GitHub stats for 2019/08/06 - 2020/01/10 (tag: 1.0.0)

These lists are automatically generated, and may be incomplete or contain duplicates.

The following 11 authors contributed 392 commits.

* Ariel Rokem
* Derek Pisner
* Eleftherios Garyfallidis
* Eric Larson
* Francois Rheault
* Jon Haitz Legarreta Gorroño
* Rafael Neto Henriques
* Ricci Woo
* Ross Lawrence
* Serge Koudoro
* Shreyas Fadnavis


We closed a total of 130 issues, 52 pull requests and 81 regular issues;
this is the full list (generated with the script
:file:`tools/github_stats.py`):

Pull Requests (52):

* :ghpull:`2030`: [MAINT] Release 1.1.1
* :ghpull:`2029`: Use the array proxy instead of get_fdata()
* :ghpull:`2023`: [Upcoming] Release 1.1.0
* :ghpull:`2021`: Examples update
* :ghpull:`2005`: [CI] add python3.8 matrix
* :ghpull:`2009`: Upgrade on DKI implementations
* :ghpull:`2022`: Update pep8speaks configuration
* :ghpull:`2020`: Update get_fnames
* :ghpull:`2018`: The Python is dead. Long live the Python
* :ghpull:`2017`: Update CONTRIBUTING.md
* :ghpull:`2016`: replace get_fdata by load_nifti_data
* :ghpull:`2012`: Add ability to analyze entire streamline in connectivity_matrix
* :ghpull:`2015`: Update nibabel minimum version (3.0.0)
* :ghpull:`1984`: Enabling TensorFlow 2 as an optional dependency
* :ghpull:`2013`: SFT constructor from another SFT
* :ghpull:`1862`: Use logging instead of print statements
* :ghpull:`1952`: Adding lpca, mppca and gibbs ringing workflows
* :ghpull:`1997`: [FIX] Avoid copy of  Streamlines data
* :ghpull:`2008`: Fix origin nomenclature sft (center/corner)
* :ghpull:`1965`: Horizon updates b1
* :ghpull:`2011`: BF: Use the `sklearn.base` interface, instead of deprecated `sklearn.linear_model.base`
* :ghpull:`2002`: BUG: Fix workflows IO `FetchFlow` `all` dataset fetching
* :ghpull:`2000`: NF: SplitFlow
* :ghpull:`1999`: add cenir_multib to dipy_fetch worflows
* :ghpull:`1998`: [DOC] Workflows Tutorial
* :ghpull:`1988`: DOC: Multi-Shell Multi-Tissue CSD Example
* :ghpull:`1975`: Azure CI
* :ghpull:`1994`: MAINT: Update gradient for SciPy deprecation
* :ghpull:`1711`: ENH: Improve the point distribution algorithm over the sphere.
* :ghpull:`1989`: DOC: Fix parameter docstring in `dipy.io.streamline.load_tractogram`
* :ghpull:`1987`: BF: use MAM distance when that is requested.
* :ghpull:`1986`: Fix the random state for SLR and Recobundles in bundle_extraction.
* :ghpull:`1977`: Add a warning on attempted import.
* :ghpull:`1983`: [Fix] one example
* :ghpull:`1981`: DOC: Improve the `doc/examples/README` file contents
* :ghpull:`1980`: DOC: Improve the `doc/README` file contents
* :ghpull:`1978`: DOC: Add extension to `doc/examples/README` file
* :ghpull:`1979`: DOC: Change the `doc/README` file extension to lowercase
* :ghpull:`1972`: Small fix in warning message
* :ghpull:`1956`: stateful_tractogram_post_1.0_fixes
* :ghpull:`1971`: fix broken link of devel
* :ghpull:`1970`: DOC: Fix typos
* :ghpull:`1929`: ENH: Relocate the `sim_response` method to allow re-use
* :ghpull:`1966`: ENH: Speed up Cython, remove warnings
* :ghpull:`1967`: ENH: Use language level 3
* :ghpull:`1962`: DOC: Display `master` branch AppVeyor status
* :ghpull:`1961`: STYLE: Remove unused import statements
* :ghpull:`1963`: DOC: Fix parameter docstring in LiFE tracking script
* :ghpull:`1900`: ENH: Add streamline deformation example and workflow
* :ghpull:`1948`: Minor Fix for Cross-Validation Example
* :ghpull:`1951`: [FIX] update some utilities scripts
* :ghpull:`1958`: Adding Missing References in SHORE docs

Issues (81):

* :ghissue:`2021`: Examples update
* :ghissue:`2005`: [CI] add python3.8 matrix
* :ghissue:`1197`: Documentation improvements
* :ghissue:`1959`: DIPY open lab meetings -- fall 2019
* :ghissue:`2003`: Artefacts in DKI reconstruction
* :ghissue:`2009`: Upgrade on DKI implementations
* :ghissue:`2022`: Update pep8speaks configuration
* :ghissue:`2020`: Update get_fnames
* :ghissue:`1777`: Maybe `read_*` should return full paths?
* :ghissue:`1634`: Use logging instead of print statements
* :ghissue:`2018`: The Python is dead. Long live the Python
* :ghissue:`2017`: Update CONTRIBUTING.md
* :ghissue:`1949`: Update Horizon
* :ghissue:`2016`: replace get_fdata by load_nifti_data
* :ghissue:`2012`: Add ability to analyze entire streamline in connectivity_matrix
* :ghissue:`2015`: Update nibabel minimum version (3.0.0)
* :ghissue:`2006`: get_data deprecated
* :ghissue:`1984`: Enabling TensorFlow 2 as an optional dependency
* :ghissue:`2013`: SFT constructor from another SFT
* :ghissue:`1862`: Use logging instead of print statements
* :ghissue:`1952`: Adding lpca, mppca and gibbs ringing workflows
* :ghissue:`1997`: [FIX] Avoid copy of  Streamlines data
* :ghissue:`2014`: Removes the appveyor configuration file.
* :ghissue:`2008`: Fix origin nomenclature sft (center/corner)
* :ghissue:`1965`: Horizon updates b1
* :ghissue:`2011`: BF: Use the `sklearn.base` interface, instead of deprecated `sklearn.linear_model.base`
* :ghissue:`2010`: Issue with sklearn update 0.22 - Attribute Error
* :ghissue:`2002`: BUG: Fix workflows IO `FetchFlow` `all` dataset fetching
* :ghissue:`1995`: ModuleNotFoundError: No module named 'numpy.testing.decorators' (Numpy 1.18)
* :ghissue:`2000`: NF: SplitFlow
* :ghissue:`1999`: add cenir_multib to dipy_fetch worflows
* :ghissue:`1998`: [DOC] Workflows Tutorial
* :ghissue:`1993`: [ENH][BF] Error handling for IVIM VarPro NLLS
* :ghissue:`1870`: Documentation example  of MSMT CSD
* :ghissue:`1988`: DOC: Multi-Shell Multi-Tissue CSD Example
* :ghissue:`1975`: Azure CI
* :ghissue:`1953`: Deprecate Reconst*Flow
* :ghissue:`1994`: MAINT: Update gradient for SciPy deprecation
* :ghissue:`1992`: MD bad / FA good
* :ghissue:`1711`: ENH: Improve the point distribution algorithm over the sphere.
* :ghissue:`184`: RF/OPT: Use scipy.optimize instead of dipy.core.sphere.disperse_charges
* :ghissue:`1989`: DOC: Fix parameter docstring in `dipy.io.streamline.load_tractogram`
* :ghissue:`1982`: small bug in recobundles for mam
* :ghissue:`1987`: BF: use MAM distance when that is requested.
* :ghissue:`1986`: Fix the random state for SLR and Recobundles in bundle_extraction.
* :ghissue:`1976`: Better warning when dipy is installed without fury
* :ghissue:`1977`: Add a warning on attempted import.
* :ghissue:`785`: WIP: TST: Add OSX to Travis.
* :ghissue:`1859`: Update Travis design
* :ghissue:`1950`: StatefullTractogram error in Horizon
* :ghissue:`1983`: [Fix] one example
* :ghissue:`1930`: dipy.io.peaks.load_peaks() IOError: This function supports only PAM5 (HDF5) files
* :ghissue:`1981`: DOC: Improve the `doc/examples/README` file contents
* :ghissue:`1980`: DOC: Improve the `doc/README` file contents
* :ghissue:`1978`: DOC: Add extension to `doc/examples/README` file
* :ghissue:`1979`: DOC: Change the `doc/README` file extension to lowercase
* :ghissue:`1968`: Broken site links (404) on new website when accessed from google
* :ghissue:`1972`: Small fix in warning message
* :ghissue:`1960`: DOC: MCSD Tutorial
* :ghissue:`1867`: WIP: CircleCI
* :ghissue:`1956`: stateful_tractogram_post_1.0_fixes
* :ghissue:`1971`: fix broken link of devel
* :ghissue:`1970`: DOC: Fix typos
* :ghissue:`1929`: ENH: Relocate the `sim_response` method to allow re-use
* :ghissue:`1966`: ENH: Speed up Cython, remove warnings
* :ghissue:`1967`: ENH: Use language level 3
* :ghissue:`1954`: WIP: Second series of updates and fixes for Horizon
* :ghissue:`1964`: Added a line break
* :ghissue:`1962`: DOC: Display `master` branch AppVeyor status
* :ghissue:`1961`: STYLE: Remove unused import statements
* :ghissue:`1963`: DOC: Fix parameter docstring in LiFE tracking script
* :ghissue:`1900`: ENH: Add streamline deformation example and workflow
* :ghissue:`1840`: Tutorial showing how to apply deformations to streamlines
* :ghissue:`1948`: Minor Fix for Cross-Validation Example
* :ghissue:`1951`: [FIX] update some utilities scripts
* :ghissue:`1841`: Lab meetings, summer 2019
* :ghissue:`1958`: Adding Missing References in SHORE docs
* :ghissue:`1955`: where is intersphinx inventory under new website domain?
* :ghissue:`1401`: Workflows documentation needs more work
* :ghissue:`1442`: ENH: Minor presentation enhancement to AffineMap object
* :ghissue:`1791`: ENH: Save seeds in TRK/TCK
