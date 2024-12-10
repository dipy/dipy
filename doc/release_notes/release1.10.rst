.. _release1.10:

=====================================
 Release notes for DIPY version 1.10
=====================================

GitHub stats for 2024/03/08 - 2024/12/09 (tag: 1.9.0)

These lists are automatically generated, and may be incomplete or contain duplicates.

The following 25 authors contributed 871 commits.

* Alex Rockhill
* Ariel Rokem
* Asa Gilmore
* Atharva Shah
* Bramsh Qamar
* Charles Poirier
* Eleftherios Garyfallidis
* Eric Larson
* Florent Wijanto
* Gabriel Girard
* Jon Haitz Legarreta Gorroño
* Jong Sung Park
* Julio Villalon
* Kaibo Tang
* Kaustav Deka
* Maharshi Gor
* Martin Kozár
* Matt Cieslak
* Matthew Feickert
* Michael R. Crusoe
* Prajwal Reddy
* Rafael Neto Henriques
* Sam Coveney
* Sandro Turriate
* Serge Koudoro


We closed a total of 445 issues, 187 pull requests and 258 regular issues;
this is the full list (generated with the script
:file:`tools/github_stats.py`):

Pull Requests (187):

* :ghpull:`3420`: CI: define output path for coverage
* :ghpull:`3415`: tests: leave no trace behind
* :ghpull:`3417`: MAINT: Address changes to API with new (1.6) version of cvxpy.
* :ghpull:`3414`: Bump codecov/codecov-action from 4 to 5 in the actions group
* :ghpull:`3412`: BF: Fixing single-slice data error in Patch2Self
* :ghpull:`3409`: [RF] Cleanup deprecated functions and remove some old legacy scripts
* :ghpull:`3407`: CI: unpin cvxpy
* :ghpull:`3408`: RF: fix step argument in DTI NLLS model
* :ghpull:`3402`: DOC: Add Exporting and importing Mapping field information in recipes and registration tutorial
* :ghpull:`3403`: Robust split idea rebase tidy
* :ghpull:`3404`: RF: Improve and Simplify version management
* :ghpull:`3395`: RF: Refactor `nn` module and deprecate `tensorflow` module
* :ghpull:`1957`: niftis2pam, pam2niftis Workflow
* :ghpull:`3396`: DOC: Adopt `sphinxcontrib-bibtex` for reconst model list refs
* :ghpull:`3399`: STYLE: Remove empty quoted paragraph in developer guide index
* :ghpull:`3398`: DOC: Improve first interaction GitHub Actions config file
* :ghpull:`2826`: [ENH] Compute fiber density and fiber spread from ODF using Bingham distributions
* :ghpull:`3303`: NF: Patch2Self3
* :ghpull:`3392`: [WIP] NF: Adding pytorch versions
* :ghpull:`3368`: [NF] DAM implementation for tissue classification using DMRI signal properties.
* :ghpull:`3390`: DOC: Update DTI tutorial title
* :ghpull:`3391`: STYLE: removing pep8speaks conf file in favor of pre-commit action
* :ghpull:`3393`: RF: fix API generation
* :ghpull:`3387`: DOC: Add first interaction GHA workflow file
* :ghpull:`3386`: DOC: Update the CI tool to GHA in `CONTRIBUTING` file
* :ghpull:`3384`: BF: Updated non_local_means
* :ghpull:`3140`: NF: Adding correct_mask to median_otsu
* :ghpull:`3345`: DOC: Skip element in documentation generation
* :ghpull:`3372`: BugFix: New Atlas OMM not working with Horizon
* :ghpull:`3381`: RF: Add support for sequential processing in Gibbs unringing
* :ghpull:`3380`: ensure all calls to a python executable are to `python3`
* :ghpull:`3376`: DOC: Use placeholder for unused variable in `streamline_tools`
* :ghpull:`3373`: DOC: Consider warnings as errors in documentation CI build
* :ghpull:`3379`: DOC: Remove example files labels
* :ghpull:`3378`: doc: Link reconstruction model list to multiple pages
* :ghpull:`3377`: DOC: Miscellaneous improvements to `PeakActor` docstring
* :ghpull:`3375`: DOC: Reference footnote in `streamline_tools`
* :ghpull:`3348`: DOC: Address remaining some warnings
* :ghpull:`3369`: ci: Bump scientific-python/upload-nightly-action from 0.6.0 to 0.6.1
* :ghpull:`3367`: Bump scientific-python/upload-nightly-action from 0.5.0 to 0.6.0 in the actions group
* :ghpull:`3366`: DOC: Make `rng` optional parameter docstrings consistent
* :ghpull:`3365`: DOC: Fix some cites.
* :ghpull:`3356`: BF: fix s390x compatibility
* :ghpull:`3360`: DOC: Remove unnecessary leading whitespace in rst doc paragraph
* :ghpull:`3357`: FIX: remove keyword only warning on examples (part2)
* :ghpull:`3343`: BF Fixing transformation function
* :ghpull:`3355`: FIX: missing keyword only arguments on example
* :ghpull:`3221`: Updating BundleWarp default value of alpha
* :ghpull:`3323`: BF: Allow passing kwargs in fit method, by moving parallelization kwargs elsewhere, including PEP 3102
* :ghpull:`3351`: DOC: Fix miscellaneous documentation build warnings (part 3)
* :ghpull:`3306`: NF: Update to examples
* :ghpull:`3293`: BF: Fix attempting to delete frame local symbol table variable
* :ghpull:`3257`: NF: Applying Decorators in Module (Reconst)
* :ghpull:`3254`: NF: Applying Decorators in Module (Direction)
* :ghpull:`3317`: DOC: Miscellaneous documentation improvements
* :ghpull:`3350`: DOC: Do not use the `scale` option for URL-based images
* :ghpull:`3344`: DOC: Fix miscellaneous documentation build warnings (part 2)
* :ghpull:`3346`: RF: Removal of keyword form Cython files
* :ghpull:`3341`: DOC: Host MNI template note references in references file
* :ghpull:`3333`: RF: Decorator fix
* :ghpull:`3335`: RF: Allow parallel processing for sphinx extension
* :ghpull:`3342`: RF: Doctest warnings
* :ghpull:`3337`: DOC: Fix miscellaneous documentation build warnings
* :ghpull:`3338`: DOC: Cite examples references using `sphinxcontrib-bibtex`
* :ghpull:`3319`: DOC: Use references bibliography file for DIPY citation file
* :ghpull:`3321`: BF: Set the superclass `fit_method` param value to the one provided
* :ghpull:`3324`: RF: Refactored for keyword arguments
* :ghpull:`3340`: CI: pin cvxpy  to 1.4.4 until 1.5.x issues are solved
* :ghpull:`3316`: DOC: Cite code base references using `sphinxcontrib-bibtex`
* :ghpull:`3332`: BF: Set the `Diso` parameter value to the one provided
* :ghpull:`3325`: DOC: Fix warnings related to displayed math expressions
* :ghpull:`3331`: DOC: Miscellaneous documentation improvements (part 3)
* :ghpull:`3329`: STYLE: Use a leading underscore to name private methods
* :ghpull:`3330`: DOC: Do not use unfinished double backticks
* :ghpull:`3320`: DOC: Miscellaneous documentation improvements (part 2)
* :ghpull:`3318`: RF: Remove unused parameters from method signature
* :ghpull:`3310`: DOC: Cite `nn` references through `sphinxcontrib-bibtex`
* :ghpull:`3315`: RF: remove legacy numpydoc
* :ghpull:`2810`: [DOC] introducing sphinxcontrib-Bibtex to improve reference management
* :ghpull:`3312`: DOC: Use `misc` for other types of BibTeX entries
* :ghpull:`3309`: DOC: Miscellaneous doc formatting fixes (part 4)
* :ghpull:`3308`: DOC: Rework the BibTeX bibliography file
* :ghpull:`3275`: FIX: remove sagital from codespellrc ignore list |# codespell:ignore sagital|
* :ghpull:`3304`: DOC: Miscellaneous doc formatting fixes (part 3)
* :ghpull:`3295`: ENH: Add a GHA workflow file to build docs
* :ghpull:`3302`: DOC: Miscellaneous doc formatting fixes (part 2)
* :ghpull:`3301`: FIX: explicit keyword argument for Horizon
* :ghpull:`3297`: DOC: Miscellaneous doc formatting fixes
* :ghpull:`3291`: FIX: nightly wheels for macOS arm64
* :ghpull:`3262`: NF: Applying Decorators in Module (Visualization)
* :ghpull:`3263`: NF: Applying Decorators in Module (Workflow)
* :ghpull:`3287`: NF: Add `__len__` to `GradientTable`
* :ghpull:`3260`: NF: Applying Decorators in Module (Tracking)
* :ghpull:`3256`: NF: Applying Decorators in Module (NeuralNetwork)
* :ghpull:`3258`: NF: Applying Decorators in Module (Segment)
* :ghpull:`3249`: NF: Applying Decorators in Module (Align)
* :ghpull:`3251`: NF: Applying Decorators in Module (Core)
* :ghpull:`3279`: FIX: Explicit type origin for long to solve the cython error during compilation
* :ghpull:`3259`: NF: Applying Decorators in Module (Sims)
* :ghpull:`3252`: NF: Applying Decorators in Module (Denoise)
* :ghpull:`3261`: NF: Applying Decorators in Module (Utils)
* :ghpull:`3255`: NF: Applying Decorators in Module (Io)
* :ghpull:`3253`: NF: Applying Decorators in Module (Data)
* :ghpull:`3233`: STYLE: Set `stacklevel` argument explicitly to warning messages
* :ghpull:`3239`: NF: Decorator for keyword-only argument
* :ghpull:`2593`: Embed parallelization into the multi_voxel_fit decorator.
* :ghpull:`3274`: RF: Update pyproject.toml for numpy 2.0
* :ghpull:`3273`: STYLE: Make statement dwell on a single line
* :ghpull:`3237`: Add support for tensor-valued spherical functions in `interp_rbf`
* :ghpull:`3245`: RF: Switch from using sparse `*_matrix` to `*_array`.
* :ghpull:`3267`: STYLE: Avoid deprecated NumPy types and methods for NumPy 2.0 compat
* :ghpull:`3264`: TEST: avoid direct comparison of floating point numbers
* :ghpull:`3268`: STYLE: Prefer using `np.asarray` to avoid copy while creating an array
* :ghpull:`3271`: RF: Do not use `np.any` for checking optional array parameters
* :ghpull:`3250`: DOC: Fix param order
* :ghpull:`3269`: STYLE: Prefer using `isin` over `in1d`
* :ghpull:`3238`: NF - add affine to peaks_from_position
* :ghpull:`3247`: STYLE: Add imported symbols to __all__ in direction module
* :ghpull:`3246`: STYLE: Import explicitly `direction.peaks` symbols
* :ghpull:`3241`: RF: Codespell fix for CI
* :ghpull:`3228`: STYLE: Fix unused loop control variable warning
* :ghpull:`3235`: STYLE: Do not allow running unintended modules as scripts
* :ghpull:`3230`: STYLE: Fix function definition loop variable binding warning
* :ghpull:`3232`: STYLE: Simplify implicitly concatenated strings
* :ghpull:`3229`: STYLE: Prefer using f-strings
* :ghpull:`3224`: BF: Rewrite list creation as `list()` instead of `[]`
* :ghpull:`3216`: STYLE: Format code using `ruff`
* :ghpull:`3178`: DOC: Fixes the AFQ tract profile tutorial.
* :ghpull:`3218`: STYLE: Fix codespell issues
* :ghpull:`3209`: [CI] Move filterwarnings from pyproject to conftest
* :ghpull:`3220`: [RF] from `os.fork` to `spawn` for multiprocessing
* :ghpull:`3214`: RF - remove buffer argument in pmf_gen.get_pmf_value(.)
* :ghpull:`3219`: [ENH] Prefer CLARABEL over ECOS as the CVXPY solver
* :ghpull:`3215`: tests: correct module-level setup
* :ghpull:`3211`: [RF] PMF Gen: from memoryview to pointer
* :ghpull:`3210`: Python 3.13: Fix tests for next Python release
* :ghpull:`3212`: STYLE: Relocate `pre-commit` and `ruff` packages to style requirements
* :ghpull:`3205`: BF: Declare variables holding integers as `cnp.npy_intp` over `double`
* :ghpull:`3174`: NF - initial directions from seed positions
* :ghpull:`3207`: DOC: Fix Cython method parameter type description
* :ghpull:`3206`: BF: Use `cnp.npy_intp` instead of `int` as counter
* :ghpull:`3204`: DOC: Fix documentation typos
* :ghpull:`3202`: [TEST] Add flag to turn warnings into errors for pytest
* :ghpull:`3158`: ENH: Remove filtering `UserWarning` warnings in test config file
* :ghpull:`3194`: MAINT: fix warning
* :ghpull:`3199`: Bump pre-commit/action from 3.0.0 to 3.0.1 in the actions group
* :ghpull:`3182`: [NF] Add DiSCo challenge data fetcher
* :ghpull:`3197`: ENH: Fix miscellaneous warnings in `dki` reconstruction module
* :ghpull:`3198`: ENH: Ensure that `arccos` argument is in the [-1,1] range
* :ghpull:`3191`: [RF] allow float and double for `trilinear_interpolate4d_c`
* :ghpull:`3151`: DKI Updates: (new radial tensor kurtosis metric, updated documentation and missing tests)
* :ghpull:`3189`: Update affine_registration to clarify returns and make them consistent with docstring
* :ghpull:`3176`: ENH: allow vol_idx in align workflow
* :ghpull:`3188`: ENH: Add `pre-commit` to project `dev` dependencies
* :ghpull:`3183`: ENH: Specify the solver for the MAP-MRI positivity constraint test
* :ghpull:`3184`: STYLE: Sort import statements using `ruff`
* :ghpull:`3181`: [PEP8] fix pep8 and docstring style in `dti.py` file
* :ghpull:`3177`: Loading Peaks faster with complete range and synchronization functionality.
* :ghpull:`3180`: BF: Fix bug in mode for isotropic tensors
* :ghpull:`3172`: [ENH] Enable range for dipy_median_otsu workflow
* :ghpull:`3171`: Clean up for tabs and tab manager
* :ghpull:`3168`: Feature/peaks tab revamp
* :ghpull:`3128`: NF: Fibonacci Hemisphere
* :ghpull:`3153`: ENH: add save peaks to dipy_fit_dti, dki
* :ghpull:`3156`: ENH: Implement NDC from Yeh2019
* :ghpull:`3161`: DOC: Fix `tri` parameter docstring in `viz.projections.sph_project`
* :ghpull:`3163`: STYLE: Make `fury` and `matplotlib` presence message in test consistent
* :ghpull:`3162`: ENH: Fix variable potentially being referenced before assignment
* :ghpull:`3144`: ROI tab revamped
* :ghpull:`2982`: [FIX] Force the use of pre-wheels
* :ghpull:`3134`: Feature/cluster revamp
* :ghpull:`3146`: [NF] Add 30 Bundle brain atlas fetcher
* :ghpull:`3150`: BUG: Fix bug with nightly wheel build
* :ghpull:`3149`: ENH: Miscellaneous cleanup
* :ghpull:`3148`: ENH: Fix HDF5 key warning when saving BUAN profile data
* :ghpull:`3138`: [CI] update CI's script
* :ghpull:`3126`: Bugfix for ROI images updates
* :ghpull:`3141`: ENH: Fix miscellaneous warnings
* :ghpull:`3139`: BF: Removing Error/Warning from Tensorflow 2.16
* :ghpull:`3132`: BF: Removed allow_break
* :ghpull:`3135`: DOC: Fix documentation URLs
* :ghpull:`3133`: grg-sphinx-theme added as dependency
* :ghpull:`3127`: Feature/viz interface tutorials
* :ghpull:`3120`: DOC - Removed unnecessary line from tracking example
* :ghpull:`3110`: Viz cli tutorial updated
* :ghpull:`3086`: [RF] Fix spherical harmonic terminology swap
* :ghpull:`3095`: [UPCOMING] Release preparation for 1.9.0

Issues (258):

* :ghissue:`3420`: CI: define output path for coverage
* :ghissue:`3415`: tests: leave no trace behind
* :ghissue:`3417`: MAINT: Address changes to API with new (1.6) version of cvxpy.
* :ghissue:`3414`: Bump codecov/codecov-action from 4 to 5 in the actions group
* :ghissue:`2469`: Error in patch2self for single-slice data
* :ghissue:`3412`: BF: Fixing single-slice data error in Patch2Self
* :ghissue:`1531`: Test suite fails with errors regarding the ConvexHull of scipy.spatial.qhull objects
* :ghissue:`3409`: [RF] Cleanup deprecated functions and remove some old legacy scripts
* :ghissue:`3410`: `radial_scale` parameter of `actor.odf_slicer` has some issues
* :ghissue:`3407`: CI: unpin cvxpy
* :ghissue:`3030`: I do not see a way to change step as used by reconst.dti.TensorModel.fit()
* :ghissue:`3408`: RF: fix step argument in DTI NLLS model
* :ghissue:`3361`: Exporting and importing SymmetricDiffeomorphicRegistration outputs
* :ghissue:`3402`: DOC: Add Exporting and importing Mapping field information in recipes and registration tutorial
* :ghissue:`3170`: Iteratively reweighted least squares for robust fitting
* :ghissue:`3358`: robust algorithm REBASE
* :ghissue:`3403`: Robust split idea rebase tidy
* :ghissue:`3115`: Fix `get_info` for release package
* :ghissue:`3404`: RF: Improve and Simplify version management
* :ghissue:`3401`: Robust split idea rebase arokem
* :ghissue:`3395`: RF: Refactor `nn` module and deprecate `tensorflow` module
* :ghissue:`1957`: niftis2pam, pam2niftis Workflow
* :ghissue:`3396`: DOC: Adopt `sphinxcontrib-bibtex` for reconst model list refs
* :ghissue:`3399`: STYLE: Remove empty quoted paragraph in developer guide index
* :ghissue:`3398`: DOC: Improve first interaction GitHub Actions config file
* :ghissue:`2826`: [ENH] Compute fiber density and fiber spread from ODF using Bingham distributions
* :ghissue:`3169`: [RF] Add peaks generation to reconst workflows
* :ghissue:`3303`: NF: Patch2Self3
* :ghissue:`3392`: [WIP] NF: Adding pytorch versions
* :ghissue:`3368`: [NF] DAM implementation for tissue classification using DMRI signal properties.
* :ghissue:`3389`: Single tensor tutorial - hard to find
* :ghissue:`3390`: DOC: Update DTI tutorial title
* :ghissue:`3391`: STYLE: removing pep8speaks conf file in favor of pre-commit action
* :ghissue:`3393`: RF: fix API generation
* :ghissue:`3387`: DOC: Add first interaction GHA workflow file
* :ghissue:`3386`: DOC: Update the CI tool to GHA in `CONTRIBUTING` file
* :ghissue:`3384`: BF: Updated non_local_means
* :ghissue:`3285`: Awkward interaction of dipy.denoise.non_local_means.non_local_means and dipy.denoise.noise_estimate.estimate_sigma
* :ghissue:`3140`: NF: Adding correct_mask to median_otsu
* :ghissue:`3345`: DOC: Skip element in documentation generation
* :ghissue:`3372`: BugFix: New Atlas OMM not working with Horizon
* :ghissue:`2757`: Use for loop when `num_processes=1` in gibbs_removal()
* :ghissue:`3381`: RF: Add support for sequential processing in Gibbs unringing
* :ghissue:`3380`: ensure all calls to a python executable are to `python3`
* :ghissue:`3376`: DOC: Use placeholder for unused variable in `streamline_tools`
* :ghissue:`3373`: DOC: Consider warnings as errors in documentation CI build
* :ghissue:`3379`: DOC: Remove example files labels
* :ghissue:`3374`: DOC: Remove `tracking_introduction_eudx` from quick start
* :ghissue:`3347`: Reconstruction model list not linked in documentation since it cannot be located
* :ghissue:`3378`: doc: Link reconstruction model list to multiple pages
* :ghissue:`2665`: DOC: Improve the CLI documentation rendering
* :ghissue:`3377`: DOC: Miscellaneous improvements to `PeakActor` docstring
* :ghissue:`3375`: DOC: Reference footnote in `streamline_tools`
* :ghissue:`3326`: Avoid Sphinx warnings from inherited third-party method documentation
* :ghissue:`3348`: DOC: Address remaining some warnings
* :ghissue:`3349`: DOC: Fix footbibliography-related errors in workflow help doc
* :ghissue:`3370`: dipy_buan_profiles CLI IndexError
* :ghissue:`3369`: ci: Bump scientific-python/upload-nightly-action from 0.6.0 to 0.6.1
* :ghissue:`3367`: Bump scientific-python/upload-nightly-action from 0.5.0 to 0.6.0 in the actions group
* :ghissue:`3366`: DOC: Make `rng` optional parameter docstrings consistent
* :ghissue:`3248`: [NF] Multicompartment DWI simulation technique implementation
* :ghissue:`3365`: DOC: Fix some cites.
* :ghissue:`3363`: Avoid SyntaxWarnings due to embedded LaTeX
* :ghissue:`2886`: test_streamwarp.py: Little-endian buffer not supported on big-endian compiler
* :ghissue:`3356`: BF: fix s390x compatibility
* :ghissue:`3360`: DOC: Remove unnecessary leading whitespace in rst doc paragraph
* :ghissue:`3357`: FIX: remove keyword only warning on examples (part2)
* :ghissue:`3343`: BF Fixing transformation function
* :ghissue:`3355`: FIX: missing keyword only arguments on example
* :ghissue:`2143`: Build template CLI
* :ghissue:`3221`: Updating BundleWarp default value of alpha
* :ghissue:`3286`: BF: Allow passing kwargs in `fit` method, by moving parallelization kwargs elsewhere
* :ghissue:`3323`: BF: Allow passing kwargs in fit method, by moving parallelization kwargs elsewhere, including PEP 3102
* :ghissue:`3351`: DOC: Fix miscellaneous documentation build warnings (part 3)
* :ghissue:`3306`: NF: Update to examples
* :ghissue:`3292`: Python 3.13: `TypeError: cannot remove variables from FrameLocalsProxy` in tests
* :ghissue:`3293`: BF: Fix attempting to delete frame local symbol table variable
* :ghissue:`3257`: NF: Applying Decorators in Module (Reconst)
* :ghissue:`3254`: NF: Applying Decorators in Module (Direction)
* :ghissue:`3317`: DOC: Miscellaneous documentation improvements
* :ghissue:`3350`: DOC: Do not use the `scale` option for URL-based images
* :ghissue:`3344`: DOC: Fix miscellaneous documentation build warnings (part 2)
* :ghissue:`3346`: RF: Removal of keyword form Cython files
* :ghissue:`2394`: Documentation References - Remove (1, 2, ...)
* :ghissue:`3341`: DOC: Host MNI template note references in references file
* :ghissue:`3333`: RF: Decorator fix
* :ghissue:`3335`: RF: Allow parallel processing for sphinx extension
* :ghissue:`3342`: RF: Doctest warnings
* :ghissue:`3337`: DOC: Fix miscellaneous documentation build warnings
* :ghissue:`3338`: DOC: Cite examples references using `sphinxcontrib-bibtex`
* :ghissue:`3319`: DOC: Use references bibliography file for DIPY citation file
* :ghissue:`3321`: BF: Set the superclass `fit_method` param value to the one provided
* :ghissue:`3339`: BUG: Bug with params
* :ghissue:`3324`: RF: Refactored for keyword arguments
* :ghissue:`3340`: CI: pin cvxpy  to 1.4.4 until 1.5.x issues are solved
* :ghissue:`3316`: DOC: Cite code base references using `sphinxcontrib-bibtex`
* :ghissue:`3332`: BF: Set the `Diso` parameter value to the one provided
* :ghissue:`3325`: DOC: Fix warnings related to displayed math expressions
* :ghissue:`3331`: DOC: Miscellaneous documentation improvements (part 3)
* :ghissue:`3329`: STYLE: Use a leading underscore to name private methods
* :ghissue:`3330`: DOC: Do not use unfinished double backticks
* :ghissue:`3320`: DOC: Miscellaneous documentation improvements (part 2)
* :ghissue:`3318`: RF: Remove unused parameters from method signature
* :ghissue:`3310`: DOC: Cite `nn` references through `sphinxcontrib-bibtex`
* :ghissue:`3315`: RF: remove legacy numpydoc
* :ghissue:`1026`: Multprocessing the multivoxel fit
* :ghissue:`2810`: [DOC] introducing sphinxcontrib-Bibtex to improve reference management
* :ghissue:`3312`: DOC: Use `misc` for other types of BibTeX entries
* :ghissue:`3309`: DOC: Miscellaneous doc formatting fixes (part 4)
* :ghissue:`3308`: DOC: Rework the BibTeX bibliography file
* :ghissue:`3223`: Remove`sagital`  from de codespell ignore list |# codespell:ignore sagital|
* :ghissue:`3275`: FIX: remove sagital from codespellrc ignore list |# codespell:ignore sagital|
* :ghissue:`3298`: Inaccurate docstring in `omp.pyx::determine_num_threads`
* :ghissue:`3304`: DOC: Miscellaneous doc formatting fixes (part 3)
* :ghissue:`3305`: How to apply NODDI sequence in dipy
* :ghissue:`3295`: ENH: Add a GHA workflow file to build docs
* :ghissue:`3056`: [WIP][RF] Use lazy loading
* :ghissue:`3302`: DOC: Miscellaneous doc formatting fixes (part 2)
* :ghissue:`3301`: FIX: explicit keyword argument for Horizon
* :ghissue:`3231`: Coverage build failing on and off in to a numpy-related statement
* :ghissue:`3297`: DOC: Miscellaneous doc formatting fixes
* :ghissue:`3300`: BF: Title Fix
* :ghissue:`3299`: Numpy compatibility issue
* :ghissue:`3291`: FIX: nightly wheels for macOS arm64
* :ghissue:`3262`: NF: Applying Decorators in Module (Visualization)
* :ghissue:`3263`: NF: Applying Decorators in Module (Workflow)
* :ghissue:`3283`: BUG: Gradient table requires at least 2 orientations
* :ghissue:`3287`: NF: Add `__len__` to `GradientTable`
* :ghissue:`3282`: Define ``__len__`` within ``GradientTable``?
* :ghissue:`3260`: NF: Applying Decorators in Module (Tracking)
* :ghissue:`3256`: NF: Applying Decorators in Module (NeuralNetwork)
* :ghissue:`3258`: NF: Applying Decorators in Module (Segment)
* :ghissue:`3249`: NF: Applying Decorators in Module (Align)
* :ghissue:`3251`: NF: Applying Decorators in Module (Core)
* :ghissue:`3279`: FIX: Explicit type origin for long to solve the cython error during compilation
* :ghissue:`3242`: Broken source installation
* :ghissue:`3259`: NF: Applying Decorators in Module (Sims)
* :ghissue:`3252`: NF: Applying Decorators in Module (Denoise)
* :ghissue:`3280`: numpy.core.multiarray failed when importing dipy.io.streamline (dipy.tracking.streamlinespeed)
* :ghissue:`3261`: NF: Applying Decorators in Module (Utils)
* :ghissue:`3255`: NF: Applying Decorators in Module (Io)
* :ghissue:`3253`: NF: Applying Decorators in Module (Data)
* :ghissue:`3233`: STYLE: Set `stacklevel` argument explicitly to warning messages
* :ghissue:`3277`: can't find dipy_buan_profiles!!!
* :ghissue:`3029`: Migrating to Keyword Only arguments (PEP 3102)
* :ghissue:`3239`: NF: Decorator for keyword-only argument
* :ghissue:`2593`: Embed parallelization into the multi_voxel_fit decorator.
* :ghissue:`3274`: RF: Update pyproject.toml for numpy 2.0
* :ghissue:`3265`: NumPy 2.0 incompatibility
* :ghissue:`3266`: NF: Call `cnp.import_array()` explicitly to use the NumPy C API
* :ghissue:`3273`: STYLE: Make statement dwell on a single line
* :ghissue:`3236`: Allow `interp_rbf` to accept tensor-valued spherical functions
* :ghissue:`3237`: Add support for tensor-valued spherical functions in `interp_rbf`
* :ghissue:`3245`: RF: Switch from using sparse `*_matrix` to `*_array`.
* :ghissue:`3267`: STYLE: Avoid deprecated NumPy types and methods for NumPy 2.0 compat
* :ghissue:`3264`: TEST: avoid direct comparison of floating point numbers
* :ghissue:`3268`: STYLE: Prefer using `np.asarray` to avoid copy while creating an array
* :ghissue:`3271`: RF: Do not use `np.any` for checking optional array parameters
* :ghissue:`3243`: Create `DiffeomorphicMap` object with saved nifti forward warp data
* :ghissue:`3250`: DOC: Fix param order
* :ghissue:`3269`: STYLE: Prefer using `isin` over `in1d`
* :ghissue:`3238`: NF - add affine to peaks_from_position
* :ghissue:`3247`: STYLE: Add imported symbols to __all__ in direction module
* :ghissue:`3246`: STYLE: Import explicitly `direction.peaks` symbols
* :ghissue:`3241`: RF: Codespell fix for CI
* :ghissue:`3228`: STYLE: Fix unused loop control variable warning
* :ghissue:`3235`: STYLE: Do not allow running unintended modules as scripts
* :ghissue:`3230`: STYLE: Fix function definition loop variable binding warning
* :ghissue:`3232`: STYLE: Simplify implicitly concatenated strings
* :ghissue:`3229`: STYLE: Prefer using f-strings
* :ghissue:`3224`: BF: Rewrite list creation as `list()` instead of `[]`
* :ghissue:`3216`: STYLE: Format code using `ruff`
* :ghissue:`3175`: Tract profiles in afq example look all wrong
* :ghissue:`3178`: DOC: Fixes the AFQ tract profile tutorial.
* :ghissue:`3218`: STYLE: Fix codespell issues
* :ghissue:`3209`: [CI] Move filterwarnings from pyproject to conftest
* :ghissue:`3220`: [RF] from `os.fork` to `spawn` for multiprocessing
* :ghissue:`3214`: RF - remove buffer argument in pmf_gen.get_pmf_value(.)
* :ghissue:`3196`: Enhancing Gradient Approximation in DTI Tests #3155
* :ghissue:`3203`: [WIP][CI] warning as error at compilation level
* :ghissue:`3219`: [ENH] Prefer CLARABEL over ECOS as the CVXPY solver
* :ghissue:`3165`: 1.9.0 system test failures
* :ghissue:`3215`: tests: correct module-level setup
* :ghissue:`3217`: ENH: Prefer `CLARABEL` over `ECOS` as the CVXPY solver
* :ghissue:`3211`: [RF] PMF Gen: from memoryview to pointer
* :ghissue:`3210`: Python 3.13: Fix tests for next Python release
* :ghissue:`3212`: STYLE: Relocate `pre-commit` and `ruff` packages to style requirements
* :ghissue:`3205`: BF: Declare variables holding integers as `cnp.npy_intp` over `double`
* :ghissue:`3174`: NF - initial directions from seed positions
* :ghissue:`3207`: DOC: Fix Cython method parameter type description
* :ghissue:`3206`: BF: Use `cnp.npy_intp` instead of `int` as counter
* :ghissue:`3204`: DOC: Fix documentation typos
* :ghissue:`3208`: BF: Cast operation explicitly to `cnp.npy_intp` in denoising Cython
* :ghissue:`3202`: [TEST] Add flag to turn warnings into errors for pytest
* :ghissue:`3201`: TEST: Turn warnings into errors when calling `pytest` in CI testing
* :ghissue:`3158`: ENH: Remove filtering `UserWarning` warnings in test config file
* :ghissue:`3200`: Check relevant warnings raised (DO NOT MERGE)
* :ghissue:`2299`: NF: Add array parsing capabilities to the CLIs
* :ghissue:`2880`: improve test_io_fetch_fetcher_datanames
* :ghissue:`3194`: MAINT: fix warning
* :ghissue:`3199`: Bump pre-commit/action from 3.0.0 to 3.0.1 in the actions group
* :ghissue:`3182`: [NF] Add DiSCo challenge data fetcher
* :ghissue:`3197`: ENH: Fix miscellaneous warnings in `dki` reconstruction module
* :ghissue:`3198`: ENH: Ensure that `arccos` argument is in the [-1,1] range
* :ghissue:`3186`: Update `trilinear_interpolate4d` to accept float and double
* :ghissue:`3191`: [RF] allow float and double for `trilinear_interpolate4d_c`
* :ghissue:`3151`: DKI Updates: (new radial tensor kurtosis metric, updated documentation and missing tests)
* :ghissue:`3185`: Improve consistency of affine_registration docstring
* :ghissue:`3189`: Update affine_registration to clarify returns and make them consistent with docstring
* :ghissue:`3187`: Setting `Legacy=True` SH basis is not possible for SH models
* :ghissue:`3176`: ENH: allow vol_idx in align workflow
* :ghissue:`3188`: ENH: Add `pre-commit` to project `dev` dependencies
* :ghissue:`3183`: ENH: Specify the solver for the MAP-MRI positivity constraint test
* :ghissue:`3184`: STYLE: Sort import statements using `ruff`
* :ghissue:`3181`: [PEP8] fix pep8 and docstring style in `dti.py` file
* :ghissue:`3177`: Loading Peaks faster with complete range and synchronization functionality.
* :ghissue:`3145`: dki_fit errors from divide by zero
* :ghissue:`3180`: BF: Fix bug in mode for isotropic tensors
* :ghissue:`3172`: [ENH] Enable range for dipy_median_otsu workflow
* :ghissue:`3171`: Clean up for tabs and tab manager
* :ghissue:`2796`: Tract profiles in the afq_profile example look terrible
* :ghissue:`1985`: Something is wonky with the AFQ  tracts profile example
* :ghissue:`3168`: Feature/peaks tab revamp
* :ghissue:`2036`: [WIP] NF - Add Closest Peak direction getter from peaks array
* :ghissue:`3128`: NF: Fibonacci Hemisphere
* :ghissue:`3122`: Add `peaks_from_model` to `dipy_fit_dti` CLI
* :ghissue:`3153`: ENH: add save peaks to dipy_fit_dti, dki
* :ghissue:`3113`: [FIX] Nlmeans Algorithm Enhancement #2950
* :ghissue:`3111`: Add support for sequential processing in Gibbs unringing #2757
* :ghissue:`3154`: ENH: Add neighboring DWI correlation QC metric
* :ghissue:`3156`: ENH: Implement NDC from Yeh2019
* :ghissue:`3161`: DOC: Fix `tri` parameter docstring in `viz.projections.sph_project`
* :ghissue:`3163`: STYLE: Make `fury` and `matplotlib` presence message in test consistent
* :ghissue:`3162`: ENH: Fix variable potentially being referenced before assignment
* :ghissue:`3144`: ROI tab revamped
* :ghissue:`2982`: [FIX] Force the use of pre-wheels
* :ghissue:`3134`: Feature/cluster revamp
* :ghissue:`3146`: [NF] Add 30 Bundle brain atlas fetcher
* :ghissue:`3150`: BUG: Fix bug with nightly wheel build
* :ghissue:`3149`: ENH: Miscellaneous cleanup
* :ghissue:`3148`: ENH: Fix HDF5 key warning when saving BUAN profile data
* :ghissue:`3138`: [CI] update CI's script
* :ghissue:`3142`: Horizon slider does not show proper 0-1 range images such as FA
* :ghissue:`3126`: Bugfix for ROI images updates
* :ghissue:`3141`: ENH: Fix miscellaneous warnings
* :ghissue:`3139`: BF: Removing Error/Warning from Tensorflow 2.16
* :ghissue:`3096`: TissueClassifierHMRF has some argument logic error
* :ghissue:`3132`: BF: Removed allow_break
* :ghissue:`3136`: conversion of cudipy.align.imwarp.DiffeomorphicMap to dipy.align.imwarp.DiffeomorphicMap
* :ghissue:`3135`: DOC: Fix documentation URLs
* :ghissue:`3133`: grg-sphinx-theme added as dependency
* :ghissue:`3127`: Feature/viz interface tutorials
* :ghissue:`3120`: DOC - Removed unnecessary line from tracking example
* :ghissue:`3116`: diffusion gradient nonlinearity correction
* :ghissue:`3110`: Viz cli tutorial updated
* :ghissue:`2970`: spherical harmonic degree/order terminology swapped
* :ghissue:`3086`: [RF] Fix spherical harmonic terminology swap
* :ghissue:`3095`: [UPCOMING] Release preparation for 1.9.0

.. |# codespell:ignore sagital| replace:: .
