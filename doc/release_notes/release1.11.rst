.. _release1.11:

=====================================
 Release notes for DIPY version 1.11
=====================================

GitHub stats for 2024/12/12 - 2025/03/14 (tag: 1.10.0)

These lists are automatically generated, and may be incomplete or contain duplicates.

The following 11 authors contributed 175 commits.

* Ariel Rokem
* Atharva Shah
* Eleftherios Garyfallidis
* Gabriel Girard
* Jon Haitz Legarreta Gorroño
* Jong Sung Park
* Maharshi Gor
* Michael R. Crusoe
* Prajwal Reddy
* Sam Coveney
* Serge Koudoro


We closed a total of 120 issues, 47 pull requests and 73 regular issues;
this is the full list (generated with the script
:file:`tools/github_stats.py`):

Pull Requests (47):

* :ghpull:`3487`: RF: update tracking cli
* :ghpull:`3490`: ENH: Updated the logging info
* :ghpull:`3489`: DOC - Renamed fast tracking example
* :ghpull:`3475`: NF: add workflow for N4 biasfield
* :ghpull:`3485`: DOC: remove modulo in dipy_math docstring
* :ghpull:`3486`: RF: Fixed Warnings of Patch2self3
* :ghpull:`3483`: ENH: Add use_cuda option to torch models
* :ghpull:`3471`: Patch2Self3 skipping b0_denoising error fixed.
* :ghpull:`3477`: NF: Add dipy_classify_tissue CLI
* :ghpull:`3461`: BF: Handling binary and small intensity value range and Volume slider not be shown if single channel provided.
* :ghpull:`3479`: DOC - Updated tracking examples to use the fast tracking framework
* :ghpull:`3482`: RF: Avoid crash of `optional_package` when dirty install
* :ghpull:`3480`: RF: Allow hcp and hcp in dipy_fetch CLI
* :ghpull:`3476`: BF - generate_tractogram needs seed coordinate in image space
* :ghpull:`3478`: RF: Small variable mismatch in the tutorial for DAM classifier
* :ghpull:`3458`: NF: Add optional fetcher
* :ghpull:`3438`: NF: Add 7 new `dipy_fit_*` workflows
* :ghpull:`3462`: NF: Reset of visualization introduced.
* :ghpull:`3470`: RF: Enforce bvecs for the cli extract_b0
* :ghpull:`3472`: RF: Rename and Deprecate dipy_sh_convert_mrtrix CLI
* :ghpull:`3449`: Adding DeepN4 PyTorch model
* :ghpull:`3465`: RF: Import fixes
* :ghpull:`3459`: NF: Add default value to docstring for the CLI
* :ghpull:`3446`: Dki constraints
* :ghpull:`3467`: RF: Changed the matrix
* :ghpull:`3457`: DOC: fix markup issues with tracking tutorial
* :ghpull:`3456`: Added tissue classification with DAM example
* :ghpull:`3444`: NF: Add 3 new ``dipy_extract_*`` workflows
* :ghpull:`3089`: Parallel Tracking Framework
* :ghpull:`3448`: Spelling error in the deprecation warning
* :ghpull:`3442`: RF: update of peak_directions to allow nogil
* :ghpull:`3445`: DOC: removed title
* :ghpull:`3400`: CI: Add python 3.13
* :ghpull:`3440`: RF - add pmf_gen argument to peaks_from_positions
* :ghpull:`3441`: ENH: Add GitHub CI workflow file to run benchmarks using `asv`
* :ghpull:`3427`: NF: add dipy_math workflow
* :ghpull:`3432`: RF: bump tensorflow minimal version to 2.18.0.
* :ghpull:`3436`: RF: Update ``scipy.special`` deprecated functions
* :ghpull:`3433`: TEST: remove skip if not have_delaunay
* :ghpull:`3434`: ENH: Transition remaining `NumPy` `RandomState` instances to `Generator`
* :ghpull:`3430`: ENH: Add type annotation information
* :ghpull:`3428`: DOC: Add implemented tractography method table to tracking example index
* :ghpull:`3426`: RF: from relative import to absolute import
* :ghpull:`3423`: Make the docs more reproducible
* :ghpull:`3421`: CI: remove python3.9 support
* :ghpull:`3422`: RF: fix joblib warning in sfm
* :ghpull:`3364`: UPCOMING:  Release 1.10.0

Issues (73):

* :ghissue:`3487`: RF: update tracking cli
* :ghissue:`3490`: ENH: Updated the logging info
* :ghissue:`3489`: DOC - Renamed fast tracking example
* :ghissue:`3475`: NF: add workflow for N4 biasfield
* :ghissue:`3485`: DOC: remove modulo in dipy_math docstring
* :ghissue:`3486`: RF: Fixed Warnings of Patch2self3
* :ghissue:`3483`: ENH: Add use_cuda option to torch models
* :ghissue:`3471`: Patch2Self3 skipping b0_denoising error fixed.
* :ghissue:`3477`: NF: Add dipy_classify_tissue CLI
* :ghissue:`3484`: Default parameter values not shown on DIPY's CLIs
* :ghissue:`3371`: Horizon miscalculating contrast when returning to previous volume
* :ghissue:`3112`: Horizon fails with a single channel 4D volume
* :ghissue:`3461`: BF: Handling binary and small intensity value range and Volume slider not be shown if single channel provided.
* :ghissue:`3479`: DOC - Updated tracking examples to use the fast tracking framework
* :ghissue:`3482`: RF: Avoid crash of `optional_package` when dirty install
* :ghissue:`3480`: RF: Allow hcp and hcp in dipy_fetch CLI
* :ghissue:`3476`: BF - generate_tractogram needs seed coordinate in image space
* :ghissue:`3478`: RF: Small variable mismatch in the tutorial for DAM classifier
* :ghissue:`3190`: Allow to define optional file to fetch
* :ghissue:`3458`: NF: Add optional fetcher
* :ghissue:`3438`: NF: Add 7 new `dipy_fit_*` workflows
* :ghissue:`3469`: horizon - SSLCertVerificationError
* :ghissue:`3152`: Horizon needs a home button which realigns the view to the z slice we are at.
* :ghissue:`2421`: DIPY Horizon Menu
* :ghissue:`3462`: NF: Reset of visualization introduced.
* :ghissue:`3470`: RF: Enforce bvecs for the cli extract_b0
* :ghissue:`3472`: RF: Rename and Deprecate dipy_sh_convert_mrtrix CLI
* :ghissue:`3449`: Adding DeepN4 PyTorch model
* :ghissue:`3164`: Simplify or remove the use of mpl_tri in viz module
* :ghissue:`3465`: RF: Import fixes
* :ghissue:`3454`: Default Value in Workflows docstring
* :ghissue:`3459`: NF: Add default value to docstring for the CLI
* :ghissue:`3446`: Dki constraints
* :ghissue:`3467`: RF: Changed the matrix
* :ghissue:`3466`: Theory for b and q values represent incorrect b matrix
* :ghissue:`3457`: DOC: fix markup issues with tracking tutorial
* :ghissue:`3463`: Remove the warnings from doc build in DIPY docs
* :ghissue:`3451`: How can be the drawing background changed from black to white using `actor.odf_slicer`
* :ghissue:`3179`: Need specific area zooming for horizon
* :ghissue:`3359`: Bug: Horizon throws errors when changing intensity range and then changing volumes
* :ghissue:`3460`: [WIP] RF: force use of header for fetcher
* :ghissue:`3394`: Create a tutorial for DAM classifier
* :ghissue:`3456`: Added tissue classification with DAM example
* :ghissue:`3444`: NF: Add 3 new ``dipy_extract_*`` workflows
* :ghissue:`1501`: Refactoring tracking and checking tutorials and workflows - high priority for next release.
* :ghissue:`834`: Multiprocessing the local tracking?
* :ghissue:`3089`: Parallel Tracking Framework
* :ghissue:`3448`: Spelling error in the deprecation warning
* :ghissue:`3442`: RF: update of peak_directions to allow nogil
* :ghissue:`3445`: DOC: removed title
* :ghissue:`3443`: NF: Replace urllib by requests to improve fetcher stability.
* :ghissue:`3400`: CI: Add python 3.13
* :ghissue:`3440`: RF - add pmf_gen argument to peaks_from_positions
* :ghissue:`3441`: ENH: Add GitHub CI workflow file to run benchmarks using `asv`
* :ghissue:`3427`: NF: add dipy_math workflow
* :ghissue:`3432`: RF: bump tensorflow minimal version to 2.18.0.
* :ghissue:`3436`: RF: Update ``scipy.special`` deprecated functions
* :ghissue:`3431`: Tests with scipy.spatial.Delaunay being skipped
* :ghissue:`3433`: TEST: remove skip if not have_delaunay
* :ghissue:`3434`: ENH: Transition remaining `NumPy` `RandomState` instances to `Generator`
* :ghissue:`3430`: ENH: Add type annotation information
* :ghissue:`3428`: DOC: Add implemented tractography method table to tracking example index
* :ghissue:`3426`: RF: from relative import to absolute import
* :ghissue:`3423`: Make the docs more reproducible
* :ghissue:`3421`: CI: remove python3.9 support
* :ghissue:`3422`: RF: fix joblib warning in sfm
* :ghissue:`2364`: Streamlines get negative coordinates in voxel space
* :ghissue:`3016`: [WIP] NF: Wigner-D Rotation Functions
* :ghissue:`3276`: [Feature] Multithreading support for reading and opening files.
* :ghissue:`2304`: [WIP] DKI ODF redux
* :ghissue:`2705`: WIP: Single-shell FWDTI
* :ghissue:`3416`: support for numpy 2.0 seems missing
* :ghissue:`3364`: UPCOMING:  Release 1.10.0
