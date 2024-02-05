.. _release1.8:

====================================
 Release notes for DIPY version 1.8
====================================
GitHub stats for 2023/04/23 - 2023/12/13 (tag: 1.7.0)

These lists are automatically generated, and may be incomplete or contain duplicates.

The following 28 authors contributed 733 commits.

* Ariel Rokem
* Atharva Shah
* Bramsh Qamar
* Charles Poirier
* Dimitri Papadopoulos
* Eleftherios Garyfallidis
* Emmanuelle Renauld
* Eric Larson
* Francois Rheault
* Gabriel Girard
* Javier Guaje
* John Kruper
* Jon Haitz Legarreta Gorroño
* Jong Sung Park
* Maharshi Gor
* Michael R. Crusoe
* Nicolas Delinte
* Paul Camacho
* Philippe Karan
* Rafael Neto Henriques
* Sam Coveney
* Samuel St-Jean
* Serge Koudoro
* Shilpi Prasad
* Theodore Ni
* Vara Lakshmi Bayanagari
* dependabot[bot]
* Étienne Mollier


We closed a total of 327 issues, 130 pull requests and 197 regular issues;
this is the full list (generated with the script
:file:`tools/github_stats.py`):

Pull Requests (129):

* :ghpull:`3009`: [DOC] Update installation instruction [ci skip]
* :ghpull:`2999`: TEST: Set explicitly `CLARABEL` as the CVXPY solver
* :ghpull:`2943`: BF: Fix bundlewarp shape analysis profile values for all `False` mask
* :ghpull:`2992`: RGB support for images
* :ghpull:`2989`: BF: Mask `1` values in leveraged, residual matrix computation
* :ghpull:`3007`: [RF] Define minimum version for some optional packages
* :ghpull:`3006`: [NF] Introduce minimum version in `optional_package`
* :ghpull:`3002`: [RF] Improve scripts and import management
* :ghpull:`3005`: Bump actions/setup-python from 4 to 5
* :ghpull:`3004`: TEST: Check and filter PCA dimensionality problem warning
* :ghpull:`2996`: RF: Fix b0 threshold warnings
* :ghpull:`2995`: [MTN] remove custom module `_importlib`
* :ghpull:`2998`: TEST: Filter SciPy 0.18.0 1-D affine transform array warning in test
* :ghpull:`3001`: RF: Create PCA denoising utils methods
* :ghpull:`3000`: RF: Prefer raising `sklearn` package warnings when required
* :ghpull:`2997`: TEST: Filter warning about resorting to `OLS` fitting method
* :ghpull:`2991`: MTN: fix byte swap ordering for numpy 2.0
* :ghpull:`2987`: STYLE: Make `cvxpy`-dependent test checking consistent in `test_mcsd`
* :ghpull:`2990`: STYLE: Use `.astype()` on uninitialized array casting
* :ghpull:`2984`: DOC: Miscellaneous documentation improvements
* :ghpull:`2988`: STYLE: Remove unused import in `dipy/core/gradients.py`
* :ghpull:`2985`: Bump conda-incubator/setup-miniconda from 2 to 3
* :ghpull:`2986`: DOC: Fix typos and grammar in `lcr_matrix` function documentation
* :ghpull:`2983`: STYLE: Fix boolean variable negation warnings
* :ghpull:`2981`: [MTN] replace the deprecated sctypes
* :ghpull:`2980`: [FIX] int_t to npy_intp
* :ghpull:`2978`: DOC: Fix issue template [ci skip]
* :ghpull:`2976`: [MTN] Update index url for PRE-Wheels dependencies
* :ghpull:`2975`: connectivity_matrix code speed up
* :ghpull:`2715`: Enable building DIPY with Meson
* :ghpull:`2964`: RF: Moving to numpy.random.Generator from numpy.random
* :ghpull:`2963`: NF: Updating EVAC+ model and adding util function
* :ghpull:`2974`: [MTN] Disable github check annotations
* :ghpull:`2956`: Adding support for btens to multi_shell_fiber_response function
* :ghpull:`2969`: bugfix for --force issue
* :ghpull:`2967`: Feature/opacity checkbox
* :ghpull:`2966`: volume slider fix
* :ghpull:`2958`: TEST: Filter legacy SH bases warnings in bootstrap direction getter test
* :ghpull:`2944`: [DOC] Remove `..figure` directive in examples
* :ghpull:`2961`: fixes for pep8 in previous PR
* :ghpull:`2922`: synchronized-slicers for same size
* :ghpull:`2924`: Additional check for horizon
* :ghpull:`2957`: STYLE: Fix typo in `msdki` reconstruction test name
* :ghpull:`2941`: TEST: Fix NumPy array to scalar value conversion warning
* :ghpull:`2932`: OPT - Optimized pmfgen
* :ghpull:`2929`: Stabilizing some test functions with set random seeds
* :ghpull:`2954`: TEST: Bring back Python 3.8 testing to GHA workflows
* :ghpull:`2946`: RF: Refactor duplicate code in `qtdmri` to `mapmri` coeff computation
* :ghpull:`2947`: RF - BootDirectionGetter
* :ghpull:`2945`: DOC: Fix miscellaneous docstrings
* :ghpull:`2940`: TEST: Filter legacy SH bases warnings in PTT direction getter test
* :ghpull:`2938`: TEST: Adding random generator with seed to icm tests
* :ghpull:`2942`: OPT: Delegate to NumPy creating a random matrix
* :ghpull:`2939`: DOC: Update jhlegarreta affiliation in developers
* :ghpull:`2933`: fixed bug with fit extra returns
* :ghpull:`2930`: Update of the tutorial apply image-based registration to streamlines
* :ghpull:`2759`: TRX integration
* :ghpull:`2923`: [DOC] Large documentation update
* :ghpull:`2825`: NF -  add initial directions to local tracking
* :ghpull:`2892`: BF - fixed random generator seed `value too large to convert to int` error
* :ghpull:`2926`: ENH: MSMT CSD module unique b-val tolerance parameter improvements
* :ghpull:`2927`: DOC: Fix package name in documentation config file comment
* :ghpull:`2925`: STYLE: Fix miscellaneous Numpy warnings
* :ghpull:`2781`: Small fixes in functions
* :ghpull:`2910`: STYLE: f-strings
* :ghpull:`2921`: [FIX] tiny fix to HBN fetcher to also grab T1 for each subject
* :ghpull:`2906`: [FIX] Pin scipy for the conda failing CI's
* :ghpull:`2920`: Mark Python3 files as such
* :ghpull:`2919`: fix various grammar errors
* :ghpull:`2915`: DOC: http:// → https://
* :ghpull:`2916`: Build(deps): Bump codespell-project/actions-codespell from 1 to 2
* :ghpull:`2914`: GitHub Actions
* :ghpull:`2816`: Correlation Tensor Imaging
* :ghpull:`2912`: MAINT: the symbol for second is s, not sec.
* :ghpull:`2902`: Short import for horizon
* :ghpull:`2904`: Apply refurb suggestions
* :ghpull:`2899`: DOC: Fix typos newly found by codespell
* :ghpull:`2891`: Apply pyupgrade suggestions
* :ghpull:`2898`: Remove zip operation in transform_tracking_output
* :ghpull:`2897`: BF: Bug when downloading hbn data.
* :ghpull:`2893`: Remove consecutive duplicate words
* :ghpull:`2894`: Get rid of trailing spaces in text files
* :ghpull:`2889`: Apply pyupgrade suggestions
* :ghpull:`2888`: Fix typos newly found by codespell
* :ghpull:`2887`: Update shm.py
* :ghpull:`2814`: [Fix] Horizon: Binary image loading
* :ghpull:`2885`: [ENH] Add minimum length to streamline generator
* :ghpull:`2875`: Increased PTT performances
* :ghpull:`2879`: Add fetcher for a sample CTI dataset
* :ghpull:`2882`: Change license_file to license_files in setup.cfg
* :ghpull:`2804`: Adding diffusion data descriptions from references to reconstruction table
* :ghpull:`2730`: Fix weighted Jacobians, return extra fit data, add adjacency function
* :ghpull:`2821`: NF -  added pft min wm parameter
* :ghpull:`2876`: Introduction of pydata theme for sphinx
* :ghpull:`2846`: Vara's Week 8 & Week 9 blog
* :ghpull:`2870`: Vara's Week 12 and Week 13 blog
* :ghpull:`2865`: Shilpi's Week0&Week1 combined
* :ghpull:`2868`: Submitting Week13.rst file
* :ghpull:`2871`: Corrected paths to static files
* :ghpull:`2863`: Shilpi's Week12 Blog.
* :ghpull:`2856`: Adding Week 11 Blog
* :ghpull:`2849`: Shilpi's 10th Blog
* :ghpull:`2847`: Pushing Week 8 + 9 blog
* :ghpull:`2836`: Shilpi's Week 5 blog
* :ghpull:`2864`: Change internal space/origin when using sft.to_x() with an empty sft.
* :ghpull:`2806`: BF - initial backward orientation of local tracking
* :ghpull:`2862`: Vara's Week 10 & Week 11 blog
* :ghpull:`2843`: Pushing 7th_blog
* :ghpull:`2841`: Vara's Week 6 & Week 7 blog
* :ghpull:`2835`: Vara's week 5 blog
* :ghpull:`2829`: Pushing 3rd blog
* :ghpull:`2828`: Vara's week 3 blog
* :ghpull:`2860`: Updates HCP fetcher dataset_description to be compatible with current BIDS
* :ghpull:`2831`: Vara's week 4 blog
* :ghpull:`2833`: Pushing 4thBlog
* :ghpull:`2840`: Shilpi's Week6 Blog
* :ghpull:`2839`: make order_from_ncoef return an int
* :ghpull:`2844`: doc/tools/: fix trailing dot in version number.
* :ghpull:`2832`: BundleWarp: added tutorial and fixed a small bug
* :ghpull:`2818`: Vara's week 0 blog
* :ghpull:`2823`: submitting clearn PR for 2nd blog
* :ghpull:`2813`: First Blog
* :ghpull:`2808`: [DOC] Fix cross referencing
* :ghpull:`2798`: Move evac+ to new module `nn`
* :ghpull:`2797`: remove Nibabel InTemporaryDirectory
* :ghpull:`2800`: Remove the Deprecating nisext
* :ghpull:`2795`: bump dependencies minimal version
* :ghpull:`2792`: Add `patch_radius` parameter to Patch2Self denoise workflow
* :ghpull:`2761`: [UPCOMING]  Release 1.7.0 - workshop release

Issues (197):

* :ghissue:`3009`: [DOC] Update installation instruction [ci skip]
* :ghissue:`2999`: TEST: Set explicitly `CLARABEL` as the CVXPY solver
* :ghissue:`2943`: BF: Fix bundlewarp shape analysis profile values for all `False` mask
* :ghissue:`2992`: RGB support for images
* :ghissue:`2989`: BF: Mask `1` values in leveraged, residual matrix computation
* :ghissue:`3007`: [RF] Define minimum version for some optional packages
* :ghissue:`3006`: [NF] Introduce minimum version in `optional_package`
* :ghissue:`1256`: script path can not be found on OSX
* :ghissue:`3002`: [RF] Improve scripts and import management
* :ghissue:`3005`: Bump actions/setup-python from 4 to 5
* :ghissue:`3004`: TEST: Check and filter PCA dimensionality problem warning
* :ghissue:`2996`: RF: Fix b0 threshold warnings
* :ghissue:`2995`: [MTN] remove custom module `_importlib`
* :ghissue:`2998`: TEST: Filter SciPy 0.18.0 1-D affine transform array warning in test
* :ghissue:`3001`: RF: Create PCA denoising utils methods
* :ghissue:`3000`: RF: Prefer raising `sklearn` package warnings when required
* :ghissue:`2997`: TEST: Filter warning about resorting to `OLS` fitting method
* :ghissue:`2979`: Prerelease wheels not NumPy 2.0.0.dev compatible
* :ghissue:`2991`: MTN: fix byte swap ordering for numpy 2.0
* :ghissue:`2987`: STYLE: Make `cvxpy`-dependent test checking consistent in `test_mcsd`
* :ghissue:`2990`: STYLE: Use `.astype()` on uninitialized array casting
* :ghissue:`2984`: DOC: Miscellaneous documentation improvements
* :ghissue:`2988`: STYLE: Remove unused import in `dipy/core/gradients.py`
* :ghissue:`2985`: Bump conda-incubator/setup-miniconda from 2 to 3
* :ghissue:`2986`: DOC: Fix typos and grammar in `lcr_matrix` function documentation
* :ghissue:`2905`: base tests
* :ghissue:`2983`: STYLE: Fix boolean variable negation warnings
* :ghissue:`2981`: [MTN] replace the deprecated sctypes
* :ghissue:`2980`: [FIX] int_t to npy_intp
* :ghissue:`2978`: DOC: Fix issue template [ci skip]
* :ghissue:`2976`: [MTN] Update index url for PRE-Wheels dependencies
* :ghissue:`2975`: connectivity_matrix code speed up
* :ghissue:`2514`: Reshape our packaging system
* :ghissue:`2715`: Enable building DIPY with Meson
* :ghissue:`2964`: RF: Moving to numpy.random.Generator from numpy.random
* :ghissue:`2736`: dipy_horizon needs --force option if there is tmp.png
* :ghissue:`2960`: add type annotation in io module
* :ghissue:`2803`: Type annotations integration
* :ghissue:`2963`: NF: Updating EVAC+ model and adding util function
* :ghissue:`2974`: [MTN] Disable github check annotations
* :ghissue:`2956`: Adding support for btens to multi_shell_fiber_response function
* :ghissue:`2969`: bugfix for --force issue
* :ghissue:`2967`: Feature/opacity checkbox
* :ghissue:`2965`: Pip installation issues with python 3.12
* :ghissue:`2968`: Pip installation issues with python 3.12
* :ghissue:`2966`: volume slider fix
* :ghissue:`2958`: TEST: Filter legacy SH bases warnings in bootstrap direction getter test
* :ghissue:`2801`: Some left-overs from sphinx-gallery conversion
* :ghissue:`2944`: [DOC] Remove `..figure` directive in examples
* :ghissue:`2961`: fixes for pep8 in previous PR
* :ghissue:`2922`: synchronized-slicers for same size
* :ghissue:`2878`: DIPY reinstall doesn't automatically update needed dependencies
* :ghissue:`2924`: Additional check for horizon
* :ghissue:`2957`: STYLE: Fix typo in `msdki` reconstruction test name
* :ghissue:`2941`: TEST: Fix NumPy array to scalar value conversion warning
* :ghissue:`2932`: OPT - Optimized pmfgen
* :ghissue:`2929`: Stabilizing some test functions with set random seeds
* :ghissue:`2954`: TEST: Bring back Python 3.8 testing to GHA workflows
* :ghissue:`2953`: [WIP] Nlmeans update
* :ghissue:`2946`: RF: Refactor duplicate code in `qtdmri` to `mapmri` coeff computation
* :ghissue:`2955`: set_number_of_points function not found for dipy 1.7.0
* :ghissue:`2947`: RF - BootDirectionGetter
* :ghissue:`2952`: Delete dipy/denoise/nlmeans.py
* :ghissue:`2949`: HBN fetcher failed
* :ghissue:`2945`: DOC: Fix miscellaneous docstrings
* :ghissue:`718`: Create an example of multi b-value SFM
* :ghissue:`2523`: Doc generation failed
* :ghissue:`2940`: TEST: Filter legacy SH bases warnings in PTT direction getter test
* :ghissue:`2928`: `test_icm_square` failing on and off
* :ghissue:`2938`: TEST: Adding random generator with seed to icm tests
* :ghissue:`2942`: OPT: Delegate to NumPy creating a random matrix
* :ghissue:`2939`: DOC: Update jhlegarreta affiliation in developers
* :ghissue:`2933`: fixed bug with fit extra returns
* :ghissue:`2936`: Automatic Fiber Bundle Extraction with RecoBundles in DIPY 1.7 broken?
* :ghissue:`2934`: demo code not working
* :ghissue:`2787`: Adds a pyproject file.
* :ghissue:`2786`: "Image based streamlines_registration: unable to warp streamlines into template"
* :ghissue:`2400`: Applying image-based deformations to streamlines example
* :ghissue:`2703`: Image based streamlines_registration: unable to warp streamlines into template space
* :ghissue:`2930`: Update of the tutorial apply image-based registration to streamlines
* :ghissue:`2759`: TRX integration
* :ghissue:`2931`: Add caption to sphinx gallery figure
* :ghissue:`2560`: MCSD Tutorial failed with `cvxpy>=1.1.15`
* :ghissue:`2794`: Add a search box to the DIPY documentation
* :ghissue:`2815`: Reconstruction table of content doesn't connect to MAP+
* :ghissue:`2923`: [DOC] Large documentation update
* :ghissue:`2790`: DTI fitting function with NLLS method raises an error.
* :ghissue:`2872`: Website image (not showing up or wrong tag showing)
* :ghissue:`2884`: WIP: trx integration
* :ghissue:`2825`: NF -  add initial directions to local tracking
* :ghissue:`2892`: BF - fixed random generator seed `value too large to convert to int` error
* :ghissue:`2926`: ENH: MSMT CSD module unique b-val tolerance parameter improvements
* :ghissue:`2927`: DOC: Fix package name in documentation config file comment
* :ghissue:`2925`: STYLE: Fix miscellaneous Numpy warnings
* :ghissue:`2777`: Error using dipy_motion_correct
* :ghissue:`2781`: Small fixes in functions
* :ghissue:`2648`: Issues with dipy_align_syn
* :ghissue:`2900`: format → f-strings?
* :ghissue:`2910`: STYLE: f-strings
* :ghissue:`2921`: [FIX] tiny fix to HBN fetcher to also grab T1 for each subject
* :ghissue:`2906`: [FIX] Pin scipy for the conda failing CI's
* :ghissue:`2920`: Mark Python3 files as such
* :ghissue:`2919`: fix various grammar errors
* :ghissue:`2915`: DOC: http:// → https://
* :ghissue:`2896`: Interactive examples for dipy
* :ghissue:`2901`: patch2self question
* :ghissue:`2916`: Build(deps): Bump codespell-project/actions-codespell from 1 to 2
* :ghissue:`2914`: GitHub Actions
* :ghissue:`2816`: Correlation Tensor Imaging
* :ghissue:`2912`: MAINT: the symbol for second is s, not sec.
* :ghissue:`2913`: DOC: fix links
* :ghissue:`2902`: Short import for horizon
* :ghissue:`2908`: Voxel correspondence between Non-Linearly aligned Volumes
* :ghissue:`2890`: Attempt to fix error in conda jobs
* :ghissue:`2907`: Temp - Gab PR
* :ghissue:`2904`: Apply refurb suggestions
* :ghissue:`2903`: Typo in the skills required section (Project 2) of Project Ideas
* :ghissue:`2899`: DOC: Fix typos newly found by codespell
* :ghissue:`2891`: Apply pyupgrade suggestions
* :ghissue:`2898`: Remove zip operation in transform_tracking_output
* :ghissue:`2897`: BF: Bug when downloading hbn data.
* :ghissue:`2893`: Remove consecutive duplicate words
* :ghissue:`2894`: Get rid of trailing spaces in text files
* :ghissue:`2889`: Apply pyupgrade suggestions
* :ghissue:`2888`: Fix typos newly found by codespell
* :ghissue:`2887`: Update shm.py
* :ghissue:`2814`: [Fix] Horizon: Binary image loading
* :ghissue:`2885`: [ENH] Add minimum length to streamline generator
* :ghissue:`1372`: Change direction getter dictionary keys from floats[3] to int
* :ghissue:`2805`: Incorrect initial direction for the backward segment of local tracking
* :ghissue:`2875`: Increased PTT performances
* :ghissue:`2883`: Adding last,Week14Blog
* :ghissue:`2879`: Add fetcher for a sample CTI dataset
* :ghissue:`2769`: DOC example for data_per_streamline usage
* :ghissue:`2774`: Added a tutorial in doc folder for saving labels.
* :ghissue:`2882`: Change license_file to license_files in setup.cfg
* :ghissue:`2881`: Adding fetcher in the test_file for #2879
* :ghissue:`2867`: Bug in PFT when changing the random function
* :ghissue:`2804`: Adding diffusion data descriptions from references to reconstruction table
* :ghissue:`2820`: fixed bug for nlls fitting
* :ghissue:`2746`: Weighted Non-Linear Fitting may be wrong
* :ghissue:`2730`: Fix weighted Jacobians, return extra fit data, add adjacency function
* :ghissue:`2821`: NF -  added pft min wm parameter
* :ghissue:`2876`: Introduction of pydata theme for sphinx
* :ghissue:`2846`: Vara's Week 8 & Week 9 blog
* :ghissue:`2870`: Vara's Week 12 and Week 13 blog
* :ghissue:`2865`: Shilpi's Week0&Week1 combined
* :ghissue:`2868`: Submitting Week13.rst file
* :ghissue:`2871`: Corrected paths to static files
* :ghissue:`2873`: Motion estimate
* :ghissue:`2863`: Shilpi's Week12 Blog.
* :ghissue:`2856`: Adding Week 11 Blog
* :ghissue:`2849`: Shilpi's 10th Blog
* :ghissue:`2847`: Pushing Week 8 + 9 blog
* :ghissue:`2836`: Shilpi's Week 5 blog
* :ghissue:`2864`: Change internal space/origin when using sft.to_x() with an empty sft.
* :ghissue:`2806`: BF - initial backward orientation of local tracking
* :ghissue:`2862`: Vara's Week 10 & Week 11 blog
* :ghissue:`2843`: Pushing 7th_blog
* :ghissue:`2841`: Vara's Week 6 & Week 7 blog
* :ghissue:`2835`: Vara's week 5 blog
* :ghissue:`2829`: Pushing 3rd blog
* :ghissue:`2828`: Vara's week 3 blog
* :ghissue:`2860`: Updates HCP fetcher dataset_description to be compatible with current BIDS
* :ghissue:`2831`: Vara's week 4 blog
* :ghissue:`1883`: Interesting dataset for linear, planar, spherical encoding
* :ghissue:`2491`: ENH: Extend Horizon to visualize 2 volumes simultaneously
* :ghissue:`2812`: Patch2self denoising followed by topup and eddy corrections worsens distortions in the orbitofrontal region
* :ghissue:`2833`: Pushing 4thBlog
* :ghissue:`2858`: Odffp
* :ghissue:`2857`: Odffp
* :ghissue:`2840`: Shilpi's Week6 Blog
* :ghissue:`2838`: Reconstruction issues using MAP-MRI  model (RTOP, RTAP, RTPP)
* :ghissue:`2845`: MAP ODF issues
* :ghissue:`2851`: How to use "synb0" in Dipy for preprocessing
* :ghissue:`2839`: make order_from_ncoef return an int
* :ghissue:`2844`: doc/tools/: fix trailing dot in version number.
* :ghissue:`2827`: BundleWarp CLI Tutorial - Missing from Website
* :ghissue:`2832`: BundleWarp: added tutorial and fixed a small bug
* :ghissue:`1627`: WIP - NF - Tracking with Initial Directions and other tracking parameters
* :ghissue:`2818`: Vara's week 0 blog
* :ghissue:`2823`: submitting clearn PR for 2nd blog
* :ghissue:`2822`: Pushing 2nd blog,
* :ghissue:`2813`: First Blog
* :ghissue:`2808`: [DOC] Fix cross referencing
* :ghissue:`2798`: Move evac+ to new module `nn`
* :ghissue:`2797`: remove Nibabel InTemporaryDirectory
* :ghissue:`2706`: FYI: Deprecating nisext in nibabel
* :ghissue:`2800`: Remove the Deprecating nisext
* :ghissue:`2689`: Installing DIPY fails with current conda version
* :ghissue:`2718`: StatefulTractogram
* :ghissue:`2795`: bump dependencies minimal version
* :ghissue:`2747`: Cannot set `dipy` as a dependency
* :ghissue:`2791`: Update Patch2Self CLI
* :ghissue:`2792`: Add `patch_radius` parameter to Patch2Self denoise workflow
* :ghissue:`2771`: BUG: Missing Python 3.11 macOS wheels
* :ghissue:`2761`: [UPCOMING]  Release 1.7.0 - workshop release
