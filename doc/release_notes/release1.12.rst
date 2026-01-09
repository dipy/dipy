
GitHub stats for 2025/03/16 - 2025/08/14 (tag: 1.11.0)

These lists are automatically generated, and may be incomplete or contain duplicates.

The following 16 authors contributed 212 commits.

* Alberto Di Biase
* Ariel Rokem
* Atharva Shah
* Daniel McCloy
* Eleftherios Garyfallidis
* Francois Rheault
* Gabriel Girard
* Jon Haitz Legarreta Gorroño
* Jong Sung Park
* Kaustav Deka
* Kesshi Jordan
* Maharshi Gor
* Prajwal Reddy
* Santiago Vila
* Serge Koudoro
* dependabot[bot]


We closed a total of 105 issues, 61 pull requests and 44 regular issues;
this is the full list (generated with the script
:file:`tools/github_stats.py`):

Pull Requests (61):

* :ghpull:`3605`: RF: Parallel `quantize_evecs`
* :ghpull:`3613`: BF: Allow `length` to work with float16 streamlines
* :ghpull:`3612`: BF: fix lower triangular bug introduced in #3563
* :ghpull:`1617`: NF: added streamline clipping function to utils
* :ghpull:`3606`: RF: Address cvxpy 1.7.0 warnings.
* :ghpull:`3609`: Bump actions/first-interaction from 1 to 2 in the actions group
* :ghpull:`3560`: Doc: explicit ``finalize_mask`` information in CLI tutorial
* :ghpull:`3602`: ENH: Adopt `pathlib` for workflows
* :ghpull:`3604`: RF: replace old LocalTracking by our new tracking interface
* :ghpull:`3587`: RF: from TRK to TRX as default file format for tracks
* :ghpull:`3603`: DOC: Fix workflows test utils parameter name in docstring
* :ghpull:`3601`: DOC: Record API changes after `pathlib` adoption
* :ghpull:`3593`: STYLE: Adopt `pathlib` for path manipulation
* :ghpull:`3599`: TEST: Restore removed test case in io workflows
* :ghpull:`3600`: MNT: Fix miscellaneous labeler regexes
* :ghpull:`3597`: STYLE: Miscellaneous style fixes
* :ghpull:`3598`: BF: Miscellaneous fixes to surfaces
* :ghpull:`3592`: STYLE: Adopt a unified logger across the code base
* :ghpull:`3429`: StatefulSurface - Class to handle surfaces
* :ghpull:`3595`: DOC: Remove default arguments from docstrings
* :ghpull:`3594`: STYLE: Miscellaneous style fixes
* :ghpull:`3591`: STYLE: Apply `ruff` manually to all files
* :ghpull:`3582`: NF: Add PR labeler workflow
* :ghpull:`3586`: MNT: Change issue template file extensions
* :ghpull:`3584`: MNT: Add GitHub issue templates
* :ghpull:`3583`: DOC: Change unused commit prefixes to some other more useful ones
* :ghpull:`3581`: SciPy deprecation of "disp" in optimizer
* :ghpull:`3556`: RF: allow the saving of S0 estimate for dti workflow
* :ghpull:`3563`: RF: Address some Zero division warnings
* :ghpull:`3565`: RF: use multi_voxel_fit for rumba
* :ghpull:`3541`: RF: Deprecate autocrop in median_otsu
* :ghpull:`3580`: TEST: uncomment and update test_cross
* :ghpull:`3515`: RF: Improve `dipy_info` printed output
* :ghpull:`3562`: DOC: Add missing opening backtick to reference syntax.
* :ghpull:`3561`: STYLE: Make affine variable naming consistent
* :ghpull:`3557`: DOC: fix streamline-tools tutorial by avoiding the use of identity affine
* :ghpull:`3559`: [RF]: Patch2self in denoising CLI tutorial
* :ghpull:`3555`: ENH: improve dipy_info message when no reference for some streamline files.
* :ghpull:`3554`: Doc: Fix typo in multiple tutorial
* :ghpull:`3546`: BF: Make `StoppingCriterion` reproducible for multi-thread execution
* :ghpull:`3548`: RF: fix typo in hcp fetcher function argument name
* :ghpull:`3549`: BF: Improves Cython enum management.
* :ghpull:`3520`: ENH: Opacity slider turned off on hide
* :ghpull:`3493`: NF: allow broadcasting in dipy_math
* :ghpull:`3488`: RF - changed min/max len from nbr pts to mm
* :ghpull:`3538`: RF: Fixed the latex
* :ghpull:`3519`: ENH: Horizon peaks fname support
* :ghpull:`3455`: Fix Bugs in #3453: Ensure Correct Weight Reshaping & Consistent Extra Output in iter_fit_tensor
* :ghpull:`3535`: CI: Ignore fork() warnings.
* :ghpull:`3528`: DOC: Changed documentation errors in dipy.sims.voxel
* :ghpull:`3533`: CI: introduce cached Data.
* :ghpull:`3531`: CI: Ignore specific cvxpy warnings to avoid CI failure.
* :ghpull:`3527`: BF: Search bar should come bigger in the center.
* :ghpull:`3530`: STYLE: Add additional emojis to first interaction message
* :ghpull:`3526`: Bump scientific-python/upload-nightly-action from 0.6.1 to 0.6.2 in the actions group
* :ghpull:`3522`: FIX: Avoid division by zero on single-CPU systems (issue #3521)
* :ghpull:`3510`: [RF]: Saving the figure in the example.
* :ghpull:`3512`: STYLE: Call `warning` instead of the deprecated `warn` function
* :ghpull:`3514`: DOC: Miscellaneous doc improvements
* :ghpull:`3495`: BF: Fixing minor issue in dipy_classify_tissue dam option
* :ghpull:`3481`: UPCOMING:  Release 1.11.0

Issues (44):

* :ghissue:`3608`: Predicting from a fitted DTI model is broken?
* :ghissue:`3588`: In-house QP solver using numba for fitting MSMT
* :ghissue:`3551`: Shutting down the gitter live chat ?
* :ghissue:`3124`: ENH: Add commentary on median otsu tutorial/change default parameters
* :ghissue:`3506`: Make sure .trx is default everywhere
* :ghissue:`3497`: dipy_fit_dti does not return S0 estimate
* :ghissue:`3418`: Division by 0 warning
* :ghissue:`3352`: Rumba and multivoxel fit decorator
* :ghissue:`3537`: Remove autocrop from median_otsu
* :ghissue:`2866`: arm64 test failure with numpy 1.24.2
* :ghissue:`1776`: make it really easy for DiPy to call graspy functions
* :ghissue:`2269`: Allow dipy_fit_dti Workflow to Use Native Orientation of Provided Mask Image
* :ghissue:`2682`: DIPY Free-Water Corrected DTI (FA output quality)
* :ghissue:`2428`: dipy.align.reslice
* :ghissue:`2399`: fitting free water mapping out of iterations
* :ghissue:`2319`: Reconstruction with MSMT CSD example resulted in Nan values
* :ghissue:`2329`: Slice to volume registration
* :ghissue:`1974`: MAP-MRI optimization & validation errors
* :ghissue:`2612`: Mean Diffusivity turns out low in CSF when reconstructing using ReconstDtiFlow
* :ghissue:`687`: Refactor imaffine class structure.
* :ghissue:`706`: Thoughts about using the SyN algorithm
* :ghissue:`2184`: discussion: continuous integration using GPUs
* :ghissue:`1991`: [DISCUSSION] b-tensor encoding gradient table format
* :ghissue:`2186`: Pure-python implementation of boundary-based registration (BBR)?
* :ghissue:`1852`: Telemetry
* :ghissue:`2338`: Team DIPY Citations
* :ghissue:`2663`: Plotting tensors in Fury: issue with right-handed coordinate system?
* :ghissue:`2850`: Orthogonal Tensor Parameter (Moment) Maps
* :ghissue:`1886`: NF - add the option to save the last point of streamlines
* :ghissue:`3382`: Check use of affine in `streamline_tools` example
* :ghissue:`2473`: Add Patch2self in our denoising CLI tutorial
* :ghissue:`2695`: Replace CENIR multishell with HBN POD2 data
* :ghissue:`2728`: Do we have multi_processes method for streamline extraction
* :ghissue:`2848`: Noise estimation of T1 brain MR images
* :ghissue:`3500`: dipy_fit_csa does not use num_processes
* :ghissue:`3517`: `dipy_info` provides misleading message when no reference is provided
* :ghissue:`3413`: Tutorial Typos
* :ghissue:`3540`: Tractography with GFA as stopping criterion is not reproducible across runs
* :ghissue:`3507`: Extract b0 interface throws value error with HCP 7T (MGH) - subject 1007
* :ghissue:`3474`: Filenames missing in horizon for peaks objects
* :ghissue:`3222`: Issue with CWLS in dki
* :ghissue:`3536`: FA latex formula does not show correctly on DTI tutorial
* :ghissue:`1404`: RF - DirectionGetter.get_direction function return
* :ghissue:`3453`: Inconistancy between documentation and implementation on dipy.reconst.dti.nlls_fit_tensor
