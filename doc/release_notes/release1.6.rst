.. _release1.6:

====================================
 Release notes for DIPY version 1.6
====================================

GitHub stats for 2022/03/11 - 2023/01/12 (tag: 1.5.0)

These lists are automatically generated, and may be incomplete or contain duplicates.

The following 15 authors contributed 242 commits.

* Ariel Rokem
* David Romero-Bascones
* Deneb Boito
* Eleftherios Garyfallidis
* Emmanuelle Renauld
* Eric Larson
* Francois Rheault
* Jacob Roberts
* Jon Haitz Legarreta Gorroño
* Malinda Dilhara
* Omar Ocegueda
* Sam Coveney
* Serge Koudoro
* Shreyas Fadnavis
* Tom Dela Haije


We closed a total of 116 issues, 41 pull requests and 75 regular issues;
this is the full list (generated with the script
:file:`tools/github_stats.py`):

Pull Requests (41):

* :ghpull:`2710`: small fixes for tutorials
* :ghpull:`2711`: [FIX] use tempfile module instead of nibabel for TemporaryDirectory
* :ghpull:`2702`: One more small fix to the hcp fetcher.
* :ghpull:`2704`: MAINT: Fixes for 3.11 and sdist
* :ghpull:`2701`: FIX: Don't print a progbar for downloading unless you need to.
* :ghpull:`2700`: [FIX] incompatible type in numpy array
* :ghpull:`2694`: NF: Add RUMBA-SD reconstruction workflow
* :ghpull:`2697`: NF: Adds a fetcher for HCP data.
* :ghpull:`2692`: RF: Improve multi-shell RUMBA test WM response parameterization
* :ghpull:`2693`: TEST: Fix `Node.js` warnings linked to GitHub actions
* :ghpull:`2687`: DOC: Fix typos in SH theory documentation page
* :ghpull:`2690`: STYLE: Remove unnecessary b-val print in CSD reconstruction flow
* :ghpull:`2688`: DOC: Document Dirac delta generation method missing member
* :ghpull:`2683`: DOC: Adds missing documentation for a kwarg.
* :ghpull:`2668`: ENH: Make the DTI fit CLI metric saving message accurate
* :ghpull:`2674`: Improve doc (tensor values' order)
* :ghpull:`2670`: ENH: Allow non-default parameters to `Patch2Self` CLI
* :ghpull:`2672`: DOC: Miscellaneous docstring fixes
* :ghpull:`2669`: DOC: Remove inaccurate `patch2self` docstring default values
* :ghpull:`2664`: DOC: Fix DTI fit CLI docstring
* :ghpull:`2553`: NF: Unbiased groupwise linear bundle registration
* :ghpull:`2369`: Transform points with DiffeormorphicMap
* :ghpull:`2631`: [FIX] Allow patch size parameter to be an int on denoise Workflow
* :ghpull:`2630`: [DOC] Remove search index
* :ghpull:`2629`: [FIX ] Handle save VTP
* :ghpull:`2618`: Use np.linalg.multi_dot instead of multiple np.dot routines
* :ghpull:`2606`: STYLE: Fix miscellaneous Python warnings
* :ghpull:`2600`: Pin Ray
* :ghpull:`2531`: NF: MAP+ constraints
* :ghpull:`2589`: Switch from using nibabel InTemporaryDirectory to standard library tmpfile module
* :ghpull:`2577`: Add positivity constraints to QTI
* :ghpull:`2595`: Temporary skip Cython 0.29.29
* :ghpull:`2591`: STYLE: Avoid array-like mutable default argument values
* :ghpull:`2592`: STYLE: Fix miscellaneous Python warnings
* :ghpull:`2579`: Generalized PCA to less than 3 spatial dims
* :ghpull:`2584`: transform_streamlines changes dtype to float64
* :ghpull:`2566`: [FIX] Update tests for the deprecated `dipy.io.bvectxt` module
* :ghpull:`2581`: Fix logger in SFT
* :ghpull:`2580`: DOC: Fix typos and grammar
* :ghpull:`2576`: DOC: Documentation fixes
* :ghpull:`2568`: DOC: Fix the docstring of `write_mapping`

Issues (75):

* :ghissue:`2710`: small fixes for tutorials
* :ghissue:`2711`: [FIX] use tempfile module instead of nibabel for TemporaryDirectory
* :ghissue:`2709`: DiffeomorphicMap object on github not the same as recent release
* :ghissue:`2708`: Provide the dataset of this code
* :ghissue:`2699`: WIP: Single shell/noreg redux
* :ghissue:`2702`: One more small fix to the hcp fetcher.
* :ghissue:`2704`: MAINT: Fixes for 3.11 and sdist
* :ghissue:`2701`: FIX: Don't print a progbar for downloading unless you need to.
* :ghissue:`2700`: [FIX] incompatible type in numpy array
* :ghissue:`2694`: NF: Add RUMBA-SD reconstruction workflow
* :ghissue:`2696`: Port HCP fetcher from pyAFQ into here
* :ghissue:`2697`: NF: Adds a fetcher for HCP data.
* :ghissue:`2692`: RF: Improve multi-shell RUMBA test WM response parameterization
* :ghissue:`2693`: TEST: Fix `Node.js` warnings linked to GitHub actions
* :ghissue:`1418`: Adding parallel_voxel_fit decorator
* :ghissue:`2687`: DOC: Fix typos in SH theory documentation page
* :ghissue:`2690`: STYLE: Remove unnecessary b-val print in CSD reconstruction flow
* :ghissue:`2688`: DOC: Document Dirac delta generation method missing member
* :ghissue:`2683`: DOC: Adds missing documentation for a kwarg.
* :ghissue:`2679`: Problems with a .nii.gz file when loading and floating
* :ghissue:`2676`: Does ```convert_sh_to_legacy``` work as intended ?
* :ghissue:`2668`: ENH: Make the DTI fit CLI metric saving message accurate
* :ghissue:`2674`: Improve doc (tensor values' order)
* :ghissue:`2670`: ENH: Allow non-default parameters to `Patch2Self` CLI
* :ghissue:`2673`: ENH: Doc: DTI format
* :ghissue:`2667`: Defaults for Patch2Self
* :ghissue:`2672`: DOC: Miscellaneous docstring fixes
* :ghissue:`2669`: DOC: Remove inaccurate `patch2self` docstring default values
* :ghissue:`2662`: Update cmd_line dipy_fit_dti
* :ghissue:`2664`: DOC: Fix DTI fit CLI docstring
* :ghissue:`2658`: Any chance of arm64 wheels for Mac / Python 3.10?
* :ghissue:`2659`: Angle var
* :ghissue:`2649`: IVIM VarPro fitting running error
* :ghissue:`2553`: NF: Unbiased groupwise linear bundle registration
* :ghissue:`2424`: Transforming individual points with SDR
* :ghissue:`2327`: Diffeomorphic transformation of coordinates
* :ghissue:`2313`: deform_streamlines for wholebrain tractogram doesn't function properly
* :ghissue:`936`: WIP: coordinate mapping with DiffeomorphicMap
* :ghissue:`2369`: Transform points with DiffeormorphicMap
* :ghissue:`2616`: dti TensorModel fitting issue
* :ghissue:`2627`: Free-Water Analysis Gradient Table Error
* :ghissue:`2635`: get_flexi_tvis_affine(tvis_hdr, nii_aff)
* :ghissue:`2634`: Fix small difference between pdfs dense 2d/3d
* :ghissue:`2559`: CLI denoise patch_size error
* :ghissue:`2631`: [FIX] Allow patch size parameter to be an int on denoise Workflow
* :ghissue:`2564`: Search, index, module links not working in the doc
* :ghissue:`2630`: [DOC] Remove search index
* :ghissue:`2572`: Saving vtp file error
* :ghissue:`2629`: [FIX ] Handle save VTP
* :ghissue:`2622`: Error with Illustrating Electrostatic Repulsion
* :ghissue:`2618`: Use np.linalg.multi_dot instead of multiple np.dot routines
* :ghissue:`2617`: DKI model fit shape broadcast error
* :ghissue:`2606`: STYLE: Fix miscellaneous Python warnings
* :ghissue:`2602`: Dear experts, how can I set different number of fiber tracks before generating streamlines?
* :ghissue:`2603`: Dear experts, how to save dti_peaks?
* :ghissue:`2600`: Pin Ray
* :ghissue:`2531`: NF: MAP+ constraints
* :ghissue:`2587`: Thread safety concerns with reliance on `nibabel.tmpdirs.InTemporaryDirectory`
* :ghissue:`2589`: Switch from using nibabel InTemporaryDirectory to standard library tmpfile module
* :ghissue:`2577`: Add positivity constraints to QTI
* :ghissue:`2450`: [NF] New tracking Algorithm:  Parallel Transport Tractography (PTT)
* :ghissue:`2594`: DIPY compilation fails with the last release of cython (0.29.29)
* :ghissue:`2595`: Temporary skip Cython 0.29.29
* :ghissue:`2591`: STYLE: Avoid array-like mutable default argument values
* :ghissue:`2592`: STYLE: Fix miscellaneous Python warnings
* :ghissue:`2579`: Generalized PCA to less than 3 spatial dims
* :ghissue:`2584`: transform_streamlines changes dtype to float64
* :ghissue:`2566`: [FIX] Update tests for the deprecated `dipy.io.bvectxt` module
* :ghissue:`2581`: Fix logger in SFT
* :ghissue:`2580`: DOC: Fix typos and grammar
* :ghissue:`2576`: DOC: Documentation fixes
* :ghissue:`2573`: Cannot Import "Feature" From "dipy.segment.metric"
* :ghissue:`2568`: DOC: Fix the docstring of `write_mapping`
* :ghissue:`2567`: This should be a `1`
* :ghissue:`2565`: compress_streamlines() not available anymore
