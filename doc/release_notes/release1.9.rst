.. _release1.9:

====================================
 Release notes for DIPY version 1.9
====================================

GitHub stats for 2023/12/14 - 2024/03/07 (tag: 1.8.0)

These lists are automatically generated, and may be incomplete or contain duplicates.

The following 17 authors contributed 175 commits.

* Ariel Rokem
* Atharva Shah
* Ebrahim Ebrahim
* Eleftherios Garyfallidis
* Gabriel Girard
* John Shen
* Jon Haitz Legarreta Gorroño
* Jong Sung Park
* Maharshi Gor
* Matthew Feickert
* Philippe Karan
* Praitayini Kanakaraj
* Sam Coveney
* Sandro
* Serge Koudoro
* dependabot[bot]
* Étienne Mollier


We closed a total of 139 issues, 60 pull requests and 81 regular issues;
this is the full list (generated with the script
:file:`tools/github_stats.py`):

Pull Requests (60):

* :ghpull:`3095`: [UPCOMING] Release preparation for 1.9.0
* :ghpull:`3086`: [RF] Fix spherical harmonic terminology swap
* :ghpull:`3105`: [doc] improve some tutorials rendering
* :ghpull:`3109`: [BF] convert_tractogram fix
* :ghpull:`3108`: enabled trx support with correct header
* :ghpull:`3107`: enabled trx support for viz
* :ghpull:`3033`: [RF] fix dki mask for nlls
* :ghpull:`3104`: Dkimaskfix
* :ghpull:`3106`: volume slices visibility fixed
* :ghpull:`3102`: Bugfix for peaks slices and synchronization.
* :ghpull:`3078`: return S0 from dki fit
* :ghpull:`3101`: [BF] Uniformize cython version
* :ghpull:`3097`: Feature/surface
* :ghpull:`3048`: [TEST] Adds support of cython for pytest
* :ghpull:`3053`: [NF] Add workflow to convert tensors in different formats
* :ghpull:`3073`: [NF] Add DSI workflow
* :ghpull:`3099`: [DOC] fix some typo [ci skip]
* :ghpull:`3098`: Removing tensorflow addon from DL models
* :ghpull:`2973`: Tab names for slice tabs as file names.
* :ghpull:`3081`: NF: Adding N4 bias correction deep learning model
* :ghpull:`3092`: Feature: volume synchronizing
* :ghpull:`3059`: Generalize special casing while loading bvecs, to include the case of transposed 2,3 vectors
* :ghpull:`3090`: RF - changed memory view to double* trilinear_interpolation_4d
* :ghpull:`3080`: Adding SH basis legacy option support to peaks_from_model
* :ghpull:`3087`: backward compatibility fixed
* :ghpull:`3088`: [TEST] Pin pytest
* :ghpull:`3084`: fixed 4d slice issue
* :ghpull:`3083`: Np.unique check removed.
* :ghpull:`3082`: Add Fedora installation instructions [ci skip]
* :ghpull:`3076`: [CI] Update scientific-python/upload-nightly-action to 0.5.0
* :ghpull:`3070`: [DOC] Fix installation link in README  [ci skip]
* :ghpull:`3069`: [DOC] Fix DTI Tutorial [ci skip]
* :ghpull:`3063`: [RF] remove cpdef in PmfGen
* :ghpull:`3054`: [DOC] Fix some links [ci skip]
* :ghpull:`3060`: Bump codecov/codecov-action from 3 to 4
* :ghpull:`3061`: [OPT] Enable openmp for macOS wheel and CI's
* :ghpull:`3049`: [MTN] code cleaning: remove some dependencies version checking
* :ghpull:`3050`: [RF] Move ``dipy.boots.resampling`` to  ``dipy.stats.resampling``
* :ghpull:`3051`: [RF] Remove dipy.io.bvectxt module
* :ghpull:`3052`: Bump scientific-python/upload-nightly-action from 3eb3a42b50671237cace9be2d18a3e4b3845d3c4 to 66bc1b6beedff9619cdff8f3361a06802c8f5874
* :ghpull:`3045`: [DOC] fix `multi_shell_fiber_response` docstring array dims [ci skip]
* :ghpull:`3041`: [NF] Add convert tractograms flow
* :ghpull:`3040`: [BW] Remove some python2 reference
* :ghpull:`3039`: [TEST] Add setup_module and teardown_module
* :ghpull:`3038`: [NF] Update `dipy_info`: allow tractogram files format
* :ghpull:`3043`: d/d/t/test_data.py: endian independent dtype.
* :ghpull:`3042`: pyproject.toml: no cython at run time.
* :ghpull:`3027`: [NF] Add Concatenate tracks workflows
* :ghpull:`3008`: NF: add SH basis conversion between dipy and mrtrix3
* :ghpull:`3025`: [TEST] Manage http errors for stateful tractograms
* :ghpull:`3031`: Bugfix: Horizon image's dtype validation
* :ghpull:`3021`: [MTN] Remove 3.8 Ci's
* :ghpull:`3026`: [RF] Fix cython 3 warnings
* :ghpull:`3022`: [DOC] Fix logo size and link [ci skip]
* :ghpull:`3013`: Added Fibonacci spiral and test for it
* :ghpull:`3019`: DOC: Fix link to toolchain roadmap page in `README`
* :ghpull:`3012`: DOC: Document observance for Scientific Python min supported versions
* :ghpull:`3018`: Bump actions/download-artifact from 3 to 4
* :ghpull:`3017`: Bump actions/upload-artifact from 3 to 4
* :ghpull:`3014`: Update release1.8.rst

Issues (81):

* :ghissue:`2970`: spherical harmonic degree/order terminology swapped
* :ghissue:`3105`: [doc] improve some tutorials rendering
* :ghissue:`3109`: [BF] convert_tractogram fix
* :ghissue:`3108`: enabled trx support with correct header
* :ghissue:`3107`: enabled trx support for viz
* :ghissue:`2994`: DKI masking
* :ghissue:`3033`: [RF] fix dki mask for nlls
* :ghissue:`3104`: Dkimaskfix
* :ghissue:`3106`: volume slices visibility fixed
* :ghissue:`3102`: Bugfix for peaks slices and synchronization.
* :ghissue:`2281`: Black Output for pam5 file with dipy_horizon
* :ghissue:`3078`: return S0 from dki fit
* :ghissue:`3101`: [BF] Uniformize cython version
* :ghissue:`3097`: Feature/surface
* :ghissue:`2719`: pytest and cdef functions
* :ghissue:`3048`: [TEST] Adds support of cython for pytest
* :ghissue:`3053`: [NF] Add workflow to convert tensors in different formats
* :ghissue:`3073`: [NF] Add DSI workflow
* :ghissue:`3099`: [DOC] fix some typo [ci skip]
* :ghissue:`3098`: Removing tensorflow addon from DL models
* :ghissue:`2973`: Tab names for slice tabs as file names.
* :ghissue:`3081`: NF: Adding N4 bias correction deep learning model
* :ghissue:`3092`: Feature: volume synchronizing
* :ghissue:`3093`: Can I use a different sort of dataset and also what if I don't have a bval , I use . mat images format PATCH2Self
* :ghissue:`3059`: Generalize special casing while loading bvecs, to include the case of transposed 2,3 vectors
* :ghissue:`3090`: RF - changed memory view to double* trilinear_interpolation_4d
* :ghissue:`3080`: Adding SH basis legacy option support to peaks_from_model
* :ghissue:`3085`: Viz Tests failing
* :ghissue:`3087`: backward compatibility fixed
* :ghissue:`3088`: [TEST] Pin pytest
* :ghissue:`3074`: Horizon for large datasets - concerns regarding np.unique
* :ghissue:`3075`: Horizon - 4D data support - slicing on the 4-th dim
* :ghissue:`3084`: fixed 4d slice issue
* :ghissue:`3083`: Np.unique check removed.
* :ghissue:`3082`: Add Fedora installation instructions [ci skip]
* :ghissue:`3065`: Add s390x test workflow
* :ghissue:`3076`: [CI] Update scientific-python/upload-nightly-action to 0.5.0
* :ghissue:`3070`: [DOC] Fix installation link in README  [ci skip]
* :ghissue:`3069`: [DOC] Fix DTI Tutorial [ci skip]
* :ghissue:`3066`: Dipy website incorrect image
* :ghissue:`3063`: [RF] remove cpdef in PmfGen
* :ghissue:`3054`: [DOC] Fix some links [ci skip]
* :ghissue:`3060`: Bump codecov/codecov-action from 3 to 4
* :ghissue:`3061`: [OPT] Enable openmp for macOS wheel and CI's
* :ghissue:`3057`:   Using the atlas HCP1065 in DIPY
* :ghissue:`3055`: [RF] replace Bunch by Enum
* :ghissue:`3049`: [MTN] code cleaning: remove some dependencies version checking
* :ghissue:`3050`: [RF] Move ``dipy.boots.resampling`` to  ``dipy.stats.resampling``
* :ghissue:`3051`: [RF] Remove dipy.io.bvectxt module
* :ghissue:`3052`: Bump scientific-python/upload-nightly-action from 3eb3a42b50671237cace9be2d18a3e4b3845d3c4 to 66bc1b6beedff9619cdff8f3361a06802c8f5874
* :ghissue:`2789`: Horizon image's dtype validation
* :ghissue:`3047`: "Editable" installation broken
* :ghissue:`3045`: [DOC] fix `multi_shell_fiber_response` docstring array dims [ci skip]
* :ghissue:`3041`: [NF] Add convert tractograms flow
* :ghissue:`3040`: [BW] Remove some python2 reference
* :ghissue:`3039`: [TEST] Add setup_module and teardown_module
* :ghissue:`3038`: [NF] Update `dipy_info`: allow tractogram files format
* :ghissue:`3043`: d/d/t/test_data.py: endian independent dtype.
* :ghissue:`3042`: pyproject.toml: no cython at run time.
* :ghissue:`3027`: [NF] Add Concatenate tracks workflows
* :ghissue:`3035`: If I want to use  6D array in "actor.odf_slicer", how can i do?
* :ghissue:`2993`: Add conversion utility between DIPY and MRtrix3 spherical harmonic basis
* :ghissue:`3008`: NF: add SH basis conversion between dipy and mrtrix3
* :ghissue:`3025`: [TEST] Manage http errors for stateful tractograms
* :ghissue:`3031`: Bugfix: Horizon image's dtype validation
* :ghissue:`3032`: Consider moving your nightly wheel away from the scipy-wheel-nightly (old location) to scientific-python-nightly-wheels
* :ghissue:`3021`: [MTN] Remove 3.8 Ci's
* :ghissue:`3003`: DIPY installation raises Cython warnings
* :ghissue:`3026`: [RF] Fix cython 3 warnings
* :ghissue:`2852`: Different behavior regarding color channels of horizon
* :ghissue:`2378`: A novice's request for advice on loading very large tractograms (.tck)
* :ghissue:`2064`: How to register MR to CT of the same person?
* :ghissue:`2601`: read_bvals_bvecs can't read double volume dwi
* :ghissue:`3022`: [DOC] Fix logo size and link [ci skip]
* :ghissue:`3013`: Added Fibonacci spiral and test for it
* :ghissue:`3020`: load_nifti import doesn't work if using submodule directly
* :ghissue:`3019`: DOC: Fix link to toolchain roadmap page in `README`
* :ghissue:`3012`: DOC: Document observance for Scientific Python min supported versions
* :ghissue:`3018`: Bump actions/download-artifact from 3 to 4
* :ghissue:`3017`: Bump actions/upload-artifact from 3 to 4
* :ghissue:`3014`: Update release1.8.rst
* :ghissue:`1525`: Clang-omp moved to boneyard on brew
