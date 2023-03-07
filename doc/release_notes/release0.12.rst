.. _release0.12:

====================================
 Release notes for DIPY version 0.12
====================================


GitHub stats for 2016/02/21 - 2017/06/26 (tag: 0.11.0)

These lists are automatically generated, and may be incomplete or contain duplicates.

The following 48 authors contributed 1491 commits.

* Alexandre Gauvin
* Antonio Ossa
* Ariel Rokem
* Bago Amirbekian
* Bishakh Ghosh
* David Reagan
* Eleftherios Garyfallidis
* Etienne St-Onge
* Gabriel Girard
* Gregory R. Lee
* Jean-Christophe Houde
* Jon Haitz Legarreta
* Julio Villalon
* Kesshi Jordan
* Manu Tej Sharma
* Marc-Alexandre Côté
* Matthew Brett
* Matthieu Dumont
* Nil Goyette
* Omar Ocegueda Gonzalez
* Rafael Neto Henriques
* Ranveer Aggarwal
* Riddhish Bhalodia
* Rutger Fick
* Samuel St-Jean
* Serge Koudoro
* Shahnawaz Ahmed
* Sourav Singh
* Stephan Meesters
* Stonge Etienne
* Guillaume Theaud
* Tingyi Wanyan
* Tom Wright
* Vibhatha Abeykoon
* Yaroslav Halchenko
* Eric Peterson
* Sven Dorkenwald
* theaverageguy


We closed a total of 511 issues, 169 pull requests and 342 regular issues;
this is the full list (generated with the script
:file:`tools/github_stats.py`):

Pull Requests (169):

* :ghpull:`1273`: Release 0.12 doc fix
* :ghpull:`1272`: small correction for debugging purpose on nlmeans
* :ghpull:`1269`: Odf slicer
* :ghpull:`1271`: Viz tut update
* :ghpull:`1268`: Following up on #1243.
* :ghpull:`1223`: local PCA using SVD
* :ghpull:`1270`: Doc cleaning deprecation warning
* :ghpull:`1267`: Adding a decorator for skipping test if openmp is not available
* :ghpull:`1090`: Documentation for command line interfaces
* :ghpull:`1243`: Better fvtk.viz error when no VTK installed
* :ghpull:`1263`: Cast Streamline attrs to numpy ints, to avoid buffer mismatch.
* :ghpull:`1254`: Automate script installation
* :ghpull:`1261`: removing absolute path in tracking module
* :ghpull:`1255`: Fix missing documentation content
* :ghpull:`1260`: removing absolute path in reconst
* :ghpull:`1241`: Csa and csd reconstruction workflow rebased
* :ghpull:`1250`: DOC: Fix reconst_dki.py DKI example documentation typos.
* :ghpull:`1244`: TEST: Decrease precision of tests for dki micro model prediction
* :ghpull:`1235`: New hdf5 file format for saving PeaksAndMetrics objects
* :ghpull:`1231`: TST: Reduce precision requirement for test of tortuosity estimation.
* :ghpull:`1233`: Feature: Added environment override for dipy_home variable
* :ghpull:`1234`: BUG: Fix non-ASCII characters in reconst_dki.py example.
* :ghpull:`1222`: A lightweight UI for medical visualizations #5: 2D Circular Slider
* :ghpull:`1228`: RF: Use cython imports instead of relying on extern
* :ghpull:`1227`: BF: Use np.npy_intp instead of assuming long for ArraySequence attributes
* :ghpull:`1226`: DKI Microstructural model
* :ghpull:`1229`: RF: allow for scipy pre-release deprecations
* :ghpull:`1225`: Add one more multi b-value data-set
* :ghpull:`1219`: MRG:Data off dropbox
* :ghpull:`1221`: NF: Check multi b-value
* :ghpull:`1212`: Follow PEP8 in reconst (part 2)
* :ghpull:`1217`: Use integer division in reconst_gqi.py
* :ghpull:`1205`: A lightweight UI for medical visualizations #4: 2D Line Slider
* :ghpull:`1166`: RF: Use the average sigma in the mask.
* :ghpull:`1216`: Use integer division to avoid errors in indexing
* :ghpull:`1214`: DOC: add a clarification note to simplify_warp_funcion_3d
* :ghpull:`1208`: Follow PEP8 in reconst (part 1)
* :ghpull:`1206`: Revert #1204, and add a filter to suppress warnings.
* :ghpull:`1196`: MRG: Use dipy's array comparisons for tests
* :ghpull:`1204`: Suppress warnings regarding one-dimensional arrays changes in scipy 0.18
* :ghpull:`1199`: A lightweight UI for medical visualizations #3: Changes to Event Handling
* :ghpull:`1202`: Use integer division to avoid errors in indexing
* :ghpull:`1198`: ENH: avoid log zero
* :ghpull:`1201`: Fix out of bounds point not being classified OUTSIDEIMAGE (binary cla…
* :ghpull:`1115`: Bayesian Tissue Classification
* :ghpull:`1052`: Conda install
* :ghpull:`1183`: A lightweight UI for medical visualizations #2: TextBox
* :ghpull:`1186`: MRG: use newer wheelhouse for installs
* :ghpull:`1195`: Make PeaksAndMetrics pickle-able
* :ghpull:`1194`: Use assert_arrays_equal when needed.
* :ghpull:`1193`: Deprecate the Accent colormap, in anticipation of changes in MPL 2.0
* :ghpull:`1140`: A lightweight UI for medical visualizations #1: Button
* :ghpull:`1171`: fix:dev: added numpy.int64 for my triangle array
* :ghpull:`1123`: Add the mask workflow
* :ghpull:`1174`: NF: added the repulsion 200 sphere.
* :ghpull:`1177`: BF: fix interpolation call with Numpy 1.12
* :ghpull:`1162`: Return S0 value for DTI fits
* :ghpull:`1147`: add this fix for newer version of pytables.
* :ghpull:`1076`: ENH: Add support for ArraySequence in `length` function
* :ghpull:`1050`: ENH: expand OpenMP utilities and move from denspeed.pyx to dipy.utils
* :ghpull:`1082`: Add documentation uploading script
* :ghpull:`1153`: Athena mapmri
* :ghpull:`1159`: TST - add tests for various affine matrices for local tracking
* :ghpull:`1157`: Replace `get_affine` with `affine` and `get_header` with `header`.
* :ghpull:`1160`: Add Shahnawaz to list of contributors.
* :ghpull:`1158`: BF: closing matplotlib plots for each file while running the examples
* :ghpull:`1151`: Define fmin() for Visual Studio
* :ghpull:`1149`: Change DKI_signal to dki_signal
* :ghpull:`1137`: Small fix to insure that fwDTI non-linear procedure does not crash
* :ghpull:`942`: NF: Added support to colorize each line points individually
* :ghpull:`1141`: Do not cover files related to benchmarks.
* :ghpull:`1098`: Adding custom interactor for visualisation
* :ghpull:`1136`: Update deprecated function.
* :ghpull:`1113`: TST: Test for invariance of model_params to splitting of the data.
* :ghpull:`1134`: Rebase of https://github.com/dipy/dipy/pull/993
* :ghpull:`1064`: Faster dti odf
* :ghpull:`1114`: flexible grid to streamline affine generation and pathlength function
* :ghpull:`1122`: Add the reconst_dti workflow
* :ghpull:`1132`: Update .travis.yml and README.md
* :ghpull:`1125`: Intensity adjustment. Find a better upper bound for interpolating images.
* :ghpull:`1130`: Minor corrections for showing surfaces
* :ghpull:`1092`: Line-based target()
* :ghpull:`1129`: Fix 1127
* :ghpull:`1034`: Viz surfaces
* :ghpull:`1060`: Fast computation of Cross Correlation metric
* :ghpull:`1124`: Small fix in free water DTI model
* :ghpull:`1058`: IVIM
* :ghpull:`1110`: WIP : Ivim linear fitting
* :ghpull:`1120`: Fix 1119
* :ghpull:`1075`: Drop26
* :ghpull:`835`: NF: Free water tensor model
* :ghpull:`1046`: BF - peaks_from_model with nbr_processes <= 0
* :ghpull:`1049`: MAINT: minor cython cleanup in align/vector_fields.pyx
* :ghpull:`1087`: Base workflow enhancements + tests
* :ghpull:`1112`: DOC: Math rendering issue in SFM gallery example.
* :ghpull:`1109`: Changed default value of mni template
* :ghpull:`1106`: Including MNI Template 2009c in Fetcher
* :ghpull:`1066`: Adaptive Denoising
* :ghpull:`1091`: Modifications for building docs with python3
* :ghpull:`1105`: Import reload function from imp module explicitly for python3
* :ghpull:`1102`: MRG: add pytables to travis-ci, Py35 full test run
* :ghpull:`1100`: Fix for Python 3 in io.dpy
* :ghpull:`1094`: Updates to FBC measures documentation
* :ghpull:`1059`: Documentation to discourage misuse of GradientTable
* :ghpull:`1063`: Fixes #1061 : Changed all S0 to 1.0
* :ghpull:`1089`: BF: fix test error on Python 3
* :ghpull:`1079`: Return a generator from `orient_by_roi`
* :ghpull:`1088`: Restored the older implementation of nlmeans
* :ghpull:`1080`: DOC: TensorModel.__init__ docstring.
* :ghpull:`828`: Fiber to bundle coherence measures
* :ghpull:`1072`: DOC: Added a coverage badge to README.rst
* :ghpull:`1025`: PEP8: Fix pep8 in segment
* :ghpull:`1077`: DOC: update fibernavigator link
* :ghpull:`1069`: DOC: Small one -- we need this additional line of white space to render.
* :ghpull:`1068`: Renamed test_shore for consistency
* :ghpull:`1067`: Generate b vectors using disperse_charges
* :ghpull:`1065`: improve OMP parallelization with scheduling
* :ghpull:`1062`: BF - fix CSD.predict to work with nd inputs.
* :ghpull:`1056`: Remove tracking interfaces
* :ghpull:`1028`: BF: Predict DKI with a volume of S0
* :ghpull:`1041`: NF - Add PMF Threshold to Tractography
* :ghpull:`1039`: Doc - fix definition of real_sph_harm functions
* :ghpull:`1019`: MRG: fix heavy dependency check; no numpy for setup
* :ghpull:`1018`: Fix: denspeed.pyx to give correct output for nlmeans
* :ghpull:`1035`: Fix for fetcher files in Windows
* :ghpull:`974`: Minor change in `tools/github_stats.py`
* :ghpull:`1021`: Added warning for VTK not installed
* :ghpull:`1024`: Documentation fix for reconst_dsid.py
* :ghpull:`981`: Fixes #979 : No figures in DKI example - Add new line after figure
* :ghpull:`958`: FIX: PEP8 in testing
* :ghpull:`1005`: FIX: Use absolute imports in io
* :ghpull:`951`: Contextual Enhancement update: fix SNR issue, fix reference
* :ghpull:`1015`: Fix progressbar of fetcher
* :ghpull:`992`: FIX: Update the import statements to use absolute import in core
* :ghpull:`1003`: FIX: Change the import statements in direction
* :ghpull:`1004`: FIX: Use absolute import in pkg_info
* :ghpull:`1006`: FIX: Use absolute import in utils and scratch
* :ghpull:`1010`: Absolute Imports in Viz
* :ghpull:`929`: Fix PEP8 in data
* :ghpull:`941`: BW: skimage.filter module name warning
* :ghpull:`976`: Fix PEP8 in sims and remove unnecessary imports
* :ghpull:`956`: FIX: PEP8 in reconst/test and reconst/benchmarks
* :ghpull:`955`: FIX: PEP8 in external
* :ghpull:`952`: FIX: PEP8 in tracking and tracking benchmarks/tests
* :ghpull:`982`: FIX: relative imports in dipy/align
* :ghpull:`972`: Fixes #901 : Added documentation for "step" in dti
* :ghpull:`971`: Add progress bar feature to dipy.data.fetcher
* :ghpull:`989`: copyright 2008-2016
* :ghpull:`977`: Relative import fix in dipy/align
* :ghpull:`957`: FIX: PEP8 in denoise
* :ghpull:`959`: FIX: PEP8 in utils
* :ghpull:`967`: Update index.rst correcting the date of release 0.11
* :ghpull:`954`: FIX: PEP8 in direction
* :ghpull:`965`: Fix typo
* :ghpull:`948`: Fix PEP8 in boots
* :ghpull:`946`: FIX: PEP8 for  test_sumsqdiff and test_scalespace
* :ghpull:`964`: FIX: PEP8 in test_imaffine
* :ghpull:`963`: FIX: PEP8 in core
* :ghpull:`947`: FIX: PEP8 for test files
* :ghpull:`897`: PEP8
* :ghpull:`926`: Fix PEP8 in fixes
* :ghpull:`937`: BF : Clamping of the value of v in winding function
* :ghpull:`907`: DOC: switch to using mathjax for maths
* :ghpull:`932`: Fixes #931 : checks if nb_points=0
* :ghpull:`927`: Fix PEP8 in io and remove duplicate definition in test_bvectxt.py
* :ghpull:`913`: Fix pep8 in workflows
* :ghpull:`935`: Setup: go on to version 0.12 development.
* :ghpull:`934`: DOC: Update github stats for 0.11 as of today.
* :ghpull:`933`: Updating release dates

Issues (342):

* :ghissue:`1273`: Release 0.12 doc fix
* :ghissue:`1272`: small correction for debugging purpose on nlmeans
* :ghissue:`1269`: Odf slicer
* :ghissue:`1143`: Slice through ODF fields
* :ghissue:`1271`: Viz tut update
* :ghissue:`1246`: WIP: Replace widget with ui components in example.
* :ghissue:`1268`: Following up on #1243.
* :ghissue:`1223`: local PCA using SVD
* :ghissue:`1265`: Test failure on OSX in test_nlmeans_4d_3dsigma_and_threads
* :ghissue:`1270`: Doc cleaning deprecation warning
* :ghissue:`1251`: Slice through ODF fields - Rebased
* :ghissue:`1267`: Adding a decorator for skipping test if openmp is not available
* :ghissue:`1090`: Documentation for command line interfaces
* :ghissue:`1243`: Better fvtk.viz error when no VTK installed
* :ghissue:`1238`: Cryptic fvtk.viz error when no VTK installed
* :ghissue:`1242`: DKI microstructure model tests still fail intermittenly
* :ghissue:`1252`: Debug PR only - Odf slicer vtk tests (do not merge)
* :ghissue:`1263`: Cast Streamline attrs to numpy ints, to avoid buffer mismatch.
* :ghissue:`1257`: revamp piesno docstring
* :ghissue:`978`: Use absolute import in align
* :ghissue:`1179`: Automate workflow generation
* :ghissue:`1253`: Automate script installation for workflows
* :ghissue:`1254`: Automate script installation
* :ghissue:`1261`: removing absolute path in tracking module
* :ghissue:`1001`: Use absolute import in tracking
* :ghissue:`1255`: Fix missing documentation content
* :ghissue:`1260`: removing absolute path in reconst
* :ghissue:`999`: Use absolute import in reconst
* :ghissue:`1258`: Fix nlmeans indexing
* :ghissue:`369`: Add TESTs for resample
* :ghissue:`1155`: csa and csd reconstruction workflow
* :ghissue:`1000`: Use absolute import in segment, testing and tests
* :ghissue:`1070`: [Docs] Examples using deprecated function
* :ghissue:`711`: Update api_changes.rst for interp_rbf
* :ghissue:`321`: Median otsu figures in example don't look good
* :ghissue:`994`: Use absolute import in dipy/core
* :ghissue:`608`: Customize at runtime the number of cores nlmeans is using
* :ghissue:`865`: PEP8 in test_imwarp
* :ghissue:`591`: Allow seed_from_mask to generate random seeds
* :ghissue:`518`: TODO: aniso2iso module will be completely removed in version 0.10.
* :ghissue:`328`: "incompatible" import of peaks_from_model in your recent publication
* :ghissue:`1241`: Csa and csd reconstruction workflow rebased
* :ghissue:`1250`: DOC: Fix reconst_dki.py DKI example documentation typos.
* :ghissue:`1244`: TEST: Decrease precision of tests for dki micro model prediction
* :ghissue:`1235`: New hdf5 file format for saving PeaksAndMetrics objects
* :ghissue:`1231`: TST: Reduce precision requirement for test of tortuosity estimation.
* :ghissue:`1210`: Switching branches in windows and pip install error
* :ghissue:`1209`: Move data files out of dropbox => persistent URL
* :ghissue:`1233`: Feature: Added environment override for dipy_home variable
* :ghissue:`1234`: BUG: Fix non-ASCII characters in reconst_dki.py example.
* :ghissue:`1222`: A lightweight UI for medical visualizations #5: 2D Circular Slider
* :ghissue:`1185`: unable to use fvtk.show after ubuntu 16.10 install
* :ghissue:`1228`: RF: Use cython imports instead of relying on extern
* :ghissue:`909`: Inconsistent output for values_from_volume
* :ghissue:`1182`: CSD vs CSA
* :ghissue:`1211`: `dipy.data.read_bundles_2_subjects` doesn't fetch data as expected
* :ghissue:`1227`: BF: Use np.npy_intp instead of assuming long for ArraySequence attributes
* :ghissue:`1027`: (DO NOT MERGE THIS PR) NF: DKI microstructural model
* :ghissue:`1226`: DKI Microstructural model
* :ghissue:`1229`: RF: allow for scipy pre-release deprecations
* :ghissue:`1225`: Add one more multi b-value data-set
* :ghissue:`1219`: MRG:Data off dropbox
* :ghissue:`1218`: [Docs] Error while generating html
* :ghissue:`1221`: NF: Check multi b-value
* :ghissue:`1212`: Follow PEP8 in reconst (part 2)
* :ghissue:`1217`: Use integer division in reconst_gqi.py
* :ghissue:`1205`: A lightweight UI for medical visualizations #4: 2D Line Slider
* :ghissue:`1166`: RF: Use the average sigma in the mask.
* :ghissue:`1216`: Use integer division to avoid errors in indexing
* :ghissue:`1215`: [Docs] Error while building examples: tracking_quick_start.py
* :ghissue:`1213`: dipy.align.vector_fields.simplify_warp_function_3d: Wrong equation in docstring
* :ghissue:`1214`: DOC: add a clarification note to simplify_warp_funcion_3d
* :ghissue:`1208`: Follow PEP8 in reconst (part 1)
* :ghissue:`1206`: Revert #1204, and add a filter to suppress warnings.
* :ghissue:`1196`: MRG: Use dipy's array comparisons for tests
* :ghissue:`1191`: Test failures for cluster code with current numpy master
* :ghissue:`1207`: Follow PEP8 in reconst
* :ghissue:`1204`: Suppress warnings regarding one-dimensional arrays changes in scipy 0.18
* :ghissue:`1107`: Dipy.align.reslice: either swallow the scipy warning or rework to avoid it
* :ghissue:`1199`: A lightweight UI for medical visualizations #3: Changes to Event Handling
* :ghissue:`1200`: [Docs] Error while generating docs
* :ghissue:`1202`: Use integer division to avoid errors in indexing
* :ghissue:`1188`: Colormap test errors for new matplotlib
* :ghissue:`1187`: Negative integer powers error with numpy 1.12
* :ghissue:`1170`: Importing vtk with dipy
* :ghissue:`1086`: ENH: avoid calling log() on zero-valued elements in anisotropic_power
* :ghissue:`1198`: ENH: avoid log zero
* :ghissue:`1201`: Fix out of bounds point not being classified OUTSIDEIMAGE (binary cla…
* :ghissue:`1115`: Bayesian Tissue Classification
* :ghissue:`1052`: Conda install
* :ghissue:`1183`: A lightweight UI for medical visualizations #2: TextBox
* :ghissue:`1173`: TST: Test on Python 3.6
* :ghissue:`1186`: MRG: use newer wheelhouse for installs
* :ghissue:`1190`: Pickle error for Python 3.6 and test_peaksFromModelParallel
* :ghissue:`1195`: Make PeaksAndMetrics pickle-able
* :ghissue:`1194`: Use assert_arrays_equal when needed.
* :ghissue:`1193`: Deprecate the Accent colormap, in anticipation of changes in MPL 2.0
* :ghissue:`1189`: Np1.12
* :ghissue:`1140`: A lightweight UI for medical visualizations #1: Button
* :ghissue:`1022`: Fixes #720 : Auto generate ipython notebooks
* :ghissue:`1139`: The shebang again! Python: bad interpreter: No such file or directory
* :ghissue:`1171`: fix:dev: added numpy.int64 for my triangle array
* :ghissue:`1123`: Add the mask workflow
* :ghissue:`1174`: NF: added the repulsion 200 sphere.
* :ghissue:`1176`: Dipy.tracking.local.interpolation.nearestneighbor_interpolate raises when used with Numpy 1.12
* :ghissue:`1177`: BF: fix interpolation call with Numpy 1.12
* :ghissue:`1162`: Return S0 value for DTI fits
* :ghissue:`1142`: pytables version and streamlines_format.py example
* :ghissue:`1147`: add this fix for newer version of pytables.
* :ghissue:`1076`: ENH: Add support for ArraySequence in `length` function
* :ghissue:`1050`: ENH: expand OpenMP utilities and move from denspeed.pyx to dipy.utils
* :ghissue:`1082`: Add documentation uploading script
* :ghissue:`1153`: Athena mapmri
* :ghissue:`1097`: Added to quantize_evecs: multiprocessing and v
* :ghissue:`1159`: TST - add tests for various affine matrices for local tracking
* :ghissue:`1163`: WIP: Combined contour function with slicer to use affine
* :ghissue:`940`: Drop python 2.6
* :ghissue:`1040`: SFM example using deprecated code
* :ghissue:`1118`: pip install dipy failing on my windows
* :ghissue:`1119`: Buildbots failing with workflow merge
* :ghissue:`1127`: Windows buildbot failures after ivim_linear merge
* :ghissue:`1128`: Support for non linear denoise?
* :ghissue:`1138`: A few broken builds
* :ghissue:`1148`: Actual S0 for DTI data
* :ghissue:`1157`: Replace `get_affine` with `affine` and `get_header` with `header`.
* :ghissue:`1160`: Add Shahnawaz to list of contributors.
* :ghissue:`740`: Improved mapmri implementation with laplacian regularization and new …
* :ghissue:`1045`: Allow affine 'shear' tolerance in LocalTracking
* :ghissue:`1154`: [Bug] connectivity matrix image in streamline_tools example
* :ghissue:`1158`: BF: closing matplotlib plots for each file while running the examples
* :ghissue:`1151`: Define fmin() for Visual Studio
* :ghissue:`1145`: DKI_signal should be dki_signal in dipy.sims.voxel
* :ghissue:`1149`: Change DKI_signal to dki_signal
* :ghissue:`1137`: Small fix to insure that fwDTI non-linear procedure does not crash
* :ghissue:`827`: Free Water Elimination DTI
* :ghissue:`942`: NF: Added support to colorize each line points individually
* :ghissue:`1141`: Do not cover files related to benchmarks.
* :ghissue:`1098`: Adding custom interactor for visualisation
* :ghissue:`1136`: Update deprecated function.
* :ghissue:`1113`: TST: Test for invariance of model_params to splitting of the data.
* :ghissue:`1134`: Rebase of https://github.com/dipy/dipy/pull/993
* :ghissue:`1064`: Faster dti odf
* :ghissue:`1114`: flexible grid to streamline affine generation and pathlength function
* :ghissue:`1122`: Add the reconst_dti workflow
* :ghissue:`1132`: Update .travis.yml and README.md
* :ghissue:`1051`: ENH: use parallel processing in the cython code for CCMetric
* :ghissue:`993`: FIX: Use absolute imports in testing,tests and segment files
* :ghissue:`673`: WIP: Workflow for syn registration
* :ghissue:`859`: [WIP] Suppress warnings in tests
* :ghissue:`983`: PEP8 in sims #884
* :ghissue:`984`: PEP8 in reconst #881
* :ghissue:`1009`: Absolute Imports in Tracking
* :ghissue:`1036`: Estimate S0 from data (DTI)
* :ghissue:`1125`: Intensity adjustment. Find a better upper bound for interpolating images.
* :ghissue:`1130`: Minor corrections for showing surfaces
* :ghissue:`1092`: Line-based target()
* :ghissue:`1129`: Fix 1127
* :ghissue:`1034`: Viz surfaces
* :ghissue:`394`: Update documentation for VTK and Anaconda
* :ghissue:`973`: Minor change in `tools/github_stats.py`
* :ghissue:`1060`: Fast computation of Cross Correlation metric
* :ghissue:`1124`: Small fix in free water DTI model
* :ghissue:`1058`: IVIM
* :ghissue:`1110`: WIP : Ivim linear fitting
* :ghissue:`1120`: Fix 1119
* :ghissue:`1121`: Recons dti workflow
* :ghissue:`1083`: nlmeans problem
* :ghissue:`1075`: Drop26
* :ghissue:`835`: NF: Free water tensor model
* :ghissue:`1046`: BF - peaks_from_model with nbr_processes <= 0
* :ghissue:`1049`: MAINT: minor cython cleanup in align/vector_fields.pyx
* :ghissue:`1087`: Base workflow enhancements + tests
* :ghissue:`1112`: DOC: Math rendering issue in SFM gallery example.
* :ghissue:`670`: Tissue classification using MAP-MRF
* :ghissue:`332`: A sample nipype interface for fit_tensor
* :ghissue:`1116`: failing to build the docs: issue with io.BufferedIOBase
* :ghissue:`1109`: Changed default value of mni template
* :ghissue:`1106`: Including MNI Template 2009c in Fetcher
* :ghissue:`1066`: Adaptive Denoising
* :ghissue:`351`: Dipy.tracking.utils.target affine parameter is misleading
* :ghissue:`1091`: Modifications for building docs with python3
* :ghissue:`912`: Unable to build documentation with Python 3
* :ghissue:`1105`: Import reload function from imp module explicitly for python3
* :ghissue:`1104`: restore_dti.py example does not work in python3
* :ghissue:`1102`: MRG: add pytables to travis-ci, Py35 full test run
* :ghissue:`1100`: Fix for Python 3 in io.dpy
* :ghissue:`1103`: BF: This raises a warning on line 367 otherwise.
* :ghissue:`1101`: Test with optional dependencies (including pytables) on Python 3.
* :ghissue:`1094`: Updates to FBC measures documentation
* :ghissue:`1059`: Documentation to discourage misuse of GradientTable
* :ghissue:`1061`: Inconsistency in specifying S0 values in multi_tensor and single_tensor
* :ghissue:`1063`: Fixes #1061 : Changed all S0 to 1.0
* :ghissue:`1089`: BF: fix test error on Python 3
* :ghissue:`1079`: Return a generator from `orient_by_roi`
* :ghissue:`1088`: Restored the older implementation of nlmeans
* :ghissue:`1080`: DOC: TensorModel.__init__ docstring.
* :ghissue:`1085`: Enhanced workflows
* :ghissue:`1081`: mean_diffusivity from the reconst.dti module returns incorrect shape
* :ghissue:`1031`: improvements for denoise/denspeed.pyx
* :ghissue:`828`: Fiber to bundle coherence measures
* :ghissue:`1072`: DOC: Added a coverage badge to README.rst
* :ghissue:`1071`: report coverage and add a badge?
* :ghissue:`1038`: BF: Should fix #1037
* :ghissue:`1078`: Fetcher for ivim data, needs md5
* :ghissue:`953`: FIX: PEP8 for segment
* :ghissue:`1025`: PEP8: Fix pep8 in segment
* :ghissue:`883`: PEP8 in segment
* :ghissue:`1077`: DOC: update fibernavigator link
* :ghissue:`1069`: DOC: Small one -- we need this additional line of white space to render.
* :ghissue:`1068`: Renamed test_shore for consistency
* :ghissue:`1067`: Generate b vectors using disperse_charges
* :ghissue:`1011`: Discrepancy with output of nlmeans.py
* :ghissue:`1055`: WIP: Ivim implementation
* :ghissue:`1065`: improve OMP parallelization with scheduling
* :ghissue:`1062`: BF - fix CSD.predict to work with nd inputs.
* :ghissue:`1057`: Workaround for https://github.com/dipy/dipy/issues/852
* :ghissue:`1037`: tracking.interfaces imports SlowAdcOpdfModel, but it is not defined
* :ghissue:`1056`: Remove tracking interfaces
* :ghissue:`813`: Windows 64-bit error in segment.featurespeed.extract
* :ghissue:`1054`: Remove tracking interfaces
* :ghissue:`1028`: BF: Predict DKI with a volume of S0
* :ghissue:`1041`: NF - Add PMF Threshold to Tractography
* :ghissue:`1039`: Doc - fix definition of real_sph_harm functions
* :ghissue:`1019`: MRG: fix heavy dependency check; no numpy for setup
* :ghissue:`1018`: Fix: denspeed.pyx to give correct output for nlmeans
* :ghissue:`1043`: DO NOT MERGE: Add a test of local tracking, using data from dipy.data.
* :ghissue:`899`: SNR in contextual enhancement example
* :ghissue:`991`: Documentation footer has 2008-2015 mentioned.
* :ghissue:`1008`: [WIP] NF: Implementation of CHARMED model
* :ghissue:`1030`: Fetcher files not found on Windows
* :ghissue:`1035`: Fix for fetcher files in Windows
* :ghissue:`1016`: viz.fvtk has no attribute 'ren'
* :ghissue:`1033`: Viz surfaces
* :ghissue:`1032`: Merge pull request #1 from nipy/master
* :ghissue:`1029`: Errors building Cython extensions on Python 3.5
* :ghissue:`974`: Minor change in `tools/github_stats.py`
* :ghissue:`1002`: Use absolute import in utils and scratch
* :ghissue:`1014`: Progress bar works only for some data
* :ghissue:`1013`: `dipy.data.make_fetcher` test fails with Python 3
* :ghissue:`1020`: Documentation does not mention Scipy as a dependency for VTK widgets.
* :ghissue:`1023`: display in dsi example is broken
* :ghissue:`1021`: Added warning for VTK not installed
* :ghissue:`882`: PEP8 in reconst tests/benchmarks
* :ghissue:`888`: PEP8 in tracking benchmarks/tests
* :ghissue:`885`: PEP8 in testing
* :ghissue:`902`: fix typo
* :ghissue:`1024`: Documentation fix for reconst_dsid.py
* :ghissue:`979`: No figures in DKI example
* :ghissue:`981`: Fixes #979 : No figures in DKI example - Add new line after figure
* :ghissue:`958`: FIX: PEP8 in testing
* :ghissue:`1005`: FIX: Use absolute imports in io
* :ghissue:`997`: Use absolute import in io
* :ghissue:`675`: Voxelwise stabilisation
* :ghissue:`951`: Contextual Enhancement update: fix SNR issue, fix reference
* :ghissue:`1015`: Fix progressbar of fetcher
* :ghissue:`1012`: TST: install the dipy.data tests.
* :ghissue:`992`: FIX: Update the import statements to use absolute import in core
* :ghissue:`1003`: FIX: Change the import statements in direction
* :ghissue:`996`: Use absolute import in dipy/direction
* :ghissue:`1004`: FIX: Use absolute import in pkg_info
* :ghissue:`998`: Use absolute import in dipy/pkg_info.py
* :ghissue:`1006`: FIX: Use absolute import in utils and scratch
* :ghissue:`1010`: Absolute Imports in Viz
* :ghissue:`1007`: Use absolute import in viz
* :ghissue:`929`: Fix PEP8 in data
* :ghissue:`874`: PEP8 in data
* :ghissue:`980`: Fix pep8 in reconst
* :ghissue:`1017`: Fixes #1016 : Raises VTK not installed
* :ghissue:`877`: PEP8 in external
* :ghissue:`887`: PEP8 in tracking
* :ghissue:`941`: BW: skimage.filter module name warning
* :ghissue:`976`: Fix PEP8 in sims and remove unnecessary imports
* :ghissue:`884`: PEP8 in sims
* :ghissue:`956`: FIX: PEP8 in reconst/test and reconst/benchmarks
* :ghissue:`955`: FIX: PEP8 in external
* :ghissue:`952`: FIX: PEP8 in tracking and tracking benchmarks/tests
* :ghissue:`982`: FIX: relative imports in dipy/align
* :ghissue:`972`: Fixes #901 : Added documentation for "step" in dti
* :ghissue:`901`: DTI `step` parameter not documented.
* :ghissue:`995`: Use absolute import in dipy/data/__init__.py
* :ghissue:`344`: Update citation page
* :ghissue:`971`: Add progress bar feature to dipy.data.fetcher
* :ghissue:`970`: Downloading data with dipy.data.fetcher does not show any progress bar
* :ghissue:`986`: "pip3 install dipy" in Installation for python3
* :ghissue:`990`: No figures in DKI example
* :ghissue:`989`: copyright 2008-2016
* :ghissue:`988`: doc/conf.py shows copyright 2008-2015. Should be 2016?
* :ghissue:`975`: Use absolute import in imaffine, imwarp, metrics
* :ghissue:`517`: TODO: Peaks to be removed from dipy.reconst in 0.10
* :ghissue:`977`: Relative import fix in dipy/align
* :ghissue:`875`: PEP8 in denoise
* :ghissue:`957`: FIX: PEP8 in denoise
* :ghissue:`960`: PEP8 in sims #884
* :ghissue:`961`: PEP8 in reconst #880
* :ghissue:`962`: PEP8 in reconst #881
* :ghissue:`889`: PEP8 in utils
* :ghissue:`959`: FIX: PEP8 in utils
* :ghissue:`866`: PEP8 in test_metrics
* :ghissue:`867`: PEP8 in test_parzenhist
* :ghissue:`868`: PEP8 in test_scalespace
* :ghissue:`869`: PEP8 in test_sumsqdiff
* :ghissue:`870`: PEP8 in test_transforms
* :ghissue:`871`: PEP8 in test_vector_fields
* :ghissue:`864`: PEP8 in `test_imaffine`
* :ghissue:`967`: Update index.rst correcting the date of release 0.11
* :ghissue:`862`: PEP8 in `test_crosscorr`
* :ghissue:`873`: PEP8 in core
* :ghissue:`831`: ACT tracking example gives weird streamlines
* :ghissue:`876`: PEP8 in direction
* :ghissue:`954`: FIX: PEP8 in direction
* :ghissue:`965`: Fix typo
* :ghissue:`968`: Use relative instead of absolute import
* :ghissue:`948`: Fix PEP8 in boots
* :ghissue:`872`: PEP8 in boots
* :ghissue:`946`: FIX: PEP8 for  test_sumsqdiff and test_scalespace
* :ghissue:`964`: FIX: PEP8 in test_imaffine
* :ghissue:`963`: FIX: PEP8 in core
* :ghissue:`966`: fix typo
* :ghissue:`947`: FIX: PEP8 for test files
* :ghissue:`920`: STYLE:PEP8 for test_imaffine
* :ghissue:`897`: PEP8
* :ghissue:`950`: PEP8 fixed in reconst/tests and reconst/benchmarks
* :ghissue:`949`: Fixed Pep8 utils tracking testing denoise
* :ghissue:`926`: Fix PEP8 in fixes
* :ghissue:`878`: PEP8 in fixes
* :ghissue:`939`: Fixed PEP8 in utils, denoise , tracking and testing
* :ghissue:`945`: FIX: PEP8 in test_scalespace
* :ghissue:`937`: BF : Clamping of the value of v in winding function
* :ghissue:`930`: pep8 fix issue  #896 - "continuation line over-indented for visual indent"
* :ghissue:`943`: BF: Removed unused code in slicer
* :ghissue:`907`: DOC: switch to using mathjax for maths
* :ghissue:`931`: dipy/tracking/streamlinespeed set_number_of_points crash when nb_points=0
* :ghissue:`932`: Fixes #931 : checks if nb_points=0
* :ghissue:`927`: Fix PEP8 in io and remove duplicate definition in test_bvectxt.py
* :ghissue:`924`: in dipy/io/tests/test_bvectxt.py function with same name is defined twice
* :ghissue:`879`: PEP8 in io
* :ghissue:`913`: Fix pep8 in workflows
* :ghissue:`891`: PEP8 in workflows
* :ghissue:`938`: PEP8 issues solved in utils, testing and denoise
* :ghissue:`935`: Setup: go on to version 0.12 development.
* :ghissue:`934`: DOC: Update github stats for 0.11 as of today.
* :ghissue:`933`: Updating release dates
