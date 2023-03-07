
.. _release1.5:

===================================
 Release notes for DIPY version 1.5
===================================

GitHub stats for 2021/05/06 - 2022/03/10 (tag: 1.4.1)

These lists are automatically generated, and may be incomplete or contain duplicates.

The following 22 authors contributed 573 commits.

* Ariel Rokem
* Dan Bullock
* David Romero-Bascones
* Derek Pisner
* Eleftherios Garyfallidis
* Eric Larson
* Francis Jerome
* Francois Rheault
* Gabriel Girard
* Giulia Bertò
* Javier Guaje
* Jon Haitz Legarreta Gorroño
* Joshua Newton
* Kenji Marshall
* Leevi Kerkela
* Leon Weninger
* Lucas Da Costa
* Nasim Anousheh
* Rafael Neto Henriques
* Sam Coveney
* Serge Koudoro
* Shreyas Fadnavis


We closed a total of 200 issues, 72 pull requests and 128 regular issues;
this is the full list (generated with the script
:file:`tools/github_stats.py`):

Pull Requests (72):

* :ghpull:`2561`: [FIX] Motion correction tutorial
* :ghpull:`2520`: Resdnn inference
* :ghpull:`2558`: BUG: Fix errant warning about starting_affine
* :ghpull:`2557`: MAINT: Fix version
* :ghpull:`2556`: [FIX] Update `dipy.segment` tutorials
* :ghpull:`2554`: Support .vtp files
* :ghpull:`2555`: Limit `peaks_from_model` number of processes in examples
* :ghpull:`2539`: Adds utilities for embarrassingly parallel loops.
* :ghpull:`2545`: Stateful Tractogram DPS and DPP keys ordering
* :ghpull:`2548`: Add timeout + concurrency to GHA
* :ghpull:`2549`: [ENH] Clarify reconst_sh tutorial
* :ghpull:`2550`: [ENH] Add sigma to DTI/DKI RESTORE workflow
* :ghpull:`2551`: [MNT] Update minimal dependencies version
* :ghpull:`2536`: Random colors fix in horizon
* :ghpull:`2533`: [FIX] Docstring cleaning: wrong underline length...
* :ghpull:`2342`: NF:  q-space trajectory imaging
* :ghpull:`2512`: Masking for affine registration
* :ghpull:`2526`: TEST: Filter legacy SH bases warnings in tests
* :ghpull:`2534`: TEST: Remove unnecessary `main` method definition in tests
* :ghpull:`2532`: STYLE: Remove unused import statements
* :ghpull:`2529`: STYLE: Remove unused import statements
* :ghpull:`2528`: TEST: Remove legacy `nose`-related dead testing code
* :ghpull:`2527`: TEST: Fix intermittent RUMBA test check failure
* :ghpull:`2493`: Fury dependency resolution
* :ghpull:`2522`: ENH: Miscellaneous cleanup
* :ghpull:`2521`: DOC: Use GitHub actions status badge in README
* :ghpull:`2420`: Documentation corrections
* :ghpull:`2482`: ENH: Improve SH bases warning messages
* :ghpull:`2423`: NF: rumba reconst
* :ghpull:`2518`: Migrations from Azure Pipeline to Github Actions
* :ghpull:`2515`: Default to False output for null streamlines in streamline_near_roi
* :ghpull:`2513`: [MNT] Drop distutils
* :ghpull:`2506`: Horizon FURY update
* :ghpull:`2510`: Optimize sfm (reboot)
* :ghpull:`2487`: ENH: Better error message
* :ghpull:`2442`: [NF] Add Motion correction workflow
* :ghpull:`2470`: Add utilities functions:  radius curvature <--> maximum deviation angle
* :ghpull:`2485`: DOC: Small updates.
* :ghpull:`2481`: ENH: Import ABCs from `collections.abc`
* :ghpull:`2480`: STYLE: Make `sklearn` import warning messages consistent
* :ghpull:`2478`: ENH: Deal appropriately with user warnings
* :ghpull:`2479`: STYLE: Improve style in misc files
* :ghpull:`2475`: ENH: Fix `complex` type `NumPy` alias deprecation warnings
* :ghpull:`2476`: ENH: Fix `dipy.io.bvectxt` deprecation warning
* :ghpull:`2472`: ENH: Return unique invalid streamline removal indices
* :ghpull:`2471`: DOC: Fix coding style guideline link
* :ghpull:`2468`: [MNT] Use windows-latest on azure pipeline
* :ghpull:`2467`: [ENH] Add fit_method option in DTI and DKI CLI
* :ghpull:`2466`: deprecate dipy.io.bvectxt module
* :ghpull:`2453`: make it compatible when number of volume is 2
* :ghpull:`2413`: Azure pipeline: from ubuntu 1604 to 2004
* :ghpull:`2447`: reduce_rois: Force input array type to bool to avoid bitwise or errors
* :ghpull:`2444`: [DOC] : Added citation for IVIM dataset
* :ghpull:`2434`: MAINT: Update import from ndimage
* :ghpull:`2435`: BUG: Backward compat support for pipeline
* :ghpull:`2436`: MAINT: Bump tolerance
* :ghpull:`2438`: BUG: Fix misplaced comma in `warn()` call from `patch2self.py`
* :ghpull:`2374`: ROIs visualizer
* :ghpull:`2390`: NF: extend the align workflow with Rigid+IsoScaling and Rigid+Scaling
* :ghpull:`2417`: OPT: Initialize `Shape` struct
* :ghpull:`2419`: Fixes the default option in the command line for Patch2Self 'ridge' -> 'ols'
* :ghpull:`2406`: Manage Approx_polygon_track with repeated points
* :ghpull:`2411`: [FIX] `c_compress_streamline` discard identical points
* :ghpull:`2416`: OPT: Prefer using a typed index to get the PMF value
* :ghpull:`2415`: Implementation multi_voxel_fit progress bar
* :ghpull:`2410`: [ENH] Improve Shore Tests
* :ghpull:`2409`: NF - Sample PMF for an input position and direction
* :ghpull:`2405`: Small correction on KFA
* :ghpull:`2407`: from random to deterministic test for deform_streamlines
* :ghpull:`2392`: Add decomposition
* :ghpull:`2389`: [Fix] bundles_distances_mdf asymmetric values
* :ghpull:`2368`: RF - Moved tracking.localtrack._local_tracker to DirectionGetter.generate_streamline.

Issues (128):

* :ghissue:`2561`: [FIX] Motion correction tutorial
* :ghissue:`2123`: WIP: Residual Deep NN
* :ghissue:`2520`: Resdnn inference
* :ghissue:`2558`: BUG: Fix errant warning about starting_affine
* :ghissue:`2557`: MAINT: Fix version
* :ghissue:`2489`: MAINT: Get Python 3.10 binaries up on scipy-wheels-nightly
* :ghissue:`2556`: [FIX] Update `dipy.segment` tutorials
* :ghissue:`2554`: Support .vtp files
* :ghissue:`2525`: Support Opening `.vtp` files
* :ghissue:`2555`: Limit `peaks_from_model` number of processes in examples
* :ghissue:`2539`: Adds utilities for embarrassingly parallel loops.
* :ghissue:`2509`: Easy robustness for streamline_near_roi and near_roi for empty streamlines?
* :ghissue:`2543`: StatefulTractogram.are_compatible compare data_per_point keys as list instead of set
* :ghissue:`2545`: Stateful Tractogram DPS and DPP keys ordering
* :ghissue:`2548`: Add timeout + concurrency to GHA
* :ghissue:`2549`: [ENH] Clarify reconst_sh tutorial
* :ghissue:`2546`: Confusing import in 'reconst_sh`
* :ghissue:`2550`: [ENH] Add sigma to DTI/DKI RESTORE workflow
* :ghissue:`2542`: DTI workflow should allow user-defined fitting method
* :ghissue:`2551`: [MNT] Update minimal dependencies version
* :ghissue:`2477`: Numpy min dependency update
* :ghissue:`2541`: Issue with coverage and pytests for numpy.min()
* :ghissue:`2507`: kernel died when use dipy.viz
* :ghissue:`2536`: Random colors fix in horizon
* :ghissue:`2533`: [FIX] Docstring cleaning: wrong underline length...
* :ghissue:`2422`: WIP-Adding math in SLR tutorial
* :ghissue:`2342`: NF:  q-space trajectory imaging
* :ghissue:`2512`: Masking for affine registration
* :ghissue:`1969`: imaffine mask support
* :ghissue:`2526`: TEST: Filter legacy SH bases warnings in tests
* :ghissue:`2456`: Horizon tests failing
* :ghissue:`2534`: TEST: Remove unnecessary `main` method definition in tests
* :ghissue:`2532`: STYLE: Remove unused import statements
* :ghissue:`2524`: Add concurrency + timeout to Github Actions (GHA)
* :ghissue:`2529`: STYLE: Remove unused import statements
* :ghissue:`2528`: TEST: Remove legacy `nose`-related dead testing code
* :ghissue:`2527`: TEST: Fix intermittent RUMBA test check failure
* :ghissue:`2493`: Fury dependency resolution
* :ghissue:`2522`: ENH: Miscellaneous cleanup
* :ghissue:`2521`: DOC: Use GitHub actions status badge in README
* :ghissue:`2420`: Documentation corrections
* :ghissue:`2482`: ENH: Improve SH bases warning messages
* :ghissue:`2449`: Nonsense deprecation warning
* :ghissue:`2423`: NF: rumba reconst
* :ghissue:`2179`: NF: Complete masking implementation in affine registration with MI
* :ghissue:`2518`: Migrations from Azure Pipeline to Github Actions
* :ghissue:`2492`: Move to GitHub actions / reusable actions
* :ghissue:`2515`: Default to False output for null streamlines in streamline_near_roi
* :ghissue:`2497`: Remove python 3.6 from Azure pipelines
* :ghissue:`2495`: Remove Distutils (deprecated)
* :ghissue:`2513`: [MNT] Drop distutils
* :ghissue:`2506`: Horizon FURY update
* :ghissue:`2305`: [WIP] Brain Tumor Image Segmentation Code
* :ghissue:`2499`: Problem generating Connectivity Matrix: "Slice step cannot be zero"
* :ghissue:`2510`: Optimize sfm (reboot)
* :ghissue:`2488`: Minimize memory footprint wherever possible, add joblib support for …
* :ghissue:`2504`: Why are there many small dots on the fwdwi image?
* :ghissue:`2502`: Can i read specific b-values from my own multishell data?
* :ghissue:`2500`: MAP issue
* :ghissue:`2490`: [BUG] MRI-CT alignment failure
* :ghissue:`2487`: ENH: Better error message
* :ghissue:`2402`: Dipy 1.4.1 breaks nipype.interfaces.dipy.dipy_to_nipype_interface
* :ghissue:`2486`: Wrong doc in interpolation
* :ghissue:`2442`: [NF] Add Motion correction workflow
* :ghissue:`2470`: Add utilities functions:  radius curvature <--> maximum deviation angle
* :ghissue:`2485`: DOC: Small updates.
* :ghissue:`2484`: [ENH] Add grid search to `AffineRegistration.optimize`
* :ghissue:`2483`: [DOC] Stable/Latest Documentation Structure
* :ghissue:`2481`: ENH: Import ABCs from `collections.abc`
* :ghissue:`2480`: STYLE: Make `sklearn` import warning messages consistent
* :ghissue:`2478`: ENH: Deal appropriately with user warnings
* :ghissue:`2479`: STYLE: Improve style in misc files
* :ghissue:`2475`: ENH: Fix `complex` type `NumPy` alias deprecation warnings
* :ghissue:`2476`: ENH: Fix `dipy.io.bvectxt` deprecation warning
* :ghissue:`2472`: ENH: Return unique invalid streamline removal indices
* :ghissue:`2471`: DOC: Fix coding style guideline link
* :ghissue:`2468`: [MNT] Use windows-latest on azure pipeline
* :ghissue:`2467`: [ENH] Add fit_method option in DTI and DKI CLI
* :ghissue:`2463`: DTI RESTORE on the CLI
* :ghissue:`2466`: deprecate dipy.io.bvectxt module
* :ghissue:`2460`: Deprecate and Remove dipy.io.bvectxt
* :ghissue:`2429`: random_colors flag in dipy_horizon does not work as before
* :ghissue:`2461`: Patch2Self: Less than 10 3D Volumes Bug
* :ghissue:`2464`: Typo on the homepage
* :ghissue:`2453`: make it compatible when number of volume is 2
* :ghissue:`2457`: Choosing sigma_diff and radius parameters for SyN registration
* :ghissue:`2413`: Azure pipeline: from ubuntu 1604 to 2004
* :ghissue:`2454`: Can I show fiber with vtk?
* :ghissue:`2446`: Use of bitwise or with non-bool inputs results in ufunc 'bitwise_or' error
* :ghissue:`2447`: reduce_rois: Force input array type to bool to avoid bitwise or errors
* :ghissue:`2444`: [DOC] : Added citation for IVIM dataset
* :ghissue:`2443`: Citation for IVIM dataset not present in docs
* :ghissue:`2434`: MAINT: Update import from ndimage
* :ghissue:`2441`: Horizon error - disk position outside the slider line
* :ghissue:`2435`: BUG: Backward compat support for pipeline
* :ghissue:`2436`: MAINT: Bump tolerance
* :ghissue:`2438`: BUG: Fix misplaced comma in `warn()` call from `patch2self.py`
* :ghissue:`2430`: dipy.align.reslice
* :ghissue:`2431`: dipy.align.reslice interpolation order for downsampling
* :ghissue:`2432`: How to apply MI metric in dipy？
* :ghissue:`2374`: ROIs visualizer
* :ghissue:`2390`: NF: extend the align workflow with Rigid+IsoScaling and Rigid+Scaling
* :ghissue:`2417`: OPT: Initialize `Shape` struct
* :ghissue:`2419`: Fixes the default option in the command line for Patch2Self 'ridge' -> 'ols'
* :ghissue:`2406`: Manage Approx_polygon_track with repeated points
* :ghissue:`2314`: Approx_polygon_track with repeated points gives an error
* :ghissue:`2411`: [FIX] `c_compress_streamline` discard identical points
* :ghissue:`1805`: `c_compress_streamline` keeps identical points when it shouldn't
* :ghissue:`2418`: kernel failure when importing mask from dipy.segment
* :ghissue:`2416`: OPT: Prefer using a typed index to get the PMF value
* :ghissue:`2415`: Implementation multi_voxel_fit progress bar
* :ghissue:`2410`: [ENH] Improve Shore Tests
* :ghissue:`365`: Code review items for `dipy.reconst.shore`
* :ghissue:`2409`: NF - Sample PMF for an input position and direction
* :ghissue:`2404`: Change affine in StatefulTractogram
* :ghissue:`2405`: Small correction on KFA
* :ghissue:`2407`: from random to deterministic test for deform_streamlines
* :ghissue:`2392`: Add decomposition
* :ghissue:`717`: Download each shell of the CENIR data separately?
* :ghissue:`2209`: _pytest.pathlib.ImportPathMismatchError:
* :ghissue:`1934`: Random lpca denoise
* :ghissue:`2312`: DIPY open group meetings, Spring 2021
* :ghissue:`2383`: error in mcsd model fitting (DCPError)
* :ghissue:`2391`: error performing cross-validation on diffusion HCP data
* :ghissue:`2393`: Add a function to read streamline from the result generated by the command "probtrackx2" in FMRIB's Diffusion Toolbox
* :ghissue:`2389`: [Fix] bundles_distances_mdf asymmetric values
* :ghissue:`2310`: `bundles_distances_mdf` asymmetric values
* :ghissue:`2368`: RF - Moved tracking.localtrack._local_tracker to DirectionGetter.generate_streamline.
