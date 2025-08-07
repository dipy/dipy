.. _release0.10:

====================================
 Release notes for DIPY version 0.10
====================================

GitHub stats for 2015/03/18 - 2015/11/19 (tag: 0.9.2)

These lists are automatically generated, and may be incomplete or contain duplicates.

The following 20 authors (alphabetically ordered) contributed 1022 commits:

* Alexandre Gauvin
* Ariel Rokem
* Bago Amirbekian
* David Qixiang Chen
* Dimitris Rozakis
* Eleftherios Garyfallidis
* Gabriel Girard
* Gonzalo Sanguinetti
* Jean-Christophe Houde
* Marc-Alexandre Côté
* Matthew Brett
* Mauro Zucchelli
* Maxime Descoteaux
* Michael Paquette
* Omar Ocegueda
* Oscar Esteban
* Rafael Neto Henriques
* Rohan Prinja
* Samuel St-Jean
* Stefan van der Walt


We closed a total of 232 issues, 94 pull requests and 138 regular issues;
this is the full list (generated with the script
:file:`tools/github_stats.py`):

Pull Requests (94):

* :ghpull:`769`: RF: Remove aniso2iso altogether.
* :ghpull:`772`: DOC: Use xvfb when building the docs in a headless machine.
* :ghpull:`754`: DOC: Should we add a side-car gitter chat to the website?
* :ghpull:`753`: TST: Test DSI with b0s.
* :ghpull:`767`: Offscreen is False for test_slicer
* :ghpull:`768`: Document dipy.reconst.dti.iter_fit_tensor params
* :ghpull:`766`: Add fit_tensor iteration decorator
* :ghpull:`751`: Reorient tracks according to ROI
* :ghpull:`765`: BF: Typo in data file name.
* :ghpull:`757`: Optimize dipy.align.reslice
* :ghpull:`587`: Fvtk 2.0 PR1
* :ghpull:`749`: Fixed deprecation warning in skimage
* :ghpull:`748`: TST: added test for _to_voxel_tolerance.
* :ghpull:`678`: BF: added tolerance for negative streamline coordinates checks
* :ghpull:`714`: RF: use masks in predictions and cross-validation
* :ghpull:`739`: Set number of OpenMP threads during runtime
* :ghpull:`733`: Add RTOP, RTAP and RTPP and the relative test
* :ghpull:`743`: BF: memleaks with typed memory views in Cython
* :ghpull:`724`: @sinkpoint's power map - refactored
* :ghpull:`741`: ENH: it is preferable to use choice rather than randint to not have
* :ghpull:`727`: Optimize tensor fitting
* :ghpull:`726`: NF - CSD response from a mask
* :ghpull:`729`: BF: tensor predict
* :ghpull:`736`: Added installation of python-tk package for VTK travis bot
* :ghpull:`735`: Added comment on nlmeans example about selecting one volume
* :ghpull:`732`: WIP: Test with vtk on Travis
* :ghpull:`731`: Np 1.10
* :ghpull:`640`: MAPMRI
* :ghpull:`682`: Created list of examples for available features and metrics
* :ghpull:`716`: Refactor data module
* :ghpull:`699`: Added gaussian noise option to estimate_sigma
* :ghpull:`712`: DOC: API changes in gh707.
* :ghpull:`713`: RF: In case a user just wants to use a single integer.
* :ghpull:`700`: TEST: add tests for AffineMap
* :ghpull:`677`: DKI PR3 - NF: Adding standard kurtosis statistics on module dki.py
* :ghpull:`721`: TST: Verify that output of estimate_sigma is a proper input to nlmeans.
* :ghpull:`572`: NF : nlmeans now support arrays of noise std
* :ghpull:`708`: Check for bval dimensionality on read.
* :ghpull:`707`: BF: Keep up with changes in scipy 0.16
* :ghpull:`709`: DOC: Use the `identity` variable in the resampling transformation.
* :ghpull:`703`: Fix syn-3d example
* :ghpull:`705`: Fix example in function compress_streamline
* :ghpull:`635`: Select streamlines based on logical operations on ROIs
* :ghpull:`702`: BF: Use only validated examples when building docs.
* :ghpull:`689`: Streamlines compression
* :ghpull:`698`: DOC: added NI citation
* :ghpull:`681`: RF + DOC: Add MNI template reference. Also import it into the dipy.da…
* :ghpull:`696`: Change title of piesno example
* :ghpull:`691`: CENIR 'HCP-like' multi b-value data
* :ghpull:`661`: Test DTI eigenvectors
* :ghpull:`690`: BF: nan entries cause segfault
* :ghpull:`667`: DOC: Remove Sourceforge related makefile things. Add the gh-pages upl…
* :ghpull:`676`: TST: update Travis config to use container infrastructure.
* :ghpull:`533`: MRG: some Cython refactorings
* :ghpull:`686`: BF: Make buildbot Pyhon26-32 happy
* :ghpull:`683`: Fixed initial estimation in piesno
* :ghpull:`654`: Affine registration PR 3/3
* :ghpull:`684`: BF: Fixed memory leak in QuickBundles.
* :ghpull:`674`: NF: Function to sample perpendicular directions relative to a given vector
* :ghpull:`679`: BF + NF: Provide dipy version info when running dipy.get_info()
* :ghpull:`680`: NF: Fetch and Read the MNI T1 and/or T2 template.
* :ghpull:`664`: DKI fitting (DKI PR2)
* :ghpull:`671`: DOC: move mailing list links to neuroimaging
* :ghpull:`663`: changed samuel st-jean email to the usherbrooke one
* :ghpull:`648`: Improve check of collinearity in vec2vec_rotmat
* :ghpull:`582`: DKI project: PR#1 Simulations to test DKI
* :ghpull:`660`: BF: If scalar-color input has len(shape)<4, need to fill that in.
* :ghpull:`612`: BF: Differences in spherical harmonic calculations wrt scipy 0.15
* :ghpull:`651`: Added estimate_sigma bias correction + update example
* :ghpull:`659`: BF: If n_frames is larger than one use path-numbering.
* :ghpull:`658`: FIX: resaved npy file causing load error for py 33
* :ghpull:`657`: Fix compilation error caused by inline functions
* :ghpull:`628`: Affine registration PR 2/3
* :ghpull:`629`: Quickbundles 2.1
* :ghpull:`637`: DOC: Fix typo in docstring of Identity class.
* :ghpull:`639`: DOC: Render the following line in the code cell.
* :ghpull:`614`: Seeds from mask random
* :ghpull:`633`: BF - no import of TissueTypes
* :ghpull:`632`: fixed typo in dti example
* :ghpull:`627`: BF: Add missing opacity property to point actor
* :ghpull:`626`: Use LooseVersion to check for scipy versions
* :ghpull:`625`: DOC: Include the PIESNO example.
* :ghpull:`624`: DOC: Corrected typos in Restore tutorial and docstring.
* :ghpull:`619`: DOC: Added missing contributor to developer list
* :ghpull:`618`: Update README file
* :ghpull:`616`: Raise ValueError when invalid matrix is given
* :ghpull:`576`: Piesno example
* :ghpull:`615`: bugfix for double word in doc example issue #387
* :ghpull:`610`: Added figure with HBM 2015
* :ghpull:`609`: Update website documentation
* :ghpull:`607`: DOC: Detailed github stats for 0.9
* :ghpull:`606`: Removed the word new
* :ghpull:`605`: Release mode: updating Changelog and Authors
* :ghpull:`594`: DOC + PEP8: Mostly just line-wrapping.

Issues (138):

* :ghissue:`769`: RF: Remove aniso2iso altogether.
* :ghissue:`772`: DOC: Use xvfb when building the docs in a headless machine.
* :ghissue:`754`: DOC: Should we add a side-car gitter chat to the website?
* :ghissue:`771`: Should we remove the deprecated quickbundles module?
* :ghissue:`753`: TST: Test DSI with b0s.
* :ghissue:`761`: reading dconn.nii
* :ghissue:`723`: WIP: Assign streamlines to an existing cluster map via QuickBundles
* :ghissue:`738`: Import tkinter
* :ghissue:`767`: Offscreen is False for test_slicer
* :ghissue:`752`: TST: Install vtk and mesa on Travis to test the fvtk module.
* :ghissue:`768`: Document dipy.reconst.dti.iter_fit_tensor params
* :ghissue:`763`: Tensor Fitting Overflows Memory
* :ghissue:`766`: Add fit_tensor iteration decorator
* :ghissue:`751`: Reorient tracks according to ROI
* :ghissue:`765`: BF: Typo in data file name.
* :ghissue:`764`: 404: Not Found when loading Stanford labels
* :ghissue:`757`: Optimize dipy.align.reslice
* :ghissue:`587`: Fvtk 2.0 PR1
* :ghissue:`286`: WIP - FVTK refactor/cleanup
* :ghissue:`755`: dipy.reconst.tests.test_shm.test_sf_to_sh: TypeError: Cannot cast ufunc add output from dtype('float64') to dtype('uint16') with casting rule 'same_kind'
* :ghissue:`749`: Fixed deprecation warning in skimage
* :ghissue:`748`: TST: added test for _to_voxel_tolerance.
* :ghissue:`678`: BF: added tolerance for negative streamline coordinates checks
* :ghissue:`714`: RF: use masks in predictions and cross-validation
* :ghissue:`739`: Set number of OpenMP threads during runtime
* :ghissue:`733`: Add RTOP, RTAP and RTPP and the relative test
* :ghissue:`743`: BF: memleaks with typed memory views in Cython
* :ghissue:`737`: Possibly set_number_of_points doesn't delete memory
* :ghissue:`672`: Power map
* :ghissue:`724`: @sinkpoint's power map - refactored
* :ghissue:`741`: ENH: it is preferable to use choice rather than randint to not have
* :ghissue:`730`: numpy 1.10 breaks master
* :ghissue:`727`: Optimize tensor fitting
* :ghissue:`726`: NF - CSD response from a mask
* :ghissue:`729`: BF: tensor predict
* :ghissue:`736`: Added installation of python-tk package for VTK travis bot
* :ghissue:`735`: Added comment on nlmeans example about selecting one volume
* :ghissue:`732`: WIP: Test with vtk on Travis
* :ghissue:`734`: WIP: Fvtk 2.0 with travis vtk support
* :ghissue:`688`: dipy.test() fails on centos 6.x / python2.6
* :ghissue:`731`: Np 1.10
* :ghissue:`725`: WIP: TST: Install vtk on travis with conda.
* :ghissue:`640`: MAPMRI
* :ghissue:`611`: OSX test fail 'we check the default value of lambda ...'
* :ghissue:`715`: In current segment_quickbundles tutorial there is no example for changing number of points
* :ghissue:`719`: Fixes #715
* :ghissue:`682`: Created list of examples for available features and metrics
* :ghissue:`716`: Refactor data module
* :ghissue:`699`: Added gaussian noise option to estimate_sigma
* :ghissue:`712`: DOC: API changes in gh707.
* :ghissue:`713`: RF: In case a user just wants to use a single integer.
* :ghissue:`700`: TEST: add tests for AffineMap
* :ghissue:`677`: DKI PR3 - NF: Adding standard kurtosis statistics on module dki.py
* :ghissue:`721`: TST: Verify that output of estimate_sigma is a proper input to nlmeans.
* :ghissue:`693`: WIP: affine map tests
* :ghissue:`694`: Memory errors / timeouts with affine registration on Windows
* :ghissue:`572`: NF : nlmeans now support arrays of noise std
* :ghissue:`708`: Check for bval dimensionality on read.
* :ghissue:`697`: dipy.io.gradients read_bvals_bvecs does not check bvals length
* :ghissue:`707`: BF: Keep up with changes in scipy 0.16
* :ghissue:`710`: Test dipy.core.tests.test_sphere.test_interp_rbf fails fails on Travis
* :ghissue:`709`: DOC: Use the `identity` variable in the resampling transformation.
* :ghissue:`649`: ROI seeds not placed at the center of the voxels
* :ghissue:`656`: Build-bot status
* :ghissue:`701`: Changes in `syn_registration_3d` example
* :ghissue:`703`: Fix syn-3d example
* :ghissue:`705`: Fix example in function compress_streamline
* :ghissue:`704`: Buildbots failure: related to streamline compression?
* :ghissue:`635`: Select streamlines based on logical operations on ROIs
* :ghissue:`702`: BF: Use only validated examples when building docs.
* :ghissue:`689`: Streamlines compression
* :ghissue:`698`: DOC: added NI citation
* :ghissue:`621`: piesno example not rendering correctly on the website
* :ghissue:`650`: profiling hyp1f1
* :ghissue:`681`: RF + DOC: Add MNI template reference. Also import it into the dipy.da…
* :ghissue:`696`: Change title of piesno example
* :ghissue:`691`: CENIR 'HCP-like' multi b-value data
* :ghissue:`661`: Test DTI eigenvectors
* :ghissue:`690`: BF: nan entries cause segfault
* :ghissue:`667`: DOC: Remove Sourceforge related makefile things. Add the gh-pages upl…
* :ghissue:`676`: TST: update Travis config to use container infrastructure.
* :ghissue:`533`: MRG: some Cython refactorings
* :ghissue:`686`: BF: Make buildbot Pyhon26-32 happy
* :ghissue:`622`: Fast shm from scipy 0.15.0 does not work on rc version
* :ghissue:`683`: Fixed initial estimation in piesno
* :ghissue:`233`: WIP: Dki
* :ghissue:`654`: Affine registration PR 3/3
* :ghissue:`684`: BF: Fixed memory leak in QuickBundles.
* :ghissue:`674`: NF: Function to sample perpendicular directions relative to a given vector
* :ghissue:`679`: BF + NF: Provide dipy version info when running dipy.get_info()
* :ghissue:`680`: NF: Fetch and Read the MNI T1 and/or T2 template.
* :ghissue:`664`: DKI fitting (DKI PR2)
* :ghissue:`539`: WIP: BF: Catching initial fodf creation of SDT
* :ghissue:`671`: DOC: move mailing list links to neuroimaging
* :ghissue:`663`: changed samuel st-jean email to the usherbrooke one
* :ghissue:`287`: Fvtk sphere origin
* :ghissue:`648`: Improve check of collinearity in vec2vec_rotmat
* :ghissue:`582`: DKI project: PR#1 Simulations to test DKI
* :ghissue:`660`: BF: If scalar-color input has len(shape)<4, need to fill that in.
* :ghissue:`612`: BF: Differences in spherical harmonic calculations wrt scipy 0.15
* :ghissue:`651`: Added estimate_sigma bias correction + update example
* :ghissue:`659`: BF: If n_frames is larger than one use path-numbering.
* :ghissue:`652`: MAINT: work around scipy bug in sph_harm
* :ghissue:`653`: Revisit naming when Matthew is back from Cuba
* :ghissue:`658`: FIX: resaved npy file causing load error for py 33
* :ghissue:`657`: Fix compilation error caused by inline functions
* :ghissue:`655`: Development documentation instructs to remove `master`
* :ghissue:`628`: Affine registration PR 2/3
* :ghissue:`629`: Quickbundles 2.1
* :ghissue:`638`: tutorial example, code in text format
* :ghissue:`637`: DOC: Fix typo in docstring of Identity class.
* :ghissue:`639`: DOC: Render the following line in the code cell.
* :ghissue:`614`: Seeds from mask random
* :ghissue:`633`: BF - no import of TissueTypes
* :ghissue:`632`: fixed typo in dti example
* :ghissue:`630`: Possible documentation bug (?)
* :ghissue:`627`: BF: Add missing opacity property to point actor
* :ghissue:`459`: streamtubes opacity kwarg
* :ghissue:`626`: Use LooseVersion to check for scipy versions
* :ghissue:`625`: DOC: Include the PIESNO example.
* :ghissue:`623`: DOC: Include the PIESNO example in the documentation.
* :ghissue:`624`: DOC: Corrected typos in Restore tutorial and docstring.
* :ghissue:`619`: DOC: Added missing contributor to developer list
* :ghissue:`604`: Retired ARM buildbot
* :ghissue:`613`: Possible random failure in test_vector_fields.test_reorient_vector_field_2d
* :ghissue:`618`: Update README file
* :ghissue:`616`: Raise ValueError when invalid matrix is given
* :ghissue:`617`: Added build status icon to readme
* :ghissue:`576`: Piesno example
* :ghissue:`615`: bugfix for double word in doc example issue #387
* :ghissue:`600`: Use of nanmean breaks dipy for numpy < 1.8
* :ghissue:`610`: Added figure with HBM 2015
* :ghissue:`609`: Update website documentation
* :ghissue:`390`: WIP: New PIESNO example and small corrections
* :ghissue:`607`: DOC: Detailed github stats for 0.9
* :ghissue:`606`: Removed the word new
* :ghissue:`605`: Release mode: updating Changelog and Authors
* :ghissue:`594`: DOC + PEP8: Mostly just line-wrapping.
