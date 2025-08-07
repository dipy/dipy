.. _release0.8:

===================================
 Release notes for DIPY version 0.8
===================================

GitHub stats for 2013/12/24 - 2014/12/26 (tag: 0.7.0)

The following 19 authors contributed 1176 commits.

* Andrew Lawrence
* Ariel Rokem
* Bago Amirbekian
* Demian Wassermann
* Eleftherios Garyfallidis
* Gabriel Girard
* Gregory R. Lee
* Jean-Christophe Houde
* Kesshi jordan
* Marc-Alexandre Cote
* Matthew Brett
* Matthias Ekman
* Matthieu Dumont
* Mauro Zucchelli
* Maxime Descoteaux
* Michael Paquette
* Omar Ocegueda
* Samuel St-Jean
* Stefan van der Walt


We closed a total of 388 issues, 155 pull requests and 233 regular issues;
this is the full list (generated with the script
:file:`tools/github_stats.py`):

Pull Requests (155):

* :ghpull:`544`: Refactor propspeed - updated
* :ghpull:`543`: MRG: update to plot_2d fixes and tests
* :ghpull:`537`: NF: add requirements.txt file
* :ghpull:`534`: BF: removed ftmp variable
* :ghpull:`536`: Update Changelog
* :ghpull:`535`: Happy New Year PR!
* :ghpull:`531`: BF: extend pip timeout to reduce install failures
* :ghpull:`527`: Remove npymath library from cython extensions
* :ghpull:`528`: MRG: move conditional compiling to C
* :ghpull:`530`: BF: work round ugly MSVC manifest bug
* :ghpull:`529`: MRG: a couple of small cleanup fixes
* :ghpull:`526`: Readme.rst and info.py update about the license
* :ghpull:`525`: Added shore gpl warning in the readme
* :ghpull:`524`: Replaced DiPy with DIPY in readme.rst and info.py
* :ghpull:`523`: RF: copy includes list for extensions
* :ghpull:`522`: DOC: Web-site release notes, and some updates on front page.
* :ghpull:`521`: Life bots
* :ghpull:`520`: Relaxing precision for win32
* :ghpull:`519`: Christmas PR! Correcting typos, linking and language for max odf tracking
* :ghpull:`513`: BF + TST: Reinstated eig_from_lo_tri
* :ghpull:`508`: Tests for reslicing
* :ghpull:`515`: TST: Increasing testing on life.
* :ghpull:`516`: TST: Reduce sensitivity on these tests.
* :ghpull:`495`: NF - Deterministic Maximum Direction Getter
* :ghpull:`514`: Website update
* :ghpull:`510`: BF: another fvtk 5 to 6 incompatibility
* :ghpull:`509`: DOC: Small fixes in documentation.
* :ghpull:`497`: New sphere for ODF reconstruction
* :ghpull:`460`: Sparse Fascicle Model
* :ghpull:`499`: DOC: Warn about the GPL license of SHORE.
* :ghpull:`491`: RF - Make peaks_from_model part of dipy.direction
* :ghpull:`501`: TST: Test for both data with and w/0 b0.
* :ghpull:`507`: BF - use different sort method to avoid mergsort for older numpy.
* :ghpull:`504`: Bug fix float overflow in estimate_sigma
* :ghpull:`494`: Fix round
* :ghpull:`503`: Fixed compatibility issues between vtk 5 and 6
* :ghpull:`498`: DTI `min_signal`
* :ghpull:`471`: Use importlib instead of __import__
* :ghpull:`419`: LiFE
* :ghpull:`489`: Fix diffeomorphic registration test failures
* :ghpull:`484`: Clear tabs from examples for website
* :ghpull:`490`: DOC: corrected typos in the tracking PR
* :ghpull:`341`: Traco Redesign
* :ghpull:`483`: NF: Find the closest vertex on a sphere for an input vector.
* :ghpull:`488`: BF: fix travis version setting
* :ghpull:`485`: RF: deleted unused files
* :ghpull:`482`: Skipping tests for different versions of Scipy for optimize.py
* :ghpull:`480`: Enhance SLR to allow for series of registrations
* :ghpull:`479`: Report on coverage for old scipy.
* :ghpull:`481`: BF - make examples was confusing files with similar names, fixed
* :ghpull:`476`: Fix optimize defaults for older scipy versions for L-BFGS-B
* :ghpull:`478`: TST: Increase the timeout on the Travis pip install
* :ghpull:`477`: MAINT+TST: update minimum nibabel dependency
* :ghpull:`474`: RF: switch travis tests to use virtualenvs
* :ghpull:`473`: TST: Make Travis provide verbose test outputs.
* :ghpull:`472`: ENH: GradientTable now calculates qvalues
* :ghpull:`469`: Fix evolution save win32
* :ghpull:`463`: DOC: update RESTORE tutorial to use new noise estimation technique
* :ghpull:`466`: BF: cannot quote command for Windows
* :ghpull:`465`: BF: increased SCIPY version definition flag to 0.12
* :ghpull:`462`: BF: fix writing history to file in Python 3
* :ghpull:`433`: Added local variance estimation
* :ghpull:`458`: DOC:  docstring fixes in dipy/align/crosscorr.pyx
* :ghpull:`448`: BF: fix link to npy_math function
* :ghpull:`447`: BF: supposed fix for the gh-439, but still unable to reproduce OP.
* :ghpull:`443`: Fix buildbots errors introduced with the registration module
* :ghpull:`456`: MRG: relax threshold for failing test + cleanup
* :ghpull:`454`: DOC: fix docstring for compile-time checker
* :ghpull:`453`: BF: refactor conditional compiling again
* :ghpull:`446`: Streamline-based Linear Registration
* :ghpull:`445`: NF: generate config.pxi file with Cython DEF vars
* :ghpull:`440`: DOC - add info on how to change default tempdir (multiprocessing).
* :ghpull:`431`: Change the writeable flag back to its original state when finished.
* :ghpull:`408`: Symmetric diffeomorphic non-linear registration
* :ghpull:`438`: Missing a blank line in examples/tracking_quick_start.py
* :ghpull:`405`: fixed frozen windows executable issue
* :ghpull:`418`: RF: move script running code into own module
* :ghpull:`437`: Update Cython download URL
* :ghpull:`435`: BF: replaced non-ascii character in dipy.reconst.dti line 956
* :ghpull:`434`: DOC: References for the DTI ODF calculation.
* :ghpull:`430`: Revert "Support read-only numpy array."
* :ghpull:`427`: Support read-only numpy array.
* :ghpull:`421`: Fix nans in gfa
* :ghpull:`422`: BF: Use the short version to verify scipy version.
* :ghpull:`415`: RF - move around some of the predict stuff
* :ghpull:`420`: Rename README.txt to README.rst
* :ghpull:`413`: Faster spherical harmonics
* :ghpull:`416`: Removed memory_leak unittest in test_strealine.py
* :ghpull:`417`: Fix streamlinespeed tests
* :ghpull:`411`: Fix memory leak in cython functions length and set_number_of_points
* :ghpull:`409`: minor corrections to pipe function
* :ghpull:`396`: TST : this is not exactly equal on some platforms.
* :ghpull:`407`: BF: fixed problem with NANs in odfdeconv
* :ghpull:`406`: Revert "Merge pull request #346 from omarocegueda/syn_registration"
* :ghpull:`402`: Fix AE test error in test_peak_directions_thorough
* :ghpull:`403`: Added mask shape check in tenfit
* :ghpull:`346`: Symmetric diffeomorphic non-linear registration
* :ghpull:`401`: BF: fix skiptest invocation for missing mpl
* :ghpull:`340`: CSD fit issue
* :ghpull:`397`: BF: fix import statement for get_cmap
* :ghpull:`393`: RF: update Cython dependency
* :ghpull:`382`: Cythonized version of streamlines' resample() and length() functions.
* :ghpull:`386`: DOC: Small fix in the xval example.
* :ghpull:`335`: Xval
* :ghpull:`352`: Fix utils docs and affine
* :ghpull:`384`: odf_sh_sharpening function fix and new test
* :ghpull:`374`: MRG: bumpy numpy requirement to 1.5 / compat fixes
* :ghpull:`380`: DOC: Update a few Dipy links to link to the correct repo
* :ghpull:`378`: Fvtk cleanup
* :ghpull:`379`: fixed typos in shm.py
* :ghpull:`339`: FVTK small improvement: Arbitrary matplotlib colormaps can be used to color spherical functions
* :ghpull:`373`: Fixed discrepancies between doc and code
* :ghpull:`371`: RF: don't use -fopenmp flag if it doesn't work
* :ghpull:`372`: BF: set integer type for crossplatform compilation
* :ghpull:`337`: Piesno
* :ghpull:`370`: Tone down the front page a bit.
* :ghpull:`364`: Add the mode param for border management.
* :ghpull:`368`: New banner for website
* :ghpull:`367`: MRG: refactor API generation for sharing
* :ghpull:`363`: RF: make cvxopt optional for tests
* :ghpull:`362`: Changes to fix issue #361: matrix sizing in tracking.utils.connectivity_matrix
* :ghpull:`360`: Added missing $ sign
* :ghpull:`355`: DOC: Updated API change document to add target function change
* :ghpull:`357`: Changed the logo to full black as the one that I sent as suggestion for HBM and ISMRM
* :ghpull:`356`: Auto-generate API docs
* :ghpull:`349`: Added api changes file to track breaks of backwards compatibility
* :ghpull:`348`: Website update
* :ghpull:`347`: DOC: Updating citations
* :ghpull:`345`: TST: Make travis look at test coverage.
* :ghpull:`338`: Add positivity constraint on the propagator
* :ghpull:`334`: Fix vec2vec
* :ghpull:`324`: Constrained optimisation for SHORE to set E(0)=1 when the CVXOPT package is available
* :ghpull:`320`: Denoising images using non-local means
* :ghpull:`331`: DOC: correct number of seeds in streamline_tools example
* :ghpull:`326`: Fix brain extraction example
* :ghpull:`327`: add small and big delta
* :ghpull:`323`: Shore pdf grid speed improvement
* :ghpull:`319`: DOC: Updated the highlights to promote the release and the upcoming paper
* :ghpull:`318`: Corrected some rendering problems with the installation instructions
* :ghpull:`317`: BF: more problems with path quoting in windows
* :ghpull:`316`: MRG: more fixes for windows script tests
* :ghpull:`315`: BF: EuDX odf_vertices param has no default value
* :ghpull:`305`: DOC: Some more details in installation instructions.
* :ghpull:`314`: BF - callable response does not work
* :ghpull:`311`: Bf seeds from mask
* :ghpull:`309`: MRG: Windows test fixes
* :ghpull:`308`: typos + pep stuf
* :ghpull:`303`: BF: try and fix nibabel setup requirement
* :ghpull:`304`: Update README.txt
* :ghpull:`302`: Time for 0.8.0.dev!
* :ghpull:`299`: BF: Put back utils.length.
* :ghpull:`301`: Updated info.py and copyright year
* :ghpull:`300`: Bf fetcher bug on windows
* :ghpull:`298`: TST - rework tests so that we do not need to download any data
* :ghpull:`290`: DOC: Started generating 0.7 release notes.

Issues (233):

* :ghissue:`544`: Refactor propspeed - updated
* :ghissue:`540`: MRG: refactor propspeed
* :ghissue:`542`: TST: Testing regtools
* :ghissue:`543`: MRG: update to plot_2d fixes and tests
* :ghissue:`541`: BUG:   plot_2d_diffeomorphic_map
* :ghissue:`439`: ValueError in RESTORE
* :ghissue:`538`: WIP: TEST: relaxed precision
* :ghissue:`449`: local variable 'ftmp' referenced before assignment
* :ghissue:`537`: NF: add requirements.txt file
* :ghissue:`534`: BF: removed ftmp variable
* :ghissue:`536`: Update Changelog
* :ghissue:`535`: Happy New Year PR!
* :ghissue:`512`: reconst.dti.eig_from_lo_tri
* :ghissue:`467`: Optimize failure on Windows
* :ghissue:`464`: Diffeomorphic registration test failures on PPC
* :ghissue:`531`: BF: extend pip timeout to reduce install failures
* :ghissue:`527`: Remove npymath library from cython extensions
* :ghissue:`528`: MRG: move conditional compiling to C
* :ghissue:`530`: BF: work round ugly MSVC manifest bug
* :ghissue:`529`: MRG: a couple of small cleanup fixes
* :ghissue:`526`: Readme.rst and info.py update about the license
* :ghissue:`525`: Added shore gpl warning in the readme
* :ghissue:`524`: Replaced DiPy with DIPY in readme.rst and info.py
* :ghissue:`523`: RF: copy includes list for extensions
* :ghissue:`522`: DOC: Web-site release notes, and some updates on front page.
* :ghissue:`521`: Life bots
* :ghissue:`520`: Relaxing precision for win32
* :ghissue:`519`: Christmas PR! Correcting typos, linking and language for max odf tracking
* :ghissue:`513`: BF + TST: Reinstated eig_from_lo_tri
* :ghissue:`508`: Tests for reslicing
* :ghissue:`515`: TST: Increasing testing on life.
* :ghissue:`516`: TST: Reduce sensitivity on these tests.
* :ghissue:`495`: NF - Deterministic Maximum Direction Getter
* :ghissue:`514`: Website update
* :ghissue:`510`: BF: another fvtk 5 to 6 incompatibility
* :ghissue:`511`: Error estimating tensors on hcp dataset
* :ghissue:`509`: DOC: Small fixes in documentation.
* :ghissue:`497`: New sphere for ODF reconstruction
* :ghissue:`460`: Sparse Fascicle Model
* :ghissue:`499`: DOC: Warn about the GPL license of SHORE.
* :ghissue:`491`: RF - Make peaks_from_model part of dipy.direction
* :ghissue:`501`: TST: Test for both data with and w/0 b0.
* :ghissue:`507`: BF - use different sort method to avoid mergsort for older numpy.
* :ghissue:`505`: stable/wheezy debian -- ar.argsort(kind='mergesort') causes TypeError: requested sort not available for type (
* :ghissue:`506`: RF: Use integer datatype for unique_rows sorting.
* :ghissue:`504`: Bug fix float overflow in estimate_sigma
* :ghissue:`399`: Multiprocessing runtime error in Windows 64 bit
* :ghissue:`383`: typo in multi tensor fit example
* :ghissue:`350`: typo in SNR example
* :ghissue:`424`: test more python versions with travis
* :ghissue:`493`: BF - older C compilers do not have round in math.h, using dpy_math instead
* :ghissue:`494`: Fix round
* :ghissue:`503`: Fixed compatibility issues between vtk 5 and 6
* :ghissue:`500`: SHORE hyp2F1
* :ghissue:`502`: Fix record vtk6
* :ghissue:`498`: DTI `min_signal`
* :ghissue:`496`: Revert "BF: supposed fix for the gh-439, but still unable to reproduce O...
* :ghissue:`492`: TST - new DTI test to help develop min_signal handling
* :ghissue:`471`: Use importlib instead of __import__
* :ghissue:`419`: LiFE
* :ghissue:`489`: Fix diffeomorphic registration test failures
* :ghissue:`484`: Clear tabs from examples for website
* :ghissue:`490`: DOC: corrected typos in the tracking PR
* :ghissue:`341`: Traco Redesign
* :ghissue:`410`: Faster spherical harmonics implementation
* :ghissue:`483`: NF: Find the closest vertex on a sphere for an input vector.
* :ghissue:`487`: Travis Problem
* :ghissue:`488`: BF: fix travis version setting
* :ghissue:`485`: RF: deleted unused files
* :ghissue:`486`: cvxopt is gpl licensed
* :ghissue:`482`: Skipping tests for different versions of Scipy for optimize.py
* :ghissue:`480`: Enhance SLR to allow for series of registrations
* :ghissue:`479`: Report on coverage for old scipy.
* :ghissue:`481`: BF - make examples was confusing files with similar names, fixed
* :ghissue:`428`: WIP: refactor travis building
* :ghissue:`429`: WIP: Refactor travising
* :ghissue:`476`: Fix optimize defaults for older scipy versions for L-BFGS-B
* :ghissue:`478`: TST: Increase the timeout on the Travis pip install
* :ghissue:`477`: MAINT+TST: update minimum nibabel dependency
* :ghissue:`475`: Does the optimizer still need `tmp_files`?
* :ghissue:`474`: RF: switch travis tests to use virtualenvs
* :ghissue:`473`: TST: Make Travis provide verbose test outputs.
* :ghissue:`470`: Enhance SLR with applying series of transformations and fix optimize bug for parameter missing in old scipy versions
* :ghissue:`472`: ENH: GradientTable now calculates qvalues
* :ghissue:`469`: Fix evolution save win32
* :ghissue:`463`: DOC: update RESTORE tutorial to use new noise estimation technique
* :ghissue:`466`: BF: cannot quote command for Windows
* :ghissue:`461`: Buildbot failures with missing 'nit' key in dipy.core.optimize
* :ghissue:`465`: BF: increased SCIPY version definition flag to 0.12
* :ghissue:`462`: BF: fix writing history to file in Python 3
* :ghissue:`433`: Added local variance estimation
* :ghissue:`432`: auto estimate the standard deviation globally for nlmeans
* :ghissue:`451`: Warning for DTI normalization
* :ghissue:`458`: DOC:  docstring fixes in dipy/align/crosscorr.pyx
* :ghissue:`448`: BF: fix link to npy_math function
* :ghissue:`447`: BF: supposed fix for the gh-439, but still unable to reproduce OP.
* :ghissue:`443`: Fix buildbots errors introduced with the registration module
* :ghissue:`456`: MRG: relax threshold for failing test + cleanup
* :ghissue:`455`: Test failure on `master`
* :ghissue:`454`: DOC: fix docstring for compile-time checker
* :ghissue:`450`: Find if replacing matrix44 from streamlinear with compose_matrix from dipy.core.geometry is a good idea
* :ghissue:`453`: BF: refactor conditional compiling again
* :ghissue:`446`: Streamline-based Linear Registration
* :ghissue:`452`: Replace raise by auto normalization when creating a gradient table with un-normalized bvecs.
* :ghissue:`398`: assert AE < 2. failure in test_peak_directions_thorough
* :ghissue:`444`: heads up - MKL error in parallel mode
* :ghissue:`445`: NF: generate config.pxi file with Cython DEF vars
* :ghissue:`440`: DOC - add info on how to change default tempdir (multiprocessing).
* :ghissue:`431`: Change the writeable flag back to its original state when finished.
* :ghissue:`408`: Symmetric diffeomorphic non-linear registration
* :ghissue:`333`: Bundle alignment
* :ghissue:`438`: Missing a blank line in examples/tracking_quick_start.py
* :ghissue:`426`: nlmeans_3d breaks with mask=None
* :ghissue:`405`: fixed frozen windows executable issue
* :ghissue:`418`: RF: move script running code into own module
* :ghissue:`437`: Update Cython download URL
* :ghissue:`435`: BF: replaced non-ascii character in dipy.reconst.dti line 956
* :ghissue:`434`: DOC: References for the DTI ODF calculation.
* :ghissue:`425`: NF added class to save streamlines in vtk format
* :ghissue:`430`: Revert "Support read-only numpy array."
* :ghissue:`427`: Support read-only numpy array.
* :ghissue:`421`: Fix nans in gfa
* :ghissue:`422`: BF: Use the short version to verify scipy version.
* :ghissue:`415`: RF - move around some of the predict stuff
* :ghissue:`420`: Rename README.txt to README.rst
* :ghissue:`413`: Faster spherical harmonics
* :ghissue:`416`: Removed memory_leak unittest in test_strealine.py
* :ghissue:`417`: Fix streamlinespeed tests
* :ghissue:`411`: Fix memory leak in cython functions length and set_number_of_points
* :ghissue:`412`: Use simple multiplication instead exponentiation and pow
* :ghissue:`409`: minor corrections to pipe function
* :ghissue:`396`: TST : this is not exactly equal on some platforms.
* :ghissue:`407`: BF: fixed problem with NANs in odfdeconv
* :ghissue:`406`: Revert "Merge pull request #346 from omarocegueda/syn_registration"
* :ghissue:`402`: Fix AE test error in test_peak_directions_thorough
* :ghissue:`403`: Added mask shape check in tenfit
* :ghissue:`346`: Symmetric diffeomorphic non-linear registration
* :ghissue:`401`: BF: fix skiptest invocation for missing mpl
* :ghissue:`340`: CSD fit issue
* :ghissue:`397`: BF: fix import statement for get_cmap
* :ghissue:`393`: RF: update Cython dependency
* :ghissue:`391`: memory usage: 16GB wasn't sufficient
* :ghissue:`382`: Cythonized version of streamlines' resample() and length() functions.
* :ghissue:`386`: DOC: Small fix in the xval example.
* :ghissue:`385`: cross_validation example doesn't render properly
* :ghissue:`335`: Xval
* :ghissue:`352`: Fix utils docs and affine
* :ghissue:`384`: odf_sh_sharpening function fix and new test
* :ghissue:`374`: MRG: bumpy numpy requirement to 1.5 / compat fixes
* :ghissue:`381`: Bago fix utils docs and affine
* :ghissue:`380`: DOC: Update a few Dipy links to link to the correct repo
* :ghissue:`378`: Fvtk cleanup
* :ghissue:`379`: fixed typos in shm.py
* :ghissue:`376`: BF: Adjust the dimensionality of the peak_values, if provided.
* :ghissue:`377`: Demianw fvtk colormap
* :ghissue:`339`: FVTK small improvement: Arbitrary matplotlib colormaps can be used to color spherical functions
* :ghissue:`373`: Fixed discrepancies between doc and code
* :ghissue:`371`: RF: don't use -fopenmp flag if it doesn't work
* :ghissue:`372`: BF: set integer type for crossplatform compilation
* :ghissue:`337`: Piesno
* :ghissue:`370`: Tone down the front page a bit.
* :ghissue:`364`: Add the mode param for border management.
* :ghissue:`368`: New banner for website
* :ghissue:`367`: MRG: refactor API generation for sharing
* :ghissue:`359`: cvxopt dependency
* :ghissue:`363`: RF: make cvxopt optional for tests
* :ghissue:`361`: Matrix size wrong for tracking.utils.connectivity_matrix
* :ghissue:`362`: Changes to fix issue #361: matrix sizing in tracking.utils.connectivity_matrix
* :ghissue:`360`: Added missing $ sign
* :ghissue:`358`: typo in doc
* :ghissue:`355`: DOC: Updated API change document to add target function change
* :ghissue:`357`: Changed the logo to full black as the one that I sent as suggestion for HBM and ISMRM
* :ghissue:`356`: Auto-generate API docs
* :ghissue:`349`: Added api changes file to track breaks of backwards compatibility
* :ghissue:`348`: Website update
* :ghissue:`347`: DOC: Updating citations
* :ghissue:`345`: TST: Make travis look at test coverage.
* :ghissue:`338`: Add positivity constraint on the propagator
* :ghissue:`334`: Fix vec2vec
* :ghissue:`343`: Please Ignore this PR!
* :ghissue:`324`: Constrained optimisation for SHORE to set E(0)=1 when the CVXOPT package is available
* :ghissue:`277`: WIP: PIESNO framework for estimating the underlying std of the gaussian distribution
* :ghissue:`336`: Demianw shore e0 constrained
* :ghissue:`235`: WIP: Cross-validation
* :ghissue:`329`: WIP: Fix vec2vec
* :ghissue:`320`: Denoising images using non-local means
* :ghissue:`331`: DOC: correct number of seeds in streamline_tools example
* :ghissue:`330`: DOC: number of seeds per voxel, inconsistent documentation?
* :ghissue:`326`: Fix brain extraction example
* :ghissue:`327`: add small and big delta
* :ghissue:`323`: Shore pdf grid speed improvement
* :ghissue:`319`: DOC: Updated the highlights to promote the release and the upcoming paper
* :ghissue:`318`: Corrected some rendering problems with the installation instructions
* :ghissue:`317`: BF: more problems with path quoting in windows
* :ghissue:`316`: MRG: more fixes for windows script tests
* :ghissue:`315`: BF: EuDX odf_vertices param has no default value
* :ghissue:`312`: Sphere and default used through the code
* :ghissue:`305`: DOC: Some more details in installation instructions.
* :ghissue:`314`: BF - callable response does not work
* :ghissue:`16`: quickie: 'from raw data to tractographies' documentation implies dipy can't do anything with nonisotropic voxel sizes
* :ghissue:`311`: Bf seeds from mask
* :ghissue:`307`: Streamline_tools example stops working when I change density from 1 to 2
* :ghissue:`241`: Wrong normalization in peaks_from_model
* :ghissue:`248`: Clarify dsi example
* :ghissue:`220`: Add ndindex to peaks_from_model
* :ghissue:`253`: Parallel peaksFromModel timing out on buildbot
* :ghissue:`256`: writing data to /tmp peaks_from_model
* :ghissue:`278`: tenmodel.bvec, not existing anymore?
* :ghissue:`282`: fvtk documentation is incomprehensible
* :ghissue:`228`: buildbot error in mask.py
* :ghissue:`197`: DOC: some docstrings are not rendered correctly
* :ghissue:`181`: OPT: Change dipy.core.sphere_stats.random_uniform_on_sphere
* :ghissue:`177`: Extension test in dipy_fit_tensor seems brittle
* :ghissue:`171`: Fix auto_attrs
* :ghissue:`31`: Plotting in test suite
* :ghissue:`42`: RuntimeWarning in dti.py
* :ghissue:`43`: Problems with edges and faces in create_half_unit_sphere
* :ghissue:`53`: Is ravel_multi_index a new thing?
* :ghissue:`64`: Fix examples that rely on old API and removed data-sets
* :ghissue:`67`: viz.projections.sph_projection is broken
* :ghissue:`92`: dti.fa division by 0 warning in tests
* :ghissue:`306`: Tests fail after windows 32 bit installation and running import dipy; dipy.test()
* :ghissue:`310`: Windows test failure for tracking test_rmi
* :ghissue:`309`: MRG: Windows test fixes
* :ghissue:`308`: typos + pep stuf
* :ghissue:`303`: BF: try and fix nibabel setup requirement
* :ghissue:`304`: Update README.txt
* :ghissue:`302`: Time for 0.8.0.dev!
* :ghissue:`299`: BF: Put back utils.length.
* :ghissue:`301`: Updated info.py and copyright year
* :ghissue:`300`: Bf fetcher bug on windows
* :ghissue:`298`: TST - rework tests so that we do not need to download any data
* :ghissue:`290`: DOC: Started generating 0.7 release notes.
