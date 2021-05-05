.. _release1.4.1:

=====================================
 Release notes for DIPY version 1.4.1
=====================================

GitHub stats for 2021/03/14 - 2021/05/05 (tag: 1.4.0)

These lists are automatically generated, and may be incomplete or contain duplicates.

The following 11 authors contributed 153 commits.

* Ariel Rokem
* Bramsh Qamar Chandio
* David Romero-Bascones
* Eleftherios Garyfallidis
* Etienne St-Onge
* Felix Liu
* Gabriel Girard
* John Kruper
* Nasim Anousheh
* Serge Koudoro
* Shreyas Fadnavis


We closed a total of 89 issues, 28 pull requests and 61 regular issues;
this is the full list (generated with the script
:file:`tools/github_stats.py`):

Pull Requests (28):

* :ghpull:`2367`: [Upcoming] Release 1.4.1
* :ghpull:`2387`: added all examples of CST and updated AFQ file name
* :ghpull:`2386`: Adding CST_L back in Bundle Segmentation Tutorial
* :ghpull:`2375`: Expanding Bundle Segmentation Tutorial
* :ghpull:`2382`: Updated docs for using P2S optimally
* :ghpull:`2385`: RF: Standardize the argument name for the number of threads/cores
* :ghpull:`2384`: RF - Removed deprecated tracking code
* :ghpull:`2351`: Updating Vec2vec_rotmat to deal with numerical issues
* :ghpull:`2381`: Adds the NIPY code of conduct to our repo.
* :ghpull:`2371`: [Fix] Add "None" options in the CLIs
* :ghpull:`2352`: RF: configure num_threads==-1 as the value to use all cores
* :ghpull:`2373`: [FIX] warning if not the same number of points
* :ghpull:`2372`: Expand patch radius if input is int
* :ghpull:`2348`: RF: Use new name for this function.
* :ghpull:`2363`: [ENH] Adding cython file(*.pyx) in documentation
* :ghpull:`2365`: [DOC]: Change defaults in Patch2Self example
* :ghpull:`2349`: [ENH] Allow for other statistics, like median, in afq_profile
* :ghpull:`2350`: [FIX] Use npy_intp variables instead of int and size_t to iterate over numpy arrays
* :ghpull:`2346`: [MNT]  Update and fix Cython warnings and use cnp.PyArray_DATA wherever possible
* :ghpull:`2347`: Replacing Data in NLMeans Tutorial
* :ghpull:`2340`: [FIX] reactivate codecov
* :ghpull:`2344`: [FIX] Tractogram Header in RecoBundles Tutorial
* :ghpull:`2339`: [FIX] Cleanup deprecated np.float, np.bool, np.int
* :ghpull:`1648`: Mesh seeding (surface)
* :ghpull:`2337`: BF: Change patch2self defaults.
* :ghpull:`2333`: Add __str__ to GradientTable
* :ghpull:`2335`: RF: Replaces deprecated basis by its new name.
* :ghpull:`2332`: [FIX] fix tests for all new deprecated functions

Issues (61):

* :ghissue:`2375`: Expanding Bundle Segmentation Tutorial
* :ghissue:`1973`: Recobundles documentation
* :ghissue:`2382`: Updated docs for using P2S optimally
* :ghissue:`2385`: RF: Standardize the argument name for the number of threads/cores
* :ghissue:`2377`: RF: standardize the argument name for the number of threads/cores
* :ghissue:`2384`: RF - Removed deprecated tracking code
* :ghissue:`2351`: Updating Vec2vec_rotmat to deal with numerical issues
* :ghissue:`2381`: Adds the NIPY code of conduct to our repo.
* :ghissue:`2380`: Community and governance
* :ghissue:`2371`: [Fix] Add "None" options in the CLIs
* :ghissue:`2300`: NF: Add "None" options in the CLIs
* :ghissue:`2352`: RF: configure num_threads==-1 as the value to use all cores
* :ghissue:`2373`: [FIX] warning if not the same number of points
* :ghissue:`2320`: RecoBundles distances
* :ghissue:`2372`: Expand patch radius if input is int
* :ghissue:`2341`: Allow use of all threads in the gibbs ringing workflow
* :ghissue:`2348`: RF: Use new name for this function.
* :ghissue:`2353`: How to create tractogram from a multi-shell data for RecoBundles
* :ghissue:`1311`: Adding cython file(*.pyx) in documentation
* :ghissue:`2363`: [ENH] Adding cython file(*.pyx) in documentation
* :ghissue:`1302`: [DOC] cython (pyx) files are not parsed
* :ghissue:`366`: Some doc missing
* :ghissue:`2365`: [DOC]: Change defaults in Patch2Self example
* :ghissue:`1672`: Dipy Segmentation fault when visualizing
* :ghissue:`1444`: Move general registration tools into own package?
* :ghissue:`562`: Multiprocessing the tensor reconstruction
* :ghissue:`13`: Cordinate maps stuff
* :ghissue:`2324`: Dipy for VR/AR
* :ghissue:`2345`: Saving and/or importing nonlinear warps
* :ghissue:`2349`: [ENH] Allow for other statistics, like median, in afq_profile
* :ghissue:`2350`: [FIX] Use npy_intp variables instead of int and size_t to iterate over numpy arrays
* :ghissue:`423`: Use npy_intp variables instead of int and size_t to iterate over numpy arrays
* :ghissue:`837`: Should we enforce float32 in tractography results?
* :ghissue:`636`: Get a standard interface for the functions using the noise variance
* :ghissue:`861`: open mp defaults to one core, is that a good idea?
* :ghissue:`2346`: [MNT]  Update and fix Cython warnings and use cnp.PyArray_DATA wherever possible
* :ghissue:`1895`: Cython warnings
* :ghissue:`545`: Use cnp.PyArray_DATA wherever possible
* :ghissue:`2347`: Replacing Data in NLMeans Tutorial
* :ghissue:`1847`: Replacing Data in NLMeans Tutorial
* :ghissue:`2340`: [FIX] reactivate codecov
* :ghissue:`1872`: Did we lose our coverage reporting?
* :ghissue:`1646`: Fetcher should not be under coverage
* :ghissue:`1635`: Track from mesh
* :ghissue:`2344`: [FIX] Tractogram Header in RecoBundles Tutorial
* :ghissue:`2309`: Tractogram Header in RecoBundles Tutorial
* :ghissue:`2334`: Aphysical signal after running patch2self
* :ghissue:`1873`: ERROR while import data
* :ghissue:`2343`: Missing Python 3.9 wheels
* :ghissue:`1996`: Documentation not being rendered correctly
* :ghissue:`2311`: Accuracy of DKI measures
* :ghissue:`2274`: DKI metrics' accuracy
* :ghissue:`2339`: [FIX] Cleanup deprecated np.float, np.bool, np.int
* :ghissue:`1648`: Mesh seeding (surface)
* :ghissue:`1675`: WIP: Integer indices
* :ghissue:`2316`: TranslationTransform2D Exact X-Y Shift
* :ghissue:`2337`: BF: Change patch2self defaults.
* :ghissue:`2333`: Add __str__ to GradientTable
* :ghissue:`2331`: gtab.info does not print anything
* :ghissue:`2335`: RF: Replaces deprecated basis by its new name.
* :ghissue:`2332`: [FIX] fix tests for all new deprecated functions
