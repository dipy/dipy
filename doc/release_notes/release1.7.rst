.. _release1.7:

====================================
 Release notes for DIPY version 1.7
====================================

GitHub stats for 2023/01/16 - 2023/04/23 (tag: 1.6.0)

These lists are automatically generated, and may be incomplete or contain duplicates.

The following 21 authors contributed 496 commits.

* Ariel Rokem
* Bramsh Qamar
* Charles Poirier
* Dogu Baran Aydogan
* Eleftherios Garyfallidis
* Etienne St-Onge
* Francois Rheault
* Gabriel Girard
* Javier Guaje
* Jong Sung Park
* Martino Pilia
* Mitesh Gulecha
* Rahul Ubale
* Sam Coveney
* Serge Koudoro
* Shilpi
* Tom Dela Haije
* Yaroslav Halchenko
* karp2601
* lb-97
* ujjwal-shekhar


We closed a total of 87 issues, 34 pull requests and 53 regular issues;
this is the full list (generated with the script
:file:`tools/github_stats.py`):

Pull Requests (34):

* :ghpull:`2765`: Sphinx-gallery integration
* :ghpull:`2788`: Remove NoseTester
* :ghpull:`2768`: BundleWarp, streamline-based nonlinear registration of white matter tracts
* :ghpull:`2749`: adding a new getitem method
* :ghpull:`2744`: Horizon Tabs
* :ghpull:`2785`: EVAC+ workflow
* :ghpull:`2540`: Updates the default value of rm_small_clusters variable in slr_with_qbx function
* :ghpull:`2609`: NF: DKI+ constraints
* :ghpull:`2596`: NF - Parallel Transport Tractography (PTT)
* :ghpull:`2740`: Integration of Denoising Method for DWI with 1D CNN
* :ghpull:`2773`: Including EVAC+ and util function
* :ghpull:`2783`: Fix test_roi_images
* :ghpull:`2782`: [MTN] Fix CI codecov upload
* :ghpull:`2780`: Added option to set Legacy=False in PmfGenDirectionGetter.from_shcoeff
* :ghpull:`2778`: BF: QBX and merge clusters should return streamlines
* :ghpull:`2767`: NF - add utility functions to fast_numpy
* :ghpull:`2626`: Adding Synb0
* :ghpull:`2763`: Update dki.py
* :ghpull:`2751`: [ENH] Asymmetric peak_directions
* :ghpull:`2762`: Remove Python 3.7 from CI
* :ghpull:`2753`: Update adaptive_soft_matching.py
* :ghpull:`2722`: fixed pca for features > samples, and fixed pca_noise_estimate
* :ghpull:`2741`: Fixing solve_qp error
* :ghpull:`2739`: codespell: config, workflow, typos fixed
* :ghpull:`2590`: Fast Streamline Search algorithm implementation
* :ghpull:`2733`: Update SynRegistrationFlow for #2648
* :ghpull:`2723`: TRX integration, requires new attributes for SFT
* :ghpull:`2727`: Fix EXTRAS_REQUIRE
* :ghpull:`2725`: DOC - Update RUMBA-SD data requirement
* :ghpull:`2716`: NF - Added cython utility functions
* :ghpull:`2717`: fixed bug for non-linear fitting with masks
* :ghpull:`2628`: resolve some CI's script typo
* :ghpull:`2713`: Empty vtk support
* :ghpull:`2625`: [Upcoming] Release 1.6.0

Issues (53):

* :ghissue:`2537`: Importing an example in another example - doc
* :ghissue:`1778`: jupyter notebooks from examples
* :ghissue:`720`: Auto-convert the examples into IPython notebooks
* :ghissue:`1990`: [WIP] Sphinx-Gallery integration
* :ghissue:`2765`: Sphinx-gallery integration
* :ghissue:`2788`: Remove NoseTester
* :ghissue:`2768`: BundleWarp, streamline-based nonlinear registration of white matter tracts
* :ghissue:`1073`: Add a method to slice gtab using bvals (Eg : gtab[bvals>200])
* :ghissue:`2749`: adding a new getitem method
* :ghissue:`2744`: Horizon Tabs
* :ghissue:`2785`: EVAC+ workflow
* :ghissue:`2530`: slr_with_qbx breaks when bundle has only one streamline
* :ghissue:`2540`: Updates the default value of rm_small_clusters variable in slr_with_qbx function
* :ghissue:`2609`: NF: DKI+ constraints
* :ghissue:`2596`: NF - Parallel Transport Tractography (PTT)
* :ghissue:`2756`: Remove unused inplace param from gibbs_removal()
* :ghissue:`2754`: [Question] dipy/denoise/gibbs.py
* :ghissue:`2740`: Integration of Denoising Method for DWI with 1D CNN
* :ghissue:`2773`: Including EVAC+ and util function
* :ghissue:`2783`: Fix test_roi_images
* :ghissue:`2782`: [MTN] Fix CI codecov upload
* :ghissue:`2775`: NF - Add option to set Legacy=False in PmfGenDirectionGetter.from_shcoeff(.)
* :ghissue:`2780`: Added option to set Legacy=False in PmfGenDirectionGetter.from_shcoeff
* :ghissue:`2778`: BF: QBX and merge clusters should return streamlines
* :ghissue:`2767`: NF - add utility functions to fast_numpy
* :ghissue:`2626`: Adding Synb0
* :ghissue:`2770`: BF - update viz.py
* :ghissue:`2763`: Update dki.py
* :ghissue:`2751`: [ENH] Asymmetric peak_directions
* :ghissue:`2762`: Remove Python 3.7 from CI
* :ghissue:`2753`: Update adaptive_soft_matching.py
* :ghissue:`2722`: fixed pca for features > samples, and fixed pca_noise_estimate
* :ghissue:`2750`: Adding tests for gradient.py.
* :ghissue:`2741`: Fixing solve_qp error
* :ghissue:`2745`: Dipy Segmentation Core Dumped - Windows.Record
* :ghissue:`2742`: ValueError: slice step cannot be zero
* :ghissue:`2739`: codespell: config, workflow, typos fixed
* :ghissue:`2590`: Fast Streamline Search algorithm implementation
* :ghissue:`2733`: Update SynRegistrationFlow for #2648
* :ghissue:`2723`: TRX integration, requires new attributes for SFT
* :ghissue:`2729`: Numpy Version Incompatibility, AttributeError in dipy.align
* :ghissue:`2726`: Setup broken on Python 3.10.9 setuptools 67.2.0
* :ghissue:`2727`: Fix EXTRAS_REQUIRE
* :ghissue:`2725`: DOC - Update RUMBA-SD data requirement
* :ghissue:`2707`: Fixdenoise
* :ghissue:`2575`: [WIP]  Define curvature and stepsize as default parameter instead of max_angle for tractography
* :ghissue:`2414`: AffineMap.transform with option: interpolation='nearest' returns: "TypeError: No matching signature found"
* :ghissue:`2716`: NF - Added cython utility functions
* :ghissue:`2717`: fixed bug for non-linear fitting with masks
* :ghissue:`2628`: resolve some CI's script typo
* :ghissue:`2713`: Empty vtk support
* :ghissue:`2599`: Support empty ArraySequence in transform_streamlines
* :ghissue:`2625`: [Upcoming] Release 1.6.0
