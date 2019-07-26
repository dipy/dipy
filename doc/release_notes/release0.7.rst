.. _release0.7:

===================================
 Release notes for DIPY version 0.7
===================================

GitHub stats for 2013/03/29 - 2013/12/23 (tag: 0.6.0)

The following 16 authors contributed 814 commits.

* Ariel Rokem
* Bago Amirbekian
* Eleftherios Garyfallidis
* Emmanuel Caruyer
* Erik Ziegler
* Gabriel Girard
* Jean-Christophe Houde
* Kimberly Chan
* Matthew Brett
* Matthias Ekman
* Matthieu Dumont
* Mauro Zucchelli
* Maxime Descoteaux
* Samuel St-Jean
* Stefan van der Walt
* Sylvain Merlet


We closed a total of 84 pull requests; this is the full list (generated with the script
:file:`tools/github_stats.py`):

Pull Requests (84):

* :ghpull:`292`: Streamline tools
* :ghpull:`289`: Examples checked for peaks_from_model
* :ghpull:`288`: Link shore examples
* :ghpull:`279`: Update release 0.7 examples' system
* :ghpull:`257`: Continuous modelling: SHORE
* :ghpull:`285`: Bad seeds cause segfault in EuDX
* :ghpull:`274`: Peak directions update
* :ghpull:`275`: Restore example
* :ghpull:`261`: R2 term response function for Sharpening Deconvolution Transform (SDT)
* :ghpull:`273`: Fixed typos + autopep8
* :ghpull:`268`: Add gfa shmfit
* :ghpull:`260`: NF: Command line interface to QuickBundles.
* :ghpull:`270`: Removed minmax_normalize from dipy.reconst.peaks
* :ghpull:`247`: Model base
* :ghpull:`267`: Refactoring peaks_from_model_parallel
* :ghpull:`219`: Update forward sdeconv mat
* :ghpull:`266`: BF - join pool before trying to delete temp directory
* :ghpull:`265`: Peak from model issue #253
* :ghpull:`264`: peak_from_model tmp files
* :ghpull:`263`: Refactoring peaks calculations to be out of odf.py
* :ghpull:`262`: Handle cpu count exception
* :ghpull:`255`: Fix peaks_from_model_parallel
* :ghpull:`259`: Release 0.7 a few cleanups
* :ghpull:`252`: Clean cc
* :ghpull:`243`: NF Added norm input to interp_rbf and angle as an alternative norm.
* :ghpull:`251`: Another cleanup for fvtk. This time the slicer function was simplified
* :ghpull:`249`: Dsi metrics 2
* :ghpull:`239`: Segmentation based on rgb threshold + examples
* :ghpull:`240`: Dsi metrics
* :ghpull:`245`: Fix some rewording
* :ghpull:`242`: A new streamtube visualization method and different fixes and cleanups for the fvtk module
* :ghpull:`237`: WIP: cleanup docs / small refactor for median otsu
* :ghpull:`221`: peaks_from_model now return peaks directions
* :ghpull:`234`: BF: predict for cases when the ADC is multi-D and S0 is provided as a volume
* :ghpull:`232`: Fix peak extraction default value of relative_peak_threshold
* :ghpull:`227`: Fix closing upon download completion in fetcher
* :ghpull:`230`: Tensor predict
* :ghpull:`229`: BF: input.dtype is used per default
* :ghpull:`210`: Brainextraction
* :ghpull:`226`: SetInput in vtk5 is now SetInputData in vtk6
* :ghpull:`225`: fixed typo
* :ghpull:`212`: Tensor visualization
* :ghpull:`223`: Fix make examples for windows.
* :ghpull:`222`: Fix restore bug
* :ghpull:`217`: RF - update csdeconv to use SphHarmFit class to reduce code duplication.
* :ghpull:`208`: Shm coefficients in peaks_from_model
* :ghpull:`216`: BF - fixed mask_voxel_size bug and added test. Replaced promote_dtype wi...
* :ghpull:`211`: Added a md5 check to each dataset.
* :ghpull:`54`: Restore
* :ghpull:`213`: Update to a more recent version of `six.py`.
* :ghpull:`204`: Maxime's [Gallery] Reconst DTI example revisited
* :ghpull:`207`: Added two new datasets online and updated fetcher.py.
* :ghpull:`209`: Fixed typos in reconst/dti.py
* :ghpull:`206`: DOC: update the docs to say that we support python 3
* :ghpull:`205`: RF: Minor corrections in index.rst and CSD example
* :ghpull:`173`: Constrained Spherical Deconvolution and the Spherical Deconvolution Transform
* :ghpull:`203`: RF: Rename tensor statistics to remove "tensor\_" from them.
* :ghpull:`202`: Typos
* :ghpull:`201`: Bago's Rename sph basis functions corrected after rebasing and other minor lateral fixes
* :ghpull:`191`: DOC - clarify docs for SphHarmModel
* :ghpull:`199`: FIX: testfail due to Non-ASCII character \xe2 in markov.py
* :ghpull:`189`: Shm small fixes
* :ghpull:`196`: DOC: add reference section to ProbabilisticOdfWeightedTracker
* :ghpull:`190`: BF - fix fit-tensor handling of file extensions and mask=none
* :ghpull:`182`: RF - fix disperse_charges so that a large constant does not cause the po...
* :ghpull:`183`: OPT: Modified dipy.core.sphere_stats.random_uniform_on_sphere, cf issue #181
* :ghpull:`185`: DOC: replace soureforge.net links with nipy.org
* :ghpull:`180`: BF: fix Cython TypeError from negative indices to tuples
* :ghpull:`179`: BF: doctest output difference workarounds
* :ghpull:`176`: MRG: Py3 compat
* :ghpull:`178`: RF: This function is superseded by read_bvals_bvecs.
* :ghpull:`170`: Westin stats
* :ghpull:`174`: RF: use $PYTHON variable for python invocation
* :ghpull:`172`: DOC: Updated index.rst and refactored example segment_quickbundles.py
* :ghpull:`169`: RF: refactor pyx / c file stamping for packaging
* :ghpull:`168`: DOC: more updates to release notes
* :ghpull:`167`: Merge maint
* :ghpull:`166`: BF: pyc and created trk files were in eg archive
* :ghpull:`160`: NF: add script to build dmgs from buildbot mpkgs
* :ghpull:`164`: Calculation for mode of a tensor
* :ghpull:`163`: Remove dti tensor
* :ghpull:`161`: DOC: typo in the probabilistic tracking example.
* :ghpull:`162`: DOC: update release notes
* :ghpull:`159`: Rename install test scripts
