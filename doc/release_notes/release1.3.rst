.. _release1.3:

====================================
 Release notes for DIPY version 1.3
====================================

GitHub stats for 2020/09/09 - 2020/11/02 (tag: 1.2.0)


The following 14 authors contributed 284 commits.

* Areesha Tariq
* Ariel Rokem
* Basile Pinsard
* Bramsh Qamar
* Charles Poirier
* Eleftherios Garyfallidis
* Eric Larson
* Gregory Lee
* Jaewon Chung
* Jon Haitz Legarreta Gorroño
* Philippe Karan
* Rafael Neto Henriques
* Serge Koudoro
* Siddharth Kapoor


We closed a total of 134 issues, 49 pull requests and 85 regular issues;
this is the full list (generated with the script
:file:`tools/github_stats.py`):

Pull Requests (49):

* :ghpull:`2181`: BUAN bundle highlight option in horizon
* :ghpull:`2223`: [NF] new linear transforms for Rigid+IsoScaling and Rigid+Scaling
* :ghpull:`2238`: [FIX] fix cython error from pre matrix
* :ghpull:`2265`: Gibbs denoising: fft only along the axis of interest
* :ghpull:`2206`: NF: Update definitions of SH bases and documentation
* :ghpull:`2266`: STYLE: minor refactoring in TissueClassifierHMRF
* :ghpull:`2255`: Modifying dti.design_matrix to take gtab.btens into account
* :ghpull:`2271`: Increase Azure pipeline timeout
* :ghpull:`2263`: [FIX] update multiple models due to cvxpy 1.1 (part2)
* :ghpull:`2259`: [Fix]  Allow read_bvals_bvecs to have 1 or 2 dwi volumes only
* :ghpull:`2264`: BF: Fix `dipy_align_syn` default value assumptions
* :ghpull:`2268`: BUG: Fix literal
* :ghpull:`2267`: BUG: Fix string literal
* :ghpull:`2262`: [FIX] update tests to respect numpy NEP 34
* :ghpull:`2244`: DOC : Denoising CLI
* :ghpull:`2119`: RecoBundles updated to read and save .trk files from Old API
* :ghpull:`2260`: [Fix] Better error handling in Diffeomorphic map `get_map`
* :ghpull:`2258`: [FIX] update Azure OSX CI + remove Azure Linux CI's
* :ghpull:`2257`: [Fix] warning if not the same number of points
* :ghpull:`2261`: [DOC]:Removed tracking evaluation section
* :ghpull:`1919`: [DOC] Add an overview of reconstruction models
* :ghpull:`2256`: update BUAN citations
* :ghpull:`2253`: Improve FFT efficiency in gibbs_removal
* :ghpull:`2240`: [ENH] Deprecate interp parameter name in AffineMap
* :ghpull:`2198`: Make single and multi tensor simulations compatible with btensors
* :ghpull:`2025`: Adds an align.api module, which provides simplified API to registration functions
* :ghpull:`2197`: Estimate smt2 metrics from mean signal kurtosis
* :ghpull:`2227`: RF: Replaces our own custom progressbar with a tqdm progressbar.
* :ghpull:`2250`: [ENH] Add parallelization to gibbs denoising
* :ghpull:`2252`: BUG: Set tau factor to parameter value in local PCA
* :ghpull:`2248`: [DOC] fetching dataset
* :ghpull:`2249`: [fix] fix value_range in HORIZON
* :ghpull:`2247`: BF: In  LiFE, set nan signals to 0.
* :ghpull:`2246`: [DOC] Replace simple backticks with double backticks
* :ghpull:`2239`: [ENH] Add inplace kwarg to gibbs_removal
* :ghpull:`2242`: maintenance of bundle_shape_similarity function
* :ghpull:`2241`: STYLE: Exclude package information file from PEP8 checks
* :ghpull:`2235`: DOC: Add tips to the documentation build section
* :ghpull:`2234`: DOC: Improve some of the links in the `info.py` file
* :ghpull:`2233`: Clarifying msmt response function docstrings
* :ghpull:`2231`: DOC: Fix HTML tag in dataset list documentation table
* :ghpull:`2221`: Robustify solve_qp for possible SolverError in one odd voxel
* :ghpull:`2226`: STYLE: Conform to `reStructuredText` syntax in examples sections
* :ghpull:`2225`: [CI] Replace Rackspace by https://anaconda.org/scipy-wheels-nightly
* :ghpull:`2224`: Replace pytest.xfail by npt.assert_raises
* :ghpull:`2220`: [DOC] move Denoising on its own section
* :ghpull:`2218`: DOC : inconsistent save_seeds documentation
* :ghpull:`2217`: Fixing numpy version rcond issue in numpy.linalg.lstsq
* :ghpull:`2211`: [FIX] used numerical indices for references

Issues (85):

* :ghissue:`2181`: BUAN bundle highlight option in horizon
* :ghissue:`2272`: DOC : Registration CLI
* :ghissue:`2223`: [NF] new linear transforms for Rigid+IsoScaling and Rigid+Scaling
* :ghissue:`2180`: [NF] add new linear transforms for Rigid+IsoScaling and Rigid+Scaling
* :ghissue:`2238`: [FIX] fix cython error from pre matrix
* :ghissue:`2265`: Gibbs denoising: fft only along the axis of interest
* :ghissue:`2206`: NF: Update definitions of SH bases and documentation
* :ghissue:`392`: mrtrix 0.3 default basis is different from mrtrix 0.2
* :ghissue:`2266`: STYLE: minor refactoring in TissueClassifierHMRF
* :ghissue:`2255`: Modifying dti.design_matrix to take gtab.btens into account
* :ghissue:`2271`: Increase Azure pipeline timeout
* :ghissue:`2054`: Discrepancy between dipy.gibbs.gibbs_removal and reisert/unring/
* :ghissue:`2263`: [FIX] update multiple models due to cvxpy 1.1 (part2)
* :ghissue:`2190`: Reconstruction with Multi-Shell Multi-Tissue CSD
* :ghissue:`2259`: [Fix]  Allow read_bvals_bvecs to have 1 or 2 dwi volumes only
* :ghissue:`2046`: read_bvals_bvecs can't read a single volume dwi
* :ghissue:`2264`: BF: Fix `dipy_align_syn` default value assumptions
* :ghissue:`2268`: BUG: Fix literal
* :ghissue:`2267`: BUG: Fix string literal
* :ghissue:`2262`: [FIX] update tests to respect numpy NEP 34
* :ghissue:`2132`: Generating ndarrays with different shapes generates NumPy warning at testing
* :ghissue:`1266`: test_mapmri_isotropic_static_scale_factor failure on OSX buildbot
* :ghissue:`1264`: FBC measures test failure on PPC
* :ghissue:`2244`: DOC : Denoising CLI
* :ghissue:`2119`: RecoBundles updated to read and save .trk files from Old API
* :ghissue:`2117`: RecoBundles workflow still using old API
* :ghissue:`2260`: [Fix] Better error handling in Diffeomorphic map `get_map`
* :ghissue:`2202`: Add error handling in Diffeomorphic map `get_map`
* :ghissue:`2258`: [FIX] update Azure OSX CI + remove Azure Linux CI's
* :ghissue:`2257`: [Fix] warning if not the same number of points
* :ghissue:`342`: Missing a warning if not the same number of points
* :ghissue:`2261`: [DOC]:Removed tracking evaluation section
* :ghissue:`2115`: Independent section on Fiber tracking evaluation not necessary
* :ghissue:`1744`: [WIP] [NF] Free Water Elimination for single-shell DTI (updated version)
* :ghissue:`1919`: [DOC] Add an overview of reconstruction models
* :ghissue:`1489`: Documentation: how to know which models support multi-shell?
* :ghissue:`2256`: update BUAN citations
* :ghissue:`2253`: Improve FFT efficiency in gibbs_removal
* :ghissue:`2240`: [ENH] Deprecate interp parameter name in AffineMap
* :ghissue:`2192`: Bringing AffineMap and DiffeomorphicMap a little closer together
* :ghissue:`2198`: Make single and multi tensor simulations compatible with btensors
* :ghissue:`2025`: Adds an align.api module, which provides simplified API to registration functions
* :ghissue:`2201`: Gradient table message error
* :ghissue:`2232`: This should be len(np.unique(gtab.bvals)) - 1 or somesuch
* :ghissue:`2197`: Estimate smt2 metrics from mean signal kurtosis
* :ghissue:`2227`: RF: Replaces our own custom progressbar with a tqdm progressbar.
* :ghissue:`2219`: Replace fetcher progress bar with tqdm
* :ghissue:`2250`: [ENH] Add parallelization to gibbs denoising
* :ghissue:`2236`: Parallelize gibbs_removal
* :ghissue:`2254`: Trackvis header saved with Dipy (nibabel) is not read  by Matlab or other tools
* :ghissue:`2252`: BUG: Set tau factor to parameter value in local PCA
* :ghissue:`2251`: localpca tau_factor is hard coded to 2.3
* :ghissue:`2248`: [DOC] fetching dataset
* :ghissue:`2249`: [fix] fix value_range in HORIZON
* :ghissue:`2243`: Unable to visualize data through dipy_horizon
* :ghissue:`2247`: BF: In  LiFE, set nan signals to 0.
* :ghissue:`2246`: [DOC] Replace simple backticks with double backticks
* :ghissue:`2239`: [ENH] Add inplace kwarg to gibbs_removal
* :ghissue:`2237`: gibbs_removal overwrites input data when inputting 3-d or 4-d data.
* :ghissue:`2245`: DOC: Fix Sphinx verbatim syntax in coding style guide
* :ghissue:`2242`: maintenance of bundle_shape_similarity function
* :ghissue:`2241`: STYLE: Exclude package information file from PEP8 checks
* :ghissue:`2235`: DOC: Add tips to the documentation build section
* :ghissue:`2234`: DOC: Improve some of the links in the `info.py` file
* :ghissue:`2222`: How can I track different streamlines in DIPY?
* :ghissue:`2233`: Clarifying msmt response function docstrings
* :ghissue:`2231`: DOC: Fix HTML tag in dataset list documentation table
* :ghissue:`2230`: TST: Assert the shape of the output based on the docstring.
* :ghissue:`2228`: Best practices for saving a tissue classifier object?
* :ghissue:`2221`: Robustify solve_qp for possible SolverError in one odd voxel
* :ghissue:`2109`: DIPY lab meetings, Spring 2020
* :ghissue:`2226`: STYLE: Conform to `reStructuredText` syntax in examples sections
* :ghissue:`2225`: [CI] Replace Rackspace by https://anaconda.org/scipy-wheels-nightly
* :ghissue:`2214`: Rackspace does not exist anymore -> Update PRE-matrix on Travis and Azure required
* :ghissue:`2224`: Replace pytest.xfail by npt.assert_raises
* :ghissue:`2220`: [DOC] move Denoising on its own section
* :ghissue:`2218`: DOC : inconsistent save_seeds documentation
* :ghissue:`2217`: Fixing numpy version rcond issue in numpy.linalg.lstsq
* :ghissue:`2216`: test_multi_shell_fiber_response failed with Numpy 1.13.3
* :ghissue:`2211`: [FIX] used numerical indices for references
* :ghissue:`2185`: Inconsistency in stating references in dti.py
* :ghissue:`2215`: problem with fetching stanford data
* :ghissue:`1762`: Font on instructions is small on mac
* :ghissue:`1354`: strange tracks
* :ghissue:`325`: streamline extraction from eudx is failing - but perhaps eudx is failing
