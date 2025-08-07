.. _release1.2:

====================================
 Release notes for DIPY version 1.2
====================================

GitHub stats for 2020/01/10 - 2020/09/08 (tag: 1.1.1)

These lists are automatically generated, and may be incomplete or contain duplicates.

The following 27 authors contributed 491 commits.

* Ariel Rokem
* Aryansh Omray
* Bramsh Qamar
* Charles Poirier
* Derek Pisner
* Eleftherios Garyfallidis
* Fabio Nery
* Francois Rheault
* Gabriel Girard
* Gregory Lee
* Jean-Christophe Houde
* Jirka Borovec
* Jon Haitz Legarreta Gorroño
* Leevi Kerkela
* Leon Weninger
* Martijn Nagtegaal
* Rafael Neto Henriques
* Sarath Chandra
* Serge Koudoro
* Shrishti Hore
* Shubham Shaswat
* Takis Panagopoulos
* Tashrif Billah
* Kunal Mehta
* svabhishek29
* Areesha Tariq
* Philippe Karan

We closed a total of 256 issues, 94 pull requests and 163 regular issues;
this is the full list (generated with the script
:file:`tools/github_stats.py`):

Pull Requests (94):

* :ghpull:`2191`: Multi-shell multi-tissue constrained spherical deconvolution
* :ghpull:`2212`: [FIX] renamed Direct Streamline Normalization tutorial title
* :ghpull:`2207`: [FIX] Horizon shader warning
* :ghpull:`2208`: Removing Python 3.5 from Travis and Azure due to its end of life (2020-09-13)
* :ghpull:`2157`: BF: Fix variable may be used uninitialized warning
* :ghpull:`2205`: Add optional installs for different settings.
* :ghpull:`2204`: from Renderer to scene
* :ghpull:`2183`: _streamlines_in_mask bounds check
* :ghpull:`2203`: End of line period in sft
* :ghpull:`2142`: ENH: function to calculate size/shape parameters of b-tensors and vice-versa
* :ghpull:`2195`: [ENH] Validate streamlines pre-LiFE
* :ghpull:`2161`: Fix a memory overlap bug in multi_median (and median_otsu)
* :ghpull:`2163`: BF: Fix Cython label defined but not used warning
* :ghpull:`2174`: Improve performance of tissue classification
* :ghpull:`2168`: add fig_kwargs
* :ghpull:`2178`: Add full SH basis
* :ghpull:`2193`: BUAN_flow.rst to buan_flow.rst
* :ghpull:`2196`: [Fix] update save_vtk_streamlines and load_vtk_streamlines
* :ghpull:`2188`: [Fix] update mapmri due to cvxpy 1.1
* :ghpull:`2176`: [DOC] Update SH basis documentation
* :ghpull:`2173`: Install ssl certificate for azure pipeline windows
* :ghpull:`2171`: Gitter url update: from nipy/dipy to dipy/dipy
* :ghpull:`2154`: Bundle segmentation CLI tutorial
* :ghpull:`2162`: BF: Fix string literal comparison warning
* :ghpull:`2156`: BF: Fix Cython signed vs. unsigned integer comparison warning
* :ghpull:`2160`: TST: Change assert_equal(statement, True) to assert_(statement).
* :ghpull:`2158`: BF: Fix Cython floating point absolute value warning
* :ghpull:`2155`: [Fix] sfm RuntimeWarning
* :ghpull:`2147`: BF: Fix Cython function  override warning
* :ghpull:`2148`: BF: Fix `distutils` Python version requirement option warning
* :ghpull:`2150`: [Fix] some warning on clustering test
* :ghpull:`2149`: [Fix] some warning on stats module
* :ghpull:`2145`: Rename dipy_track command line
* :ghpull:`2152`: changed buan_flow.rst to BUAN_flow.rst
* :ghpull:`2146`: Cluster threshold parameter in dipy_buan_shapes workflow
* :ghpull:`2134`: Slicing, adding function of StatefulTractogram
* :ghpull:`2001`: Basic processing documentation for CLI.
* :ghpull:`2135`: [Fix] shm.py RuntimeWarning
* :ghpull:`2141`: [FIX] doc generation issue
* :ghpull:`2136`: [Fix]  laplacian_regularization on MAPMRI + cleanup warning
* :ghpull:`2140`: BF: Fix NumPy warning when creating arrays from ragged sequences
* :ghpull:`2139`: BF: Use equality check instead of identity check
* :ghpull:`2108`: [Horizon] Update clipping range on slicer
* :ghpull:`2121`: BF: ensure `btens` attribute of `GradientTable` is initialised
* :ghpull:`2129`: BF: Fix sequence stacking warning in LiFE tracking
* :ghpull:`2133`: BF: Fix NumPy warning when creating arrays from ragged sequences
* :ghpull:`2125`: ENH: function to calculate anisotropy of b-tensors
* :ghpull:`2124`: BUAN framework documentation
* :ghpull:`2033`: RF - Direction getters naming
* :ghpull:`2111`: Added an epsilon to bounding_box check
* :ghpull:`2086`: WIP Issue 1996
* :ghpull:`2091`: Modified the model for multiple hidden layers
* :ghpull:`2057`: DOC: Add DIPY dataset list to documentation.
* :ghpull:`2103`: Documentation typos, grammar corrections & imports
* :ghpull:`2088`: BUAN paper code and CLIs
* :ghpull:`2120`: rename var sp to sph
* :ghpull:`2113`: BF: refer to cigar_tensor
* :ghpull:`2116`: fixed code tags and minor changes
* :ghpull:`2100`: Fixed typos, grammatical errors and time import
* :ghpull:`2101`: Minor typos and imports
* :ghpull:`2095`: Fixed typos in dipy.align
* :ghpull:`2099`: Minor Typos and imports in the beginning
* :ghpull:`2102`: Modules imported in the beginning
* :ghpull:`2055`: Multidimensional gradient table
* :ghpull:`2097`: Replace manual sform values with get_best_affine
* :ghpull:`2104`: Fixed minor typos
* :ghpull:`2065`: Some typos and grammatical errors
* :ghpull:`2090`: Small fix reorient_bvecs
* :ghpull:`2067`: Some spelling and grammatical mistakes
* :ghpull:`2093`: Placed all imports in the beginning
* :ghpull:`2077`: Fixed minor typos in tutorials
* :ghpull:`2071`: Some change backs
* :ghpull:`2084`: Kunakl07 patch 7
* :ghpull:`2085`: Kunakl07 patch 8
* :ghpull:`2068`: Some spelling and grammatical errors
* :ghpull:`2069`: some typos
* :ghpull:`2063`: Gibbs tutorial patch
* :ghpull:`2045`: [Fix] workflow variable string
* :ghpull:`2060`: Replace old function with cython version
* :ghpull:`2058`: DOC: Fix `Sphinx` link in `CONTRIBUTING.md`
* :ghpull:`2059`: DOC: Add `Azure Pipelines` to CI tools in `CONTRIBUTING.md`
* :ghpull:`2056`: MAINT: Up Numpy version to 1.12.
* :ghpull:`2053`: Correct TYPO on the note about n_points in _gibbs_removal_2d()
* :ghpull:`2043`: [NF] Add a Deprecation system
* :ghpull:`2047`: [fix] Doc generation issue
* :ghpull:`2044`: [FIX] check seeds dtype
* :ghpull:`2041`: BF: SFM prediction with mask
* :ghpull:`2039`: Remove __future__
* :ghpull:`2042`: Add tests to rng module
* :ghpull:`2040`: RF: Swallow a couple of warnings that are safe to ignore.
* :ghpull:`2038`: [DOC] Update repo path
* :ghpull:`2037`: DOC: fix typo in FW DTI tutorial
* :ghpull:`2028`: Adapted for patch_radius with radii differing xyz direction.
* :ghpull:`2035`: DOC: Update DKI documentation according to the new get data functions

Issues (164):

* :ghissue:`2114`: Tutorial title in Streamline-based registration section is misleading
* :ghissue:`2207`: [FIX] Horizon shader warning
* :ghissue:`2208`: Removing Python 3.5 from Travis and Azure due to its end of life (2020-09-13)
* :ghissue:`1793`: ENH: Improving Command Line and WF Docs
* :ghissue:`2007`: Reference for `load_trk` in recobundles example
* :ghissue:`2061`: Is dipy.viz supported in Colab or Kaggle Notebooks?
* :ghissue:`2070`: Python has stopped working during import
* :ghissue:`2107`: PNG images saved by window.record in the tutorial example are always black
* :ghissue:`2153`: lowercase .rst file names
* :ghissue:`2138`: Basic introduction to CLI needs better dataset to showcase capabilities
* :ghissue:`2194`: LiFE  model won't fit ?
* :ghissue:`2157`: BF: Fix variable may be used uninitialized warning
* :ghissue:`2177`: VIZ: streamline actor failing on Windows + macOS due to the new VTK9
* :ghissue:`2205`: Add optional installs for different settings.
* :ghissue:`2204`: from Renderer to scene
* :ghissue:`2183`: _streamlines_in_mask bounds check
* :ghissue:`2182`: `target_line_based` might read out of bounds
* :ghissue:`2203`: End of line period in sft
* :ghissue:`2200`: BF: Fix `'tp_print' is deprecated` Cython warning
* :ghissue:`2142`: ENH: function to calculate size/shape parameters of b-tensors and vice-versa
* :ghissue:`2199`: BUG: Fix NumPy and Cython deprecation and initialization warnings
* :ghissue:`2195`: [ENH] Validate streamlines pre-LiFE
* :ghissue:`2161`: Fix a memory overlap bug in multi_median (and median_otsu)
* :ghissue:`2163`: BF: Fix Cython label defined but not used warning
* :ghissue:`2174`: Improve performance of tissue classification
* :ghissue:`2168`: add fig_kwargs
* :ghissue:`2178`: Add full SH basis
* :ghissue:`2193`: BUAN_flow.rst to buan_flow.rst
* :ghissue:`2196`: [Fix] update save_vtk_streamlines and load_vtk_streamlines
* :ghissue:`2175`: Save streamlines as vtk polydata to a supported format file updated t…
* :ghissue:`2188`: [Fix] update mapmri due to cvxpy 1.1
* :ghissue:`2190`: Reconstruction with Multi-Shell Multi-Tissue CSD
* :ghissue:`2051`: BF: Non-negative Least Squares for IVIM
* :ghissue:`2176`: [DOC] Update SH basis documentation
* :ghissue:`2173`: Install ssl certificate for azure pipeline windows
* :ghissue:`2172`: fetch_gold_standard_io fetcher failed regularly
* :ghissue:`2169`: saving tracts in obj format
* :ghissue:`2170`: Output of utils.density_map() using tck file is different than MRTrix
* :ghissue:`2171`: Gitter url update: from nipy/dipy to dipy/dipy
* :ghissue:`2144`: Move gitter to dipy/dipy?
* :ghissue:`2154`: Bundle segmentation CLI tutorial
* :ghissue:`2162`: BF: Fix string literal comparison warning
* :ghissue:`2156`: BF: Fix Cython signed vs. unsigned integer comparison warning
* :ghissue:`2160`: TST: Change assert_equal(statement, True) to assert_(statement).
* :ghissue:`2158`: BF: Fix Cython floating point absolute value warning
* :ghissue:`2155`: [Fix] sfm RuntimeWarning
* :ghissue:`2159`: BF: Fix Cython different sign integer comparison warning
* :ghissue:`2147`: BF: Fix Cython function  override warning
* :ghissue:`2148`: BF: Fix `distutils` Python version requirement option warning
* :ghissue:`2151`: [DOC] Fixed minor typos, grammar errors and moved imports up in all examples
* :ghissue:`2130`: Checking empty Cluster objects generates NumPy warning
* :ghissue:`2131`: Elementwise comparison failure warning in multi_voxel test
* :ghissue:`2150`: [Fix] some warning on clustering test
* :ghissue:`2149`: [Fix] some warning on stats module
* :ghissue:`2145`: Rename dipy_track command line
* :ghissue:`2152`: changed buan_flow.rst to BUAN_flow.rst
* :ghissue:`2146`: Cluster threshold parameter in dipy_buan_shapes workflow
* :ghissue:`2128`: Registration Module failing with pre-matrix on Travis with future Numpy/Scipy release
* :ghissue:`2134`: Slicing, adding function of StatefulTractogram
* :ghissue:`2001`: Basic processing documentation for CLI.
* :ghissue:`2135`: [Fix] shm.py RuntimeWarning
* :ghissue:`2141`: [FIX] doc generation issue
* :ghissue:`2136`: [Fix]  laplacian_regularization on MAPMRI + cleanup warning
* :ghissue:`1765`: Refactor  dipy/stats/analysis.py
* :ghissue:`2122`: [WIP] Add build template
* :ghissue:`2140`: BF: Fix NumPy warning when creating arrays from ragged sequences
* :ghissue:`2139`: BF: Use equality check instead of identity check
* :ghissue:`2127`: DOC : Minor grammar fixes and moved imports up with respective comments
* :ghissue:`2108`: [Horizon] Update clipping range on slicer
* :ghissue:`2121`: BF: ensure `btens` attribute of `GradientTable` is initialised
* :ghissue:`2129`: BF: Fix sequence stacking warning in LiFE tracking
* :ghissue:`2133`: BF: Fix NumPy warning when creating arrays from ragged sequences
* :ghissue:`2125`: ENH: function to calculate anisotropy of b-tensors
* :ghissue:`2124`: BUAN framework documentation
* :ghissue:`2126`: dipy / fury fails to install on Ubuntu 18 with pip3
* :ghissue:`2033`: RF - Direction getters naming
* :ghissue:`2111`: Added an epsilon to bounding_box check
* :ghissue:`2112`: [WIP] BUndle ANalytics (BUAN) pipeline documentation
* :ghissue:`2086`: WIP Issue 1996
* :ghissue:`2091`: Modified the model for multiple hidden layers
* :ghissue:`2096`: Deep Code
* :ghissue:`2057`: DOC: Add DIPY dataset list to documentation.
* :ghissue:`2103`: Documentation typos, grammar corrections & imports
* :ghissue:`2088`: BUAN paper code and CLIs
* :ghissue:`2120`: rename var sp to sph
* :ghissue:`2118`: Local namespace of Scipy is same as a variable name
* :ghissue:`1861`: WIP: Refactoring the stats module
* :ghissue:`2113`: BF: refer to cigar_tensor
* :ghissue:`2116`: fixed code tags and minor changes
* :ghissue:`2024`: DIPY open lab meetings, Winter 2020
* :ghissue:`2100`: Fixed typos, grammatical errors and time import
* :ghissue:`2101`: Minor typos and imports
* :ghissue:`2094`: Detailed Beginner Friendly Tutorials
* :ghissue:`2095`: Fixed typos in dipy.align
* :ghissue:`2099`: Minor Typos and imports in the beginning
* :ghissue:`2102`: Modules imported in the beginning
* :ghissue:`2055`: Multidimensional gradient table
* :ghissue:`2097`: Replace manual sform values with get_best_affine
* :ghissue:`2105`: Tutorial Symmetric Regn 3D patch 3
* :ghissue:`2104`: Fixed minor typos
* :ghissue:`2078`: Applying affine transform to streamlines in a SFT object
* :ghissue:`2065`: Some typos and grammatical errors
* :ghissue:`1305`: Questions/policies about writing .rst/web doc files
* :ghissue:`2090`: Small fix reorient_bvecs
* :ghissue:`2067`: Some spelling and grammatical mistakes
* :ghissue:`2093`: Placed all imports in the beginning
* :ghissue:`2077`: Fixed minor typos in tutorials
* :ghissue:`2089`: Transforming bvecs after registration
* :ghissue:`2071`: Some change backs
* :ghissue:`2084`: Kunakl07 patch 7
* :ghissue:`2085`: Kunakl07 patch 8
* :ghissue:`2072`: Some typos and grammatical errors in faq.rst
* :ghissue:`2073`: Some minor grammatical fixes old_highlights.txt
* :ghissue:`2074`: Small typos
* :ghissue:`2075`: Some grammatical changes in maintainer_workflow.rst
* :ghissue:`2076`: Some grammatical changes in maintainer_workflow.rst
* :ghissue:`2079`: Some minor typos in gimbal_lock.rst
* :ghissue:`2080`: Some minor grammatical errors fixes
* :ghissue:`2081`: Some typos and grammatical corrections in Changelog
* :ghissue:`2082`: Grammatical fixes in readme.rst
* :ghissue:`2083`: Fixes in regtools.py
* :ghissue:`2066`: Fixed some typos
* :ghissue:`2068`: Some spelling and grammatical errors
* :ghissue:`2069`: some typos
* :ghissue:`2063`: Gibbs tutorial patch
* :ghissue:`2045`: [Fix] workflow variable string
* :ghissue:`2060`: Replace old function with cython version
* :ghissue:`2058`: DOC: Fix `Sphinx` link in `CONTRIBUTING.md`
* :ghissue:`2059`: DOC: Add `Azure Pipelines` to CI tools in `CONTRIBUTING.md`
* :ghissue:`1363`: MDF not working properly
* :ghissue:`2056`: MAINT: Up Numpy version to 1.12.
* :ghissue:`1871`: apply transformation to all volumes in a series of DWI (4D)
* :ghissue:`2052`: dipy.tracking.utils.density_map order of arguments changed by mistake
* :ghissue:`1785`: Could we use gifti for streamlines?
* :ghissue:`1728`: Bug in shm_coeff computation?
* :ghissue:`1699`: Details in aparc-reduced.nii.gz
* :ghissue:`1671`: Question about shm_coeff
* :ghissue:`1552`: dti.py - quantize_evecs - error
* :ghissue:`1373`: How to convert from converted isotropic to original resolution (anisotropic)
* :ghissue:`1364`: SNR estimation troubleshooting
* :ghissue:`1152`: nan gfa and odf values when mask includes voxels with 0 dwi signal
* :ghissue:`1047`: Gradient flipped in the x-direction - FSL bvecs handling
* :ghissue:`2019`: Apply deformation map to render "registered" image
* :ghissue:`2049`: KFA calculation
* :ghissue:`2048`: Group analysis
* :ghissue:`2053`: Correct TYPO on the note about n_points in _gibbs_removal_2d()
* :ghissue:`2043`: [NF] Add a Deprecation system
* :ghissue:`218`: Callable response broken in csd module
* :ghissue:`2047`: [fix] Doc generation issue
* :ghissue:`313`: csdeconv response as callable
* :ghissue:`1848`: Add Dipy to MRI-Hub (ISMRM Reproducible Research Study Group)
* :ghissue:`2044`: [FIX] check seeds dtype
* :ghissue:`2034`: Using the tutorial on Euler method on my data
* :ghissue:`2041`: BF: SFM prediction with mask
* :ghissue:`1724`: Failure on Windows/Python 3.5
* :ghissue:`1938`: Auto-clearing the AppVeyor queue backlog
* :ghissue:`2039`: Remove __future__
* :ghissue:`2042`: Add tests to rng module
* :ghissue:`1864`: Add tests to dipy.core.rng
* :ghissue:`2040`: RF: Swallow a couple of warnings that are safe to ignore.
* :ghissue:`2038`: [DOC] Update repo path
* :ghissue:`2037`: DOC: fix typo in FW DTI tutorial
* :ghissue:`2028`: Adapted for patch_radius with radii differing xyz direction.
* :ghissue:`2035`: DOC: Update DKI documentation according to the new get data functions
