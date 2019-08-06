.. _release1.0:

====================================
 Release notes for DIPY version 1.0
====================================

GitHub stats for 2019/03/11 - 2019/08/05 (tag: 0.16.0)

These lists are automatically generated, and may be incomplete or contain duplicates.

The following 17 authors contributed 707 commits.

* Adam Richie-Halford
* Antoine Theberge
* Ariel Rokem
* Clint Greene
* Eleftherios Garyfallidis
* Francois Rheault
* Gabriel Girard
* Jean-Christophe Houde
* Jon Haitz Legarreta Gorroño
* Kevin Sitek
* Marc-Alexandre Côté
* Matt Cieslak
* Rafael Neto Henriques
* Scott Trinkle
* Serge Koudoro
* Shreyas Fadnavis


We closed a total of 289 issues, 97 pull requests and 192 regular issues;
this is the full list (generated with the script
:file:`tools/github_stats.py`):

Pull Requests (97):

* :ghpull:`1924`: Some updates in Horizon fixing some issues for upcoming release
* :ghpull:`1946`: Fix empty tractogram loading saving
* :ghpull:`1947`: DOC: fixing examples links
* :ghpull:`1942`: Remove dipy.io.trackvis
* :ghpull:`1917`: A functional implementation of Random matrix local pca.
* :ghpull:`1940`: Increase affine consistency in dipy.tracking.streamlines
* :ghpull:`1909`: [WIP] - MTMS-CSD Tutorial
* :ghpull:`1931`: [BF] IVIM fixes
* :ghpull:`1944`: Update DKI, WMTI, fwDTI examples and give more evidence to WMTI and fwDTI models
* :ghpull:`1939`: Increase affine consistency in dipy.tracking.utils
* :ghpull:`1943`: Increase affine consistency in dipy.tracking.life and dipy.stats.analysis
* :ghpull:`1941`: Remove some viz tutorial
* :ghpull:`1926`: RF - dipy.tracking.local
* :ghpull:`1935`: Remove dipy.external and dipy.fixes packages
* :ghpull:`1903`: Skip some tests on big endian architecture (like s390x)
* :ghpull:`1892`: Use the correct (row) order of the tensor components
* :ghpull:`1804`: BF: added check to avoid infinite loop on consecutive coordinates.
* :ghpull:`1937`: Add a warning about future changes that will happen in dipy.stats.
* :ghpull:`1928`: Update streamlines formats example
* :ghpull:`1925`: FIX: Stateful tractogram examples
* :ghpull:`1927`: BF - move import to top level
* :ghpull:`1923`: [Fix] removing minmax_norm parameter from peak_direction
* :ghpull:`1894`: Default sphere: From symmetric724 to repulsion724
* :ghpull:`1812`: ENH: Statefull tractogram, robust spatial handling and IO
* :ghpull:`1922`: Remove deprecated functions from imaffine
* :ghpull:`1885`: BF - remove single pts streamline
* :ghpull:`1913`: RF - EuDX legacy code/test
* :ghpull:`1915`: Doc generation under Windows
* :ghpull:`1630`: [Fix] remove Userwarning message
* :ghpull:`1896`: New module: dipy.core.interpolation
* :ghpull:`1912`: Remove deprecated parameter voxel_size
* :ghpull:`1916`: Spherical deconvolution model CANNOT be constructed without specifying a response
* :ghpull:`1918`: ENH: Remove unused `warning` package import
* :ghpull:`1881`: DOC - RF tracking examples
* :ghpull:`1911`: Add python_requires
* :ghpull:`1914`: [Fix] vol_idx missing in snr_in_cc Tutorial
* :ghpull:`1907`: DOC: Fix examples documentation generation warnings
* :ghpull:`1908`: DOC: Fix typos
* :ghpull:`1887`: DOC - updated streamline_tools example with the LocalTracking Framework
* :ghpull:`1905`: ENH: Remove deprecated SH bases
* :ghpull:`1849`: Adds control for number of iterations in CSD recon
* :ghpull:`1902`: Warn users if they don't have FURY installed
* :ghpull:`1904`: DOC: Improve documentation
* :ghpull:`1771`: Gibbs removal
* :ghpull:`1899`: Fix: Byte ordering error on Python 3.5
* :ghpull:`1898`: Replace SingleTensor by single_tensor
* :ghpull:`1897`: DOC: Fix typos
* :ghpull:`1893`: Remove scratch folder
* :ghpull:`1891`: Move the tests from test_refine_rb to test_bundles.
* :ghpull:`1888`: BF - fix eudx tracking for npeaks=1
* :ghpull:`1879`: DOC - explicitly run the streamline generators before saving the trk
* :ghpull:`1884`: Clean up: Remove streamlines memory patch
* :ghpull:`1875`: ENH: Add binary tissue classifier option for tracking workflow
* :ghpull:`1882`: DOC - clarified the state of the tracking process once stopped
* :ghpull:`1880`: DOC: Fix typos and improve documentation
* :ghpull:`1878`: Clean up: Remove NUMPY_LESS_0.8.x
* :ghpull:`1877`: Clean up: Remove all SCIPY_LESS_0.x.x
* :ghpull:`1876`: DOC: Fix typos
* :ghpull:`1874`: DOC: Fix documentation oversights.
* :ghpull:`1858`: NF: MSMT - CSD
* :ghpull:`1843`: [NF] new workflow: FetchFlow
* :ghpull:`1866`: MAINT: Drop support for Python 3.4
* :ghpull:`1850`: NF: Add is_hemispherical test
* :ghpull:`1855`: Pin scipy version for bots that need statsmodels.
* :ghpull:`1835`: [Fix]  Workflow mask documentation
* :ghpull:`1836`: Corrected median_otsu function declaration that was breaking tutorials
* :ghpull:`1792`: [NF]: Add seeds to TRK
* :ghpull:`1851`: DOC: Add single-module test/coverage instructions
* :ghpull:`1842`: [Fix] Remove tput from fetcher
* :ghpull:`1800`: Update command line documentation generation
* :ghpull:`1830`: Delete six module
* :ghpull:`1821`: Fixes 238, by requiring vol_idx input with 4D images.
* :ghpull:`1775`: Remove Python 2 dependency.
* :ghpull:`1816`: Remove Deprecated function  dipy.data.get_data
* :ghpull:`1818`: [DOC] fix rank order typo
* :ghpull:`1827`: Remove deprecated module dipy.segment.quickbundes
* :ghpull:`1824`: Remove deprecated module dipy.reconst.peaks
* :ghpull:`1819`: [Fix] Diffeormorphic + CCMetric on small image
* :ghpull:`1823`: Remove accent colormap
* :ghpull:`1814`: [Fix]  add a basic check on dipy_horizon
* :ghpull:`1815`: [FIX] median_otsu deprecated parameter
* :ghpull:`1813`: [Fix] Add Readme for doc generation
* :ghpull:`1766`: NF - add tracking workflow parameters
* :ghpull:`1772`: BF: changes min_signal defaults from 1 to 1e-5
* :ghpull:`1810`: [Bug FIx]  dipy_fit_csa and dipy_fit_csd workflow
* :ghpull:`1806`: Plot both IVIM fits on the same axis
* :ghpull:`1789`: VarPro Fit Example IVIM
* :ghpull:`1770`: Parallel reconst workflows
* :ghpull:`1796`: [Fix] stripping in workflow documentation
* :ghpull:`1795`: [Fix] workflows description
* :ghpull:`1768`: Add afq to stats
* :ghpull:`1788`: Add test for different dtypes
* :ghpull:`1769`: Change "is" check for 'GCV'
* :ghpull:`1767`: BF: self.self
* :ghpull:`1759`: Add one more acknowledgement
* :ghpull:`1230`: Mean Signal DKI
* :ghpull:`1760`: Implements the inverse of decfa

Issues (192):

* :ghissue:`1798`: ploting denoised img
* :ghissue:`1924`: Some updates in Horizon fixing some issues for upcoming release
* :ghissue:`1946`: Fix empty tractogram loading saving
* :ghissue:`1947`: DOC: fixing examples links
* :ghissue:`1942`: Remove dipy.io.trackvis
* :ghissue:`1917`: A functional implementation of Random matrix local pca.
* :ghissue:`1940`: Increase affine consistency in dipy.tracking.streamlines
* :ghissue:`1909`: [WIP] - MTMS-CSD Tutorial
* :ghissue:`1931`: [BF] IVIM fixes
* :ghissue:`1817`: Unusual behavior in Dipy IVIM implementation/example
* :ghissue:`1774`: Split up DKI example
* :ghissue:`1944`: Update DKI, WMTI, fwDTI examples and give more evidence to WMTI and fwDTI models
* :ghissue:`1939`: Increase affine consistency in dipy.tracking.utils
* :ghissue:`1943`: Increase affine consistency in dipy.tracking.life and dipy.stats.analysis
* :ghissue:`1941`: Remove some viz tutorial
* :ghissue:`1926`: RF - dipy.tracking.local
* :ghissue:`1935`: Remove dipy.external and dipy.fixes packages
* :ghissue:`1903`: Skip some tests on big endian architecture (like s390x)
* :ghissue:`1587`: Could tests for functionality not supported on big endians just skip?
* :ghissue:`1890`: Tensor I/O in dipy_fit_dti
* :ghissue:`1892`: Use the correct (row) order of the tensor components
* :ghissue:`1804`: BF: added check to avoid infinite loop on consecutive coordinates.
* :ghissue:`1937`: Add a warning about future changes that will happen in dipy.stats.
* :ghissue:`1933`: Remove deprecated voxel_size from seed_from_mask
* :ghissue:`1928`: Update streamlines formats example
* :ghissue:`985`: Getting started example should be commented at each step
* :ghissue:`1558`: Example of creating Trackvis compatible streamlines is needed
* :ghissue:`1925`: FIX: Stateful tractogram examples
* :ghissue:`1910`: BF: IVIM fixes
* :ghissue:`1927`: BF - move import to top level
* :ghissue:`1923`: [Fix] removing minmax_norm parameter from peak_direction
* :ghissue:`389`: minmax_norm in peaks_directions does nothing
* :ghissue:`1894`: Default sphere: From symmetric724 to repulsion724
* :ghissue:`590`: Change default sphere
* :ghissue:`1722`: Error when using TCK files written by dipy
* :ghissue:`1832`: Tracking workflow header affine issue & fix
* :ghissue:`1812`: ENH: Statefull tractogram, robust spatial handling and IO
* :ghissue:`1922`: Remove deprecated functions from imaffine
* :ghissue:`1885`: BF - remove single pts streamline
* :ghissue:`1913`: RF - EuDX legacy code/test
* :ghissue:`283`: Spherical deconvolution model can be constructed without specifying a response
* :ghissue:`1915`: Doc generation under Windows
* :ghissue:`1630`: [Fix] remove Userwarning message
* :ghissue:`1896`: New module: dipy.core.interpolation
* :ghissue:`728`: Many interpolation functions in different places can they all go to same module?
* :ghissue:`1912`: Remove deprecated parameter voxel_size
* :ghissue:`1920`: How can I get streamlines using fiber orientation by bedpostx of MRtrix3?
* :ghissue:`1432`: DOC/RF - update/standardize tracking examples
* :ghissue:`1779`: Probabilistic Direction Getter gallery example
* :ghissue:`1916`: Spherical deconvolution model CANNOT be constructed without specifying a response
* :ghissue:`1918`: ENH: Remove unused `warning` package import
* :ghissue:`1881`: DOC - RF tracking examples
* :ghissue:`1906`: Add python_requires=">=3.5"
* :ghissue:`1911`: Add python_requires
* :ghissue:`1901`: window.record() function shows the coronal view
* :ghissue:`1914`: [Fix] vol_idx missing in snr_in_cc Tutorial
* :ghissue:`1718`: cannot import name window
* :ghissue:`1747`: CI error that sometimes shows up (Python 2.7)
* :ghissue:`1907`: DOC: Fix examples documentation generation warnings
* :ghissue:`1908`: DOC: Fix typos
* :ghissue:`1887`: DOC - updated streamline_tools example with the LocalTracking Framework
* :ghissue:`1839`: [WIP] IVIM fixes
* :ghissue:`1905`: ENH: Remove deprecated SH bases
* :ghissue:`583`: Make a cython style guide
* :ghissue:`1849`: Adds control for number of iterations in CSD recon
* :ghissue:`1902`: Warn users if they don't have FURY installed
* :ghissue:`1904`: DOC: Improve documentation
* :ghissue:`1694`: Intermittent test failures in `test_streamline`
* :ghissue:`1724`: Failure on Windows/Python 3.5
* :ghissue:`1771`: Gibbs removal
* :ghissue:`1899`: Fix: Byte ordering error on Python 3.5
* :ghissue:`1898`: Replace SingleTensor by single_tensor
* :ghissue:`844`: Refactor behavior of dipy.sims.voxel.single_tensor vs SingleTensor
* :ghissue:`1752`: Intermittent failure on Python 3.4
* :ghissue:`1856`: Figure out how to get a "used by" button
* :ghissue:`1897`: DOC: Fix typos
* :ghissue:`1807`: tracking fails when npeaks=1 for peaks_from_model with tensor model
* :ghissue:`1889`: `segment.bundles` package not being tested
* :ghissue:`1893`: Remove scratch folder
* :ghissue:`1713`: Clean up "scratch"
* :ghissue:`1891`: Move the tests from test_refine_rb to test_bundles.
* :ghissue:`1888`: BF - fix eudx tracking for npeaks=1
* :ghissue:`668`: Add transformation matrix output and input
* :ghissue:`592`: Shouldn't TRACKPOINT be renamed to NODIRECTION?
* :ghissue:`1879`: DOC - explicitly run the streamline generators before saving the trk
* :ghissue:`1884`: Clean up: Remove streamlines memory patch
* :ghissue:`1875`: ENH: Add binary tissue classifier option for tracking workflow
* :ghissue:`1811`: Add binary tissue classifier option for the tracking workflow
* :ghissue:`1846`: streamlines to array
* :ghissue:`1831`: bvec file dimension  prob
* :ghissue:`1882`: DOC - clarified the state of the tracking process once stopped
* :ghissue:`1880`: DOC: Fix typos and improve documentation
* :ghissue:`1857`: point outside data error
* :ghissue:`1878`: Clean up: Remove NUMPY_LESS_0.8.x
* :ghissue:`1877`: Clean up: Remove all SCIPY_LESS_0.x.x
* :ghissue:`1863`: Clean up core.optimize
* :ghissue:`1876`: DOC: Fix typos
* :ghissue:`1874`: DOC: Fix documentation oversights.
* :ghissue:`1781`: [WIP] Random lpca
* :ghissue:`1858`: NF: MSMT - CSD
* :ghissue:`1843`: [NF] new workflow: FetchFlow
* :ghissue:`1869`: get rotation and translation parameters of a rigid transformation
* :ghissue:`1844`: Statsmodels import error
* :ghissue:`1866`: MAINT: Drop support for Python 3.4
* :ghissue:`1865`: Drop Python 3.4?
* :ghissue:`1850`: NF: Add is_hemispherical test
* :ghissue:`1860`: Dependency Graph: Dependents?
* :ghissue:`1855`: Pin scipy version for bots that need statsmodels.
* :ghissue:`1168`: Nf mtms csd model
* :ghissue:`1854`: Testing the CI. DO NOT MERGE
* :ghissue:`1835`: [Fix]  Workflow mask documentation
* :ghissue:`1764`: DTI metrics workflow: mask is optional, but crashes when no mask provided
* :ghissue:`1836`: Corrected median_otsu function declaration that was breaking tutorials
* :ghissue:`1792`: [NF]: Add seeds to TRK
* :ghissue:`1731`: Plan for dropping Python 2 support.
* :ghissue:`1851`: DOC: Add single-module test/coverage instructions
* :ghissue:`1845`: Signal to noise
* :ghissue:`1842`: [Fix] Remove tput from fetcher
* :ghissue:`1829`: When fetching ... 'tput' is not reco...
* :ghissue:`1606`: Cleaned PR for Visualization Modules to Assess the quality of Registration Qualitatively.
* :ghissue:`1837`: labels
* :ghissue:`1786`: Upcoming DIPY lab meetings
* :ghissue:`1828`: IVIM VarPro implementation throws infeasible 'x0'
* :ghissue:`1833`: Affine registration of similar images
* :ghissue:`1834`: Which file to convert  from dicom to nifti?!
* :ghissue:`1800`: Update command line documentation generation
* :ghissue:`1830`: Delete six module
* :ghissue:`1721`: using code style
* :ghissue:`238`: Median_otsu b0slices too implicit?
* :ghissue:`1821`: Fixes 238, by requiring vol_idx input with 4D images.
* :ghissue:`1775`: Remove Python 2 dependency.
* :ghissue:`1816`: Remove Deprecated function  dipy.data.get_data
* :ghissue:`1818`: [DOC] fix rank order typo
* :ghissue:`1499`: Possible mistake about B matrix in documentation "DIY Stuff about b and q"
* :ghissue:`1827`: Remove deprecated module dipy.segment.quickbundes
* :ghissue:`1822`: .trk file
* :ghissue:`1824`: Remove deprecated module dipy.reconst.peaks
* :ghissue:`1825`: Fury visualizing bug - plane only visible for XY-slice of FODs
* :ghissue:`1819`: [Fix] Diffeormorphic + CCMetric on small image
* :ghissue:`1048`: divide by zero error in DiffeomorphicRegistration of small image volumes
* :ghissue:`1823`: Remove accent colormap
* :ghissue:`1797`: function parameters
* :ghissue:`1802`: crossing fibers & fractional anisotropy
* :ghissue:`1787`: RF - change default tracking algorithm for dipy_track_local to EuDX
* :ghissue:`1763`: Threshold default in QballBaseModel
* :ghissue:`1814`: [Fix]  add a basic check on dipy_horizon
* :ghissue:`1756`: Error using dipy_horizon
* :ghissue:`1815`: [FIX] median_otsu deprecated parameter
* :ghissue:`1761`: Deprecation warning when running median_otsu
* :ghissue:`795`: dipy.tracking: Converting an array with ndim > 0 to an index will result in an error
* :ghissue:`620`: Extend the AUTHOR list with more information
* :ghissue:`1813`: [Fix] Add Readme for doc generation
* :ghissue:`436`: Doc won't build without cvxopt
* :ghissue:`1758`: additional parameters for dipy_track_local workflow
* :ghissue:`1766`: NF - add tracking workflow parameters
* :ghissue:`1772`: BF: changes min_signal defaults from 1 to 1e-5
* :ghissue:`1810`: [Bug FIx]  dipy_fit_csa and dipy_fit_csd workflow
* :ghissue:`1808`: `dipy_fit_csd` CLI is broken?
* :ghissue:`1806`: Plot both IVIM fits on the same axis
* :ghissue:`1794`: Removed/renamed DetTrackPAMFlow?
* :ghissue:`1801`: segmentation
* :ghissue:`1803`: tools
* :ghissue:`1809`: datasets
* :ghissue:`1799`: steps from nifiti file to tracts
* :ghissue:`1712`: dipy.reconst.peak_direction_getter.PeaksAndMetricsDirectionGetter.initial_direction (dipy/reconst/peak_direction_getter.c:3075) IndexError: point outside data
* :ghissue:`1789`: VarPro Fit Example IVIM
* :ghissue:`1770`: Parallel reconst workflows
* :ghissue:`1796`: [Fix] stripping in workflow documentation
* :ghissue:`1795`: [Fix] workflows description
* :ghissue:`1768`: Add afq to stats
* :ghissue:`1783`: Make trilinear_interpolate4d work with more dtypes.
* :ghissue:`1784`: Generalize trilinear_interpolate4d to other dtypes.
* :ghissue:`1788`: Add test for different dtypes
* :ghissue:`1790`: ValueError: operands could not be broadcast together with remapped shapes [original->remapped]: (13,13)->(13,13) (10000,10)->(10000,newaxis,10)
* :ghissue:`1782`: Conversion from MRTrix SH basis to dipy
* :ghissue:`1769`: Change "is" check for 'GCV'
* :ghissue:`1320`: WIP: Bias correction
* :ghissue:`1245`: non_local_means : patch size argument for local mean and variance
* :ghissue:`1240`: WIP: Improve the axonal water fraction estimation.
* :ghissue:`1237`: DOC: Flesh out front page example.
* :ghissue:`1192`: Error handling in SDT
* :ghissue:`1096`: Robust Brain Extraction
* :ghissue:`832`: trilinear_interpolate4d only works on float64
* :ghissue:`578`: WIP: try out Stefan Behnel's cython coverage
* :ghissue:`1780`: [WIP]: Randommatrix localpca
* :ghissue:`1022`: Fixes #720 : Auto generate ipython notebooks
* :ghissue:`1126`: Publishing in JOSS : Added paper summary for IVIM
* :ghissue:`1603`: [WIP] - Free water elimination algorithm for single-shell DTI
* :ghissue:`1767`: BF: self.self
* :ghissue:`1759`: Add one more acknowledgement
* :ghissue:`1230`: Mean Signal DKI
* :ghissue:`1760`: Implements the inverse of decfa
