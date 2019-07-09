.. _release0.11:

====================================
 Release notes for DIPY version 0.11
====================================

GitHub stats for 2015/12/03 - 2016/02/21 (tag: 0.10)

The following 16 authors contributed 271 commits.

* Ariel Rokem
* Bago Amirbekian
* Bishakh Ghosh
* Eleftherios Garyfallidis
* Gabriel Girard
* Gregory R. Lee
* Himanshu Mishra
* Jean-Christophe Houde
* Marc-Alexandre Côté
* Matthew Brett
* Matthieu Dumont
* Omar Ocegueda
* Sagun Pai
* Samuel St-Jean
* Stephan Meesters
* Vatsala Swaroop


We closed a total of 144 issues, 55 pull requests and 89 regular issues;
this is the full list (generated with the script 
:file:`tools/github_stats.py`):

Pull Requests (55):

* :ghpull:`933`: Updating release dates
* :ghpull:`925`: fix typos
* :ghpull:`915`: BF: correct handling of output paths in dipy_quickbundles.
* :ghpull:`922`: Fix PEP8 in top-level tests
* :ghpull:`921`: fix typo
* :ghpull:`918`: Fix PEP8 in test_expectmax
* :ghpull:`917`: Website 0.11 update and more devs
* :ghpull:`916`: Getting website ready for 0.11 release
* :ghpull:`914`: DOC: Update release notes for 0.11
* :ghpull:`910`: Singleton sl vals
* :ghpull:`908`: Fix pep8 errors in viz
* :ghpull:`911`: fix typo
* :ghpull:`904`: fix typo
* :ghpull:`851`: Tissue Classifier tracking example - changed seeding mask to wm only voxels
* :ghpull:`858`: Updates for upcoming numpy 1.11 release
* :ghpull:`856`: Add reference to gitter chat room in the README
* :ghpull:`762`: Contextual enhancements of ODF/FOD fields
* :ghpull:`857`: DTI memory: use the same step in prediction as you use in fitting.
* :ghpull:`816`: A few fixes to SFM.
* :ghpull:`811`: Extract values from an image based on streamline coordinates.
* :ghpull:`853`: miscellaneous Python 3 compatibility problem fixes in fvtk
* :ghpull:`849`: nlmeans use num threads option in 3d
* :ghpull:`850`: DOC: fix typo
* :ghpull:`848`: DOC: fix typo
* :ghpull:`847`: DOC: fix typo
* :ghpull:`845`: DOC: Add kurtosis example to examples_index
* :ghpull:`846`: DOC: fix typo
* :ghpull:`826`: Return numpy arrays instead of memory views from cython functions
* :ghpull:`841`: Rename CONTRIBUTING to CONTRIBUTING.md
* :ghpull:`839`: DOC: Fix up the docstring for the CENIR data
* :ghpull:`819`: DOC: Add the DKI reconstruction example to the list of valid examples.
* :ghpull:`843`: Drop 3.2
* :ghpull:`838`: "Contributing"
* :ghpull:`833`: Doc: Typo
* :ghpull:`817`: RF: Convert nan values in bvectors to 0's
* :ghpull:`836`: fixed typo
* :ghpull:`695`: Introducing workflows
* :ghpull:`829`: Fixes issue #813 by not checking data type explicitly.
* :ghpull:`830`: Fixed doc of SDT
* :ghpull:`825`: Updated toollib and doc tools (#802)
* :ghpull:`760`: NF - random seeds from mask
* :ghpull:`824`: Updated copyright to 2016
* :ghpull:`815`: DOC: The previous link doesn't exist anymore.
* :ghpull:`669`: Function to reorient gradient directions according to moco parameters
* :ghpull:`809`: MRG: refactor and test setup.py
* :ghpull:`821`: BF: revert accidentally committed COMMIT_INFO.txt
* :ghpull:`818`: Round coords life
* :ghpull:`797`: Update csdeconv.py
* :ghpull:`806`: Relax regression tests
* :ghpull:`814`: TEST: compare array shapes directly
* :ghpull:`808`: MRG: pull in discarded changes from maintenance
* :ghpull:`745`: faster version of piesno
* :ghpull:`807`: BF: fix shebang lines for scripts
* :ghpull:`794`: RF: Allow setting the verbosity of the AffineRegistration while running it
* :ghpull:`801`: TST: add Python 3.5 to travis-ci test matrix

Issues (89):

* :ghissue:`933`: Updating release dates
* :ghissue:`925`: fix typos
* :ghissue:`915`: BF: correct handling of output paths in dipy_quickbundles.
* :ghissue:`922`: Fix PEP8 in top-level tests
* :ghissue:`886`: PEP8 in top-level tests
* :ghissue:`921`: fix typo
* :ghissue:`918`: Fix PEP8 in test_expectmax
* :ghissue:`863`: PEP8 in test_expectmax
* :ghissue:`919`: STYLE:PEP8 workflows
* :ghissue:`896`: STYLE: PEP8 for workflows folder
* :ghissue:`917`: Website 0.11 update and more devs
* :ghissue:`900`: SLR example needs updating
* :ghissue:`906`: Compiling the website needs too much memory
* :ghissue:`916`: Getting website ready for 0.11 release
* :ghissue:`914`: DOC: Update release notes for 0.11
* :ghissue:`910`: Singleton sl vals
* :ghissue:`908`: Fix pep8 errors in viz
* :ghissue:`890`: PEP8 in viz
* :ghissue:`911`: fix typo
* :ghissue:`905`: math is broken in doc
* :ghissue:`904`: fix typo
* :ghissue:`851`: Tissue Classifier tracking example - changed seeding mask to wm only voxels
* :ghissue:`858`: Updates for upcoming numpy 1.11 release
* :ghissue:`856`: Add reference to gitter chat room in the README
* :ghissue:`762`: Contextual enhancements of ODF/FOD fields
* :ghissue:`857`: DTI memory: use the same step in prediction as you use in fitting.
* :ghissue:`816`: A few fixes to SFM.
* :ghissue:`898`: Pep8 #891
* :ghissue:`811`: Extract values from an image based on streamline coordinates.
* :ghissue:`892`: PEP8 workflows
* :ghissue:`894`: PEP8 utils
* :ghissue:`895`: PEP8 Tracking
* :ghissue:`893`: PEP8 Viz
* :ghissue:`860`: Added Travis-CI badge
* :ghissue:`692`: Refactor fetcher.py
* :ghissue:`742`: LinAlgError on tracking quickstart, with python 3.4
* :ghissue:`822`: Could you help me ?  "URLError：<urlopen error [Errno 10060]>"
* :ghissue:`840`: Make dti reconst less memory hungry
* :ghissue:`855`: 0.9.3rc
* :ghissue:`853`: miscellaneous Python 3 compatibility problem fixes in fvtk
* :ghissue:`849`: nlmeans use num threads option in 3d
* :ghissue:`850`: DOC: fix typo
* :ghissue:`848`: DOC: fix typo
* :ghissue:`153`: DiffusionSpectrumModel assumes 1 b0 and fails with data with more than 1 b0
* :ghissue:`93`: GradientTable mask does not account for nan's in b-values
* :ghissue:`665`: Online tutorial of quickbundles does not work for released version on macosx
* :ghissue:`758`: One viz test still failing on mac os
* :ghissue:`847`: DOC: fix typo
* :ghissue:`845`: DOC: Add kurtosis example to examples_index
* :ghissue:`846`: DOC: fix typo
* :ghissue:`826`: Return numpy arrays instead of memory views from cython functions
* :ghissue:`841`: Rename CONTRIBUTING to CONTRIBUTING.md
* :ghissue:`839`: DOC: Fix up the docstring for the CENIR data
* :ghissue:`842`: New pip fails on 3.2
* :ghissue:`819`: DOC: Add the DKI reconstruction example to the list of valid examples.
* :ghissue:`843`: Drop 3.2
* :ghissue:`838`: "Contributing"
* :ghissue:`833`: Doc: Typo
* :ghissue:`817`: RF: Convert nan values in bvectors to 0's
* :ghissue:`836`: fixed typo
* :ghissue:`695`: Introducing workflows
* :ghissue:`829`: Fixes issue #813 by not checking data type explicitly.
* :ghissue:`805`: Multiple failures on Windows Python 3.5 build
* :ghissue:`802`: toollib and doc tools need update to 3.5
* :ghissue:`812`: Python 2.7 doctest failures on 64-bit Windows
* :ghissue:`685`: (WIP) DKI PR5 - NF: DKI-ODF estimation
* :ghissue:`830`: Fixed doc of SDT
* :ghissue:`825`: Updated toollib and doc tools (#802)
* :ghissue:`760`: NF - random seeds from mask
* :ghissue:`824`: Updated copyright to 2016
* :ghissue:`666`: Parallelized local tracking branch so now you can actually look at my code :)
* :ghissue:`815`: DOC: The previous link doesn't exist anymore.
* :ghissue:`747`: TEST: make test faster
* :ghissue:`631`: NF - multiprocessing multi voxel fit
* :ghissue:`669`: Function to reorient gradient directions according to moco parameters
* :ghissue:`809`: MRG: refactor and test setup.py
* :ghissue:`820`: dipy.get_info() returns wrong commit hash
* :ghissue:`821`: BF: revert accidentally committed COMMIT_INFO.txt
* :ghissue:`818`: Round coords life
* :ghissue:`810`: Wrong input type for `_voxel2stream` on 64-bit Windows
* :ghissue:`803`: Windows 7 Pro VM Python 2.7 gives 5 test errors with latest release 0.10.1
* :ghissue:`797`: Update csdeconv.py
* :ghissue:`806`: Relax regression tests
* :ghissue:`814`: TEST: compare array shapes directly
* :ghissue:`808`: MRG: pull in discarded changes from maintenance
* :ghissue:`745`: faster version of piesno
* :ghissue:`807`: BF: fix shebang lines for scripts
* :ghissue:`794`: RF: Allow setting the verbosity of the AffineRegistration while running it
* :ghissue:`801`: TST: add Python 3.5 to travis-ci test matrix