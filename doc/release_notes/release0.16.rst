.. _release0.16:

====================================
 Release notes for DIPY version 0.16
====================================

GitHub stats for 2018/12/12 - 2019/03/10 (tag: 0.15.0)

These lists are automatically generated, and may be incomplete or contain duplicates.

The following 14 authors contributed 361 commits.

* Ariel Rokem
* Bramsh Qamar
* Clément Zotti
* Eleftherios Garyfallidis
* Francois Rheault
* Gabriel Girard
* Jean-Christophe Houde
* Jon Haitz Legarreta Gorroño
* Katrin Leinweber
* Kesshi Jordan
* Parichit Sharma
* Serge Koudoro
* Shreyas Fadnavis
* Yijun Liu


We closed a total of 103 issues, 41 pull requests and 62 regular issues;
this is the full list (generated with the script 
:file:`tools/github_stats.py`):

Pull Requests (41):

* :ghpull:`1755`: Bundle Analysis and Linear mixed Models Workflows
* :ghpull:`1748`: [REBASED ] Symmetric Diffeomorphic registration workflow
* :ghpull:`1714`: DOC: Add Cython style guideline for DIPY.
* :ghpull:`1726`: IVIM MIX Model (MicroLearn)
* :ghpull:`1753`: BF: Fixes #1751, by adding a `scale` key-word arg to `decfa`
* :ghpull:`1743`: Horizon
* :ghpull:`1749`: [Fix] Tutorial syntax
* :ghpull:`1739`: IVIM Workflow
* :ghpull:`1695`: A few tractometry functions 
* :ghpull:`1741`: [Fix]  Pre-build : From relative to absolute import
* :ghpull:`1742`: [REBASED] Apply Transform Workflow
* :ghpull:`1745`: Adjust number of threads for SLR in Recobundles
* :ghpull:`1746`: [DOC] Add NIH to sponsor list
* :ghpull:`1735`: [Rebase] Affine Registration Workflow with Supporting Quality Metrics 
* :ghpull:`1738`: [FIX] Python 3 compatibility for some tools script
* :ghpull:`1740`: [FIX] mapmri  with cvxpy 1.0.15
* :ghpull:`1730`: [Fix] Cython syntax error
* :ghpull:`1666`: Remove the points added oustide of a mask. Fix the related tests.
* :ghpull:`1737`: Doc Fix
* :ghpull:`1733`: Minor Doc Fix
* :ghpull:`1732`: MAIIncrement Python version for testing to 3.6
* :ghpull:`1716`: DOC: Add `Imports` section on package shorthands recommendations.
* :ghpull:`1640`: Workflows - Adding PFT, probabilistic, closestpeaks tracking
* :ghpull:`1652`: Switching tests to pytest
* :ghpull:`1720`: [WIP] DIPY Workshop link on current website
* :ghpull:`1719`: BF: Syntax fix example images not rendering
* :ghpull:`1715`: DOC: Avoid the bullet points being interpreted as quoted blocks.
* :ghpull:`1706`: BUG: Fix Cython one-liner non-trivial type declaration warnings.
* :ghpull:`1705`: BUG: Fix Numpy `.random.random_integer` deprecation warning.
* :ghpull:`1704`: DOC: Fix typo in `linear_fascicle_evaluation.py` script.
* :ghpull:`1701`: BUG: Fix Sphinx math notation documentation warnings.
* :ghpull:`1707`: BUG: Address `numpy.matrix` `PendingDeprecation` warnings.
* :ghpull:`1703`: BUG: Fix blank image being recorded at `linear_fascicle_evaluation.py`.
* :ghpull:`1700`: DOC: Use triple double-quoted strings for docstrings.
* :ghpull:`1708`: BF: Clip the values before passing to arccos, instead of fixing nans.
* :ghpull:`1710`: DOC: fix typo in instruction below sample snippet 
* :ghpull:`1702`: BUG: Fix `dipy.io.trackvis` deprecation warnings.
* :ghpull:`1697`: Hyperlink DOIs to preferred resolver
* :ghpull:`1696`: Transfer release notes to a specific folder
* :ghpull:`1690`: Typo in release notes
* :ghpull:`1693`: Changed the GSoC conduction years

Issues (62):

* :ghissue:`1757`: NI Learn info from sensors designed to acseptible eeg format?
* :ghissue:`1755`: Bundle Analysis and Linear mixed Models Workflows
* :ghissue:`1748`: [REBASED ] Symmetric Diffeomorphic registration workflow
* :ghissue:`1714`: DOC: Add Cython style guideline for DIPY.
* :ghissue:`1726`: IVIM MIX Model (MicroLearn)
* :ghissue:`1751`: Scale values in `decfa`
* :ghissue:`1753`: BF: Fixes #1751, by adding a `scale` key-word arg to `decfa`
* :ghissue:`1754`: DeprecationWarning: This function is deprecated. Please call randint(0, 10000 + 1) instead    C:\Users\ranji\Anaconda3\lib\site-packages\ipykernel_launcher.py:14: DeprecationWarning: `imsave` is deprecated! `imsave` is deprecated in SciPy 1.0.0, and will be removed in 1.2.0. Use ``imageio.imwrite`` instead.
* :ghissue:`1743`: Horizon
* :ghissue:`1749`: [Fix] Tutorial syntax
* :ghissue:`1616`: Implementing the Diffeomorphic registration 
* :ghissue:`1739`: IVIM Workflow
* :ghissue:`1695`: A few tractometry functions 
* :ghissue:`1741`: [Fix]  Pre-build : From relative to absolute import
* :ghissue:`1742`: [REBASED] Apply Transform Workflow
* :ghissue:`1745`: Adjust number of threads for SLR in Recobundles
* :ghissue:`1746`: [DOC] Add NIH to sponsor list
* :ghissue:`1605`: Cleaned PR for Apply Transform Workflow to quickly register a set of moving NIFTI images using a given affine matrix.
* :ghissue:`1735`: [Rebase] Affine Registration Workflow with Supporting Quality Metrics 
* :ghissue:`1738`: [FIX] Python 3 compatibility for some tools script
* :ghissue:`1740`: [FIX] mapmri  with cvxpy 1.0.15
* :ghissue:`1730`: [Fix] Cython syntax error
* :ghissue:`1661`: Local Tracking - BinaryTissueClassifier - dropping the last point
* :ghissue:`1666`: Remove the points added oustide of a mask. Fix the related tests.
* :ghissue:`1737`: Doc Fix
* :ghissue:`1604`: Cleaned PR for Affine Registration Workflow with Supporting Quality Metrics
* :ghissue:`1734`: Documentation Error
* :ghissue:`1733`: Minor Doc Fix
* :ghissue:`1565`: New warnings with `PRE=1`
* :ghissue:`1732`: MAIIncrement Python version for testing to 3.6
* :ghissue:`1716`: DOC: Add `Imports` section on package shorthands recommendations.
* :ghissue:`1640`: Workflows - Adding PFT, probabilistic, closestpeaks tracking
* :ghissue:`1729`: N3/N4 Bias correction?
* :ghissue:`1652`: Switching tests to pytest
* :ghissue:`1280`: Switch testing to pytest?
* :ghissue:`1727`: Upgrade Scipy for MIX framework?
* :ghissue:`1723`: Failure on 
* :ghissue:`1720`: [WIP] DIPY Workshop link on current website
* :ghissue:`1718`: cannot import name window
* :ghissue:`1719`: BF: Syntax fix example images not rendering
* :ghissue:`1717`: cannot import packages of dipy.viz
* :ghissue:`1715`: DOC: Avoid the bullet points being interpreted as quoted blocks.
* :ghissue:`1706`: BUG: Fix Cython one-liner non-trivial type declaration warnings.
* :ghissue:`1705`: BUG: Fix Numpy `.random.random_integer` deprecation warning.
* :ghissue:`1664`: Deprecation warnings on testing
* :ghissue:`1704`: DOC: Fix typo in `linear_fascicle_evaluation.py` script.
* :ghissue:`1701`: BUG: Fix Sphinx math notation documentation warnings.
* :ghissue:`1707`: BUG: Address `numpy.matrix` `PendingDeprecation` warnings.
* :ghissue:`1633`: Blank png saved/displayed on documentation
* :ghissue:`1703`: BUG: Fix blank image being recorded at `linear_fascicle_evaluation.py`.
* :ghissue:`1700`: DOC: Use triple double-quoted strings for docstrings.
* :ghissue:`1708`: BF: Clip the values before passing to arccos, instead of fixing nans.
* :ghissue:`1698`: test failure in travis
* :ghissue:`1710`: DOC: fix typo in instruction below sample snippet 
* :ghissue:`1702`: BUG: Fix `dipy.io.trackvis` deprecation warnings.
* :ghissue:`1697`: Hyperlink DOIs to preferred resolver
* :ghissue:`1696`: Transfer release notes to a specific folder
* :ghissue:`1691`: Doc folder for release notes
* :ghissue:`1690`: Typo in release notes
* :ghissue:`1693`: Changed the GSoC conduction years
* :ghissue:`1692`: Minor doc fix
* :ghissue:`1632`: Link broken on website
