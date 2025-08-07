.. _release0.15:

====================================
 Release notes for DIPY version 0.15
====================================

GitHub stats for 2018/05/01 - 2018/12/12 (tag: 0.14.0)

These lists are automatically generated, and may be incomplete or contain duplicates.

The following 30 authors contributed 676 commits.

* Ariel Rokem
* Bramsh Qamar
* Chris Filo Gorgolewski
* David Reagan
* Demian Wassermann
* Eleftherios Garyfallidis
* Enes Albay
* Gabriel Girard
* Guillaume Theaud
* Javier Guaje
* Jean-Christophe Houde
* Jiri Borovec
* Jon Haitz Legarreta Gorroño
* Karandeep
* Kesshi Jordan
* Marc-Alexandre Côté
* Matt Cieslak
* Matthew Brett
* Parichit Sharma
* Ricci Woo
* Rutger Fick
* Serge Koudoro
* Shreyas Fadnavis
* Chandan Gangwar
* Daniel Enrico Cahall
* David Hunt
* Francois Rheault
* Jakob Wasserthal


We closed a total of 287 issues, 93 pull requests and 194 regular issues;
this is the full list (generated with the script
:file:`tools/github_stats.py`):

Pull Requests (93):

* :ghpull:`1684`: [FIX] testing line-based target function
* :ghpull:`1686`: Standardize workflow
* :ghpull:`1685`: [Fix] Typo on examples
* :ghpull:`1663`: Stats, SNR_in_CC workflow
* :ghpull:`1681`: fixed issue with cst orientation in bundle_extraction example
* :ghpull:`1680`: [Fix] workflow variable string
* :ghpull:`1683`: test for new error in IVIM
* :ghpull:`1667`: Changing the default b0_threshold in gtab
* :ghpull:`1677`: [FIX] workflow help msg
* :ghpull:`1678`: Numpy matrix deprecation
* :ghpull:`1676`: [FIX] Example Update
* :ghpull:`1283`: get_data consistence
* :ghpull:`1670`: fixed RecoBundle workflow, SLR reference, and updated fetcher.py
* :ghpull:`1669`: Flow csd sh order
* :ghpull:`1659`: From dipy.viz to FURY
* :ghpull:`1621`: workflows : warn user for strange b0 threshold
* :ghpull:`1657`: DOC: Add spherical harmonics basis documentation.
* :ghpull:`1660`: OPT - moved the tolerance check outside of the for loop
* :ghpull:`1658`: STYLE: Honor 'descoteaux'and 'tournier' SH basis naming.
* :ghpull:`1281`: Representing qtau- signal attenuation using qtau-dMRI functional basis
* :ghpull:`1651`: Add save/load tck
* :ghpull:`1656`: Link to the dipy tag on neurostars
* :ghpull:`1624`: NF: Outlier scoring
* :ghpull:`1655`: [Fix] decrease tolerance on forecast
* :ghpull:`1650`: Increase codecov tolerance
* :ghpull:`1649`: Path Length Map example rebase
* :ghpull:`1556`: RecoBundles and SLR workflows
* :ghpull:`1645`: Fix workflows creation tutorial error
* :ghpull:`1647`: DOC: Fix duplicate link and AppVeyor badge.
* :ghpull:`1644`: Adds an Appveyor badge
* :ghpull:`1643`: Add hash for SCIL b0 file
* :ghpull:`787`: TST: Add an appveyor starter file.
* :ghpull:`1642`: Test that you can use the 724 symmetric sphere in PAM.
* :ghpull:`1641`: changed vertices to float64 in evenly_distributed_sphere_642.npz
* :ghpull:`1564`: Added scroll bar to ListBox2D
* :ghpull:`1636`: Fixed broken link.
* :ghpull:`1584`: Added Examples
* :ghpull:`1554`: Checking if the input file or directory exists when running a workflow
* :ghpull:`1528`: Show spheres with different radii, colors and opacities + add timers + add exit a + resolve issue with imread
* :ghpull:`1526`: Eigenvalue - eigenvector array compatibility check
* :ghpull:`1628`: Adding python 3.7 on travis
* :ghpull:`1623`: NF: Convert between 4D DEC FA and 3D 24 bit representation.
* :ghpull:`1622`: [Fix] viz slice example
* :ghpull:`1626`: RF - removed duplicate tests
* :ghpull:`1619`: [DOC] update VTK version
* :ghpull:`1592`: Added File Menu element to viz.ui
* :ghpull:`1559`: Checkbox and RadioButton elements for viz.ui
* :ghpull:`1583`: Fix the relative SF threshold Issue
* :ghpull:`1602`: Fix random seed in tracking
* :ghpull:`1609`: [DOC] update dependencies file
* :ghpull:`1560`: Removed affine matrices from tracking.
* :ghpull:`1593`: Removed event.abort for release events
* :ghpull:`1597`: Upgrade nibabel minimum version
* :ghpull:`1601`: Fix: Decrease Nosetest warning
* :ghpull:`1515`: RF: Use the new Streamlines API for orienting of streamlines.
* :ghpull:`1590`: Revert 1570 file menu
* :ghpull:`1589`: Fix calculation of highest order for a sh basis set
* :ghpull:`1580`: Allow PRE=1 job to fail
* :ghpull:`1533`: Show message if number of arguments mismatch between the doc string and the run method.
* :ghpull:`1523`: Showing help when no input parameters are given and suppress warnings for cmds
* :ghpull:`1543`: Update the default out_strategy to create the output in the current working directory
* :ghpull:`1574`: Fixed Bug in PR #1547
* :ghpull:`1561`: add example SDR for binary and fuzzy images
* :ghpull:`1578`: BF - bad condition in maximum dg
* :ghpull:`1570`: Added File Menu element to viz.ui
* :ghpull:`1563`: Replacing major_version in viz.ui
* :ghpull:`1557`: Range slider element for viz.ui
* :ghpull:`1547`:  Changed the icon set in Button2D from Dictionary to List of Tuples
* :ghpull:`1555`: Fix bug in actor.label
* :ghpull:`1522`: Image element in dipy.viz.ui
* :ghpull:`1355`: WIP: ENH: UI Listbox
* :ghpull:`1540`: fix potential zero division in demon register.
* :ghpull:`1548`: Fixed references per request of @garyfallidis.
* :ghpull:`1542`: fix for using cvxpy solver
* :ghpull:`1546`: References to reference
* :ghpull:`1545`: Adding a reference in README.rst
* :ghpull:`1492`: Enh ui components positioning (with code refactoring)
* :ghpull:`1538`: Explanation that is mistakenly rendered as code fixed in example of DKI
* :ghpull:`1536`: DOC: Update Rafael's current institution.
* :ghpull:`1537`: removed unnecessary imported from sims example
* :ghpull:`1530`: Wrong default value for parameter 'symmetric' connectivity_matrix function
* :ghpull:`1529`: minor typo fix in quickstart
* :ghpull:`1520`: Updating the documentation for the workflow creation tutorial.
* :ghpull:`1524`: Values from streamlines object
* :ghpull:`1521`: Moved some older highlights and announcements to the old news files.
* :ghpull:`1518`: DOC: updated some developers affiliations.
* :ghpull:`1517`: Dev info update
* :ghpull:`1516`: [DOC] Installation instruction update
* :ghpull:`1514`: Adding pep8speak config file
* :ghpull:`1513`: fix typo in example of quick_start
* :ghpull:`1510`: copyright updated to 2008-2018
* :ghpull:`1508`: Adds whitespace, to appease the sphinx.
* :ghpull:`1506`: moving to 0.15.0 dev

Issues (194):

* :ghissue:`1684`: [FIX] testing line-based target function
* :ghissue:`1679`: Intermittent issue in testing line-based target function
* :ghissue:`1220`: RF: Replaces 1997 definitions of tensor geometric params with 1999 definitions.
* :ghissue:`1686`: Standardize workflow
* :ghissue:`746`: New fetcher returns filenames as dictionary keys in a tuple
* :ghissue:`1685`: [Fix] Typo on examples
* :ghissue:`1663`: Stats, SNR_in_CC workflow
* :ghissue:`1637`: Advice for saving results from MAPMRI
* :ghissue:`1673`: CST Image in bundle extraction is not oriented well
* :ghissue:`1681`: fixed issue with cst orientation in bundle_extraction example
* :ghissue:`1680`: [Fix] workflow variable string
* :ghissue:`1338`: Variable string input does not work with self.get_io_iterator() in workflows
* :ghissue:`1683`: test for new error in IVIM
* :ghissue:`1682`: Add tests for IVIM for new Error
* :ghissue:`634`: BinaryTissueClassifier segfaults on corner case
* :ghissue:`742`: LinAlgError on tracking quickstart, with python 3.4
* :ghissue:`852`: Problem with spherical harmonics computations on some Anaconda python versions
* :ghissue:`1667`: Changing the default b0_threshold in gtab
* :ghissue:`1500`: Updating streamlines API in streamlinear.py
* :ghissue:`944`: Slicer fix
* :ghissue:`1111`: WIP: A lightweight UI for medical visualizations based on VTK-Python
* :ghissue:`1099`: Needed PRs for merging recobundles into Dipy's master
* :ghissue:`1544`: Plans for viz module
* :ghissue:`641`: Tests raise a deprecation warning
* :ghissue:`643`: Use appveyor for Windows CI?
* :ghissue:`400`: Add travis-ci test without matplotlib installed
* :ghissue:`1677`: [FIX] workflow help msg
* :ghissue:`1674`: Workflows should print out help per default
* :ghissue:`1678`: Numpy matrix deprecation
* :ghissue:`1397`: Running dipy 'Intro to Basic Tracking' code and keep getting error. On Linux Centos
* :ghissue:`1676`: [FIX] Example Update
* :ghissue:`10`: data.get_data() should be consistent across datasets
* :ghissue:`1283`: get_data consistence
* :ghissue:`1670`: fixed RecoBundle workflow, SLR reference, and updated fetcher.py
* :ghissue:`1669`: Flow csd sh order
* :ghissue:`1668`: One issue on handling HCP data -- HCP b vectors raise NaN in the gradient table
* :ghissue:`1662`: Remove the points added outside of a mask. Fix the related tests.
* :ghissue:`1659`: From dipy.viz to FURY
* :ghissue:`1621`: workflows : warn user for strange b0 threshold
* :ghissue:`1657`: DOC: Add spherical harmonics basis documentation.
* :ghissue:`1296`: Need of a travis bot that runs ana/mini/conda and vtk=7.1.0+
* :ghissue:`1660`: OPT - moved the tolerance check outside of the for loop
* :ghissue:`1658`: STYLE: Honor 'descoteaux'and 'tournier' SH basis naming.
* :ghissue:`1281`: Representing qtau- signal attenuation using qtau-dMRI functional basis
* :ghissue:`1653`: STYLE: Honor 'descoteaux' SH basis naming.
* :ghissue:`1651`: Add save/load tck
* :ghissue:`1656`: Link to the dipy tag on neurostars
* :ghissue:`1624`: NF: Outlier scoring
* :ghissue:`1655`: [Fix] decrease tolerance on forecast
* :ghissue:`1654`: Test failure in FORECAST
* :ghissue:`1414`: [WIP] Switching tests to pytest and removing nose dependencies
* :ghissue:`1650`: Increase codecov tolerance
* :ghissue:`1093`: WIP: Add functionality to clip streamlines between ROIs in `orient_by_rois`
* :ghissue:`1611`: Preloader element for viz.ui
* :ghissue:`1615`: Color Picker element for viz.ui
* :ghissue:`1631`: Path Length Map example
* :ghissue:`1649`: Path Length Map example rebase
* :ghissue:`1556`: RecoBundles and SLR workflows
* :ghissue:`1645`: Fix workflows creation tutorial error
* :ghissue:`1647`: DOC: Fix duplicate link and AppVeyor badge.
* :ghissue:`1644`: Adds an Appveyor badge
* :ghissue:`1638`: Fetcher downloads data every time it is called
* :ghissue:`1643`: Add hash for SCIL b0 file
* :ghissue:`1600`: NODDIx 2 fibers crossing
* :ghissue:`1618`: viz.ui.FileMenu2D
* :ghissue:`1569`: viz.ui.ListBoxItem2D text overflow
* :ghissue:`1532`: dipy test failed on mac osx sierra with ananoda python.
* :ghissue:`1420`: window.record() resolution limit
* :ghissue:`1396`: Visualization problem with tensors ?
* :ghissue:`1295`: Reorienting peak_slicer and ODF_slicer
* :ghissue:`1232`: With VTK 6.3, streamlines color map bar text disappears when using streamtubes
* :ghissue:`928`: dipy.viz.colormap crash on single fibers
* :ghissue:`923`: change size of colorbar in viz module
* :ghissue:`854`: VTK and Python 3 support in fvtk
* :ghissue:`759`: How to resolve python-vtk6 link issues in Ubuntu
* :ghissue:`647`: fvtk contour function ignores voxsz parameter
* :ghissue:`646`: Dipy visualization with missing (?) affine parameter
* :ghissue:`645`: Dipy visualization (fvtk) crash when saving series of images
* :ghissue:`353`: fvtk.label won't show up if called twice
* :ghissue:`787`: TST: Add an appveyor starter file.
* :ghissue:`1642`: Test that you can use the 724 symmetric sphere in PAM.
* :ghissue:`1641`: changed vertices to float64 in evenly_distributed_sphere_642.npz
* :ghissue:`1203`: Some bots might need a newer version of nibabel
* :ghissue:`1156`: Deterministic tracking workflow
* :ghissue:`642`: WIP - NF parallel framework
* :ghissue:`1135`: WIP : Multiprocessing - implemented a parallel_voxel_fit decorator
* :ghissue:`387`: References do not render correctly in SHORE example
* :ghissue:`442`: Allow length and set_number_of_points to work with generators
* :ghissue:`558`: Allow setting of the zoom on fvtk ren objects
* :ghissue:`1236`: bundle visualisation using nibabel API: wrong colormap
* :ghissue:`1389`: VTK 8: minimal version?
* :ghissue:`1519`: Scipy stopped supporting scipy.misc.imread
* :ghissue:`1596`: Reproducibility in PFT tracking
* :ghissue:`1614`: for GSoC NODDIx_PR
* :ghissue:`1576`: [WIP] Needs Optimization and Cleaning
* :ghissue:`1564`: Added scroll bar to ListBox2D
* :ghissue:`1636`: Fixed broken link.
* :ghissue:`1584`: Added Examples
* :ghissue:`1568`: Multi_io axis out of bounds error
* :ghissue:`1554`: Checking if the input file or directory exists when running a workflow
* :ghissue:`1528`: Show spheres with different radii, colors and opacities + add timers + add exit a + resolve issue with imread
* :ghissue:`1108`: Local PCA Slow Version
* :ghissue:`1526`: Eigenvalue - eigenvector array compatibility check
* :ghissue:`1628`: Adding python 3.7 on travis
* :ghissue:`1623`: NF: Convert between 4D DEC FA and 3D 24 bit representation.
* :ghissue:`1622`: [Fix] viz slice example
* :ghissue:`1629`: [WIP][fix] remove Userwarning message
* :ghissue:`1591`: PRE is failing :  module 'cvxpy' has no attribute 'utilities'
* :ghissue:`1626`: RF - removed duplicate tests
* :ghissue:`1582`: SF threshold in PMF is not relative
* :ghissue:`1575`: Website: warning about python versions
* :ghissue:`1619`: [DOC] update VTK version
* :ghissue:`1592`: Added File Menu element to viz.ui
* :ghissue:`1559`: Checkbox and RadioButton elements for viz.ui
* :ghissue:`1583`: Fix the relative SF threshold Issue
* :ghissue:`1602`: Fix random seed in tracking
* :ghissue:`1620`: 3.7 wheels
* :ghissue:`1598`: Apply Transform workflow for transforming a collection of moving images.
* :ghissue:`1595`: Workflow for visualizing the quality of the registered data with DIPY
* :ghissue:`1581`: Image registration Workflow with quality metrics
* :ghissue:`1588`: Dipy.reconst.shm.calculate_max_order only works on specific cases.
* :ghissue:`1608`: Parallelized affine registration
* :ghissue:`1610`: Tortoise - sub
* :ghissue:`1607`: Reminder to add in the docs that users will need to update nibabel to 2.3.0 during the next release
* :ghissue:`1609`: [DOC] update dependencies file
* :ghissue:`1560`: Removed affine matrices from tracking.
* :ghissue:`1593`: Removed event.abort for release events
* :ghissue:`1586`: Slider breaks interaction in viz_advanced example
* :ghissue:`1597`: Upgrade nibabel minimum version
* :ghissue:`1601`: Fix: Decrease Nosetest warning
* :ghissue:`1515`: RF: Use the new Streamlines API for orienting of streamlines.
* :ghissue:`1585`: Add a random seed for reproducibility
* :ghissue:`1594`: Integrating the support for the visualization in Affine registration
* :ghissue:`1590`: Revert 1570 file menu
* :ghissue:`1589`: Fix calculation of highest order for a sh basis set
* :ghissue:`1577`: Revert "Added File Menu element to viz.ui"
* :ghissue:`1571`: WIP: multi-threaded on affine registration
* :ghissue:`1580`: Allow PRE=1 job to fail
* :ghissue:`1533`: Show message if number of arguments mismatch between the doc string and the run method.
* :ghissue:`1523`: Showing help when no input parameters are given and suppress warnings for cmds
* :ghissue:`1579`: Error on PRE=1 (cython / numpy)
* :ghissue:`1543`: Update the default out_strategy to create the output in the current working directory
* :ghissue:`1433`: New version of h5py messing with us?
* :ghissue:`1541`: demon registration, unstable?
* :ghissue:`1574`: Fixed Bug in PR #1547
* :ghissue:`1573`: Failure in test_ui_listbox_2d
* :ghissue:`1561`: add example SDR for binary and fuzzy images
* :ghissue:`1578`: BF - bad condition in maximum dg
* :ghissue:`1566`: Bad condition in local tracking
* :ghissue:`1570`: Added File Menu element to viz.ui
* :ghissue:`1572`: [WIP]
* :ghissue:`1567`: WIP: NF: multi-threaded on affine registration
* :ghissue:`1563`: Replacing major_version in viz.ui
* :ghissue:`1557`: Range slider element for viz.ui
* :ghissue:`1547`:  Changed the icon set in Button2D from Dictionary to List of Tuples
* :ghissue:`1555`: Fix bug in actor.label
* :ghissue:`1551`: Actor.label not working anymore
* :ghissue:`1522`: Image element in dipy.viz.ui
* :ghissue:`1549`: CVXPY installation on >3.5
* :ghissue:`1355`: WIP: ENH: UI Listbox
* :ghissue:`1562`: Should we retire our Python 3.5 travis builds?
* :ghissue:`1550`: Memory error when running rigid transform
* :ghissue:`1540`: fix potential zero division in demon register.
* :ghissue:`1548`: Fixed references per request of @garyfallidis.
* :ghissue:`1527`: New version of CVXPY changes API
* :ghissue:`1542`: fix for using cvxpy solver
* :ghissue:`1534`: Changed the icon set in Button2D from Dictionary to List of Tuples
* :ghissue:`1546`: References to reference
* :ghissue:`1545`: Adding a reference in README.rst
* :ghissue:`1492`: Enh ui components positioning (with code refactoring)
* :ghissue:`1538`: Explanation that is mistakenly rendered as code fixed in example of DKI
* :ghissue:`1536`: DOC: Update Rafael's current institution.
* :ghissue:`1487`: Commit for updated check_scratch.py script.
* :ghissue:`1486`: Parichit dipy flows
* :ghissue:`1539`: Changing the default behavior of the workflows to create the output file(s) in the current working directory.
* :ghissue:`1537`: removed unnecessary imported from sims example
* :ghissue:`1535`: removed some unnecessary imports from sims example
* :ghissue:`1530`: Wrong default value for parameter 'symmetric' connectivity_matrix function
* :ghissue:`1529`: minor typo fix in quickstart
* :ghissue:`1520`: Updating the documentation for the workflow creation tutorial.
* :ghissue:`1524`: Values from streamlines object
* :ghissue:`1521`: Moved some older highlights and announcements to the old news files.
* :ghissue:`1518`: DOC: updated some developers affiliations.
* :ghissue:`1517`: Dev info update
* :ghissue:`1516`: [DOC] Installation instruction update
* :ghissue:`1514`: Adding pep8speak config file
* :ghissue:`1507`: Mathematical expressions are not rendered correctly in reference page
* :ghissue:`1513`: fix typo in example of quick_start
* :ghissue:`1510`: copyright updated to 2008-2018
* :ghissue:`1508`: Adds whitespace, to appease the sphinx.
* :ghissue:`1512`: Fix typo in example of quick_start
* :ghissue:`1511`: Fix typo in exaample quick_start
* :ghissue:`1509`: DOC: fix math rendering for some dki functions
* :ghissue:`1506`: moving to 0.15.0 dev
