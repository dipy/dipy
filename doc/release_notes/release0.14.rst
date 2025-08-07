.. _release0.14:

====================================
 Release notes for DIPY version 0.14
====================================

GitHub stats for 2017/10/24 - 2018/05/01 (tag: 0.13.0)

These lists are automatically generated, and may be incomplete or contain duplicates.

The following 24 authors contributed 645 commits.

* Ariel Rokem
* Bago Amirbekian
* Bennet Fauber
* Conor Corbin
* David Reagan
* Eleftherios Garyfallidis
* Gabriel Girard
* Jean-Christophe Houde
* Jiri Borovec
* Jon Haitz Legarreta Gorroño
* Jon Mendoza
* Karandeep Singh Juneja
* Kesshi Jordan
* Kumar Ashutosh
* Marc-Alexandre Côté
* Matthew Brett
* Nil Goyette
* Pradeep Reddy Raamana
* Ricci Woo
* Serge Koudoro
* Shreyas Fadnavis
* Aman Arya
* Mauro Zucchelli


We closed a total of 215 issues, 70 pull requests and 145 regular issues;
this is the full list (generated with the script
:file:`tools/github_stats.py`):

Pull Requests (70):

* :ghpull:`1504`: Fix test_whole_brain_slr precision
* :ghpull:`1503`: fix plotting issue on Mapmri example
* :ghpull:`1424`: ENH: Use the CFIN dataset rather than the CENIR dataset.
* :ghpull:`1502`: [DOC] Fix doc generation
* :ghpull:`1498`: BF: fix bug in example of segment_quickbundles.py
* :ghpull:`1431`: NF - Bootstrap Direction Getter (cythonized)
* :ghpull:`1443`: RecoBundles - recognition of bundles
* :ghpull:`1398`: Deform streamlines
* :ghpull:`1447`: DOC: Link Coding Style Guide ref in CONTRIBUTING to the website.
* :ghpull:`1423`: DOC: Fix `reconst_mapmri` example markup.
* :ghpull:`1493`: Fix surface rendering bugs in VTK8
* :ghpull:`1497`: BF: fix bug in example of fiber_to_bundle_coherence.py
* :ghpull:`1496`: BF: fix bug in example of streamline_tools.py
* :ghpull:`1495`: BF: fix bug in example of sfm_reconst.py
* :ghpull:`1494`: BF: fix bug in example of reconst_csd.py
* :ghpull:`1474`: DOC: Fix typo on website & examples
* :ghpull:`1471`: Code Cleaning
* :ghpull:`1457`: Fix for "Sliders in examples don't react properly to clicks"
* :ghpull:`1491`: Fix documentation typos
* :ghpull:`1468`: DOC: correct error when doing 'make html'
* :ghpull:`1484`: DOC: Use the correct punctuation marks for `et al.`.
* :ghpull:`1475`: Refactor demon registration - _iterate
* :ghpull:`1482`: DOC: Fix typo in `test_mapmri.py` file.
* :ghpull:`1460`: Fix for "DiskSlider does not rotate actor in opposite direction"
* :ghpull:`1452`: actor.slicer.copy() copies opacity set via actor.slicer.opacity()
* :ghpull:`1466`: DOC: Limit the DIPY logo height in README for better rendering.
* :ghpull:`1464`: DOC: Use the correct DIPY logo as the banner in `README`.
* :ghpull:`1465`: Fixed the Progit book link in doc
* :ghpull:`1451`: DOC: Add the DIPY banner to the README file.
* :ghpull:`1379`: New streamlines API integration on dipy examples
* :ghpull:`1445`: repr and get methods for AffineMap, w/ precise exceptions
* :ghpull:`1450`: [Fix] Manage multiple space delimiter
* :ghpull:`1425`: DOC: Add different GitHub badges to the `README.rst` file.
* :ghpull:`1446`: DOC: Fix bad hyperlink format to CONTRIBUTING.md from README.rst.
* :ghpull:`1437`: DOC: Fix missing reference to QuickBundles paper.
* :ghpull:`1440`: Raise Error when MDFmetric is used in QB or QBX
* :ghpull:`1428`: Mapmri workflow rebased
* :ghpull:`1385`: Enh textblock
* :ghpull:`1422`: [MRG] Improves delimiter in read_bvals_bvecs
* :ghpull:`1434`: QuickBundlesX
* :ghpull:`1430`: BF - replaced non-ascii character in workflows/reconst.py
* :ghpull:`1421`: DOC: Fix reStructuredText formatting issues in coding style guideline.
* :ghpull:`1416`: Updated links
* :ghpull:`1413`: BF::Fix inspect.getargspec deprecation warning in Python 3
* :ghpull:`1393`: Adds a DKI workflow.
* :ghpull:`1294`: Suppress a warning in geometry.
* :ghpull:`1419`: Suppress rcond warning
* :ghpull:`1358`: Det track workflow  rebased (merge)
* :ghpull:`1384`: NF - Particle Filtering Tractography (merge)
* :ghpull:`1411`: Added eddy_rotated_bvecs extension
* :ghpull:`1407`: [MRG] Default colormap changed in examples
* :ghpull:`1408`: Updated color map in reconst_csa.py and reconst_forecast.py
* :ghpull:`1406`: [MRG] assert_true which checks for equality replaced with assert_equal
* :ghpull:`1347`: Replacing fvtk by the new viz API
* :ghpull:`1322`: Forecast
* :ghpull:`1326`: BUG: Fix factorial import module in test_mapmri.py.
* :ghpull:`1400`: BF: fixes #1399, removing an un-needed singleton dimension.
* :ghpull:`1391`: Re-entering conflict-free typos from deleted PR 1331
* :ghpull:`1386`: Possible fix for the inline compilation problem
* :ghpull:`1165`: Make vtk contour take an affine
* :ghpull:`1300`: RF: Remove patch for older numpy ravel_multi_index.
* :ghpull:`1381`: DOC - re-orientation of figures in the DKI example
* :ghpull:`1375`: Fix piesno type
* :ghpull:`1342`: Cythonize DirectionGetter and whatnot
* :ghpull:`1378`: Fix: numpy legacy print again...
* :ghpull:`1377`: FIX: update printing format for numpy 1.14
* :ghpull:`1374`: FIX: Viz test correction
* :ghpull:`1368`: DOC: Update developers' affiliations.
* :ghpull:`1370`: TST - add tracking tests for PeaksAndMetricsDirectionGetter
* :ghpull:`1369`: MRG: add procedure for building, uploading wheels

Issues (145):

* :ghissue:`1504`: Fix test_whole_brain_slr precision
* :ghissue:`1418`: Adding parallel_voxel_fit decorator
* :ghissue:`1503`: fix plotting issue on Mapmri example
* :ghissue:`1291`: Existing MAPMRI tutorial does not render correctly and MAPL looks hidden in the existing tutorial.
* :ghissue:`1424`: ENH: Use the CFIN dataset rather than the CENIR dataset.
* :ghissue:`1502`: [DOC] Fix doc generation
* :ghissue:`1498`: BF: fix bug in example of segment_quickbundles.py
* :ghissue:`1431`: NF - Bootstrap Direction Getter (cythonized)
* :ghissue:`1443`: RecoBundles - recognition of bundles
* :ghissue:`644`: Dipy visualization: it does not seem possible to position tensor ellipsoid slice in fvtk
* :ghissue:`1398`: Deform streamlines
* :ghissue:`1447`: DOC: Link Coding Style Guide ref in CONTRIBUTING to the website.
* :ghissue:`1423`: DOC: Fix `reconst_mapmri` example markup.
* :ghissue:`1493`: Fix surface rendering bugs in VTK8
* :ghissue:`1490`: Streamtube visualization problem with vtk 8.1
* :ghissue:`1469`: Errors in generating documents (.rst) of examples
* :ghissue:`1497`: BF: fix bug in example of fiber_to_bundle_coherence.py
* :ghissue:`1496`: BF: fix bug in example of streamline_tools.py
* :ghissue:`1495`: BF: fix bug in example of sfm_reconst.py
* :ghissue:`1494`: BF: fix bug in example of reconst_csd.py
* :ghissue:`1474`: DOC: Fix typo on website & examples
* :ghissue:`1485`: BF: Fix bug in example of segment_quickbundles.py
* :ghissue:`1483`: BF: Fix bug in example of fiber_to_bundle_coherence.py
* :ghissue:`1480`: BF: Fix bug in example of streamline_tool.py
* :ghissue:`1479`: BF: Fix bug in example of sfm_reconst.py
* :ghissue:`1477`: BF: Fix bug in example of reconst_csd.py
* :ghissue:`1448`: Enh ui components positioning
* :ghissue:`1471`: Code Cleaning
* :ghissue:`1481`: BF: Fix bug of no attribute 'GlobalImmediateModeRenderingOn' in actor.py
* :ghissue:`1454`: Sliders in examples don't react properly to clicks
* :ghissue:`1457`: Fix for "Sliders in examples don't react properly to clicks"
* :ghissue:`1491`: Fix documentation typos
* :ghissue:`1468`: DOC: correct error when doing 'make html'
* :ghissue:`1467`: Error on "from dipy.core.gradients import gradient_table"
* :ghissue:`1488`: Unexpected behavior in the DIPY workflow script
* :ghissue:`1484`: DOC: Use the correct punctuation marks for `et al.`.
* :ghissue:`1475`: Refactor demon registration - _iterate
* :ghissue:`1482`: DOC: Fix typo in `test_mapmri.py` file.
* :ghissue:`1478`: DOC: Add comment about package CVXPY in example of reconst_mapmri.py
* :ghissue:`1476`: BF: Fix bug in example of reconst_csd.py
* :ghissue:`1470`: simplify SDR iterate
* :ghissue:`1458`: DiskSlider does not rotate actor in opposite direction
* :ghissue:`1460`: Fix for "DiskSlider does not rotate actor in opposite direction"
* :ghissue:`1452`: actor.slicer.copy() copies opacity set via actor.slicer.opacity()
* :ghissue:`1438`: actor.slicer.copy() doesn't copy opacity if set via actor.slicer.opacity()
* :ghissue:`1473`: Uploading Windows wheels
* :ghissue:`1466`: DOC: Limit the DIPY logo height in README for better rendering.
* :ghissue:`1472`: Invalid dims failure in 32-bit Python on Windows
* :ghissue:`1464`: DOC: Use the correct DIPY logo as the banner in `README`.
* :ghissue:`1462`: Logo/banner on README not the correct one!
* :ghissue:`1461`: Broken link in  Documentation: Git Resources
* :ghissue:`1465`: Fixed the Progit book link in doc
* :ghissue:`1463`: Fixed the Progit book link in the docs
* :ghissue:`1455`: Using pyautogui to adapt to users' monitor size in viz examples
* :ghissue:`1459`: Fix for "DiskSlider does not rotate actor in opposite direction"
* :ghissue:`1456`: Fix for "Sliders in examples don't react properly to clicks"
* :ghissue:`1453`: changed window.record() to a large value
* :ghissue:`1451`: DOC: Add the DIPY banner to the README file.
* :ghissue:`1379`: New streamlines API integration on dipy examples
* :ghissue:`1339`: Deprecate dipy.io.trackvis?
* :ghissue:`1445`: repr and get methods for AffineMap, w/ precise exceptions
* :ghissue:`1441`: Cleaning UI and improving positioning of Panel2D
* :ghissue:`1450`: [Fix] Manage multiple space delimiter
* :ghissue:`1449`: read_bvals_bvecs crash with bvec rotated eddy
* :ghissue:`1425`: DOC: Add different GitHub badges to the `README.rst` file.
* :ghissue:`1446`: DOC: Fix bad hyperlink format to CONTRIBUTING.md from README.rst.
* :ghissue:`1437`: DOC: Fix missing reference to QuickBundles paper.
* :ghissue:`1371`: Quickbundles tutorials miss reference
* :ghissue:`1362`: Make more use of TextBlock2D constructor
* :ghissue:`1440`: Raise Error when MDFmetric is used in QB or QBX
* :ghissue:`1395`: Mapmri workflow
* :ghissue:`1428`: Mapmri workflow rebased
* :ghissue:`1385`: Enh textblock
* :ghissue:`1436`: Fixed delimiter issue #1417
* :ghissue:`1422`: [MRG] Improves delimiter in read_bvals_bvecs
* :ghissue:`1417`: Improve delimiter on read_bvals_bvecs()
* :ghissue:`1435`: compilation failed with the new cython version (0.28)
* :ghissue:`1439`: BF: Avoid using memview in struct (Cython 0.28)
* :ghissue:`1434`: QuickBundlesX
* :ghissue:`1184`: Bootstrap direction getter
* :ghissue:`1380`: WIP: QuickBundlesX
* :ghissue:`1429`: BUG - SyntaxError (Non-ASCII character '\xe2' in file dipy/workflows/reconst.py on line 596
* :ghissue:`1430`: BF - replaced non-ascii character in workflows/reconst.py
* :ghissue:`1421`: DOC: Fix reStructuredText formatting issues in coding style guideline.
* :ghissue:`1390`: coding_style_guideline.rst does not render correctly
* :ghissue:`1427`: Add delimiter to read_bvals_bvecs()
* :ghissue:`1426`: Add delimiter parameter to numpy.loadtxt
* :ghissue:`1416`: Updated links
* :ghissue:`987`: Practical FAQs don't have hyperlinks to modules/libraries.
* :ghissue:`1327`: Fix inspect.getargspec deprecation warning in Python 3
* :ghissue:`1413`: BF::Fix inspect.getargspec deprecation warning in Python 3
* :ghissue:`1393`: Adds a DKI workflow.
* :ghissue:`1294`: Suppress a warning in geometry.
* :ghissue:`1181`: peaks warning while CSD reconstructing
* :ghissue:`1419`: Suppress rcond warning
* :ghissue:`1150`: Line-based version of streamline_mapping
* :ghissue:`1358`: Det track workflow  rebased (merge)
* :ghissue:`1384`: NF - Particle Filtering Tractography (merge)
* :ghissue:`1409`: create documentation in multiple languages
* :ghissue:`1415`: NF: check compiler flags before compiling
* :ghissue:`1117`: .eddy_rotated_bvecs file throws error from io.gradients read_bvals_bvecs function
* :ghissue:`1411`: Added eddy_rotated_bvecs extension
* :ghissue:`1412`: BF:Fix inspect.getargspec deprecation warning in Python 3
* :ghissue:`791`: Possible divide by zero in reconst.sfm.py
* :ghissue:`1410`: BF: Added .eddy_rotated_bvecs extension support
* :ghissue:`1407`: [MRG] Default colormap changed in examples
* :ghissue:`1403`: Avoid promoting jet color map in examples
* :ghissue:`1408`: Updated color map in reconst_csa.py and reconst_forecast.py
* :ghissue:`1406`: [MRG] assert_true which checks for equality replaced with assert_equal
* :ghissue:`1387`: Assert equality, instead of asserting that `a == b` is true
* :ghissue:`1405`: Error using CSD model on data
* :ghissue:`1347`: Replacing fvtk by the new viz API
* :ghissue:`1402`: [Question] rint() or round()
* :ghissue:`1321`: mapfit_laplacian_aniso (high non-Gaussianity, NG values in CSF)
* :ghissue:`1161`: fvtk volume doesn't handle affine (crashes notebook)
* :ghissue:`1394`: Deprecation warning in newer versions of scipy, because `scipy.misc` is going away
* :ghissue:`1382`: is there any defined function that reads locally stored data or is all downloaded? I refer to nii or nifti files
* :ghissue:`1322`: Forecast
* :ghissue:`1326`: BUG: Fix factorial import module in test_mapmri.py.
* :ghissue:`1399`: New test errors on Python 3 Travis bots
* :ghissue:`1400`: BF: fixes #1399, removing an un-needed singleton dimension.
* :ghissue:`1350`: WIP: Add mapmri flow
* :ghissue:`1392`: Gitter chat box not visible on chrome?
* :ghissue:`1391`: Re-entering conflict-free typos from deleted PR 1331
* :ghissue:`1331`: Update gradients_spheres.py
* :ghissue:`1388`: Mapmri workflow
* :ghissue:`1386`: Possible fix for the inline compilation problem
* :ghissue:`1165`: Make vtk contour take an affine
* :ghissue:`1340`: NF - Particle Filtering Tractography
* :ghissue:`1383`: Mmriflow
* :ghissue:`1299`: test_rmi on 32 bit:  invalid dims: array size defined by dims is larger than the maximum possible size.
* :ghissue:`1300`: RF: Remove patch for older numpy ravel_multi_index.
* :ghissue:`1381`: DOC - re-orientation of figures in the DKI example
* :ghissue:`1301`: Brains need re-orientation in plotting in DKI example
* :ghissue:`1375`: Fix piesno type
* :ghissue:`1342`: Cythonize DirectionGetter and whatnot
* :ghissue:`1378`: Fix: numpy legacy print again...
* :ghissue:`1376`: New test failures with pre-release numpy
* :ghissue:`1377`: FIX: update printing format for numpy 1.14
* :ghissue:`1343`: ActiveAx model fitting using MIX framework
* :ghissue:`1374`: FIX: Viz test correction
* :ghissue:`1282`: Tests fail on viz module
* :ghissue:`1368`: DOC: Update developers' affiliations.
* :ghissue:`1370`: TST - add tracking tests for PeaksAndMetricsDirectionGetter
* :ghissue:`1369`: MRG: add procedure for building, uploading wheels
