.. _release0.13:

====================================
 Release notes for DIPY version 0.13
====================================


GitHub stats for 2017/06/27 - 2017/10/24 (tag: 0.12.0)

These lists are automatically generated, and may be incomplete or contain duplicates.

The following 13 authors contributed 212 commits.

* Ariel Rokem
* Bennet Fauber
* David Reagan
* Eleftherios Garyfallidis
* Guillaume Theaud
* Jon Haitz Legarreta Gorroño
* Marc-Alexandre Côté
* Matthieu Dumont
* Rafael Neto Henriques
* Ranveer Aggarwal
* Rutger Fick
* Saber Sheybani
* Serge Koudoro


We closed a total of 115 issues, 39 pull requests and 76 regular issues;
this is the full list (generated with the script
:file:`tools/github_stats.py`):

Pull Requests (39):

* :ghpull:`1367`: BF: Import Streamlines object directly from nibabel.
* :ghpull:`1361`: Windows instructions update + citation update
* :ghpull:`1316`: DOC: Add a coding style guideline.
* :ghpull:`1360`: [FIX] references order in workflow
* :ghpull:`1348`: ENH: Add support for ArraySequence in`set_number_of_points` function
* :ghpull:`1357`: Update tracking_quick_start.py rebase of #1332
* :ghpull:`1239`: Enable memory profiling of the examples.
* :ghpull:`1356`: Add picking to slicer - rebase
* :ghpull:`1351`: From tables to h5py
* :ghpull:`1353`: FIX : improve epilogue accessibility for workflow
* :ghpull:`1262`: VIZ: A lightweight UI for medical visualizations #6: 2D File Selector
* :ghpull:`1352`: use legacy float array for printing numpy array
* :ghpull:`1314`: DOC: Fix typos and formatting in .rst files and Python examples.
* :ghpull:`1345`: DOC: Format README.rst file code blocks.
* :ghpull:`1330`: ENH: Add Travis badge to README.rst.
* :ghpull:`1315`: Remove GPL from our README.
* :ghpull:`1328`: BUG: Address small_delta vs. big_delta flipped parameters.
* :ghpull:`1329`: DOC: Fix typos in multi_io.py workflow file docstring.
* :ghpull:`1336`: Test modification for windows 10 / numpy 1.14
* :ghpull:`1335`: Catch a more specific warning in test_csdeconv
* :ghpull:`1319`: Correct white-space for fwdti example.
* :ghpull:`1297`: Added eigh version of localpca to svd version
* :ghpull:`1298`: Make TextActor2D extend UI instead of object
* :ghpull:`1312`: Flags correction for windows
* :ghpull:`1285`: mapmri using cvxpy instead of cvxopt
* :ghpull:`1307`: PyTables Error-handling
* :ghpull:`1310`: Fix error message
* :ghpull:`1308`: Fix inversion in the dti mode doc
* :ghpull:`1304`: DOC: Fix typos in dti.py reconstruction file doc.
* :ghpull:`1303`: DOC: Add missing label to reciprocal space eq.
* :ghpull:`1289`: MRG: Suppress a divide-by-zero warning.
* :ghpull:`1288`: NF Add the parameter fa_operator in auto_response function
* :ghpull:`1290`: Corrected a small error condition
* :ghpull:`1279`: UI advanced fix
* :ghpull:`1287`: Fix doc errors
* :ghpull:`1286`: Last doc error fix on 0.12.x
* :ghpull:`1284`: Added missing tutorials
* :ghpull:`1278`: Moving ahead with 0.13 (dev version)
* :ghpull:`1277`: One test (decimal issue) and a fix in viz_ui tutorial.

Issues (76):

* :ghissue:`1367`: BF: Import Streamlines object directly from nibabel.
* :ghissue:`1366`: Circular imports in dipy.tracking.utils?
* :ghissue:`1146`: Installation instructions for windows need to be updated
* :ghissue:`1084`: Installation for windows developers using Anaconda needs to be updated
* :ghissue:`1361`: Windows instructions update + citation update
* :ghissue:`1248`: Windows doc installation update is needed for Python 3, Anaconda and VTK support
* :ghissue:`1316`: DOC: Add a coding style guideline.
* :ghissue:`1360`: [FIX] references order in workflow
* :ghissue:`1359`: Epilogue's reference should be last not first
* :ghissue:`1324`: WIP: Det track workflow and other improvements in workflows
* :ghissue:`1348`: ENH: Add support for ArraySequence in`set_number_of_points` function
* :ghissue:`1357`: Update tracking_quick_start.py rebase of #1332
* :ghissue:`1332`: Update tracking_quick_start.py
* :ghissue:`1239`: Enable memory profiling of the examples.
* :ghissue:`1356`: Add picking to slicer - rebase
* :ghissue:`1334`: Add picking to slicer
* :ghissue:`1351`: From tables to h5py
* :ghissue:`1353`: FIX : improve epilogue accessibility for workflow
* :ghissue:`1344`: Check accessibility of epilogue in Workflows
* :ghissue:`1262`: VIZ: A lightweight UI for medical visualizations #6: 2D File Selector
* :ghissue:`1352`: use legacy float array for printing numpy array
* :ghissue:`1346`: Test broken in numpy 1.14
* :ghissue:`1333`: Trying QuickBundles (Python3 and vtk--> using: conda install -c clinicalgraphics vtk)
* :ghissue:`1044`: Reconstruction FOD
* :ghissue:`1247`: Interactor bug in viz_ui example
* :ghissue:`1314`: DOC: Fix typos and formatting in .rst files and Python examples.
* :ghissue:`1345`: DOC: Format README.rst file code blocks.
* :ghissue:`1349`: Doctest FIX : use legacy printing
* :ghissue:`1330`: ENH: Add Travis badge to README.rst.
* :ghissue:`1337`: Coveralls seems baggy let's remove it
* :ghissue:`1341`: ActiveAx model fitting using MIX framework
* :ghissue:`1315`: Remove GPL from our README.
* :ghissue:`1325`: Small is Big - Big is small (mapl - mapmri)
* :ghissue:`1328`: BUG: Address small_delta vs. big_delta flipped parameters.
* :ghissue:`1329`: DOC: Fix typos in multi_io.py workflow file docstring.
* :ghissue:`1336`: Test modification for windows 10 / numpy 1.14
* :ghissue:`1323`: Warnings raised in csdeconv for upcoming numpy 1.14
* :ghissue:`1335`: Catch a more specific warning in test_csdeconv
* :ghissue:`1042`: RF - move direction getters to dipy/direction/
* :ghissue:`1319`: Correct white-space for fwdti example.
* :ghissue:`1317`: reconst_fwdti.py example figures not being rendered
* :ghissue:`1297`: Added eigh version of localpca to svd version
* :ghissue:`1313`: No module named 'vtkCommonCore'
* :ghissue:`1318`: Mix framework with Cythonized func_mul
* :ghissue:`1167`: Potential replacement for CVXOPT?
* :ghissue:`1180`: WIP: replacing cvxopt with cvxpy.
* :ghissue:`1298`: Make TextActor2D extend UI instead of object
* :ghissue:`375`: Peak directiions test error on PPC
* :ghissue:`1312`: Flags correction for windows
* :ghissue:`804`: Wrong openmp flag on Windows
* :ghissue:`1285`: mapmri using cvxpy instead of cvxopt
* :ghissue:`662`: dipy/align/mattes.pyx doctest import error
* :ghissue:`1307`: PyTables Error-handling
* :ghissue:`1306`: Error-handling when pytables not installed
* :ghissue:`1309`: step_helpers gives a wrong error message
* :ghissue:`1310`: Fix error message
* :ghissue:`1308`: Fix inversion in the dti mode doc
* :ghissue:`1304`: DOC: Fix typos in dti.py reconstruction file doc.
* :ghissue:`1303`: DOC: Add missing label to reciprocal space eq.
* :ghissue:`1289`: MRG: Suppress a divide-by-zero warning.
* :ghissue:`1293`: Garyfallidis recobundles
* :ghissue:`1292`: Garyfallidis recobundles
* :ghissue:`1288`: NF Add the parameter fa_operator in auto_response function
* :ghissue:`1290`: Corrected a small error condition
* :ghissue:`1279`: UI advanced fix
* :ghissue:`1287`: Fix doc errors
* :ghissue:`1286`: Last doc error fix on 0.12.x
* :ghissue:`1284`: Added missing tutorials
* :ghissue:`322`: Missing content in tracking.utils' documentation
* :ghissue:`570`: The documentation for `dipy.viz` is not in the API reference
* :ghissue:`1053`: WIP: Local pca and noise estimation
* :ghissue:`881`: PEP8 in reconst
* :ghissue:`880`: PEP8 in reconst
* :ghissue:`1169`: Time for a new release - scipy 0.18?
* :ghissue:`1278`: Moving ahead with 0.13 (dev version)
* :ghissue:`1277`: One test (decimal issue) and a fix in viz_ui tutorial.
