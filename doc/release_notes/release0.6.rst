.. _release0.6:

===================================
 Release notes for DIPY version 0.6
===================================

GitHub stats for 2011/02/12 - 2013/03/20

The following 13 authors contributed 972 commits.

* Ariel Rokem
* Bago Amirbekian
* Eleftherios Garyfallidis
* Emanuele Olivetti
* Ian Nimmo-Smith
* Maria Luisa Mandelli
* Matthew Brett
* Maxime Descoteaux
* Michael Paquette
* Samuel St-Jean
* Stefan van der Walt
* Yaroslav Halchenko
* endolith

We closed a total of 225 issues, 100 pull requests and 125 regular issues;
this is the full list (generated with the script
:file:`tools/github_stats.py`):

Pull Requests (100):

* :ghpull:`146`: BF - allow Bootstrap Wrapper to work with markov tracking
* :ghpull:`143`: Garyfallidis tutorials 0.6
* :ghpull:`145`: Mdesco dti metrics
* :ghpull:`141`: Peak extraction isbi
* :ghpull:`142`: RF - always use theta and phi in that order, (not "phi, theta")
* :ghpull:`140`: Sf2sh second try at correcting suggestions
* :ghpull:`139`: Spherical function to spherical harmonics and back
* :ghpull:`138`: Coding style fix for dsi_deconv
* :ghpull:`137`: BF - check shapes before allclose
* :ghpull:`136`: BF: add top-level benchmarking command
* :ghpull:`135`: Refactor local maxima
* :ghpull:`134`: BF - fix shm tests to accept antipodal directions as the same
* :ghpull:`133`: Corrected test for Deconvolution after the discrete direction finder was removed
* :ghpull:`124`: Remove direction finder
* :ghpull:`77`: Rework tracking
* :ghpull:`132`: A new fvtk function for visualizing fields of odfs
* :ghpull:`131`: Add missing files
* :ghpull:`130`: Implementation of DSI deconvolution from E.J. Canales-Rodriguez
* :ghpull:`128`: Colorfa
* :ghpull:`129`: RF - minor cleanup of pdf_odf code
* :ghpull:`127`: Adding multi-tensor simulation
* :ghpull:`126`: Improve local maxima
* :ghpull:`122`: Removed calculation of gfa and other functions from inside the odf(sphere) of DSI and GQI
* :ghpull:`103`: Major update of the website, with a few examples and with some additional minor RFs
* :ghpull:`121`: NF: Allow the smoothing parameter to come through to rbf interpolation.
* :ghpull:`120`: Fast squash fix
* :ghpull:`116`: RF: common dtype for squash without result_type
* :ghpull:`117`: Fix directions on TensorFit and add getitem
* :ghpull:`119`: RF: raise errors for Python version dependencies
* :ghpull:`118`: Seperate fa
* :ghpull:`111`: RF - clean up _squash in multi_voxel and related code
* :ghpull:`112`: RF: fix vec_val_vect logic, generalize for shape
* :ghpull:`114`: BF: fix face and edge byte order for sphere load
* :ghpull:`109`: Faster einsum
* :ghpull:`110`: TST: This is only almost equal on XP, for some reason.
* :ghpull:`108`: TST + STY: Use and assert_equal so that we get more information upon failure
* :ghpull:`107`: RF: A.dot(B) => np.dot(A, B) for numpy < 1.5
* :ghpull:`102`: BF - Allow ndindex to work with older numpy than 1.6.
* :ghpull:`106`: RF: allow optional scipy.spatial.Delaunay
* :ghpull:`105`: Skip doctest decorator
* :ghpull:`104`: RF: remove deprecated old parametric testing
* :ghpull:`101`: WIP: Fix isnan windows
* :ghpull:`100`: Small stuff
* :ghpull:`94`: Multivoxel dsi and gqi are back!
* :ghpull:`96`: ENH: Implement masking for the new TensorModel implementation.
* :ghpull:`95`: NF fetch publicly available datasets
* :ghpull:`26`: Noise
* :ghpull:`84`: Non linear peak finding
* :ghpull:`82`: DTI new api
* :ghpull:`91`: Shm new api
* :ghpull:`88`: NF - wrapper function for multi voxel models
* :ghpull:`86`: DOC: Fixed some typos, etc in the FAQ
* :ghpull:`90`: A simpler ndindex using generators.
* :ghpull:`87`: RF - Provide shape as argument to ndindex.
* :ghpull:`85`: Add fast ndindex.
* :ghpull:`81`: RF - fixup peaks_from_model to take use remove_similar_vertices and
* :ghpull:`79`: BF: Fixed projection plots.
* :ghpull:`80`: RF - remove some old functions tools
* :ghpull:`71`: ENH: Make the internals of the io module visible on tab completion in ip...
* :ghpull:`76`: Yay, more gradient stuff
* :ghpull:`75`: Rename L2norm to vector_norm
* :ghpull:`74`: Gradient rf
* :ghpull:`73`: RF/BF - removed duplicate vector_norm/L2norm
* :ghpull:`72`: Mr bago model api
* :ghpull:`68`: DSI seems working again - Have a look
* :ghpull:`65`: RF: Make the docstring and call consistent with scipy.interpolate.Rbf.
* :ghpull:`61`: RF - Refactor direction finding.
* :ghpull:`60`: NF - Add key-value cache for use in models.
* :ghpull:`63`: TST - Disable reconstruction methods that break the test suite.
* :ghpull:`62`: BF - Fix missing import in peak finding tests.
* :ghpull:`37`: cleanup refrences in the code to E1381S6_edcor* (these were removed from...
* :ghpull:`55`: Ravel multi index
* :ghpull:`58`: TST - skip doctest when matplotlib is not available
* :ghpull:`59`: optional_traits is not needed anymore
* :ghpull:`56`: TST: Following change to API in dipy.segment.quickbundles.
* :ghpull:`52`: Matplotlib optional
* :ghpull:`50`: NF - added subdivide method to sphere
* :ghpull:`51`: Fix tracking utils
* :ghpull:`48`: BF - Brought back _filter peaks and associated test.
* :ghpull:`47`: RF - Removed reduce_antipodal from sphere.
* :ghpull:`41`: NF - Add radial basis function interpolation on the sphere.
* :ghpull:`39`: GradientTable
* :ghpull:`40`: BF - Fix axis specification in sph_project.
* :ghpull:`28`: Odf+shm api update
* :ghpull:`36`: Nf hemisphere preview
* :ghpull:`34`: RF - replace _filter_peaks with unique_vertices
* :ghpull:`35`: BF - Fix imports from dipy.core.sphere.
* :ghpull:`21`: Viz 2d
* :ghpull:`32`: NF - Sphere class.
* :ghpull:`30`: RF: Don't import all this every time.
* :ghpull:`24`: TST: Fixing tests in reconst module.
* :ghpull:`27`: DOC - Add reference to white matter diffusion values.
* :ghpull:`25`: NF - Add prolate white matter as defaults for multi-tensor signal sim.
* :ghpull:`22`: Updating my fork with the nipy master
* :ghpull:`20`: RF - create OptionalImportError for traits imports
* :ghpull:`19`: DOC: add comments and example to commit codes
* :ghpull:`18`: DOC: update gitwash from source
* :ghpull:`17`: Optional traits
* :ghpull:`14`: DOC - fix frontpage example
* :ghpull:`12`: BF(?): cart2sphere and sphere2cart are now invertible.
* :ghpull:`11`: BF explicit type declaration and initialization for longest_track_len[AB] -- for cython 0.15 compatibility

Issues (125):

* :ghissue:`99`: RF - Separate direction finder from model fit.
* :ghissue:`143`: Garyfallidis tutorials 0.6
* :ghissue:`144`: DTI metrics
* :ghissue:`145`: Mdesco dti metrics
* :ghissue:`123`: Web content and examples for 0.6
* :ghissue:`141`: Peak extraction isbi
* :ghissue:`142`: RF - always use theta and phi in that order, (not "phi, theta")
* :ghissue:`140`: Sf2sh second try at correcting suggestions
* :ghissue:`139`: Spherical function to spherical harmonics and back
* :ghissue:`23`: qball not properly import-able
* :ghissue:`29`: Don't import everything when you import dipy
* :ghissue:`138`: Coding style fix for dsi_deconv
* :ghissue:`137`: BF - check shapes before allclose
* :ghissue:`136`: BF: add top-level benchmarking command
* :ghissue:`135`: Refactor local maxima
* :ghissue:`134`: BF - fix shm tests to accept antipodal directions as the same
* :ghissue:`133`: Corrected test for Deconvolution after the discrete direction finder was removed
* :ghissue:`124`: Remove direction finder
* :ghissue:`77`: Rework tracking
* :ghissue:`132`: A new fvtk function for visualizing fields of odfs
* :ghissue:`125`: BF: Remove 'mayavi' directory, to avoid triggering mayavi import warning...
* :ghissue:`131`: Add missing files
* :ghissue:`130`: Implementation of DSI deconvolution from E.J. Canales-Rodriguez
* :ghissue:`128`: Colorfa
* :ghissue:`129`: RF - minor cleanup of pdf_odf code
* :ghissue:`127`: Adding multi-tensor simulation
* :ghissue:`126`: Improve local maxima
* :ghissue:`97`: BF - separate out storing of fit values in gqi
* :ghissue:`122`: Removed calculation of gfa and other functions from inside the odf(sphere) of DSI and GQI
* :ghissue:`103`: Major update of the website, with a few examples and with some additional minor RFs
* :ghissue:`121`: NF: Allow the smoothing parameter to come through to rbf interpolation.
* :ghissue:`120`: Fast squash fix
* :ghissue:`116`: RF: common dtype for squash without result_type
* :ghissue:`117`: Fix directions on TensorFit and add getitem
* :ghissue:`119`: RF: raise errors for Python version dependencies
* :ghissue:`118`: Seperate fa
* :ghissue:`113`: RF - use min_diffusivity relative to 1 / max(bval)
* :ghissue:`111`: RF - clean up _squash in multi_voxel and related code
* :ghissue:`112`: RF: fix vec_val_vect logic, generalize for shape
* :ghissue:`114`: BF: fix face and edge byte order for sphere load
* :ghissue:`109`: Faster einsum
* :ghissue:`110`: TST: This is only almost equal on XP, for some reason.
* :ghissue:`98`: This is an update of PR #94 mostly typos and coding style
* :ghissue:`108`: TST + STY: Use and assert_equal so that we get more information upon failure
* :ghissue:`107`: RF: A.dot(B) => np.dot(A, B) for numpy < 1.5
* :ghissue:`102`: BF - Allow ndindex to work with older numpy than 1.6.
* :ghissue:`106`: RF: allow optional scipy.spatial.Delaunay
* :ghissue:`105`: Skip doctest decorator
* :ghissue:`104`: RF: remove deprecated old parametric testing
* :ghissue:`101`: WIP: Fix isnan windows
* :ghissue:`100`: Small stuff
* :ghissue:`94`: Multivoxel dsi and gqi are back!
* :ghissue:`96`: ENH: Implement masking for the new TensorModel implementation.
* :ghissue:`95`: NF fetch publicly available datasets
* :ghissue:`26`: Noise
* :ghissue:`84`: Non linear peak finding
* :ghissue:`82`: DTI new api
* :ghissue:`91`: Shm new api
* :ghissue:`88`: NF - wrapper function for multi voxel models
* :ghissue:`86`: DOC: Fixed some typos, etc in the FAQ
* :ghissue:`89`: Consisten ndindex behaviour
* :ghissue:`90`: A simpler ndindex using generators.
* :ghissue:`87`: RF - Provide shape as argument to ndindex.
* :ghissue:`85`: Add fast ndindex.
* :ghissue:`81`: RF - fixup peaks_from_model to take use remove_similar_vertices and
* :ghissue:`83`: Non linear peak finding
* :ghissue:`78`: This PR replaces PR 70
* :ghissue:`79`: BF: Fixed projection plots.
* :ghissue:`80`: RF - remove some old functions tools
* :ghissue:`70`: New api dti
* :ghissue:`71`: ENH: Make the internals of the io module visible on tab completion in ip...
* :ghissue:`76`: Yay, more gradient stuff
* :ghissue:`69`: New api and tracking refacotor
* :ghissue:`75`: Rename L2norm to vector_norm
* :ghissue:`74`: Gradient rf
* :ghissue:`73`: RF/BF - removed duplicate vector_norm/L2norm
* :ghissue:`72`: Mr bago model api
* :ghissue:`66`: DOCS - docs for model api
* :ghissue:`49`: Reworking tracking code.
* :ghissue:`68`: DSI seems working again - Have a look
* :ghissue:`65`: RF: Make the docstring and call consistent with scipy.interpolate.Rbf.
* :ghissue:`61`: RF - Refactor direction finding.
* :ghissue:`60`: NF - Add key-value cache for use in models.
* :ghissue:`63`: TST - Disable reconstruction methods that break the test suite.
* :ghissue:`62`: BF - Fix missing import in peak finding tests.
* :ghissue:`37`: cleanup refrences in the code to E1381S6_edcor* (these were removed from...
* :ghissue:`55`: Ravel multi index
* :ghissue:`46`: BF: Trying to fix test failures.
* :ghissue:`57`: TST: Reverted back to optional definition of the function to make TB hap...
* :ghissue:`58`: TST - skip doctest when matplotlib is not available
* :ghissue:`59`: optional_traits is not needed anymore
* :ghissue:`56`: TST: Following change to API in dipy.segment.quickbundles.
* :ghissue:`52`: Matplotlib optional
* :ghissue:`50`: NF - added subdivide method to sphere
* :ghissue:`51`: Fix tracking utils
* :ghissue:`48`: BF - Brought back _filter peaks and associated test.
* :ghissue:`47`: RF - Removed reduce_antipodal from sphere.
* :ghissue:`41`: NF - Add radial basis function interpolation on the sphere.
* :ghissue:`33`: Gradients Table class
* :ghissue:`39`: GradientTable
* :ghissue:`45`: BF - Fix sphere creation in triangle_subdivide.
* :ghissue:`38`: Subdivide octahedron
* :ghissue:`40`: BF - Fix axis specification in sph_project.
* :ghissue:`28`: Odf+shm api update
* :ghissue:`36`: Nf hemisphere preview
* :ghissue:`34`: RF - replace _filter_peaks with unique_vertices
* :ghissue:`35`: BF - Fix imports from dipy.core.sphere.
* :ghissue:`21`: Viz 2d
* :ghissue:`32`: NF - Sphere class.
* :ghissue:`30`: RF: Don't import all this every time.
* :ghissue:`24`: TST: Fixing tests in reconst module.
* :ghissue:`27`: DOC - Add reference to white matter diffusion values.
* :ghissue:`25`: NF - Add prolate white matter as defaults for multi-tensor signal sim.
* :ghissue:`22`: Updating my fork with the nipy master
* :ghissue:`20`: RF - create OptionalImportError for traits imports
* :ghissue:`8`: X error BadRequest with fvtk.show
* :ghissue:`19`: DOC: add comments and example to commit codes
* :ghissue:`18`: DOC: update gitwash from source
* :ghissue:`17`: Optional traits
* :ghissue:`15`: Octahedron in dipy.core.triangle_subdivide has wrong faces
* :ghissue:`14`: DOC - fix frontpage example
* :ghissue:`12`: BF(?): cart2sphere and sphere2cart are now invertible.
* :ghissue:`11`: BF explicit type declaration and initialization for longest_track_len[AB] -- for cython 0.15 compatibility
* :ghissue:`5`: Add DSI reconstruction in Dipy
* :ghissue:`9`: Bug in dipy.tracking.metrics.downsampling when we downsample a track to more than 20 points
