============
API changes
============

Here we provide information about functions or classes that have been removed,
renamed or are deprecated (not recommended) during different release circles.

DIPY 1.10.0 changes
------------------

**Workflows**
- The `vol_idx` parameter datatype from ``dipy_median_otsu`` has been changed from `variable int` to `str`.
  this change allows user to provide a range of values for the `vol_idx` parameter. e.g: `--vol_idx 0,1,2` or `--vol_idx 4,5,12-20,22`.

DIPY 1.9.0 changes
------------------

**General**

- The module ``dipy.boots.resampling`` has moved to ``dipy.stats.resampling``.
- The package ``dipy.boots`` has been removed.
- FURY minimum version is 0.10.0.
- Multiple deprecated parameters have been removed from the codebase.

**IO**

- ``dipy.io.bvectxt`` module is removed

DIPY 1.8.0 changes
------------------
**Gradients**

- Change in ``dipy.core.gradients``, function ``reorient_bvecs`` now requires the affine to have a shape of (4, 4, n) or (3, 3, n)

**Direction**
- Change in ``dipy.direction.bootstrap_direction_getter``.
    - The parent class was changes from ``PmfGenDirectionGetter`` to ``DirectionGetter``. The ``BootPmfGen`` functions were merged in ``BootDirectionGetter``.
    - The class constructor parameter `pmfgen` was removed. Parameters `data`, `model` and `sh_order=0` were added.
    - The class method `BootDirectionGetter.from_data()` was not changed.
- Change in ``dipy.direction.pmf``. The class ``BootPmfGen`` was removed; its functions were merged in ``BootDirectionGetter``.

DIPY 1.7.0 changes
------------------

**Denoising**

- Change in ``dipy.denoise.localpca``, function ``genpca`` can use fewer images than patch voxels.
- Change in ``dipy.denoise.pca_noise_estimate``, function ``pca_noise_estimate`` has new argument ``images_as_samples``

DIPY 1.6.0 changes
------------------


DIPY 1.5.0 changes
------------------
**General**

- FURY minimum version is 0.8.0
- Distutils has been dropped
- ``dipy.io.bvectxt`` module is deprecated and will be removed

**Denoising**

- The default option in the command line for Patch2Self 'ridge' -> 'ols'

**Tracking**

- Change in ``dipy.tracking.pmf``
    - The parent class ``PmfGen`` has new mandatory parameter ``sphere``. The sphere vertices correspond to the spherical distribution of the pmf values.
    - The parent class ``PmfGen`` has new function ``get_pmf_value(point, xyz)`` which return the pmf value at location ``point`` and orientation ``xyz``.

**Segment**

- The deprecated ``from dipy.segment.metric import ResampleFeature`` was removed and replaced by ``from dipy.segment.featurespeed import ResampleFeature``.


DIPY 1.4.1 changes
------------------

**General**

- The name of the argument for the number of cores/threads has been standardized to:
    - ``num_threads`` for OpenMP parallelization.
    - ``num_processes`` for parallelization using multiprocessing package.
- Change in the parallelization logic when using OpenMP:
    - If ``num_threads = None`` the value of ``OMP_NUM_THREADS`` environment variable is used. If it is not set then all available threads are used.
    - If ``num_threads > 0`` that number is used as the number of threads.
    - If ``num_threads < 0`` the maximum between ``1`` and ``num_cpu_cores - |num_threads + 1|`` is selected. If ``-1`` then all available threads are used.
    - If ``num_threads = 0`` an error is raised.
- Change in the parallelization logic when using multiprocessing package:
    - The same as with OpenMP with the difference that ``num_processes = None`` uses all cores directly.

**Tracking**

- Change in DirectionGetters:
    - The deprecated ``dipy.direction.closest_peak_direction_getter.BaseDirectionGetter`` was removed and replaced by ``dipy.direction.closest_peak_direction_getter.BasePmfDirectionGetter``.
    - The deprecated ``dipy.reconst.EuDXDirectionGetter`` was removed and replaced by ``dipy.reconst.eudx_direction_getter.EuDXDirectionGetter``.

DIPY 1.4.0 changes
------------------

- Migration from Tavis to Azure

DIPY 1.3.0 changes
------------------

- new dependency added: tqdm

**Registration**

- The argument `interp` of the method `dipy.align.imaffine.AffineMap.transform`  has been renamed `interpolation`.
- The argument `interp` of the method `dipy.align.imaffine.AffineMap.transform_inverse`  has been renamed `interpolation`.

**Segmentation**

- The tissue segmentation method ``dipy.segment.TissueClassifierHMRF`` now checks the tolerance-based stopping criterion at every iteration (previously it was only checked every 10th iteration). This may result in earlier termination of iterations than with previous releases.

DIPY 1.2.0 changes
------------------

**Reconstruction**

The ``dipy.reconst.csdeconv.auto_response`` has been renamed
``dipy.reconst.csdeconv.auto_response_ssst``.

The ``dipy.reconst.csdeconv.response_from_mask`` has been renamed
``dipy.reconst.csdeconv.response_from_mask_ssst``.

The ``dipy.sims.voxel.multi_shell_fiber_response`` has been moved to
``dipy.reconst.mcsd.multi_shell_fiber_response``.

**Segmentation**

In prior releases, for users with SciPy < 1.5, a memory overlap bug occurs in
``multi_median``, causing an overly smooth output. This has now been fixed,
regardless of the user's installed SciPy version. Users of this function via
``median_otsu`` thresholding should check the output of their image processing
pipelines after the 1.2.0 release to make sure thresholding is still operating
as expected (if not, try readjusting the ``median_radius`` parameter).

**Tracking**

The ``dipy.reconst.peak_direction_getter.EuDXDirectionGetter`` has
been renamed ``dipy.reconst.eudx_direction_getter.EuDXDirectionGetter``.

The command line ``dipy_track_local`` has been renamed ``dipy_track``.


**Others**

The ``dipy.core.gradients.unique_bvals`` has been renamed
``dipy.core.gradients.unique_bvals_magnitude``.


**Visualization**

- Use ``window.Scene()`` instead of ``window.Renderer()``.
- Use ``scene.clear()`` instead of ``window.rm_all(scene)``.
- Use ``scene.clear()`` instead of ``window.clear(scene)``.


DIPY 1.1.1 changes
------------------

**IO**

``img.get_data()`` is deprecated since Nibabel 3.0.0. Using ``np.asanyarray(img.dataobj)`` instead of ``img.get_data()``.

**Tractogram**

``dipy.io.streamlines.StatefulTractogram`` can be created by another one.

**Workflows**

``dipy_nlmeans`` command lines have been renamed ``dipy_denoise_nlmeans``.

**Others**

``get_data`` has been deprecated by Nibabel and replaced by ``get_fdata``. This modification has been
applied to all the codebase. The default datatype is now float64.


DIPY 1.0.0 changes
------------------
Some of the changes introduced in the 1.0 release will break backward
compatibility with previous versions. This release is compatible with Python 3.5+

**Reconstruction**

The spherical harmonics bases ``mrtrix`` and ``fibernav`` have been renamed to
``tournier07`` and ``descoteaux07`` after the deprecation cycle started in the
0.15 release.

We changed ``dipy.data.default_sphere`` from symmetric724 to repulsion724 which is
more evenly distributed.

**Segmentation**

The API of ``dipy.segment.mask.median_otsu`` has changed in the following ways:
if you are providing a 4D volume, `vol_idx` is now a required argument.
The order of parameters has also changed.

**Tractogram loading and saving**

The API of ``dipy.io.streamlines.load_tractogram`` and
``dipy.io.streamlines.save_tractogram`` has changed in the following ways:
When loading trk, tck, vtk, fib, or dpy) a reference nifti file is needed to
guarantee proper spatial transformation handling.

**Spatial transformation handling**

Functions from ``dipy.tracking.streamlines`` were modified to enforce the
affine parameter and uniform docstrings. ``deform_streamlines``
``select_by_rois``, ``orient_by_rois``, ``_extract_vals``
and ``values_from_volume``.

Functions from ``dipy.tracking.utils`` were modified to enforce the
affine parameter and uniform docstring. ``density_map``
``connectivity_matrix``, ``seeds_from_mask``, ``random_seeds_from_mask``,
``target``, ``target_line_based``, ``near_roi``, ``length`` and
``path_length`` were all modified.

The function ``affine_for_trackvis``, ``move_streamlines``,
``flexi_tvis_affine`` and ``get_flexi_tvis_affine`` were deleted.

Functions from ``dipy.tracking.life`` were modified to enforce the
affine parameter and uniform docstring. ``voxel2streamline``,
``setup`` and ``fit`` from class ``FiberModel`` were all modified.

``afq_profile`` from ``dipy.stats.analysis`` was modified similarly.

**Simulations**

- ``dipy.sims.voxel.SingleTensor`` has been replaced by ``dipy.sims.voxel.single_tensor``
- ``dipy.sims.voxel.MultiTensor`` has been replaced by ``dipy.sims.voxel.multi_tensor``
- ``dipy.sims.voxel.SticksAndBall`` has been replaced by ``dipy.sims.voxel.sticks_and_ball``

**Interpolation**

All interpolation functions have been moved to a new module name `dipy.core.interpolation`

**Tracking**

The `voxel_size` parameter has been removed from the following function:

- ``dipy.tracking.utils.connectivity_matrix``
- ``dipy.tracking.utils.density_map``
- ``dipy.tracking.utils.stremline_mapping``
- ``dipy.tracking._util._mapping_to_voxel``

The ``dipy.reconst.peak_direction_getter.PeaksAndMetricsDirectionGetter`` has
been renamed ``dipy.reconst.peak_direction_getter.EuDXDirectionGetter``.

The `LocalTracking` and `ParticleFilteringTracking` functions were moved from
``dipy.tracking.local.localtracking`` to ``dipy.tracking.local_tracking``.
They now need to be imported from ``dipy.tracking.local_tracking``.

- functions argument `tissue_classifier` were renamed `stopping_criterion`

The `TissueClassifier` were renamed `StoppingCriterion` and moved from
``dipy.tracking.local.tissue_classifier`` to ``dipy.tracking.stopping_criterion``.
They now need to be imported from ``dipy.tracking.stopping_criterion``.

- `TissueClassifier` -> `StoppingCriterion`
- `BinaryTissueClassifier` -> `BinaryStoppingCriterion`
- `ThresholdTissueClassifier` -> `ThresholdStoppingCriterion`
- `ConstrainedTissueClassifier` -> `AnatomicalStoppingCriterion`
- `ActTissueClassifier` -> `ActStoppingCriterion`
- `CmcTissueClassifier` -> `CmcStoppingCriterion`

The ``dipy.tracking.local.tissue_classifier.TissueClass`` was renamed
``dipy.tracking.stopping_criterion.StreamlineStatus``.

The `EuDX` tracking function has been removed. EuDX tractography can be
performed using ``dipy.tracking.local_tracking`` using
``dipy.reconst.peak_direction_getter.EuDXDirectionGetter``.

**Streamlines**

``dipy.io.trackvis`` has been removed. Use ``dipy.io.streamline`` instead.

**Other**

- ``dipy.external`` package has been removed.
- ``dipy.fixes`` package has been removed.
- ``dipy.segment.quickbundes`` module has been removed.
- ``dipy.reconst.peaks`` module has been removed.
- Compatibility with Python 2.7 has been removed.

DIPY 0.16 Changes
-----------------

**Stats**

Welcome to the new module ``dipy.viz.stats``. This module will be used to integrate various analyses.

**Tracking**

- New option to adjust the number of threads for SLR in Recobundles
- The tracking algorithm excludes the stop point inside the mask during the tracking process.

**Notes**

- Replacement of Nose by Pytest


DIPY 0.15 Changes
-----------------

**IO**

``load_tck`` and ``save_tck`` from ``dipy.io.streamline`` have been added. They are highly recommended for managing streamlines.

**Gradient Table**

The default value of ``b0_thresold`` has been changed(from 0 to 50). This change can impact your algorithm.
If you want to assure that your code runs in exactly the same manner as before, please initialize your gradient table with the keyword argument ``b0_threshold`` set to 0.

**Visualization**

``dipy.viz.fvtk`` module has been removed. Use ``dipy.viz.*`` instead. This implies the following important changes:
- Use ``from dipy.viz import window, actor`` instead of ``from dipy.viz import fvtk`.
- Use ``window.Renderer()`` instead of ``fvtk.ren()``.
- All available actors are in ``dipy.viz.actor`` instead of ``dipy.fvtk.actor``.
- UI elements are available in ``dipy.viz.ui``.

``dipy.viz`` depends on the FURY package. To learn more about FURY, go to https://fury.gl


DIPY 0.14 Changes
-----------------

**Streamlines**

``dipy.io.trackvis`` module is deprecated. Use ``dipy.io.streamline`` instead. Furthermore,
``load_trk`` and ``save_trk`` from ``dipy.io.streamline`` is highly recommended for managing streamlines.
When you create streamlines, you should use ``from dipy.tracking.streamlines import Streamlines``. This new
object uses much less memory and it is easier to process.

**Visualization**

``dipy.viz.fvtk`` module is deprecated. Use ``dipy.viz.*`` instead. This implies the following important changes:
- Use ``from dipy.viz import window, actor`` instead of ``from dipy.viz import fvtk`.
- Use ``window.Renderer()`` instead of ``fvtk.ren()``.
- All available actors are in ``dipy.viz.actor`` instead of ``dipy.fvtk.actor``.
- UI elements are available in ``dipy.viz.ui``.


DIPY 0.13 Changes
-----------------

No major API changes.

**Notes**

``dipy.io.trackvis`` module will be deprecated on release 0.14. Use ``dipy.io.streamline`` instead.
``dipy.viz.fvtk`` module will be deprecated on release 0.14. Use ``dipy.viz.ui`` instead.


DIPY 0.12 Changes
-----------------
**Dropped support for Python 2.6***

It has been 6 years since the release of Python 2.7, and multiple other
versions have been released since. As far as we know, DIPY still works well
on Python 2.6, but we no longer test on this version, and we recommend that
users upgrade to Python 2.7 or newer to use DIPY.


**Tracking**

``probabilistic_direction_getter.ProbabilisticDirectionGetter`` input parameters
have changed. Now the optional parameter ``pmf_threshold=0.1`` (previously fixed
to 0.0) removes directions with probability lower than ``pmf_threshold`` from
the probability mass function (pmf) when selecting the tracking direction.

**DKI**

The default of DKI model fitting was changed from "OLS" to "WLS".

The default max_kurtosis of the functions axial_kurtosis, mean_kurtosis,
radial_kurotis was changed from 3 to 10.

**Visualization**

Prefer using the UI elements in ``dipy.viz.ui`` rather than
``dipy.viz.widgets``.

**IO**

Use the module ``nibabel.streamlines`` for saving trk files and not
``nibabel.trackvis``. Requires upgrading to nibabel 2+.

DIPY 0.10 Changes
-----------------

**New visualization module**

``fvtk.slicer`` input parameters have changed. Now the slicer function is
more powerful and supports RGB images too. See tutorial ``viz_slice.py`` for
more information.

**Interpolation**
The default behavior of the function `core.sphere.interp_rbf` has changed.
The default smoothing parameter is now set to 0.1 (previously 0). In addition,
the default norm is now `angle` (was previously `euclidean_norm`). Note that
the use of `euclidean_norm` is discouraged, and this norm will be deprecated
in the 0.11 release cycle.

**Registration**

The following utility functions from ``vector_fields`` module were renamed:

``warp_2d_affine`` is now ``transform_2d_affine``
``warp_2d_affine_nn`` is now ``transform_2d_affine_nn``
``warp_3d_affine`` is now ``transform_3d_affine``
``warp_3d_affine_nn`` is now ``transform_3d_affine_nn``


DIPY 0.9 Changes
----------------

**GQI integration length**

The calculation of integration length in GQI2 now matches the calculation in the
'standard' method. Using values of 1-1.3 for either is recommended (see
docs and references therein).


DIPY 0.8 Changes
----------------

**Peaks**

The module ``peaks`` is now available from ``dipy.direction`` and it can still
be accessed from ``dipy.reconst`` but it will be completely removed in version
0.10.

**Resample**

The function ``resample`` from ``dipy.align.aniso2iso`` is deprecated. Please,
use instead ``reslice`` from ``dipy.align.reslice``. The module ``aniso2iso``
will be completely removed in version 0.10.


Changes between 0.7.1 and 0.6
------------------------------

**Peaks_from_model**

The function ``peaks_from_model`` is now available from ``dipy.reconst.peaks``
. Please replace all imports like::

    from dipy.reconst.odf import peaks_from_model

with::

    from dipy.reconst.peaks import peaks_from_model

**Target**

The function ``target`` from ``dipy.tracking.utils`` now takes an affine
transform instead of a voxel sizes array. Please update all code using
``target`` in a way similar to this::

    img = nib.load(anat)
    voxel_dim = img.header['pixdim'][1:4]
    streamlines = utils.target(streamlines, img.get_data(), voxel_dim)

to something similar to::

    img = nib.load(anat)
    streamlines = utils.target(streamlines, img.get_data(), img.affine)
