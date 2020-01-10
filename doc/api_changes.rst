============
API changes
============

Here we provide information about functions or classes that have been removed,
renamed or are deprecated (not recommended) during different release circles.

DIPY 1.1.1 changes
----------------

**IO**

`img.get_data()` is deprecated since Nibabel 3.0.0. Using `np.asanyarray(img.dataobj)` instead of `img.get_data()`.

**Tractogram**

``dipy.io.streamlines.StatefulTractogram``can be created by an other one.

**Workflows**

``dipy_nlmeans`` command lines has been renamed ``dipy_denoise_nlmeans``.

**Others**

``get_data`` has been deprecated by Nibabel and replaced by ``get_fdata``. This modification has been
applied to all the code base. The default datatype is now float64.


DIPY 1.0.0 changes
----------------
Some of the changes introduced in the 1.0 release will break backwards
compatibility with previous versions. This release is compatible with Python 3.5+

**Reconstruction**

The spherical harmonics bases `mrtrix` and `fibernav` have been renamed to
`tournier07` and `descoteaux07` after the deprecation cycle started in the
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

``afq_profile`` from ``dipy.stats.analysis`` was modified in a similar way.

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

Welcome to the new module ``dipy.viz.stats``. This module will be used to integrate various analysis.

**Tracking**

- New option to adjust number of threads for SLR in Recobundles
- The tracking algoritm excludes the stop point inside the mask during tracking process.

**Notes**

- Replacement of Nose by Pytest


DIPY 0.15 Changes
-----------------

**IO**

``load_tck`` and ``save_tck`` from ``dipy.io.streamline`` has been added. They are highly recommended for managing streamlines.

**Gradient Table**

The default value of ``b0_thresold`` has been changed(from 0 to 50). This change can impact your algorithm.
If you want to assure that your code runs in exactly the same manner as before, please initialize your gradient table with the keyword argument ``b0_threshold`` set to 0.

**Visualization**

``dipy.viz.fvtk`` module has been removed. Use ``dipy.viz.*`` instead. This implies the following important changes:
- Use ``from dipy.viz import window, actor`` instead of ``from dipy.viz import fvtk`.
- Use ``window.Renderer()`` instead of ``fvtk.ren()``.
- All available actors are in ``dipy.viz.actor`` instead of ``dipy.fvtk.actor``.
- UI elements are available in ``dipy.viz.ui``.

``dipy.viz`` depends on FURY package. To get more informations about FURY, go to https://fury.gl


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

Default of DKI model fitting was changed from "OLS" to "WLS".

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

** New visualization module**

``fvtk.slicer`` input parameters have changed. Now the slicer function is
more powerfull and supports RGB images too. See tutorial ``viz_slice.py`` for
more information.

**Interpolation**
The default behavior of the function `core.sphere.interp_rbf` has changed.
The default smoothing parameter is now set to 0.1 (previously 0). In addition,
the default norm is now `angle` (was previously `euclidean_norm`). Note that
the use of `euclidean_norm` is discouraged, and this norm will be deprecated
in the 0.11 release cycle.

**Registration**

The following utilty functions from ``vector_fields`` module were renamed:

``warp_2d_affine`` is now ``transform_2d_affine``
``warp_2d_affine_nn`` is now ``transform_2d_affine_nn``
``warp_3d_affine`` is now ``transform_3d_affine``
``warp_3d_affine_nn`` is now ``transform_3d_affine_nn``


DIPY 0.9 Changes
----------------

**GQI integration length**

Calculation of integration length in GQI2 now matches the calculation in the
'standard' method. Using values of 1-1.3 for either is recommended (see
docs and references therein).


DIPY 0.8 Changes
----------------

**Peaks**

The module ``peaks`` is now available from ``dipy.direction`` and it can still
be accessed from ``dipy.reconst`` but it will be completelly removed in version
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
