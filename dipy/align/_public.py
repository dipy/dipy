"""

Registration API: simplified API for registration of MRI data and of
streamlines.


"""
import collections
import numbers
import numpy as np
import nibabel as nib
from dipy.align.metrics import CCMetric, EMMetric, SSDMetric
from dipy.align.imwarp import (SymmetricDiffeomorphicRegistration,
                               DiffeomorphicMap)

from dipy.align.imaffine import (transform_centers_of_mass,
                                 AffineMap,
                                 MutualInformationMetric,
                                 AffineRegistration)

from dipy.align.transforms import (TranslationTransform3D,
                                   RigidTransform3D,
                                   AffineTransform3D)


import dipy.core.gradients as dpg
import dipy.data as dpd
from dipy.align.streamlinear import StreamlineLinearRegistration
from dipy.tracking.streamline import set_number_of_points
from dipy.tracking.utils import transform_tracking_output
from dipy.io.streamline import load_trk
from dipy.io.utils import read_img_arr_or_path
from dipy.io.image import load_nifti, save_nifti

__all__ = ["syn_registration", "register_dwi_to_template",
           "write_mapping", "read_mapping", "resample",
           "center_of_mass", "translation", "rigid", "affine",
           "affine_registration", "register_series",
           "register_dwi_series", "streamline_registration"]

# Global dicts for choosing metrics for registration:
syn_metric_dict = {'CC': CCMetric,
                   'EM': EMMetric,
                   'SSD': SSDMetric}

affine_metric_dict = {'MI': MutualInformationMetric}


def _handle_pipeline_inputs(moving, static, static_affine=None,
                            moving_affine=None, starting_affine=None):
    """
    Helper function to prepare inputs for pipeline functions

    Parameters
    ----------
    moving, static: Either as a 3D/4D array or as a nifti image object, or as
        a string containing the full path to a nifti file.

    static_affine, moving_affine: 2D arrays.
        The array associated with the static/moving images.

    starting_affine : 2D array, optional.
        This is the registration matrix that is inherited from previous steps
        in the pipeline. Default: 4-by-4 identity matrix.
    """
    static, static_affine = read_img_arr_or_path(static,
                                                 affine=static_affine)
    moving, moving_affine = read_img_arr_or_path(moving,
                                                 affine=moving_affine)
    if starting_affine is None:
        starting_affine = np.eye(4)

    return static, static_affine, moving, moving_affine, starting_affine


def syn_registration(moving, static,
                     moving_affine=None,
                     static_affine=None,
                     step_length=0.25,
                     metric='CC',
                     dim=3,
                     level_iters=None,
                     prealign=None,
                     **metric_kwargs):
    """Register a 2D/3D source image (moving) to a 2D/3D target image (static).

    Parameters
    ----------
    moving, static : array or nib.Nifti1Image or str.
        Either as a 2D/3D array or as a nifti image object, or as
        a string containing the full path to a nifti file.
    moving_affine, static_affine : 4x4 array, optional.
        Must be provided for `data` provided as an array. If provided together
        with Nifti1Image or str `data`, this input will over-ride the affine
        that is stored in the `data` input. Default: use the affine stored
        in `data`.
    metric : string, optional
        The metric to be optimized. One of `CC`, `EM`, `SSD`,
        Default: 'CC' => CCMetric.
    dim: int (either 2 or 3), optional
       The dimensions of the image domain. Default: 3
    level_iters : list of int, optional
        the number of iterations at each level of the Gaussian Pyramid (the
        length of the list defines the number of pyramid levels to be
        used). Default: [10, 10, 5].
    metric_kwargs : dict, optional
        Parameters for initialization of the metric object. If not provided,
        uses the default settings of each metric.

    Returns
    -------
    warped_moving : ndarray
        The data in `moving`, warped towards the `static` data.
    forward : ndarray (..., 3)
        The vector field describing the forward warping from the source to the
        target.
    backward : ndarray (..., 3)
        The vector field describing the backward warping from the target to the
        source.
    """
    level_iters = level_iters or [10, 10, 5]

    static, static_affine, moving, moving_affine, _ = \
        _handle_pipeline_inputs(moving, static,
                                moving_affine=moving_affine,
                                static_affine=static_affine,
                                starting_affine=None)

    use_metric = syn_metric_dict[metric.upper()](dim, **metric_kwargs)

    sdr = SymmetricDiffeomorphicRegistration(use_metric, level_iters,
                                             step_length=step_length)

    mapping = sdr.optimize(static, moving,
                           static_grid2world=static_affine,
                           moving_grid2world=moving_affine,
                           prealign=prealign)

    warped_moving = mapping.transform(moving)
    return warped_moving, mapping


def register_dwi_to_template(dwi, gtab, dwi_affine=None, template=None,
                             template_affine=None, reg_method="syn",
                             **reg_kwargs):
    """
    Register DWI data to a template through the B0 volumes.

    Parameters
    -----------
    dwi : 4D array, nifti image or str
        Containing the DWI data, or full path to a nifti file with DWI.
    gtab : GradientTable or sequence of strings
        The gradients associated with the DWI data, or a sequence with
        (fbval, fbvec), full paths to bvals and bvecs files.
    dwi_affine : 4x4 array, optional
        An affine transformation associated with the DWI. Required if data
        is provided as an array. If provided together with nifti/path,
        will over-ride the affine that is in the nifti.
    template : 3D array, nifti image or str
        Containing the data for the template, or full path to a nifti file
        with the template data.
    template_affine : 4x4 array, optional
        An affine transformation associated with the template. Required if data
        is provided as an array. If provided together with nifti/path,
        will over-ride the affine that is in the nifti.

    reg_method : str,
        One of "syn" or "aff", which designates which registration method is
        used. Either syn, which uses the :func:`syn_registration` function
        or :func:`affine_registration` function. Default: "syn".
    reg_kwargs : key-word arguments for :func:`syn_registration` or
        :func:`affine_registration`

    Returns
    -------
    warped_b0, mapping: The fist is an array with the b0 volume warped to the
    template. If reg_method is "syn", the second is a DiffeomorphicMap class
    instance that can be used to transform between the two spaces. Otherwise,
    if reg_method is "aff", this is a 4x4 matrix encoding the affine transform.

    Notes
    -----
    This function assumes that the DWI data is already internally registered.
    See :func:`register_dwi_series`.

    """
    dwi_data, dwi_affine = read_img_arr_or_path(dwi, affine=dwi_affine)

    if template is None:
        template = dpd.read_mni_template()

    template_data, template_affine = read_img_arr_or_path(
                                       template,
                                       affine=template_affine)

    if not isinstance(gtab, dpg.GradientTable):
        gtab = dpg.gradient_table(*gtab)

    mean_b0 = np.mean(dwi_data[..., gtab.b0s_mask], -1)
    if reg_method.lower() == "syn":
        warped_b0, mapping = syn_registration(mean_b0, template_data,
                                              moving_affine=dwi_affine,
                                              static_affine=template_affine,
                                              **reg_kwargs)
    elif reg_method.lower() == "aff":
        warped_b0, mapping = affine_registration(mean_b0, template_data,
                                                 moving_affine=dwi_affine,
                                                 static_affine=template_affine,
                                                 **reg_kwargs)
    else:
        raise ValueError("reg_method should be one of 'aff' or 'syn', but you"
                         " provided %s" % reg_method)

    return warped_b0, mapping


def write_mapping(mapping, fname):
    """
    Write out a syn registration mapping to a nifti file

    Parameters
    ----------
    mapping : a DiffeomorphicMap object derived from :func:`syn_registration`
    fname : str
        Full path to the nifti file storing the mapping

    Notes
    -----
    The data in the file is organized with shape (X, Y, Z, 2, 3, 3), such
    that the forward mapping in each voxel is in `data[i, j, k, 0, :, :]` and
    the backward mapping in each voxel is in `data[i, j, k, 0, :, :]`.
    """
    mapping_data = np.array([mapping.forward.T, mapping.backward.T]).T
    save_nifti(fname, mapping_data, mapping.codomain_world2grid)


def read_mapping(disp, domain_img, codomain_img, prealign=None):
    """
    Read a syn registration mapping from a nifti file

    Parameters
    ----------
    disp : str or Nifti1Image
        A file of image containing the mapping displacement field in each voxel
        Shape (x, y, z, 3, 2)

    domain_img : str or Nifti1Image

    codomain_img : str or Nifti1Image

    Returns
    -------
    A :class:`DiffeomorphicMap` object.

    Notes
    -----
    See :func:`write_mapping` for the data format expected.
    """
    if isinstance(disp, str):
        disp_data, disp_affine = load_nifti(disp)

    if isinstance(domain_img, str):
        domain_img = nib.load(domain_img)

    if isinstance(codomain_img, str):
        codomain_img = nib.load(codomain_img)

    mapping = DiffeomorphicMap(3, disp_data.shape[:3],
                               disp_grid2world=np.linalg.inv(disp_affine),
                               domain_shape=domain_img.shape[:3],
                               domain_grid2world=domain_img.affine,
                               codomain_shape=codomain_img.shape,
                               codomain_grid2world=codomain_img.affine,
                               prealign=prealign)

    mapping.forward = disp_data[..., 0]
    mapping.backward = disp_data[..., 1]
    mapping.is_inverse = True

    return mapping


def resample(moving, static, moving_affine=None, static_affine=None,
             between_affine=None):
    """Resample an image (moving) from one space to another (static).

    Parameters
    ----------
    moving : array, nifti image or str
        Containing the data for the moving object, or full path to a nifti file
        with the moving data.

    moving_affine : 4x4 array, optional
        An affine transformation associated with the moving object. Required if
        data is provided as an array. If provided together with nifti/path,
        will over-ride the affine that is in the nifti.

    static : array, nifti image or str
        Containing the data for the static object, or full path to a nifti file
        with the moving data.

    static_affine : 4x4 array, optional
        An affine transformation associated with the static object. Required if
        data is provided as an array. If provided together with nifti/path,
        will over-ride the affine that is in the nifti.

    between_affine: 4x4 array, optional
        If an additional affine is needed betweeen the two spaces.
        Default: identity (no additional registration).

    Returns
    -------
    A Nifti1Image class instance with the data from the moving object
    resampled into the space of the static object.
    """

    static, static_affine, moving, moving_affine, between_affine = \
        _handle_pipeline_inputs(moving, static,
                                moving_affine=moving_affine,
                                static_affine=static_affine,
                                starting_affine=between_affine)
    affine_map = AffineMap(between_affine,
                           static.shape, static_affine,
                           moving.shape, moving_affine)
    resampled = affine_map.transform(moving)
    return nib.Nifti1Image(resampled, static_affine)


def center_of_mass(moving, static, static_affine=None, moving_affine=None,
                   starting_affine=None, reg=None):
    """
    Implements a center of mass transform

    Parameters
    ----------
    moving : array, nifti image or str
        Containing the data for the moving object, or full path to a nifti file
        with the moving data.

    moving_affine : 4x4 array, optional
        An affine transformation associated with the moving object. Required if
        data is provided as an array. If provided together with nifti/path,
        will over-ride the affine that is in the nifti.

    static : array, nifti image or str
        Containing the data for the static object, or full path to a nifti file
        with the moving data.

    static_affine : 4x4 array, optional
        An affine transformation associated with the static object. Required if
        data is provided as an array. If provided together with nifti/path,
        will over-ride the affine that is in the nifti.

    starting_affine: 4x4 array, optional
        Initial guess for the transformation between the spaces.

    reg : not needed here. Use None

    Returns
    -------
    transformed, transform.affine : array with moving data resampled to the
    static space after computing the center of mass transformation and the
    affine 4x4 associated with the transformation.
    """
    static, static_affine, moving, moving_affine, starting_affine = \
        _handle_pipeline_inputs(moving, static,
                                moving_affine=moving_affine,
                                static_affine=static_affine,
                                starting_affine=starting_affine)
    transform = transform_centers_of_mass(static, static_affine,
                                          moving, moving_affine)
    return transform.affine


def translation(moving, static, static_affine=None, moving_affine=None,
                starting_affine=None, reg=None):
    """
    Implements a translation transform

    Parameters
    ----------
    moving : array, nifti image or str
        Containing the data for the moving object, or full path to a nifti file
        with the moving data.

    moving_affine : 4x4 array, optional
        An affine transformation associated with the moving object. Required if
        data is provided as an array. If provided together with nifti/path,
        will over-ride the affine that is in the nifti.

    static : array, nifti image or str
        Containing the data for the static object, or full path to a nifti file
        with the moving data.

    static_affine : 4x4 array, optional
        An affine transformation associated with the static object. Required if
        data is provided as an array. If provided together with nifti/path,
        will over-ride the affine that is in the nifti.

    starting_affine: 4x4 array, optional
        Initial guess for the transformation between the spaces.

    reg : AffineRegistration class instance.

    Returns
    -------
    transformed, transform.affine : array with moving data resampled to the
    static space after computing the translation transformation and the
    affine 4x4 associated with the transformation.
    """
    static, static_affine, moving, moving_affine, starting_affine = \
        _handle_pipeline_inputs(moving, static,
                                moving_affine=moving_affine,
                                static_affine=static_affine,
                                starting_affine=starting_affine)
    transform = TranslationTransform3D()
    translation = reg.optimize(static, moving, transform, None,
                               static_affine, moving_affine,
                               starting_affine=starting_affine)

    return translation.affine


def rigid(moving, static, static_affine=None, moving_affine=None,
          starting_affine=None, reg=None):
    """
    Implements a rigid transform

    Parameters
    ----------
    moving : array, nifti image or str
        Containing the data for the moving object, or full path to a nifti file
        with the moving data.

    moving_affine : 4x4 array, optional
        An affine transformation associated with the moving object. Required if
        data is provided as an array. If provided together with nifti/path,
        will over-ride the affine that is in the nifti.

    static : array, nifti image or str
        Containing the data for the static object, or full path to a nifti file
        with the moving data.

    static_affine : 4x4 array, optional
        An affine transformation associated with the static object. Required if
        data is provided as an array. If provided together with nifti/path,
        will over-ride the affine that is in the nifti.

    starting_affine: 4x4 array, optional
        Initial guess for the transformation between the spaces.

    reg : AffineRegistration class instance.

    Returns
    -------
    transformed, transform.affine : array with moving data resampled to the
    static space after computing the rigid transformation and the affine 4x4
    associated with the transformation.
    """
    static, static_affine, moving, moving_affine, starting_affine = \
        _handle_pipeline_inputs(moving, static,
                                moving_affine=moving_affine,
                                static_affine=static_affine,
                                starting_affine=starting_affine)
    transform = RigidTransform3D()
    rigid = reg.optimize(static, moving, transform, None,
                         static_affine, moving_affine,
                         starting_affine=starting_affine)
    return rigid.affine


def affine(moving, static, static_affine=None, moving_affine=None,
           starting_affine=None, reg=None):
    """
    Implements a translation transform

    Parameters
    ----------
    moving : array, nifti image or str
        Containing the data for the moving object, or full path to a nifti file
        with the moving data.

    moving_affine : 4x4 array, optional
        An affine transformation associated with the moving object. Required if
        data is provided as an array. If provided together with nifti/path,
        will over-ride the affine that is in the nifti.

    static : array, nifti image or str
        Containing the data for the static object, or full path to a nifti file
        with the moving data.

    static_affine : 4x4 array, optional
        An affine transformation associated with the static object. Required if
        data is provided as an array. If provided together with nifti/path,
        will over-ride the affine that is in the nifti.

    starting_affine: 4x4 array, optional
        Initial guess for the transformation between the spaces.

    reg : AffineRegistration class instance.

    Returns
    -------
    transformed, transform.affine : array with moving data resampled to the
    static space after computing the affine transformation and the affine
    4x4 associated with the transformation.
    """

    static, static_affine, moving, moving_affine, starting_affine = \
        _handle_pipeline_inputs(moving, static,
                                moving_affine=moving_affine,
                                static_affine=static_affine,
                                starting_affine=starting_affine)
    transform = AffineTransform3D()
    xform = reg.optimize(static, moving, transform, None,
                         static_affine, moving_affine,
                         starting_affine=starting_affine)

    return xform.affine


def affine_registration(moving, static,
                        moving_affine=None,
                        static_affine=None,
                        pipeline=None,
                        starting_affine=None,
                        metric='MI',
                        level_iters=None,
                        sigmas=None,
                        factors=None,
                        **metric_kwargs):

    """
    Find the affine transformation between two 3D images.

    Parameters
    ----------
    moving : array, nifti image or str
        Containing the data for the moving object, or full path to a nifti file
        with the moving data.

    moving_affine : 4x4 array, optional
        An affine transformation associated with the moving object. Required if
        data is provided as an array. If provided together with nifti/path,
        will over-ride the affine that is in the nifti.

    static : array, nifti image or str
        Containing the data for the static object, or full path to a nifti file
        with the moving data.

    static_affine : 4x4 array, optional
        An affine transformation associated with the static object. Required if
        data is provided as an array. If provided together with nifti/path,
        will over-ride the affine that is in the nifti.

    pipeline : sequence, optional
        Sequence of transforms to use in the gradual fitting of the full
        affine. Default: (executed from left to right):
        `[center_of_mass, translation, rigid, affine]`

    starting_affine: 4x4 array, optional
        Initial guess for the transformation between the spaces.
        Default: identity.

    metric : str, optional.
        Currently only supports 'MI' for MutualInformationMetric.

    nbins : int, optional
        MutualInformationMetric key-word argument: the number of bins to be
        used for computing the intensity histograms. The default is 32.

    sampling_proportion : None or float in interval (0, 1], optional
        MutualInformationMetric key-word argument: There are two types of
        sampling: dense and sparse. Dense sampling uses all voxels for
        estimating the (joint and marginal) intensity histograms, while
        sparse sampling uses a subset of them. If `sampling_proportion` is
        None, then dense sampling is used. If `sampling_proportion` is a
        floating point value in (0,1] then sparse sampling is used,
        where `sampling_proportion` specifies the proportion of voxels to
        be used. The default is None (dense sampling).

    level_iters : sequence, optional
        AffineRegistration key-word argument: the number of iterations at each
        scale of the scale space. `level_iters[0]` corresponds to the coarsest
        scale, `level_iters[-1]` the finest, where n is the length of the
        sequence. By default, a 3-level scale space with iterations
        sequence equal to [10000, 1000, 100] will be used.

    sigmas : sequence of floats, optional
        AffineRegistration key-word argument: custom smoothing parameter to
        build the scale space (one parameter for each scale). By default,
        the sequence of sigmas will be [3, 1, 0].

    factors : sequence of floats, optional
        AffineRegistration key-word argument: custom scale factors to build the
        scale space (one factor for each scale). By default, the sequence of
        factors will be [4, 2, 1].

    Returns
    -------
    transformed, affine : array with moving data resampled to the static space
    after computing the affine transformation and the affine 4x4
    associated with the transformation.


    Notes
    -----
    Performs a gradual registration between the two inputs, using a pipeline
    that gradually approximates the final registration. If the final default
    step (`affine`) is ommitted, the resulting affine may not have all 12
    degrees of freedom adjusted.
    """
    pipeline = pipeline or [center_of_mass, translation, rigid, affine]
    level_iters = level_iters or [10000, 1000, 100]
    sigmas = sigmas or [3, 1, 0.0]
    factors = factors or [4, 2, 1]

    static, static_affine, moving, moving_affine, starting_affine = \
        _handle_pipeline_inputs(moving, static,
                                moving_affine=moving_affine,
                                static_affine=static_affine,
                                starting_affine=starting_affine)

    # Define the Affine registration object we'll use with the chosen metric.
    # For now, there is only one metric (mutual information)
    use_metric = affine_metric_dict[metric](**metric_kwargs)

    affreg = AffineRegistration(metric=use_metric,
                                level_iters=level_iters,
                                sigmas=sigmas,
                                factors=factors)

    # Go through the selected transformation:
    for func in pipeline:
        starting_affine = func(moving, static,
                               static_affine=static_affine,
                               moving_affine=moving_affine,
                               starting_affine=starting_affine,
                               reg=affreg)

    # After doing all that, resample once at the end:
    affine_map = AffineMap(starting_affine,
                           static.shape, static_affine,
                           moving.shape, moving_affine)

    resampled = affine_map.transform(moving)

    return resampled, starting_affine


def register_series(series, ref, pipeline=None, series_affine=None,
                    ref_affine=None):
    """Register a series to a reference image.

    Parameters
    ----------
    series : 4D array or nib.Nifti1Image class instance or str
        The data is 4D with the last dimension separating different 3D volumes

    ref : int or 3D array or nib.Nifti1Image class instance or str
        If this is an int, this is the index of the reference image within the
        series. Otherwise it is an array of data to register with (associated
        with a `ref_affine` required) or a nifti img or full path to a file
        containing one.

    pipeline : sequence, optional
        Sequence of transforms to do for each volume in the series.
        Default: (executed from left to right):
        `[center_of_mass, translation, rigid, affine]`

    series_affine, ref_affine : 4x4 arrays, optional.
        The affine. If provided, this input will over-ride the affine provided
        together with the nifti img or file.

    Returns
    -------
    xformed, affines : 4D array with transformed data and a (4,4,n) array
    with 4x4 matrices associated with each of the volumes of the input moving
    data that was used to transform it into register with the static data.
    """
    pipeline = pipeline or [center_of_mass, translation, rigid, affine]

    series, series_affine = read_img_arr_or_path(series,
                                                 affine=series_affine)
    if isinstance(ref, numbers.Number):
        ref_as_idx = ref
        idxer = np.zeros(series.shape[-1]).astype(bool)
        idxer[ref] = True
        ref = series[..., idxer].squeeze()
        ref_affine = series_affine
    else:
        ref_as_idx = False
        ref, ref_affine = read_img_arr_or_path(ref, affine=ref_affine)
        if len(ref.shape) != 3:
            raise ValueError("The reference image should be a single volume",
                             " or the index of one or more volumes")

    xformed = np.zeros(series.shape)
    affines = np.zeros((4, 4, series.shape[-1]))
    for ii in range(series.shape[-1]):
        this_moving = series[..., ii]
        if isinstance(ref_as_idx, numbers.Number) and ii == ref_as_idx:
            # This is the reference! No need to move and the xform is I(4):
            xformed[..., ii] = this_moving
            affines[..., ii] = np.eye(4)
        else:
            transformed, reg_affine = affine_registration(
                this_moving, ref,
                moving_affine=series_affine,
                static_affine=ref_affine,
                pipeline=pipeline)
            xformed[..., ii] = transformed
            affines[..., ii] = reg_affine

    return xformed, affines


def register_dwi_series(data, gtab, affine=None, b0_ref=0, pipeline=None):
    """
    Register a DWI series to the mean of the B0 images in that series (all
    first registered to the first B0 volume)

    Parameters
    ----------
    data : 4D array or nibabel Nifti1Image class instance or str
        Diffusion data. Either as a 4D array or as a nifti image object, or as
        a string containing the full path to a nifti file.

    gtab : a GradientTable class instance or tuple of strings
        If provided as a tuple of strings, these are assumed to be full paths
        to the bvals and bvecs files (in that order).

    affine : 4x4 array, optional.
        Must be provided for `data` provided as an array. If provided together
        with Nifti1Image or str `data`, this input will over-ride the affine
        that is stored in the `data` input. Default: use the affine stored
        in `data`.

    b0_ref : int, optional.
        Which b0 volume to use as reference. Default: 0

    pipeline : list of callables, optional.
        The transformations to perform in sequence (from left to right):
        Default: `[center_of_mass, translation, rigid, affine]`


    Returns
    -------
    xform_img, affine_array: a Nifti1Image containing the registered data and
    using the affine of the original data and a list containing the affine
    transforms associated with each of the

    """
    if pipeline is None:
        [center_of_mass, translation, rigid, affine]

    data, affine = read_img_arr_or_path(data, affine=affine)
    if isinstance(gtab, collections.Sequence):
        gtab = dpg.gradient_table(*gtab)

    if np.sum(gtab.b0s_mask) > 1:
        # First, register the b0s into one image and average:
        b0_img = nib.Nifti1Image(data[..., gtab.b0s_mask], affine)
        trans_b0, b0_affines = register_series(b0_img, ref=b0_ref,
                                               pipeline=pipeline)
        ref_data = np.mean(trans_b0, -1)
    else:
        # There's only one b0 and we register everything to it
        trans_b0 = ref_data = data[..., gtab.b0s_mask]
        b0_affines = np.eye(4)[..., np.newaxis]

    # Construct a series out of the DWI and the registered mean B0:
    moving_data = data[..., ~gtab.b0s_mask]
    series_arr = np.concatenate([ref_data, moving_data], -1)
    series = nib.Nifti1Image(series_arr, affine)

    xformed, affines = register_series(series, ref=0, pipeline=pipeline)
    # Cut out the part pertaining to that first volume:
    affines = affines[..., 1:]
    xformed = xformed[..., 1:]
    affine_array = np.zeros((4, 4, data.shape[-1]))
    affine_array[..., gtab.b0s_mask] = b0_affines
    affine_array[..., ~gtab.b0s_mask] = affines

    data_array = np.zeros(data.shape)
    data_array[..., gtab.b0s_mask] = trans_b0
    data_array[..., ~gtab.b0s_mask] = xformed

    return nib.Nifti1Image(data_array, affine), affine_array


def streamline_registration(moving, static, n_points=100,
                            native_resampled=False):
    """
    Register two collections of streamlines ('bundles') to each other

    Parameters
    ----------
    moving, static : lists of 3 by n, or str
        The two bundles to be registered. Given either as lists of arrays with
        3D coordinates, or strings containing full paths to these files.

    n_points : int, optional
        How many points to resample to. Default: 100.

    native_resampled : bool, optional
        Whether to return the moving bundle in the original space, but
        resampled in the static space to n_points.

    Returns
    -------
    aligned : list
        Streamlines from the moving group, moved to be closely matched to
        the static group.

    matrix : array (4, 4)
        The affine transformation that takes us from 'moving' to 'static'
    """
    # Load the streamlines, if you were given a file-name
    if isinstance(moving, str):
        moving = load_trk(moving, 'same', bbox_valid_check=False).streamlines
    if isinstance(static, str):
        static = load_trk(static, 'same', bbox_valid_check=False).streamlines

    srr = StreamlineLinearRegistration()
    srm = srr.optimize(static=set_number_of_points(static, n_points),
                       moving=set_number_of_points(moving, n_points))

    aligned = srm.transform(moving)
    if native_resampled:
        aligned = set_number_of_points(aligned, n_points)
        aligned = transform_tracking_output(aligned, np.linalg.inv(srm.matrix))

    return aligned, srm.matrix
