import numpy as np
from scipy.ndimage import affine_transform

from dipy.align.reslice import reslice
from dipy.testing.decorators import warning_for_keywords
from dipy.utils.deprecator import deprecated_params


@warning_for_keywords()
def normalize(image, *, min_v=None, max_v=None, new_min=-1, new_max=1):
    """
    normalization function

    Parameters
    ----------
    image : np.ndarray
        Image to be normalized.
    min_v : int or float, optional
        minimum value range for normalization
        intensities below min_v will be clipped
        if None it is set to min value of image
        Default : None
    max_v : int or float, optional
        maximum value range for normalization
        intensities above max_v will be clipped
        if None it is set to max value of image
        Default : None
    new_min : int or float, optional
        new minimum value after normalization
        Default : 0
    new_max : int or float, optional
        new maximum value after normalization
        Default : 1

    Returns
    -------
    np.ndarray
        Normalized image from range new_min to new_max
    """
    if min_v is None:
        min_v = np.min(image)
    if max_v is None:
        max_v = np.max(image)
    return np.interp(image, (min_v, max_v), (new_min, new_max))


def unnormalize(image, norm_min, norm_max, min_v, max_v):
    """
    unnormalization function

    Parameters
    ----------
    image : np.ndarray
    norm_min : int or float
        minimum value of normalized image
    norm_max : int or float
        maximum value of normalized image
    min_v : int or float
        minimum value of unnormalized image
    max_v : int or float
        maximum value of unnormalized image

    Returns
    -------
    np.ndarray
        unnormalized image from range min_v to max_v
    """
    return (image - norm_min) / (norm_max - norm_min) * (max_v - min_v) + min_v


def set_logger_level(log_level, logger):
    """Change the logger to one of the following:
    DEBUG, INFO, WARNING, CRITICAL, ERROR

    Parameters
    ----------
    log_level : str
        Log level for the logger
    """
    logger.setLevel(level=log_level)


def get_bounds(shape, affine, considered_points="corners"):
    """
    Function to get the bounds of an image after applying affine

    Parameters
    ----------
    shape : tuple (3,)
        Shape of the image
    affine : np.ndarray
        Affine matrix provided by the image file.
    considered_points : str, optional
        Considered points when calculating the transformed shape.
        \"corners\" will consider only corners of the image shape.
        \"all\" will consider all voxels. Might be needed when shearing is applied.

    Returns
    -------
    min_bounds : np.ndarray (3,)
        Minimum bounds of the transformed image
    max_bounds : np.ndarray (3,)
        Maximum bounds of the transformed image
    """

    if considered_points == "corners":
        points = np.array(
            [
                [0, 0, 0, 1],
                [shape[0] - 1, 0, 0, 1],
                [0, shape[1] - 1, 0, 1],
                [0, 0, shape[2] - 1, 1],
                [shape[0] - 1, shape[1] - 1, shape[2] - 1, 1],
                [shape[0] - 1, 0, shape[2] - 1, 1],
                [0, shape[1] - 1, shape[2] - 1, 1],
                [shape[0] - 1, shape[1] - 1, 0, 1],
            ],
            dtype=np.float64,
        )
    elif considered_points == "all":
        temp1 = np.arange(shape[0])
        temp2 = np.arange(shape[1])
        temp3 = np.arange(shape[2])
        grid1, grid2, grid3 = np.meshgrid(temp1, temp2, temp3)
        points = np.vstack([grid1.ravel(), grid2.ravel(), grid3.ravel()]).T
        points = np.hstack([points, np.full((points.shape[0], 1), 1)])
        points = points.astype(np.float64)
    else:
        ValueError('considered points should be "corners" or "all"')

    transformed_corners = (affine @ points.T).T
    min_bounds = transformed_corners.min(axis=0)[:3]
    max_bounds = transformed_corners.max(axis=0)[:3]

    return min_bounds, max_bounds


@warning_for_keywords()
@deprecated_params("scale", new_name="ratio", since="1.12", until="1.14")
def transform_img(
    image,
    affine,
    *,
    voxsize=(1, 1, 1),
    target_voxsize=None,
    considered_points="corners",
    init_shape=None,
    ratio=None,
    final_size=None,
    need_isotropic=False,
    order=3,
):
    """
    Function to transform images for Deep Learning models

    Parameters
    ----------
    image : np.ndarray
        Image to transform.
    affine : np.ndarray
        Affine matrix provided by the image file.
    voxsize : tuple (3,), optional
        Voxel size provided by the image file.
    target_voxsize : tuple (3,), optional
        The voxel size we want to start from.
        Ignored if need_isotropic is True.
    considered_points : str, optional
        Considered points when calculating the transformed shape.
        \"corners\" will consider only corners of the image shape.
        \"all\" will consider all voxels. Might be needed when shearing is applied.
    init_shape : tuple (3,), optional
        What we want the initial shape to be before last resizing step.
        Ignored if need_isotropic is True.
    ratio : float, optional
        The ratio of change in the last resizing step.
        Ignored if need_isotropic is True.
    final_size : tuple (3,), optional
        The final size of the image array.
    need_isotropic : bool, optional
        Whether the output needs to be isotropic in the end.
    order : int, optional
        The order of the spline interpolation.
        The order has to be in the range 0-5.
        If transforming an int image, order 0 is recommended.



    Returns
    -------
    new_image : np.ndarray
        Transformed image to be used in the Deep Learning model.
    params : dict
        Parameters that are used when recovering the original image space.
    """
    ori_shape = image.shape
    if need_isotropic:
        target_voxsize = tuple(np.max(voxsize) * np.ones(3))
    if target_voxsize is not None and np.any(target_voxsize != voxsize):
        resliced_image, resliced_affine = reslice(
            image, affine, voxsize, target_voxsize
        )
    else:
        resliced_image = image.copy()
        resliced_affine = affine.copy()

    shape = resliced_image.shape

    min_bounds, max_bounds = get_bounds(shape, resliced_affine)

    # Calculate the required offset to ensure
    # all necessary coordinates are positive
    offset = np.ceil(-min_bounds)
    new_shape = (np.ceil(max_bounds) + offset).astype(int)
    offset_array = np.array(
        [[1, 0, 0, offset[0]], [0, 1, 0, offset[1]], [0, 0, 1, offset[2]], [0, 0, 0, 1]]
    )

    new_affine = resliced_affine.copy()
    new_affine = np.matmul(offset_array, new_affine)

    inv_affine = np.linalg.inv(new_affine)
    new_image = np.zeros(tuple(new_shape))
    affine_transform(
        resliced_image,
        inv_affine,
        output_shape=tuple(new_shape),
        output=new_image,
        order=order,
    )

    crop_vs = None
    pad_vs = None
    if not need_isotropic:
        if init_shape:
            new_image, pad_vs, crop_vs = pad_crop(new_image, init_shape)

        if (ratio is not None and ratio != 1) and final_size is None:
            new_image, _ = reslice(
                new_image, np.eye(4), (1, 1, 1), (ratio, ratio, ratio)
            )
        elif final_size:
            new_image, _ = reslice(
                new_image,
                np.eye(4),
                (1, 1, 1),
                (
                    new_image.shape[0] / final_size[0],
                    new_image.shape[1] / final_size[1],
                    new_image.shape[2] / final_size[2],
                ),
            )

    else:
        ratio = np.max(np.array(new_image.shape) / np.array(final_size))
        new_size = np.ceil(np.array(final_size) * ratio)
        new_image, pad_vs, crop_vs = pad_crop(new_image, tuple(new_size))
        new_image, _ = reslice(new_image, np.eye(4), (1, 1, 1), (ratio, ratio, ratio))

    params = {
        "inv_affine": inv_affine,
        "resliced_shape": resliced_image.shape,
        "offset": offset_array,
        "crop_value": crop_vs,
        "pad_value": pad_vs,
        "ratio": ratio,
        "voxsize": voxsize,
        "target_voxsize": target_voxsize,
        "isotropic_flag": need_isotropic,
        "final_size": final_size,
        "ori_shape": ori_shape,
        "order": order,
    }

    return new_image, params


def recover_img(image, params):
    """
    Function to recover image from transform_img

    Parameters
    ----------
    image : np.ndarray
        Image to recover.
    params : tuple
        Parameters for recover_img function.
        Returned from transform_img.

    Returns
    -------
    new_image : np.ndarray
        Recovered image
    affine : np.ndarray
        Recovered affine.
        This should be same as the original affine.
    """
    expected_keys = {
        "inv_affine",
        "resliced_shape",
        "offset",
        "crop_value",
        "pad_value",
        "ratio",
        "voxsize",
        "target_voxsize",
        "isotropic_flag",
        "final_size",
        "ori_shape",
        "order",
    }
    missing = expected_keys - set(params.keys())
    if missing:
        raise ValueError(f"params is missing keys: {missing}")

    inv_affine = params["inv_affine"]
    mid_shape = params["resliced_shape"]
    offset_array = params["offset"]
    crop_vs = params["crop_value"]
    pad_vs = params["pad_value"]
    ratio = params["ratio"]
    voxsize = params["voxsize"]
    target_voxsize = params["target_voxsize"]
    need_isotropic = params["isotropic_flag"]
    final_size = params["final_size"]
    ori_shape = params["ori_shape"]
    order = params["order"]

    new_affine = np.linalg.inv(inv_affine)
    if need_isotropic:
        new_image, _ = reslice(image, np.eye(4), (ratio, ratio, ratio), (1, 1, 1))
    else:
        if (ratio is not None and ratio != 1) and final_size is None:
            new_image, _ = reslice(image, np.eye(4), (ratio, ratio, ratio), (1, 1, 1))
        elif final_size:
            new_image, _ = reslice(
                image,
                np.eye(4),
                (
                    image.shape[0] / final_size[0],
                    image.shape[1] / final_size[1],
                    image.shape[2] / final_size[2],
                ),
                (1, 1, 1),
            )
        else:
            new_image = image

    if crop_vs is not None and pad_vs is not None:
        crop_vs = crop_vs.astype(int)
        pad_vs = pad_vs.astype(int)
        new_image = np.pad(
            new_image,
            (
                (crop_vs[0, 0], crop_vs[0, 1]),
                (crop_vs[1, 0], crop_vs[1, 1]),
                (crop_vs[2, 0], crop_vs[2, 1]),
            ),
        )
        new_image = new_image[
            pad_vs[0, 0] : new_image.shape[0] - pad_vs[0, 1],
            pad_vs[1, 0] : new_image.shape[1] - pad_vs[1, 1],
            pad_vs[2, 0] : new_image.shape[2] - pad_vs[2, 1],
        ]

    new_image = affine_transform(
        new_image, new_affine, output_shape=mid_shape, order=order
    )
    affine = np.matmul(np.linalg.inv(offset_array), new_affine)

    if target_voxsize is not None and np.any(target_voxsize != voxsize):
        new_image, affine = reslice(new_image, affine, target_voxsize, voxsize)
        if new_image.shape != ori_shape:
            new_image = pad_crop(new_image, ori_shape)

    return new_image, affine


def pad_crop(image, target_shape):
    """
    Function to figure out pad and crop range
    to fit the target shape with the image

    Parameters
    ----------
    image : np.ndarray
        Target image
    target_shape : (3,)
        Target shape

    Returns
    -------
    image : np.ndarray
        Padded/cropped image
    pad_vs : np.ndarray (3,2)
        Pad range used
    crop_vs : np.ndarray (3,2)
        Crop range used
    """
    crop_vs = np.zeros((3, 2)).astype(int)
    pad_vs = np.zeros((3, 2)).astype(int)
    if image.shape == target_shape:
        return image, pad_vs, crop_vs

    pad_crop_v = np.array(target_shape) - np.array(image.shape)
    for d in range(3):
        if pad_crop_v[d] < 0:
            crop_vs[d, 0] = -pad_crop_v[d] // 2
            crop_vs[d, 1] = -pad_crop_v[d] - crop_vs[d, 0]
        elif pad_crop_v[d] > 0:
            pad_vs[d, 0] = pad_crop_v[d] // 2
            pad_vs[d, 1] = pad_crop_v[d] - pad_vs[d, 0]
    image = np.pad(
        image,
        (
            (pad_vs[0, 0], pad_vs[0, 1]),
            (pad_vs[1, 0], pad_vs[1, 1]),
            (pad_vs[2, 0], pad_vs[2, 1]),
        ),
    )
    image = image[
        crop_vs[0, 0] : image.shape[0] - crop_vs[0, 1],
        crop_vs[1, 0] : image.shape[1] - crop_vs[1, 1],
        crop_vs[2, 0] : image.shape[2] - crop_vs[2, 1],
    ]

    return image, pad_vs, crop_vs
