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
    set_size=None,
    need_isotropic=False,
):
    """
    Function to transform images for Deep Learning models

    Parameters
    ----------
    image : np.ndarray
        Image to transform.
    affine : np.ndarray
        Affine matrix provided by the image file.
    voxsize : tuple (3,)
        Voxel size provided by the image file.
    target_voxsize : tuple (3,)
        The voxel size we want to start from.
        Ignored if need_isotropic is True.
    considered_points : str
        Considered points when calculating the transformed shape.
        \"corners\" will consider only corners of the image shape.
        \"all\" will consider all voxels. Might be needed when shearing is applied.
    init_shape : tuple (3,)
        What we want the initial shape to be before last resizing step.
        Ignored if need_isotropic is True.
    ratio : float
        The ratio of change in the last resizing step.
        Ignored if need_isotropic is True.
    set_size : tuple (3,)
        The final size of the image array.
    need_isotropic : bool
        Whether the output needs to be isotropic in the end.



    Returns
    -------
    new_image : np.ndarray
        Transformed image to be used in the Deep Learning model.
    params : tuple
        Parameters that are used when recovering the original image space.
    """
    ori_shape = image.shape
    if need_isotropic:
        target_voxsize = tuple(np.max(voxsize) * np.ones(3))
    if target_voxsize is not None and np.any(target_voxsize != np.ones(3)):
        image2, affine2 = reslice(image, affine, voxsize, target_voxsize)
    else:
        image2 = image.copy()
        affine2 = affine.copy()

    shape = image2.shape

    if considered_points == "corners":
        corners = np.array(
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
        corners = np.vstack([grid1.ravel(), grid2.ravel(), grid3.ravel()]).T
        corners = np.hstack([corners, np.full((corners.shape[0], 1), 1)])
        corners = corners.astype(np.float64)
    else:
        ValueError('considered points should be "corners" or "all"')

    transformed_corners = (affine2 @ corners.T).T
    min_bounds = transformed_corners.min(axis=0)[:3]
    max_bounds = transformed_corners.max(axis=0)[:3]

    # Calculate the required offset to ensure
    # all necessary coordinates are positive
    offset = np.ceil(-min_bounds)
    new_shape = (np.ceil(max_bounds) + offset).astype(int)
    offset_array = np.array(
        [[1, 0, 0, offset[0]], [0, 1, 0, offset[1]], [0, 0, 1, offset[2]], [0, 0, 0, 1]]
    )

    new_affine = affine2.copy()
    new_affine = np.matmul(offset_array, new_affine)

    inv_affine = np.linalg.inv(new_affine)
    new_image = np.zeros(tuple(new_shape))
    affine_transform(
        image2, inv_affine, output_shape=tuple(new_shape), output=new_image
    )

    mid_image = new_image.copy()

    crop_vs = None
    pad_vs = None
    if not need_isotropic:
        if init_shape:
            new_image, pad_vs, crop_vs = pad_crop(new_image, init_shape)

        if (ratio is not None and ratio != 1) and set_size is None:
            new_image, _ = reslice(
                new_image, np.eye(4), (1, 1, 1), (ratio, ratio, ratio)
            )
        elif set_size:
            new_image, _ = reslice(
                new_image,
                np.eye(4),
                (1, 1, 1),
                (
                    new_image.shape[0] / set_size[0],
                    new_image.shape[1] / set_size[1],
                    new_image.shape[2] / set_size[2],
                ),
            )

    else:
        ratio = np.max(np.array(mid_image.shape) / np.array(set_size))
        new_size = np.ceil(np.array(set_size) * ratio)
        new_image, pad_vs, crop_vs = pad_crop(mid_image, tuple(new_size))
        new_image, _ = reslice(new_image, np.eye(4), (1, 1, 1), (ratio, ratio, ratio))

    params = (
        inv_affine,
        image2.shape,
        offset_array,
        crop_vs,
        pad_vs,
        ratio,
        voxsize,
        target_voxsize,
        need_isotropic,
        set_size,
        ori_shape,
    )

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
    (
        inv_affine,
        mid_shape,
        offset_array,
        crop_vs,
        pad_vs,
        ratio,
        voxsize,
        target_voxsize,
        need_isotropic,
        set_size,
        ori_shape,
    ) = params
    new_affine = np.linalg.inv(inv_affine)
    if need_isotropic:
        new_image, _ = reslice(image, np.eye(4), (ratio, ratio, ratio), (1, 1, 1))
    else:
        if (ratio is not None and ratio != 1) and set_size is None:
            new_image, _ = reslice(image, np.eye(4), (ratio, ratio, ratio), (1, 1, 1))
        elif set_size:
            new_image, _ = reslice(
                image,
                np.eye(4),
                (
                    new_image.shape[0] / set_size[0],
                    new_image.shape[1] / set_size[1],
                    new_image.shape[2] / set_size[2],
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

    new_image = affine_transform(new_image, new_affine, output_shape=mid_shape)
    affine = np.matmul(np.linalg.inv(offset_array), new_affine)

    if target_voxsize is not None and np.any(target_voxsize != np.ones(3)):
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
