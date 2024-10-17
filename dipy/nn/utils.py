import numpy as np
from scipy.ndimage import affine_transform

from dipy.align.reslice import reslice
from dipy.testing.decorators import warning_for_keywords


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
def transform_img(
    image,
    affine,
    *,
    voxsize=None,
    considered_points="corners",
    init_shape=(256, 256, 256),
    scale=2,
):
    """
    Function to reshape image as an input to the model

    Parameters
    ----------
    image : np.ndarray
        Image to transform to voxelspace
    affine : np.ndarray
        Affine matrix provided by the file
    voxsize : np.ndarray (3,), optional
        Voxel size of the image
    considered_points : str, optional
        which points to consider when calculating
        the boundary of the image. If there is shearing
        in the affine, 'all' might be more accurate
    init_shape : list, tuple or numpy array (3,), optional
        Initial shape to transform the image to
    scale : float, optional
        How much we want to scale the image

    Returns
    -------
    tuple
        Tuple with variables for recover_img
    """
    if voxsize is not None and np.any(voxsize != np.ones(3)):
        image2, affine2 = reslice(image, affine, voxsize, (1, 1, 1))
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
    else:
        temp1 = np.arange(shape[0])
        temp2 = np.arange(shape[1])
        temp3 = np.arange(shape[2])
        grid1, grid2, grid3 = np.meshgrid(temp1, temp2, temp3)
        corners = np.vstack([grid1.ravel(), grid2.ravel(), grid3.ravel()]).T
        corners = np.hstack([corners, np.full((corners.shape[0], 1), 1)])
        corners = corners.astype(np.float64)

    transformed_corners = (affine2 @ corners.T).T
    min_bounds = transformed_corners.min(axis=0)[:3]
    max_bounds = transformed_corners.max(axis=0)[:3]

    # Calculate the required offset to ensure
    # all necessary coordinates are positive
    offset = np.floor(-min_bounds)
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

    new_image, pad_vs, crop_vs = pad_crop(new_image, init_shape)

    if scale != 1:
        new_image, _ = reslice(new_image, np.eye(4), (1, 1, 1), (scale, scale, scale))

    return new_image, inv_affine, image2.shape, offset_array, scale, crop_vs, pad_vs


def recover_img(
    image,
    inv_affine,
    mid_shape,
    ori_shape,
    offset_array,
    voxsize,
    scale,
    crop_vs,
    pad_vs,
):
    """
    Function to recover image from transform_img

    Parameters
    ----------
    image : np.ndarray
        Image to recover
    inv_affine : np.ndarray
        Affine matrix returned from transform_img
    mid_shape : np.ndarray (3,)
        shape of image returned from transform_img
    ori_shape : tuple (3,)
        original shape of the image
    offset_array : np.ndarray
        Affine matrix that was used in transform_img
        to translate the center
    voxsize : np.ndarray (3,)
        Voxel size used in transform_img
    scale : float
        Scale used in transform_img
    crop_vs : np.ndarray (3,2)
        crop range used in transform_img
    pad_vs : np.ndarray (3,2)
        pad range used in transform_img

    Returns
    -------
    image2 : np.ndarray
        Recovered image
    """
    new_affine = np.linalg.inv(inv_affine)
    new_image, _ = reslice(image, np.eye(4), (scale, scale, scale), (1, 1, 1))
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
    if voxsize is not None and np.any(voxsize != np.ones(3)):
        image2, _ = reslice(new_image, affine, (1, 1, 1), voxsize)
    else:
        image2 = new_image

    # because of zoom rounding errors
    image2, _, _ = pad_crop(image2, ori_shape)
    return image2


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
