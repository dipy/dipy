import numpy as np
from scipy.ndimage import affine_transform, map_coordinates

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
    max_v : int or float, optional
        maximum value range for normalization
        intensities above max_v will be clipped
        if None it is set to max value of image
    new_min : int or float, optional
        new minimum value after normalization
    new_max : int or float, optional
        new maximum value after normalization

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
def get_bounds(shape, affine, *, considered_points="corners"):
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
        \"all\" will consider all voxels.
        Might be needed when shearing is applied.

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
def transform_img(
    image,
    affine,
    *,
    target_voxsize=None,
    final_size=None,
    order=3,
    considered_points="corners",
):
    """
    Function to transform images for Deep Learning models

    Parameters
    ----------
    image : np.ndarray
        Image to transform.
    affine : np.ndarray
        Affine matrix provided by the image file.
    target_voxsize : tuple (3,), optional
        The voxel size we want to start from.
        If none is provided, we calculate the closest isotropic voxel size.
    final_size : tuple (3,), optional
        The final size of the image array.
    order : int, optional
        The order of the spline interpolation.
        The order has to be in the range 0-5.
        If transforming an int image, order 0 is recommended.
    considered_points : str, optional
        Considered points when calculating the transformed shape.
        \"corners\" will consider only corners of the image shape.
        \"all\" will consider all voxels. Might be needed when shearing is applied.

    Returns
    -------
    new_image : np.ndarray
        Transformed image to be used in the Deep Learning model.
    params : dict
        Parameters that are used when recovering the original image space.
    """
    init_shape = image.shape
    R = affine[:3, :3]
    voxsize = np.sqrt(np.sum(R * R, axis=0))
    if target_voxsize is None:
        target_voxsize = np.array([np.min(voxsize)] * 3)
    else:
        target_voxsize = np.array(target_voxsize, dtype=float)

    min_bounds, max_bounds = get_bounds(
        init_shape, affine, considered_points=considered_points
    )
    new_shape = (max_bounds - min_bounds) / target_voxsize
    new_shape = np.round(new_shape).astype(int)

    invA = np.linalg.inv(affine)
    D = np.diag(target_voxsize)
    new_origin_world = min_bounds

    matrix = (invA[:3, :3] @ D).astype(float)
    offset = (invA[:3, 3] + invA[:3, :3] @ new_origin_world).astype(float)

    new_image = np.zeros(tuple(new_shape), dtype=image.dtype)
    affine_transform(
        image,
        matrix=matrix,
        offset=offset,
        output_shape=tuple(new_shape),
        output=new_image,
        order=order,
    )

    if final_size is None:
        final_size = new_image.shape
    final_size = tuple(int(x) for x in final_size)
    new_image, pad_vs, crop_vs = pad_crop(new_image, final_size)

    params = {
        "init_shape": init_shape,
        "ori_affine": affine,
        "target_voxsize": target_voxsize,
        "new_origin_world": new_origin_world,
        "pad_value": pad_vs,
        "crop_value": crop_vs,
    }

    return new_image, params


def recover_img(image, params, *, order=3):
    """
    Function to recover image from transform_img

    Parameters
    ----------
    image : np.ndarray
        Image to recover.
    params : dict
        Parameters for recover_img function.
        Returned from transform_img.
    order : int, optional
        The order of the spline interpolation.
        The order has to be in the range 0-5.
        If recovering an int image, order 0 is recommended.

    Returns
    -------
    recovered : np.ndarray
        Recovered image
    """
    expected = {
        "init_shape",
        "ori_affine",
        "target_voxsize",
        "new_origin_world",
        "pad_value",
        "crop_value",
    }
    missing = expected - set(params.keys())
    if missing:
        raise ValueError(f"params missing: {missing}")

    init_shape = tuple(int(x) for x in params["init_shape"])
    affine = np.asarray(params["ori_affine"], dtype=float)
    target_voxsize = np.asarray(params["target_voxsize"], dtype=float)
    new_origin_world = np.asarray(params["new_origin_world"], dtype=float)
    pad_vs = params["pad_value"]
    crop_vs = params["crop_value"]

    if crop_vs is not None and pad_vs is not None:
        new_image = inv_pad_crop(image, crop_vs, pad_vs)
    else:
        new_image = image

    A = affine
    D = np.diag(target_voxsize)

    coords_idx = np.indices(init_shape, dtype=float)
    coords_idx = coords_idx.reshape((len(init_shape), -1))
    world_coords = (A[:3, :3] @ coords_idx) + A[:3, 3][:, None]
    coords_new = np.linalg.inv(D) @ (world_coords - new_origin_world[:, None])
    coords_for_map = coords_new.reshape((len(init_shape),) + tuple(init_shape))
    recovered = map_coordinates(new_image, coords_for_map, order=order)

    return recovered


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
    if np.all(np.array(image.shape) == target_shape):
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


def inv_pad_crop(image, crop_vs, pad_vs):
    """
    Function to figure out pad and crop range
    to fit the target shape with the image

    Parameters
    ----------
    image : np.ndarray
        Target image
    crop_vs : np.ndarray (3,2)
        Crop range used when padding/cropping
    pad_vs : np.ndarray (3,2)
        Pad range used when padding/cropping

    Returns
    -------
    image : np.ndarray
        Recovered image
    """
    crop_vs = crop_vs.astype(int)
    pad_vs = pad_vs.astype(int)
    image = np.pad(
        image,
        (
            (crop_vs[0, 0], crop_vs[0, 1]),
            (crop_vs[1, 0], crop_vs[1, 1]),
            (crop_vs[2, 0], crop_vs[2, 1]),
        ),
    )
    image = image[
        pad_vs[0, 0] : image.shape[0] - pad_vs[0, 1],
        pad_vs[1, 0] : image.shape[1] - pad_vs[1, 1],
        pad_vs[2, 0] : image.shape[2] - pad_vs[2, 1],
    ]

    return image
