import os
import tempfile
import time
from warnings import warn

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from tqdm import tqdm

from dipy.stats.sketching import count_sketch
from dipy.testing.decorators import warning_for_keywords
from dipy.utils.optpkg import optional_package

sklearn, has_sklearn, _ = optional_package("sklearn")
linear_model, _, _ = optional_package("sklearn.linear_model")


def _vol_split(train, vol_idx):
    """Split the 3D volumes into the train and test set.

    Parameters
    ----------
    train : numpy.ndarray
        Array of all 3D patches flattened out to be 2D.
    vol_idx : int
        The volume number that needs to be held out for training.

    Returns
    -------
    cur_x : numpy.ndarray of shape (nvolumes * patch_size) x (nvoxels)
        Array of patches corresponding to all volumes except the held out volume.
    y : numpy.ndarray of shape (patch_size) x (nvoxels)
        Array of patches corresponding to the volume that is used a target for
        denoising.

    """
    mask = np.zeros(train.shape[0], dtype=bool)
    mask[vol_idx] = True
    cur_x = train[~mask].reshape((train.shape[0] - 1) * train.shape[1], train.shape[2])
    y = train[vol_idx, train.shape[1] // 2, :]
    return cur_x, y


def _extract_3d_patches(arr, patch_radius):
    """Extract 3D patches from 4D DWI data.

    Parameters
    ----------
    arr : ndarray
        The 4D noisy DWI data to be denoised.
    patch_radius : int or array of shape (3,)
        The radius of the local patch to be taken around each voxel (in
        voxels).

    Returns
    -------
    all_patches : ndarray
        All 3D patches flattened out to be 2D corresponding to the each 3D
        volume of the 4D DWI data.

    """
    patch_radius = np.asarray(patch_radius, dtype=int)
    if patch_radius.size == 1:
        patch_radius = np.repeat(patch_radius, 3)
    elif patch_radius.size != 3:
        raise ValueError("patch_radius should have length 1 or 3")

    patch_size = 2 * patch_radius + 1
    dim = arr.shape[-1]

    # Calculate the shape of the output array
    output_shape = tuple(arr.shape[i] - 2 * patch_radius[i] for i in range(3))
    total_patches = np.prod(output_shape)

    patches = sliding_window_view(arr, tuple(patch_size) + (dim,))

    # Reshape and transpose the patches to match the original function's output shape
    all_patches = patches.reshape(total_patches, np.prod(patch_size), dim)
    all_patches = all_patches.transpose(2, 1, 0)

    return np.array(all_patches)


def _fit_denoising_model(train, vol_idx, model, alpha):
    """Fit a single 3D volume using a train and test phase.

    Parameters
    ----------
    train : ndarray
        Array of all 3D patches flattened out to be 2D.
    vol_idx : int
        The volume number that needs to be held out for training.
    model : str or sklearn.base.RegressorMixin
        This will determine the algorithm used to solve the set of linear
        equations underlying this model. If it is a string it needs to be
        one of the following: {'ols', 'ridge', 'lasso'}. Otherwise,
        it can be an object that inherits from
        `dipy.optimize.SKLearnLinearSolver` or an object with a similar
        interface from Scikit-Learn:
        `sklearn.linear_model.LinearRegression`,
        `sklearn.linear_model.Lasso` or `sklearn.linear_model.Ridge`
        and other objects that inherit from `sklearn.base.RegressorMixin`.
    alpha : float
        Regularization parameter only for ridge and lasso regression models.
    version : int
        Version 1 or 3 of Patch2Self to use.

    Returns
    -------
    model_instance : fitted linear model object
        The fitted model instance if version is 3.
    cur_x : ndarray
        The patches corresponding to all volumes except the held out volume.

    """
    if isinstance(model, str):
        if model.lower() == "ols":
            model_instance = linear_model.Ridge(copy_X=False, alpha=1e-10)
        elif model.lower() == "ridge":
            model_instance = linear_model.Ridge(copy_X=False, alpha=alpha)
        elif model.lower() == "lasso":
            model_instance = linear_model.Lasso(copy_X=False, max_iter=50, alpha=alpha)
        else:
            raise ValueError(
                f"Invalid model string: {model}. Should be 'ols', 'ridge', or 'lasso'."
            )
    elif isinstance(model, linear_model.BaseEstimator):
        model_instance = model
    else:
        raise ValueError(
            "Model should either be a string or \
                an instance of sklearn.linear_model BaseEstimator."
        )
    cur_x, y = _vol_split(train, vol_idx)
    model_instance.fit(cur_x.T, y.T)
    return model_instance, cur_x


def vol_denoise(
    data_dict, b0_idx, dwi_idx, model, alpha, b0_denoising, verbose, tmp_dir
):
    """Denoise a single 3D volume using train and test phase.

    Parameters
    ----------
    data_dict : dict
        Dictionary containing the following:
            data_name : str
                The name of the memmap file containing the memmaped data.
            data_dtype : dtype
                The dtype of the data.
            data_shape : tuple
                The shape of the data.
            data_b0s : ndarray
                Array of all 3D patches flattened out to be 2D for b0 volumes.
            data_dwi : ndarray
                Array of all 3D patches flattened out to be 2D for dwi volumes.
    b0_idx : ndarray
        The indices of the b0 volumes.
    dwi_idx : ndarray
        The indices of the dwi volumes.
    model : sklearn.base.RegressorMixin
        This is the model that is initialized from the `_fit_denoising_model` function.
    alpha : float
        Regularization parameter only for ridge and lasso regression models.
    b0_denoising : bool
        Skips denoising b0 volumes if set to False.
    verbose : bool
        Show progress of Patch2Self and time taken.
    tmp_dir : str
        The directory to save the temporary files.

    Returns
    -------
    denoised_arr.name : str
        The name of the memmap file containing the denoised array.
    denoised_arr.dtype : dtype
        The dtype of the denoised array.
    denoised_arr.shape : tuple
        The shape of the denoised array.

    """
    data_shape = data_dict["data"][2]
    data_tmp = np.memmap(
        data_dict["data"][0],
        dtype=data_dict["data"][1],
        mode="r",
        shape=data_dict["data"][2],
    ).reshape(np.prod(data_shape[:-1]), data_shape[-1])
    data_b0s = data_dict["data_b0s"]
    data_dwi = data_dict["data_dwi"]
    p = data_tmp.shape[0] // 10
    b0_counter = 0
    dwi_counter = 0
    start_idx = 0
    denoised_arr_file = tempfile.NamedTemporaryFile(
        delete=False, dir=tmp_dir, suffix="denoised_arr"
    )
    denoised_arr_file.close()
    denoised_arr = np.memmap(
        denoised_arr_file.name, dtype=data_tmp.dtype, mode="w+", shape=data_shape
    )
    idx_counter = 0
    full_result = np.empty(
        (data_shape[0], data_shape[1], data_shape[2], data_shape[3] // 5)
    )
    b0_idx = b0_idx
    dwi_idx = dwi_idx
    if data_b0s.shape[0] == 1:
        b0_denoising = False
    if not b0_denoising:
        if verbose:
            print("b0 denoising skipped....")
    for vol_idx in tqdm(
        range(data_shape[-1]), desc="Fitting and Denoising", leave=False
    ):
        if vol_idx in b0_idx.flatten():
            if b0_denoising:
                b_fit, _ = _fit_denoising_model(data_b0s, b0_counter, model, alpha)
                b_matrix = np.zeros(data_tmp.shape[-1])
                b_fit_coef = np.insert(b_fit.coef_, b0_counter, 0)
                np.put(b_matrix, b0_idx, b_fit_coef)
                result = np.zeros(data_tmp.shape[0])
                for z in range(0, data_tmp.shape[0], p):
                    end_idx = z + p
                    if end_idx > z + p:
                        end_idx = data_tmp.shape[0]
                    result[z:end_idx] = (
                        np.matmul(np.squeeze(data_tmp[z:end_idx, :]), b_matrix)
                        + b_fit.intercept_
                    )
                full_result[..., idx_counter] = result.reshape(
                    data_shape[0], data_shape[1], data_shape[2]
                )
                idx_counter += 1
                b0_counter += 1
                del b_fit_coef
                del b_matrix
                del result
            else:
                full_result[..., idx_counter] = data_tmp[..., b0_counter].reshape(
                    data_shape[0], data_shape[1], data_shape[2]
                )
                b0_counter += 1
                idx_counter += 1
        else:
            dwi_fit, _ = _fit_denoising_model(data_dwi, dwi_counter, model, alpha)
            b_matrix = np.zeros(data_tmp.shape[-1])
            dwi_fit_coef = np.insert(dwi_fit.coef_, dwi_counter, 0)
            np.put(b_matrix, dwi_idx, dwi_fit_coef)
            del dwi_fit_coef
            result = np.zeros(data_tmp.shape[0])
            for z in range(0, data_tmp.shape[0], p):
                end_idx = z + p
                if end_idx > z + p:
                    end_idx = data_tmp.shape[0]
                result[z:end_idx] = (
                    np.matmul(np.squeeze(data_tmp[z:end_idx, :]), b_matrix)
                    + dwi_fit.intercept_
                )
            full_result[..., idx_counter] = result.reshape(
                data_shape[0], data_shape[1], data_shape[2]
            )
            idx_counter += 1
            dwi_counter += 1
        if idx_counter >= data_shape[-1] // 5:
            denoised_arr[..., start_idx : vol_idx + 1] = full_result
            start_idx = vol_idx + 1
            idx_counter = 0
    denoised_arr_idx = data_shape[-1] - data_shape[-1] % 5
    full_result_idx = full_result.shape[-1] - data_shape[-1] % 5
    denoised_arr[..., denoised_arr_idx:] = full_result[..., full_result_idx:]
    del full_result
    return denoised_arr_file.name, denoised_arr.dtype, denoised_arr.shape


@warning_for_keywords()
def patch2self(
    data,
    bvals,
    *,
    patch_radius=(0, 0, 0),
    model="ols",
    b0_threshold=50,
    out_dtype=None,
    alpha=1.0,
    verbose=False,
    b0_denoising=True,
    clip_negative_vals=False,
    shift_intensity=True,
    tmp_dir=None,
    version=3,
):
    """Patch2Self Denoiser.

    See :footcite:p:`Fadnavis2020` for further details about the method.
    See :footcite:p:`Fadnavis2024` for further details about the new method.

    Parameters
    ----------
    data : ndarray
        The 4D noisy DWI data to be denoised.
    bvals : array of shape (N,)
        Array of the bvals from the DWI acquisition
    patch_radius : int or array of shape (3,), optional
        The radius of the local patch to be taken around each voxel (in
        voxels).
    model : string, or sklearn.base.RegressorMixin, optional
        This will determine the algorithm used to solve the set of linear
        equations underlying this model. If it is a string it needs to be
        one of the following: {'ols', 'ridge', 'lasso'}. Otherwise,
        it can be an object that inherits from
        `dipy.optimize.SKLearnLinearSolver` or an object with a similar
        interface from Scikit-Learn:
        `sklearn.linear_model.LinearRegression`,
        `sklearn.linear_model.Lasso` or `sklearn.linear_model.Ridge`
        and other objects that inherit from `sklearn.base.RegressorMixin`.
    b0_threshold : int, optional
        Threshold for considering volumes as b0.
    out_dtype : str or dtype, optional
        The dtype for the output array. Default: output has the same dtype as
        the input.
    alpha : float, optional
        Regularization parameter only for ridge regression model.
    verbose : bool, optional
        Show progress of Patch2Self and time taken.
    b0_denoising : bool, optional
        Skips denoising b0 volumes if set to False.
    clip_negative_vals : bool, optional
        Sets negative values after denoising to 0 using `np.clip`.
    shift_intensity : bool, optional
        Shifts the distribution of intensities per volume to give
        non-negative values.
    tmp_dir : str, optional
        The directory to save the temporary files. If None, the temporary
        files are saved in the system's default temporary directory.
    version : int, optional
        Version 1 or 3 of Patch2Self to use.

    Returns
    -------
    denoised array : ndarray
        This is the denoised array of the same size as that of the input data,
        clipped to non-negative values.

    References
    ----------
    .. footbibliography::

    """

    out_dtype, tmp_dir, patch_radius = _validate_inputs(
        data, out_dtype, patch_radius, version, tmp_dir
    )

    if version == 1:
        return _patch2self_version1(
            data,
            bvals,
            patch_radius,
            model,
            b0_threshold,
            out_dtype,
            alpha,
            verbose,
            b0_denoising,
            clip_negative_vals,
            shift_intensity,
        )
    return _patch2self_version3(
        data,
        bvals,
        model,
        b0_threshold,
        out_dtype,
        alpha,
        verbose,
        b0_denoising,
        clip_negative_vals,
        shift_intensity,
        tmp_dir,
    )


def _validate_inputs(data, out_dtype, patch_radius, version, tmp_dir):
    """Validate inputs for patch2self function.

    Parameters
    ----------
    data : ndarray
        The 4D noisy DWI data to be denoised.
    out_dtype : str or dtype
        The dtype for the output array.
    patch_radius : int or array of shape (3,)
        The radius of the local patch to be taken around each voxel (in
        voxels).
    version : int
        Version 1 or 3 of Patch2Self to use.
    tmp_dir : str
        The directory to save the temporary files. If None, the temporary
        files are saved in the system's default temporary directory.

    Raises
    ------
    ValueError
        If temporary directory is not None for Patch2Self version 1.
        If the patch_radius is not 0 for Patch2Self version 3.
        If the temporary directory does not exist.
        If the input data is not a 4D array.

    Warns
    -----
    If the input data has less than 10 3D volumes.

    Returns
    -------
    out_dtype : str or dtype
        The dtype for the output array.
    tmp_dir : str
        The directory to save the temporary files. If None, the temporary
        files are saved in the system's default temporary directory.

    """
    if out_dtype is None:
        out_dtype = data.dtype

    if tmp_dir is None and version == 3:
        tmp_dir = tempfile.gettempdir()

    if version not in [1, 3]:
        raise ValueError("Invalid version. Should be 1 or 3.")

    if version == 1 and tmp_dir is not None:
        raise ValueError(
            "Temporary directory is not supported for Patch2Self version 1. \
                Please set tmp_dir to None."
        )
    if patch_radius != (0, 0, 0) and version == 3:
        raise ValueError(
            "Patch radius is not supported for Patch2Self version 3. \
                Please do not set patch_radius."
        )

    if isinstance(patch_radius, list) and len(patch_radius) == 1:
        patch_radius = (patch_radius[0], patch_radius[0], patch_radius[0])

    if isinstance(patch_radius, int):
        patch_radius = (patch_radius, patch_radius, patch_radius)

    if version == 3 and tmp_dir is not None and not os.path.exists(tmp_dir):
        raise ValueError("The temporary directory does not exist.")
    if data.ndim != 4:
        raise ValueError("Patch2Self can only denoise on 4D arrays.", data.shape)
    if data.shape[3] < 10:
        warn(
            "The input data has less than 10 3D volumes. \
                Patch2Self may not give optimal denoising performance.",
            stacklevel=2,
        )

    return out_dtype, tmp_dir, patch_radius


def _patch2self_version1(
    data,
    bvals,
    patch_radius,
    model,
    b0_threshold,
    out_dtype,
    alpha,
    verbose,
    b0_denoising,
    clip_negative_vals,
    shift_intensity,
):
    """Patch2Self Denoiser.

    Parameters
    ----------
    data : ndarray
        The 4D noisy DWI data to be denoised.
    bvals : array of shape (N,)
        Array of the bvals from the DWI acquisition.
    patch_radius : int or array of shape (3,)
        The radius of the local patch to be taken around each voxel (in
        voxels).
    model : string, or sklearn.base.RegressorMixin
        This will determine the algorithm used to solve the set of linear
        equations underlying this model. If it is a string it needs to be
        one of the following: {'ols', 'ridge', 'lasso'}. Otherwise,
        it can be an object that inherits from
        `dipy.optimize.SKLearnLinearSolver` or an object with a similar
        interface from Scikit-Learn:
        `sklearn.linear_model.LinearRegression`,
        `sklearn.linear_model.Lasso` or `sklearn.linear_model.Ridge`
        and other objects that inherit from `sklearn.base.RegressorMixin`.
    b0_threshold : int
        Threshold for considering volumes as b0.
    out_dtype : str or dtype
        The dtype for the output array. Default: output has the same dtype as
        the input.
    alpha : float
        Regularization parameter only for ridge regression model.
    verbose : bool
        Show progress of Patch2Self and time taken.
    b0_denoising : bool
        Skips denoising b0 volumes if set to False.
    clip_negative_vals : bool
        Sets negative values after denoising to 0 using `np.clip`.
    shift_intensity : bool
        Shifts the distribution of intensities per volume to give
        non-negative values.

    Returns
    -------
    denoised array : ndarray
        This is the denoised array of the same size as that of the input data,
        clipped to non-negative values.
    """

    # We retain float64 precision, iff the input is in this precision:
    if data.dtype == np.float64:
        calc_dtype = np.float64

    # Otherwise, we'll calculate things in float32 (saving memory)
    else:
        calc_dtype = np.float32

    original_shape = data.shape
    if 1 in data.shape and data.shape[-1] != 1:
        position = data.shape.index(1)
        data = np.concatenate((data, data, data), position)

    # Segregates volumes by b0 threshold
    b0_idx = np.argwhere(bvals <= b0_threshold)
    dwi_idx = np.argwhere(bvals > b0_threshold)

    data_b0s = np.squeeze(np.take(data, b0_idx, axis=3))
    data_dwi = np.squeeze(np.take(data, dwi_idx, axis=3))

    # create empty arrays
    denoised_b0s = np.empty(data_b0s.shape, dtype=calc_dtype)
    denoised_dwi = np.empty(data_dwi.shape, dtype=calc_dtype)

    denoised_arr = np.empty(data.shape, dtype=calc_dtype)

    if verbose is True:
        t1 = time.time()

    # if only 1 b0 volume, skip denoising it
    if data_b0s.ndim == 3 or not b0_denoising:
        if verbose:
            print("b0 denoising skipped...")
        denoised_b0s = data_b0s

    else:
        train_b0 = _extract_3d_patches(
            np.pad(
                data_b0s,
                (
                    (patch_radius[0], patch_radius[0]),
                    (patch_radius[1], patch_radius[1]),
                    (patch_radius[2], patch_radius[2]),
                    (0, 0),
                ),
                mode="constant",
            ),
            patch_radius=patch_radius,
        )

        for vol_idx in range(0, data_b0s.shape[3]):
            b0_model, cur_x = _fit_denoising_model(
                train_b0, vol_idx, model, alpha=alpha
            )

            denoised_b0s[..., vol_idx] = b0_model.predict(cur_x.T).reshape(
                data_b0s.shape[0], data_b0s.shape[1], data_b0s.shape[2]
            )

        if verbose is True:
            print("Denoised b0 Volume: ", vol_idx)
    # Separate denoising for DWI volumes
    train_dwi = _extract_3d_patches(
        np.pad(
            data_dwi,
            (
                (patch_radius[0], patch_radius[0]),
                (patch_radius[1], patch_radius[1]),
                (patch_radius[2], patch_radius[2]),
                (0, 0),
            ),
            mode="constant",
        ),
        patch_radius=patch_radius,
    )

    # Insert the separately denoised arrays into the respective empty arrays
    for vol_idx in range(0, data_dwi.shape[3]):
        dwi_model, cur_x = _fit_denoising_model(train_dwi, vol_idx, model, alpha=alpha)
        denoised_dwi[..., vol_idx] = dwi_model.predict(cur_x.T).reshape(
            data_dwi.shape[0], data_dwi.shape[1], data_dwi.shape[2]
        )

        if verbose is True:
            print("Denoised DWI Volume: ", vol_idx)

    if verbose is True:
        t2 = time.time()
        print("Total time taken for Patch2Self: ", t2 - t1, " seconds")

    if data_b0s.ndim == 3:
        denoised_arr[:, :, :, b0_idx[0][0]] = denoised_b0s
    else:
        for i, idx in enumerate(b0_idx):
            denoised_arr[:, :, :, idx[0]] = np.squeeze(denoised_b0s[..., i])

    for i, idx in enumerate(dwi_idx):
        denoised_arr[:, :, :, idx[0]] = np.squeeze(denoised_dwi[..., i])

    if 1 in original_shape and original_shape[-1] != 1:
        denoised_arr = np.take(denoised_arr, [0], axis=position)

    denoised_arr = _apply_post_processing(
        denoised_arr, shift_intensity, clip_negative_vals
    )
    return np.array(denoised_arr, dtype=out_dtype)


def _patch2self_version3(
    data,
    bvals,
    model,
    b0_threshold,
    out_dtype,
    alpha,
    verbose,
    b0_denoising,
    clip_negative_vals,
    shift_intensity,
    tmp_dir,
):
    """Patch2Self Denoiser.

    Parameters
    ----------
    data : ndarray
        The 4D noisy DWI data to be denoised.
    bvals : array of shape (N,)
        Array of the bvals from the DWI acquisition.
    model : string, or sklearn.base.RegressorMixin
        This will determine the algorithm used to solve the set of linear
        equations underlying this model. If it is a string it needs to be
        one of the following: {'ols', 'ridge', 'lasso'}. Otherwise,
        it can be an object that inherits from
        `dipy.optimize.SKLearnLinearSolver` or an object with a similar
        interface from Scikit-Learn:
        `sklearn.linear_model.LinearRegression`,
        `sklearn.linear_model.Lasso` or `sklearn.linear_model.Ridge`
        and other objects that inherit from `sklearn.base.RegressorMixin`.
    b0_threshold : int
        Threshold for considering volumes as b0.
    out_dtype : str or dtype
        The dtype for the output array. Default: output has the same dtype as
        the input.
    alpha : float
        Regularization parameter only for ridge regression model.
    verbose : bool
        Show progress of Patch2Self and time taken.
    b0_denoising : bool
        Skips denoising b0 volumes if set to False.
    clip_negative_vals : bool
        Sets negative values after denoising to 0 using `np.clip`.
    shift_intensity : bool
        Shifts the distribution of intensities per volume to give
        non-negative values.
    tmp_dir : str
        The directory to save the temporary files. If None, the temporary
        files are saved in the system's default temporary directory.

    Returns
    -------
    denoised array : ndarray
        This is the denoised array of the same size as that of the input data,
        clipped to non-negative values.

    """
    tmp_file = tempfile.NamedTemporaryFile(delete=False, dir=tmp_dir, suffix="tmp_file")
    tmp_file.close()
    tmp = np.memmap(
        tmp_file.name,
        dtype=data.dtype,
        mode="w+",
        shape=(data.shape[0], data.shape[1], data.shape[2], data.shape[3]),
    )
    p = data.shape[-1] // 5
    idx_start = 0
    for z in range(0, data.shape[3], p):
        end_idx = z + p
        if end_idx > data.shape[3]:
            end_idx = data.shape[3]
        if verbose:
            print("Loading data from {} to {}".format(idx_start, end_idx))
        tmp[..., idx_start:end_idx] = data[..., idx_start:end_idx]
        idx_start = end_idx
    sketch_rows = int(0.30 * data.shape[0] * data.shape[1] * data.shape[2])
    sketched_matrix_name, sketched_matrix_dtype, sketched_matrix_shape = count_sketch(
        tmp_file.name,
        data.dtype,
        tmp.shape,
        sketch_rows=sketch_rows,
        tmp_dir=tmp_dir,
    )
    sketched_matrix = np.memmap(
        sketched_matrix_name,
        dtype=sketched_matrix_dtype,
        mode="r",
        shape=sketched_matrix_shape,
    ).T
    if verbose:
        print("Sketching done.")
    b0_idx = np.argwhere(bvals <= b0_threshold)
    dwi_idx = np.argwhere(bvals > b0_threshold)
    data_b0s = np.take(np.squeeze(sketched_matrix), b0_idx, axis=0)
    data_dwi = np.take(np.squeeze(sketched_matrix), dwi_idx, axis=0)
    data_dict = {
        "data": [tmp_file.name, data.dtype, tmp.shape],
        "data_b0s": data_b0s,
        "data_dwi": data_dwi,
    }
    if verbose:
        t1 = time.time()
    del sketched_matrix
    os.unlink(sketched_matrix_name)
    denoised_arr_name, denoised_arr_dtype, denoised_arr_shape = vol_denoise(
        data_dict,
        b0_idx,
        dwi_idx,
        model,
        alpha,
        b0_denoising,
        verbose,
        tmp_dir=tmp_dir,
    )
    denoised_arr = np.memmap(
        denoised_arr_name,
        dtype=denoised_arr_dtype,
        mode="r+",
        shape=denoised_arr_shape,
    )
    if verbose:
        t2 = time.time()
        print("Time taken for Patch2Self: ", t2 - t1, " seconds.")

    denoised_arr = _apply_post_processing(
        denoised_arr, shift_intensity, clip_negative_vals
    )
    del tmp
    os.unlink(data_dict["data"][0])
    result = np.array(denoised_arr, dtype=out_dtype)
    del denoised_arr
    os.unlink(denoised_arr_name)
    return result


def _apply_post_processing(denoised_arr, shift_intensity, clip_negative_vals):
    """Apply post-processing steps such as clipping and shifting intensities.

    Parameters
    ----------
    denoised_arr : ndarray
        The denoised array.
    shift_intensity : bool
        Shifts the distribution of intensities per volume to give
        non-negative values.
    clip_negative_vals : bool
        Sets negative values after denoising to 0 using `np.clip`.

    Returns
    -------
    denoised_arr : ndarray
        The denoised array with post-processing applied.

    """
    if shift_intensity and not clip_negative_vals:
        for i in range(denoised_arr.shape[-1]):
            shift = np.min(denoised_arr[..., i]) - np.min(denoised_arr[..., i])
            denoised_arr[..., i] += shift
    elif clip_negative_vals and not shift_intensity:
        denoised_arr.clip(min=0, out=denoised_arr)
    elif clip_negative_vals and shift_intensity:
        warn(
            "Both `clip_negative_vals` and `shift_intensity` cannot be True. \
                Defaulting to `clip_negative_vals`...",
            stacklevel=2,
        )
        denoised_arr.clip(min=0, out=denoised_arr)
    return denoised_arr
