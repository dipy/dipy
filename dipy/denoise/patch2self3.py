import atexit
import os
import tempfile
import time
from warnings import warn

import numpy as np
from tqdm import tqdm

from dipy.utils.optpkg import optional_package

sklearn, has_sklearn, _ = optional_package("sklearn")
linear_model, _, _ = optional_package("sklearn.linear_model")


def count_sketch(matrixA_name, matrixA_dtype, matrixA_shape, sketch_rows, tmp_dir):
    """Count Sketching algorithm to reduce the size of the matrix.

    Parameters
    ----------
    matrixA_name : str
        The name of the memmap file containing the matrix A.
    matrixA_dtype : dtype
        The dtype of the matrix A.
    matrixA_shape : tuple
        The shape of the matrix A.
    s : int
        The number of rows in the sketch matrix.
    tmp_dir : str
        The directory to save the temporary files.
    sketch_rows : int
        The number of rows in the sketch matrix.

    Returns
    -------
    matrixC_name : str
        The name of the memmap file containing the sketch matrix.
    matrixC_dtype : dtype
        The dtype of the sketch matrix.
    matrixC_shape : tuple
        The shape of the sketch matrix.

    """
    matrixA = np.squeeze(
        np.memmap(matrixA_name, dtype=matrixA_dtype, mode="r+", shape=matrixA_shape)
    ).reshape(np.prod(matrixA_shape[:-1]), matrixA_shape[-1])
    m, n = matrixA.shape
    matrixt_file = tempfile.NamedTemporaryFile(delete=False, dir=tmp_dir)
    matrixt = np.memmap(matrixt_file.name, dtype=matrixA_dtype, mode="w+", shape=(m, n))
    matrixC_file = tempfile.NamedTemporaryFile(delete=False, dir=tmp_dir)
    matrixC = np.memmap(
        matrixC_file.name, dtype=matrixA_dtype, mode="w+", shape=(sketch_rows, n)
    )
    hashedIndices = np.random.choice(sketch_rows, m, replace=True)
    randSigns = np.random.choice(2, m, replace=True) * 2 - 1
    for i in range(0, m, m // 20):
        end_index = min(i + m // 20, m)
        matrixt[i:end_index, :] = (
            matrixA[i:end_index, :] * randSigns[i:end_index, np.newaxis]
        )
    matrixt.flush()
    np.add.at(matrixC, hashedIndices, matrixt)
    matrixC.flush()
    os.unlink(matrixt_file.name)
    return matrixC_file.name, matrixC.dtype, matrixC.shape


def _vol_split(train, vol_idx):
    """Split the 3D volumes into the train and test set.

    Parameters
    ----------
    train : ndarray
        Array of all 3D patches flattened out to be 2D.

    vol_idx : int
        The volume number that needs to be held out for training.

    Returns
    -------
    cur_x : 2D-array (nvolumes * patch_size) x (nvoxels)
        Array of patches corresponding to all volumes except the held out volume.

    y : 1D-array
        Array of patches corresponding to the volume that is used a target for denoising
    """
    mask = np.zeros(train.shape[0], dtype=bool)
    mask[vol_idx] = True
    cur_x = train[~mask]
    cur_x = cur_x.reshape((train.shape[0] - 1) * train.shape[1], train.shape[2])
    y = train[vol_idx, train.shape[1] // 2, :]
    return cur_x, y


def _fit_denoising_model(train, vol_idx, model, alpha):
    """Fit a single 3D volume using a train and test phase.

    Parameters
    ----------
    train : ndarray
        Array of all 3D patches flattened out to be 2D.

    vol_idx : int
        The volume number that needs to be held out for training.

    model : str or initialized linear model object.
        This will determine the algorithm used to solve the set of linear
        equations underlying this model. If it is a string it needs to be
        one of the following: {'ols', 'ridge', 'lasso'}. Otherwise,
        it can be an object that inherits from
        `dipy.optimize.SKLearnLinearSolver` or an object with a similar
        interface from Scikit-Learn:
        `sklearn.linear_model.LinearRegression`,
        `sklearn.linear_model.Lasso` or `sklearn.linear_model.Ridge`
        and other objects that inherit from `sklearn.base.RegressorMixin`.
        Default: 'ridge'.

    alpha : float, optional
        Regularization parameter only for ridge and lasso regression models.
        default: 1.0.

    Returns
    -------
    model_instance : object
        The fitted model instance.
    """
    if isinstance(model, str):
        if model.lower() == "ols":
            model_instance = linear_model.LinearRegression(copy_X=False)
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
    return model_instance


def vol_denoise(
    data_dict, b0_idx, dwi_idx, model, alpha, b0_denoising, verbose, tmp_dir
):
    """Denoise a single 3D volume using train and test phase.

    Parameters
    ----------
    data_dict : dict
        Dictionary containing the following
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
    model : str or initialized linear model object.
        This is the model that is initialized from the _fit_denoising_model function.
    alpha : float, optional
        Regularization parameter only for ridge and lasso regression models.
        default: 1.0.
    b0_denoising : bool, optional
        Skips denoising b0 volumes if set to False.
    verbose : bool, optional
        Show progress of Patch2Self and time taken.
    tmp_dir : str
        The directory to save the temporary files.

    Returns
    -------
    denoised_arr_name : str
        The name of the memmap file containing the denoised array.
    denoised_arr_dtype : dtype
        The dtype of the denoised array.
    denoised_arr_shape : tuple
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
    denoised_arr_file = tempfile.NamedTemporaryFile(delete=False, dir=tmp_dir)
    denoised_arr = np.memmap(
        denoised_arr_file.name, dtype=data_tmp.dtype, mode="w+", shape=data_shape
    )
    idx_counter = 0
    full_result = np.empty(
        (data_shape[0], data_shape[1], data_shape[2], data_shape[3] // 5)
    )
    b0_idx = b0_idx
    dwi_idx = dwi_idx
    if data_b0s.shape[0] == 1 or not b0_denoising:
        if verbose:
            print("b0 denoising skipped....")
        for i in range(data_b0s.shape[0]):
            full_result[..., i] = data_tmp[..., b0_counter].reshape(
                data_shape[0], data_shape[1], data_shape[2]
            )
            b0_counter += 1
            idx_counter += 1
    for vol_idx in tqdm(
        range(data_shape[-1]), desc="Fitting and Denoising", leave=False
    ):
        if vol_idx in b0_idx.flatten():
            if b0_denoising:
                b_fit = _fit_denoising_model(data_b0s, b0_counter, model, alpha)
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
            dwi_fit = _fit_denoising_model(data_dwi, dwi_counter, model, alpha)
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
    os.unlink(data_dict["data"][0])
    return denoised_arr_file.name, denoised_arr.dtype, denoised_arr.shape


def patch2self(
    data,
    bvals,
    model="ols",
    b0_threshold=50,
    out_dtype=None,
    alpha=1.0,
    verbose=False,
    b0_denoising=True,
    clip_negative_vals=False,
    shift_intensity=True,
    as_ndarray=False,
    tmp_dir=None,
):
    """Patch2Self Denoiser.

    Parameters
    ----------
    data : ndarray
        The 4D noisy DWI data to be denoised.

    bvals : 1D array
        Array of the bvals from the DWI acquisition

    model : string, or initialized linear model object.
            This will determine the algorithm used to solve the set of linear
            equations underlying this model. If it is a string it needs to be
            one of the following: {'ols', 'ridge', 'lasso'}. Otherwise,
            it can be an object that inherits from
            `dipy.optimize.SKLearnLinearSolver` or an object with a similar
            interface from Scikit-Learn:
            `sklearn.linear_model.LinearRegression`,
            `sklearn.linear_model.Lasso` or `sklearn.linear_model.Ridge`
            and other objects that inherit from `sklearn.base.RegressorMixin`.
            Default: 'ols'.

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
        non-negative values

    as_ndarray : bool, optional
        If True, the output is returned as a numpy array else as numpy memmap. Default: False.

    tmp_dir : str, optional
        The directory to save the temporary files. If None, the temporary
        files are saved in the system's default temporary directory. Default: None.

    Returns
    -------
    denoised array : ndarray
        This is the denoised array of the same size as that of the input data.

    References
    ----------
    [Fadnavis20] S. Fadnavis, J. Batson, E. Garyfallidis, Patch2Self:
                    Denoising Diffusion MRI with Self-supervised Learning,
                    Advances in Neural Information Processing Systems 33 (2020)

    [Fadnavis20] S. Fadnavis, J. Batson, E. Garyfallidis, Patch2Self:
                    Denoising Diffusion MRI with Self-supervised Learning,
                    Advances in Neural Information Processing Systems 33 (2020)

    [Fadnavis24] S. Fadnavis, A. Chowdhury, J. Batson, P. Drineas,
                    E. Garyfallidis, Patch2Self2: Self-supervised Denoising
                    on Coresets via Matrix Sketching, Proceedings of the IEEE/CVF
                    Conference on Computer Vision and Pattern Recognition (2024),
                    27641-27651.

    """  # noqa: E501
    if tmp_dir is None:
        tmp_dir = tempfile.gettempdir()
    if os.path.exists(tmp_dir) is False:
        raise ValueError("The temporary directory does not exist.")
    if not data.ndim == 4:
        raise ValueError("Patch2Self can only denoise on 4D arrays.", data.shape)
    if data.shape[3] < 10:
        warn(
            "The input data has less than 10 3D volumes. \
                Patch2Self may not give denoising performance.",
                stacklevel=2
        )
    if out_dtype is None:
        out_dtype = data.dtype
    tmp_file = tempfile.NamedTemporaryFile(delete=False, dir=tmp_dir)
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
    sketch_rows = int(0.30 * data.size)
    sketched_matrix_name, sketched_matrix_dtype, sketched_matrix_shape = count_sketch(
        tmp_file.name, data.dtype, tmp.shape, sketch_rows=sketch_rows, tmp_dir=tmp_dir
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
    os.unlink(sketched_matrix_name)
    denoised_arr_name, denoised_arr_dtype, denoised_arr_shape = vol_denoise(
        data_dict, b0_idx, dwi_idx, model, alpha, b0_denoising, verbose, tmp_dir=tmp_dir
    )
    denoised_arr = np.memmap(
        denoised_arr_name, dtype=denoised_arr_dtype, mode="r+", shape=denoised_arr_shape
    )
    if verbose:
        t2 = time.time()
        print("Time taken for Patch2Self: ", t2 - t1, " seconds.")
    if shift_intensity and not clip_negative_vals:
        for i in range(denoised_arr.shape[-1]):
            shift = np.min(tmp[..., i]) - np.min(denoised_arr[..., i])
            denoised_arr[..., i] = denoised_arr[..., i] + shift
    elif clip_negative_vals and not shift_intensity:
        denoised_arr.clip(min=0, out=denoised_arr)
    elif clip_negative_vals and shift_intensity:
        msg = "Both `clip_negative_vals` and `shift_intensity` cannot be True."
        msg += "Defaulting to `clip_negative_vals`..."
        warn(msg,stacklevel=2)
        denoised_arr.clip(min=0, out=denoised_arr)
    atexit.register(os.remove, denoised_arr_name)
    if as_ndarray:
        return np.array(denoised_arr, dtype=out_dtype)
    else:
        return denoised_arr
