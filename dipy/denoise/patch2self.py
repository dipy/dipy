import numpy as np
from warnings import warn
import time
from dipy.utils.optpkg import optional_package
import dipy.core.optimize as opt

sklearn, has_sklearn, _ = optional_package('sklearn')
linear_model, _, _ = optional_package('sklearn.linear_model')


def _vol_split(train, vol_idx):
    """ Split the 3D volumes into the train and test set.

    Parameters
    ----------
    train : ndarray
        Array of all 3D patches flattened out to be 2D.

    vol_idx: int
        The volume number that needs to be held out for training.

    Returns
    -------
    cur_x : 2D-array (nvolumes*patch_size) x (nvoxels)
        Array of patches corresponding to all the volumes except for the
        held-out volume.

    y : 1D-array
        Array of patches corresponding to the volume that is used a target for
        denoising.
    """
    # Hold-out the target volume
    mask = np.zeros(train.shape[0])
    mask[vol_idx] = 1
    cur_x = train[mask == 0]
    cur_x = cur_x.reshape(((train.shape[0]-1)*train.shape[1],
                           train.shape[2]))

    # Center voxel of the selected block
    y = train[vol_idx, train.shape[1]//2, :]
    return cur_x, y


def _vol_denoise(train, vol_idx, model, data_shape, alpha):
    """ Denoise a single 3D volume using a train and test phase.

    Parameters
    ----------
    train : ndarray
        Array of all 3D patches flattened out to be 2D.

    vol_idx : int
        The volume number that needs to be held out for training.

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
            Default: 'ridge'.

    data_shape : ndarray
        The 4D shape of noisy DWI data to be denoised.

    alpha : float, optional
        Regularization parameter only for ridge and lasso regression models.
        default: 1.0

    Returns
    -------
    model prediction : ndarray
        Denoised array of all 3D patches flattened out to be 2D corresponding
        to the held out volume `vol_idx`.

    """
    # To add a new model, use the following API
    # We adhere to the following options as they are used for comparisons
    if model.lower() == 'ols':
        model = linear_model.LinearRegression(copy_X=False)

    elif model.lower() == 'ridge':
        model = linear_model.Ridge(copy_X=False, alpha=alpha)

    elif model.lower() == 'lasso':
        model = linear_model.Lasso(copy_X=False, max_iter=50, alpha=alpha)

    elif (isinstance(model, opt.SKLearnLinearSolver) or
          has_sklearn and isinstance(model, sklearn.base.RegressorMixin)):
        model = model

    else:
        e_s = "The `solver` key-word argument needs to be: "
        e_s += "'ols', 'ridge', 'lasso' or a "
        e_s += "`dipy.optimize.SKLearnLinearSolver` object"
        raise ValueError(e_s)

    cur_x, y = _vol_split(train, vol_idx)
    model.fit(cur_x.T, y.T)

    return model.predict(cur_x.T).reshape(data_shape[0], data_shape[1],
                                          data_shape[2])


def _extract_3d_patches(arr, patch_radius):
    """ Extract 3D patches from 4D DWI data.

    Parameters
    ----------
    arr : ndarray
        The 4D noisy DWI data to be denoised.

    patch_radius : int or 1D array
        The radius of the local patch to be taken around each voxel (in
        voxels).

    Returns
    -------
    all_patches : ndarray
        All 3D patches flattened out to be 2D corresponding to the each 3D
        volume of the 4D DWI data.

    """
    if isinstance(patch_radius, int):
        patch_radius = np.ones(3, dtype=int) * patch_radius
    if len(patch_radius) != 3:
        raise ValueError("patch_radius should have length 3")
    else:
        patch_radius = np.asarray(patch_radius, dtype=int)
    patch_size = 2 * patch_radius + 1

    dim = arr.shape[-1]

    all_patches = []

    # loop around and find the 3D patch for each direction
    for i in range(patch_radius[0], arr.shape[0] -
                   patch_radius[0], 1):
        for j in range(patch_radius[1], arr.shape[1] -
                       patch_radius[1], 1):
            for k in range(patch_radius[2], arr.shape[2] -
                           patch_radius[2], 1):

                ix1 = i - patch_radius[0]
                ix2 = i + patch_radius[0] + 1
                jx1 = j - patch_radius[1]
                jx2 = j + patch_radius[1] + 1
                kx1 = k - patch_radius[2]
                kx2 = k + patch_radius[2] + 1

                X = arr[ix1:ix2, jx1:jx2,
                        kx1:kx2].reshape(np.prod(patch_size), dim)
                all_patches.append(X)

    return np.array(all_patches).T


def patch2self(data, bvals, patch_radius=(0, 0, 0), model='ols',
               b0_threshold=50, out_dtype=None, alpha=1.0, verbose=False,
               b0_denoising=True, clip_negative_vals=False,
               shift_intensity=True):
    """ Patch2Self Denoiser.

    Parameters
    ----------
    data : ndarray
        The 4D noisy DWI data to be denoised.

    bvals : 1D array
        Array of the bvals from the DWI acquisition

    patch_radius : int or 1D array, optional
        The radius of the local patch to be taken around each voxel (in
        voxels). Default: 0 (denoise in blocks of 1x1x1 voxels).

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


    Returns
    -------
    denoised array : ndarray
        This is the denoised array of the same size as that of the input data,
        clipped to non-negative values.

    References
    ----------
    [Fadnavis20] S. Fadnavis, J. Batson, E. Garyfallidis, Patch2Self:
                    Denoising Diffusion MRI with Self-supervised Learning,
                    Advances in Neural Information Processing Systems 33 (2020)

    """
    if isinstance(patch_radius, int):
        patch_radius = np.ones(3, dtype=int) * patch_radius

    patch_radius = np.asarray(patch_radius, dtype=int)

    if not data.ndim == 4:
        raise ValueError("Patch2Self can only denoise on 4D arrays.",
                         data.shape)

    if data.shape[3] < 10:
        warn("The input data has less than 10 3D volumes. Patch2Self may not "
             "give denoising performance.")

    if out_dtype is None:
        out_dtype = data.dtype

    # We retain float64 precision, iff the input is in this precision:
    if data.dtype == np.float64:
        calc_dtype = np.float64

    # Otherwise, we'll calculate things in float32 (saving memory)
    else:
        calc_dtype = np.float32

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
        train_b0 = _extract_3d_patches(np.pad(data_b0s, ((patch_radius[0],
                                              patch_radius[0]),
                                              (patch_radius[1],
                                               patch_radius[1]),
                                              (patch_radius[2],
                                               patch_radius[2]),
                                              (0, 0)), mode='constant'),
                                       patch_radius=patch_radius)

        for vol_idx in range(0, data_b0s.shape[3]):
            denoised_b0s[..., vol_idx] = _vol_denoise(train_b0,
                                                      vol_idx, model,
                                                      data_b0s.shape,
                                                      alpha=alpha)

            if verbose is True:
                print("Denoised b0 Volume: ", vol_idx)

    # Separate denoising for DWI volumes
    train_dwi = _extract_3d_patches(np.pad(data_dwi, ((patch_radius[0],
                                                       patch_radius[0]),
                                                      (patch_radius[1],
                                                       patch_radius[1]),
                                                      (patch_radius[2],
                                                       patch_radius[2]),
                                                      (0, 0)),
                                           mode='constant'),
                                    patch_radius=patch_radius)

    # Insert the separately denoised arrays into the respective empty arrays
    for vol_idx in range(0, data_dwi.shape[3]):
        denoised_dwi[..., vol_idx] = _vol_denoise(train_dwi,
                                                  vol_idx, model,
                                                  data_dwi.shape,
                                                  alpha=alpha)

        if verbose is True:
            print("Denoised DWI Volume: ", vol_idx)

    if verbose is True:
        t2 = time.time()
        print('Total time taken for Patch2Self: ', t2-t1, " seconds")

    if data_b0s.ndim == 3:
        denoised_arr[:, :, :, b0_idx[0][0]] = denoised_b0s
    else:
        for i, idx in enumerate(b0_idx):
            denoised_arr[:, :, :, idx[0]] = np.squeeze(denoised_b0s[..., i])

    for i, idx in enumerate(dwi_idx):
        denoised_arr[:, :, :, idx[0]] = np.squeeze(denoised_dwi[..., i])

    # shift intensities per volume to handle for negative intensities
    if shift_intensity and not clip_negative_vals:
        for i in range(0, denoised_arr.shape[3]):
            shift = np.min(data[..., i]) - np.min(denoised_arr[..., i])
            denoised_arr[..., i] = denoised_arr[..., i] + shift

    # clip out the negative values from the denoised output
    elif clip_negative_vals and not shift_intensity:
        denoised_arr.clip(min=0, out=denoised_arr)

    elif clip_negative_vals and shift_intensity:
        msg = 'Both `clip_negative_vals` and `shift_intensity` cannot be True.'
        msg += ' Defaulting to `clip_negative_vals`...'
        warn(msg)
        denoised_arr.clip(min=0, out=denoised_arr)

    return np.array(denoised_arr, dtype=out_dtype)
