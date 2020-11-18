import numpy as np
import warnings
from dipy.utils.optpkg import optional_package
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
    --------
    cur_x : ndarray
        Array of patches corresponding to all the volumes except from the held
        -out volume.

    y : ndarray
        Array of patches corresponding to the volume that is used a target for
        denoising.
    """

    # Delete the f-th volume
    x1 = train[:vol_idx, :, :]
    x2 = train[vol_idx+1:, :, :]

    cur_x = np.reshape(np.concatenate((x1, x2), axis=0),
                       ((train.shape[0]-1)*train.shape[1], train.shape[2]))

    # Center voxel of the selected block
    y = train[vol_idx, train.shape[1]//2, :]
    return cur_x, y


def _vol_denoise(train, vol_idx, model, data):
    """ Denoise a single 3D volume using a train and test phase.

    Parameters
    ----------
    train : ndarray
        Array of all 3D patches flattened out to be 2D.

    vol_idx: int
        The volume number that needs to be held out for training.

    model: str, optional
        Corresponds to the object of the regressor being used for
        performing the denoising. Options: 'ols', 'ridge', 'lasso'
        default: 'ridge'.

    data: ndarray
        The 4D noisy DWI data to be denoised.

    Returns
    --------
    model prediction : ndarray
        Denoised array of all 3D patches flattened out to be 2D corresponding
        to the held out volume `vol_idx`.

    """

    # to add a new model, use the following API
    # We adhere to the following options as they are used for comparisons
    if model.lower() == 'ols':
        model = linear_model.LinearRegression(copy_X=False,
                                              fit_intercept=True,
                                              n_jobs=-1, normalize=False)

    elif model.lower() == 'ridge':
        model = linear_model.Ridge()

    elif model.lower() == 'lasso':
        model = linear_model.Lasso(max_iter=50)

    else:
        warnings.warn('Model not supported. Choose from: ols, ridge or lasso')

    cur_x, y = _vol_split(train, vol_idx)
    model.fit(cur_x.T, y.T)

    return model.predict(cur_x.T).reshape(data.shape[0], data.shape[1],
                                          data.shape[2])


def _extract_3d_patches(arr, patch_radius=[0, 0, 0]):
    """ Extract 3D patches from 4D DWI data.

    Parameters
    ----------
    arr : ndarray
        The 4D noisy DWI data to be denoised.

    patch_radius : int or 1D array (optional)
        The radius of the local patch to be taken around each voxel (in
        voxels). Default: 0 (denoise in blocks of 1x1x1 voxels).

    Returns
    --------
    all_patches : ndarray
        All 3D patches flattened out to be 2D corresponding to the each 3D
        volume of the 4D DWI data.

    """

    if isinstance(patch_radius, int):
        patch_radius = np.ones(3, dtype=int) * patch_radius
    if len(patch_radius) != 3:
        raise ValueError("patch_radius should have length 3")
    else:
        patch_radius = np.asarray(patch_radius).astype(int)
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


def patch2self(data, patch_radius=[0, 0, 0], model='ridge'):
    """ Patch2Self Denoiser.

    Parameters
    ----------
    data : ndarray
        The 4D noisy DWI data to be denoised.

    patch_radius : int or 1D array (optional)
        The radius of the local patch to be taken around each voxel (in
        voxels). Default: 0 (denoise in blocks of 1x1x1 voxels).

    model: str, optional
        Corresponds to the object of the regressor being used for
        performing the denoising. Options: 'ols', 'ridge' qnd 'lasso'
        default: 'ridge'.

    Returns
    --------
    denoised array : ndarray
        The 4D denoised DWI data.

    References
    ----------

    .. [Fadnavis20] S. Fadnavis, J. Batson, E. Garyfallidis, Patch2Self:
                    Denoising Diffusion MRI with Self-supervised Learning,
                    Advances in Neural Information Processing Systems 33 (2020)
    """

    idx_max = np.argmax([np.mean(data[..., i])
                         for i in range(0, data.shape[3])])
    data = np.insert(data, 0, data[..., idx_max], axis=3)

    train = _extract_3d_patches(np.pad(data, ((patch_radius[0],
                                               patch_radius[0]),
                                              (patch_radius[1],
                                               patch_radius[1]),
                                              (patch_radius[2],
                                               patch_radius[2]),
                                              (0, 0)), mode='constant'),
                                patch_radius=patch_radius)

    patch_radius = np.asarray(patch_radius).astype(int)
    denoised_array = np.zeros((data.shape))

    for vol_idx in range(0, data.shape[3]):
        denoised_array[..., vol_idx] = _vol_denoise(train,
                                                    vol_idx, model, data)

    denoised_array = np.delete(denoised_array, 0, axis=3)

    return denoised_array
