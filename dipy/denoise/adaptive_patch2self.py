import numpy as np
from warnings import warn
import time
from dipy.utils.optpkg import optional_package
import dipy.core.optimize as opt
try:
    from scipy.linalg.lapack import dgesvd as svd
    svd_args = [1, 0]
    # If you have an older version of scipy, we fall back
    # on the standard scipy SVD API:
except ImportError:
    from scipy.linalg import svd
    svd_args = [False]

sklearn, has_sklearn, _ = optional_package('sklearn')
linear_model, _, _ = optional_package('sklearn.linear_model')
decomposition, _, _ = optional_package('sklearn.decomposition')


if not has_sklearn:
    w = "Scikit-Learn is required to denoise the data via (Adaptive)Patch2Self."
    warn(w)


def site_weight_beam(U, pca_ind, sign, sigma, growth_func):
    """
    Return weights that increase going along the sign direction of U[:, pca_ind],
    and fall off Gaussianly going away from that axis.
    """
    u = U[:, pca_ind]
    if sign > 0:
        side = u > 0
    else:
        side = u < 0
    w = growth_func(u[side] / u[side].mean())
    radial_d2 = np.sum(U[side]**2, axis=1) - u[side]**2
    w *= np.exp(-0.5 * radial_d2 / sigma**2)
    return w, side


def site_weight_beam_linear(U, pca_ind, sign, sigma):
    return site_weight_beam(U, pca_ind, sign, sigma, lambda x: x)


def site_weight_beam_arctan(U, pca_ind, sign, sigma):
    return site_weight_beam(U, pca_ind, sign, sigma, np.arctan)


def getRandomState(seed, default=42):
    """Return a np.random.RandomState whether seed is one already, an int, or None."""
    if seed is None:
        rng = np.random.RandomState(default)
    elif isinstance(seed, int):
        rng = np.random.RandomState(seed)
    elif isinstance(seed, np.random.mtrand.RandomState):
        rng = seed
    else:
        raise ValueError("type %s for seed is not supported" % type(seed))
    return rng


def doSVD(X, n_comps=None):
    """Singular Value Decompose X; X = U.dot(S).dot(Vt)

    Parameters
    ----------
    X : (nsamples, nfeatures) array
        Hint: you can optionally demean it by passing it as X - X.mean(axis=0)[np.newaxis, :]
    n_comps : int or None
        Keep the n_comps largest components. None means all. (Does not affect speed)

    Returns
    -------
    U : (nsamples, nfeatures) array
        The principal components in sample space as orthonormal column vectors, matching the order of S.
    S : 1D array with len min(nsamples, nfeatures)
        The "eigenvalues", from largest to smallest
    Vt : (nfeatures, nfeatures) array
        The principal components in feature space as orthonormal row vectors, matching the order of S.
    """
    U, S, Vt = svd(X, *svd_args)[:3]

    # Trim down to the top n_comps principal components. I wonder if there is a way
    # to do this as part of svd.
    if n_comps and n_comps < len(S):
        U = U[:, :n_comps]

        # For completeness - they are not actually used, but take up relatively little memory.
        S = S[:n_comps]
        Vt = Vt[:n_comps]
    return U, S, Vt


def calcSVDU(X, n_comps=None):
    return doSVD(X, n_comps)[0]


def doICA(X, n_comps=None, seed=None, whiten=True):
    """Decompose X into independent components with FastICA.

    Parameters
    ----------
    X : (nsamples, nfeatures) array
        The array to decompose.
    n_comps : int or None
        Keep the n_comps largest components. None means all. (Smaller is faster)
    seed : None, int, or numpy.random.mtrand.RandomState
        Initialize the ICA's RNG to this. Default: np.random.RandomState(42)
    whiten : bool
        Iff True, whiten the input before decomposition.

    Returns
    -------
    U : (nsamples, nfeatures) array
        The principal components in sample space as orthonormal column vectors, matching the order of S.
    S : 1D array with len min(nsamples, nfeatures)
        The "eigenvalues", from largest to smallest
    Vt : (nfeatures, nfeatures) array
        The principal components in feature space as orthonormal row vectors, matching the order of S.
    """
    ica = decomposition.FastICA(n_comps, random_state=getRandomState(seed))
    return ica.fit(X).transform(X)


class AdaptivePatch2Self(object):
    """Creates a set of regressors that are trained using different neighborhoods
    of the data, spread out along the n_comps largest principal components,
    plus one at the center. Then when a value is predicted the model
    coefficients are a combination of the components of each neighborhood,
    weighted according to the argument (X)'s distance to each neighborhood.

    In other words it is a version of Patch2Self where instead of just one set
    of coefficients per volume being denoised, there is a collection of them,
    and the regressor smoothly varies the weights of each to adapt to the
    voxel's type (or mix of them), without allowing the regressor to exactly
    fit the voxel, which would prevent denoising.
    """
    def __init__(self, data, n_comps=None, dtype=np.float32, model='ols', mod_kwargs={},
                 site_weight_func=site_weight_beam_linear, site_placer=doICA):
        """ Calculate the principal components of data and initialize the set of
        regressors and neighborhood weights.

        Parameters
        ----------
        data : 2D array
            The semi-flattened data, with shape (n_voxels, n_volumes).
            The regressions will be strictly over the n_volumes axis.

        n_comps : int or None
            The number of principal components to account for in the variation
            of the model coefficients. Two Gaussian neighborhoods will be created
            for each component, centered at +/- 1 on the component axis and with unit
            standard deviation in each normalized component axis.
            If None use int(ceil(n_volumes**0.5)).

        dtype : dtype
            The data type for calculations

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

        mod_kwargs : dict
            Any keyword arguments to pass to the models.

        site_weight_func : function(U, pca_ind, sign, sigma) returning array
            How to weight the samples around each PC's sites.

        site_placer : function(data, n_comps) -> (data.shape[0], n_comps) array
            A decomposition function such as calcSVDU or doICA.
            doICA gives slightly better results, but usually the components it
            produces are effectively almost equivalent to the 1st n_comps
            principal components. calcSVDU may be useful if doICA fails due to
            a quirk of the data.
        """
        super(AdaptivePatch2Self, self).__init__()
        self.dtype = dtype

        if not n_comps:
            # Assume the number of voxel types that need distinct regression
            # coefficients is a slowly growing function of both the number of
            # voxels and the number of volumes.
            self.n_comps = int(np.ceil(min(data.shape)**0.5))
        else:
            self.n_comps = n_comps

        if mod_kwargs.get('positive'):
            self.nonnegative = True
        else:
            self.nonnegative = False
        #self.nonnegative = True
        sup_mod_kwargs = {k: mod_kwargs.pop(k) for k in ['alpha', 'max_iter']
                          if k in mod_kwargs}
        self.model = supply_model(model, **sup_mod_kwargs, mod_kwargs=mod_kwargs)

        # Do NOT demean data - that introduces an anticorrelation between the
        # volumes that makes it easy for the regression to predict any given
        # volume as -sum(other volumes).
        self.X = data.astype(self.dtype, copy=True)

        # Calculate the primary components of X in order to place fitting
        # sites covering different types of voxels.
        # This time we use the demeaned data because we want U relative to the centroid.
        # (FastICA defaults to whitening, but PCA does not.)
        # The columns of U are the normalized "eigenheads" (U.shape = (X.shape[0], n_comps))
        U = site_placer(self.X - self.X.mean(axis=0)[np.newaxis, :], self.n_comps).astype(self.dtype)

        # The weights for each site, arranged by [sample number, site number],
        # with the central site coming last.
        self.site_weights = np.zeros((self.X.shape[0], 2 * self.n_comps + 1), dtype=self.dtype)
        sigmaU = self.X.shape[0]**-0.5
        for pca_ind in range(self.n_comps):
            for sign_ind, sign in enumerate((-1, 1)):
                w, side = site_weight_func(U, pca_ind, sign, sigmaU)
                self.site_weights[side, 2 * pca_ind + sign_ind] = w

        # Add a simple Gaussian site at the origin. The 1e-7 prevents division
        # by zero where the Gaussians have withered away to nothing.
        self.site_weights[:, -1] = np.exp(-0.5 * ((U / sigmaU)**2).sum(axis=1)) + 1e-7

        self.site_weights /= self.site_weights.sum(axis=1)[:, np.newaxis]

    def _select_vol(self, vol_idx):
        mask = np.zeros(self.X.shape[1:], dtype=bool)
        mask[vol_idx] = 1
        self._cur_x = self.X[:, mask == 0]
        self._y = self.X[:, vol_idx]
        self._vol_idx = vol_idx

    def _fit(self):
        self.coef_ = np.zeros_like(self._cur_x)
        self.intercept_ = np.zeros_like(self._y)
        for site_ind in range(self.site_weights.shape[-1]):
            w = self.site_weights[:, site_ind]
            self.model.fit(self._cur_x[w > 0], self._y[w > 0], w[w > 0])
            self.coef_[w > 0] += np.outer(w[w > 0], self.model.coef_)
            self.intercept_[w > 0] += w[w > 0] * self.model.intercept_

    def _predict(self):
        r"""Return $Y = X^\prime \beta + a$, where
            $X^\prime$ is self.X without self._vol_idx's column,
            $\beta$ = self.coef_, and
            $a$ = self.intercept_
        """
        return np.einsum("...j,...j", self._cur_x, self.coef_) + self.intercept_

    def vol_denoise(self, vol_idx):
        self._select_vol(vol_idx)
        self._fit()
        return self._predict()

    def get_coefs(self, vol_idx, site_inds):
        """Return the model coefficients for vol_idx and site_inds.
        This is only needed for displaying the coefficients and how they vary by site.

        Parameters
        ----------
        vol_idx : int
            The index of the volume to be fitted.
        site_inds : int array-like
            The site indices to get coefficients for.
            The site indices for the negative and positive sides of PC index i are 2i and 2i + 1.
            The central site is -1 or 2 * n_comps + 1.

        Returns
        -------
        coefs : (nsites, nvols - 1) float array
            Each row is the coefficients by non-vol_idx volume index for the corresponding site in
            site_inds.
        """
        self._select_vol(vol_idx)
        coefs = np.zeros((len(site_inds), self._cur_x.shape[1]))
        for site_ind in site_inds:
            w = self.site_weights[:, site_ind]
            self.model.fit(self._cur_x[w > 0], self._y[w > 0], w[w > 0])
            coefs[site_ind] = self.model.coef_
        return coefs


def supply_model(model, alpha=1.0, max_iter=50, copy_X=False, mod_kwargs={}):
    """ Produce a sklearn.base.RegressorMixin object.

    Parameters
    ----------
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

    alpha : float, optional
        Regularization parameter only for ridge and lasso regression models.
        default: 1.0

    max_iter : int or None
        Maximum number of iterations for conjugate gradient solver.
        For 'sparse_cg' and 'lsqr' solvers, the default value is determined
        by scipy.sparse.linalg. For 'sag' solver, the default value is 1000.

    copy_X : bool
        If True, X will be copied; else, it may be overwritten.

    mod_kwargs : dict
        Other keyword arguments to pass to mod.


    Returns
    -------
    mod : sklearn.base.RegressorMixin object
    """
    # To add a new model, use the following API
    # We adhere to the following options as they are used for comparisons
    mod_kwargs['copy_X'] = copy_X
    mod_kwargs['alpha'] = alpha

    if model.lower() == 'ols':
        mod_kwargs.pop('alpha')
        model = linear_model.LinearRegression(**mod_kwargs)

    elif model.lower() == 'ridge':
        mod_kwargs.pop('ridge', None)
        model = linear_model.Ridge(**mod_kwargs)

    elif model.lower() == 'lasso':
        mod_kwargs['max_iter'] = max_iter
        model = linear_model.Lasso(**mod_kwargs)

    elif (isinstance(model, opt.SKLearnLinearSolver) or
          has_sklearn and isinstance(model, sklearn.base.RegressorMixin)):
        model = model

    # Both sklearn.linear_model._ridge.Ridge, etc., and opt.SKLearnLinearSolver are abc.ABCMeta.
    # Using type(opt.SKLearnLinearSolver) avoids having to import abc.
    elif isinstance(model, type(opt.SKLearnLinearSolver)):
        model = model(**mod_kwargs)

    else:
        e_s = "The `solver` key-word argument needs to be: "
        e_s += "'ols', 'ridge', 'lasso' or a "
        e_s += "`dipy.optimize.SKLearnLinearSolver` object"
        raise ValueError(e_s)
    return model


def extract_data(data, mask=None):
    """Given an n dimensional array, return a 2D copy with the first n-1
    dimensions flattened out and the last one left as is.

    Parameters
    ----------
    data : n dimensional array
        Typically 4D data

    mask : n-1 dimensional array or None, optional
        If not None, only use the voxels of each volume where mask is True.
        Its shape must match data.shape[:-1].

    Returns
    -------
    out : 2D array
        data (or data[mask == True]) rearranged

    See also
    --------
    dipy.segment.mask.applymask
    """
    if mask is None:
        rv = data.copy().reshape(np.prod(data.shape[:-1]), data.shape[-1])
    else:
        nvols = data.shape[-1]
        if mask.shape != data.shape[:-1]:
            raise ValueError("mask's shape must equal data.shape[:-1]")
        rv = np.empty((len(mask[mask > 0]), nvols), dtype=data.dtype)
        for v in range(nvols):
            rv[:, v] = data[mask > 0, v]
    return rv


def replace_data(inp, out, mask=None):
    """Pour inp into out, accounting for the shape of out and mask.

    Parameters
    ----------
    inp : 1D array
        The input.

    out : nD array
        The destination.

    mask : None or nD array, optional
        If not None, only pour inp into voxels where mask is True.
        (mask.shape must == out.shape and (mask > 0).sum() must == len(inp))

    Returns
    -------
    out : nD array
    """
    if mask is None:
        if inp.size != out.size:
            raise ValueError("shape %s cannot be poured into shape %s without a mask"
                             % (inp.shape, out.shape))
        out = inp.reshape(out.shape)
    else:
        if mask.shape != out.shape:
            raise ValueError("the mask and output shapes must match (%s vs %s)"
                             % (mask.shape, out.shape))
        nvox = (mask > 0).sum()
        if nvox != len(inp):
            e_s = "the number of values in input must match the number of nonzero voxels in mask\n"
            e_s += "(%d vs %d)" % (len(inp), nvox)
            raise ValueError(e_s)

        out[mask > 0] = inp
    return out


def denoise_volumes(data, mask, model, verbose, label, calc_dtype):
    """Return a version of data denoised with a AdaptivePatch2Self.

    Parameters
    ----------
    data : 4D array
        Noisy data. Denoising will use patch2self only along the last axis.

    mask : None or 3D array
        If not None, only denoise voxels where this is True.

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

    verbose : bool
        Iff True, print progress updates

    label : str
        A label for the kind of volumes being denoised, e.g. "b0" or "DWI".
        Only used if verbose is True.

    calc_dtype : dtype
        Data type to use for the calculations and output.

    Returns
    -------
    denoised : 4D array
    """
    extracted_data = extract_data(data, mask)
    ap2s = AdaptivePatch2Self(extracted_data, model=model, dtype=calc_dtype)

    denoised = data.astype(calc_dtype, copy=True)
    for vol_idx in range(0, data.shape[3]):
        denoised[..., vol_idx] = replace_data(ap2s.vol_denoise(vol_idx),
                                              denoised[..., vol_idx], mask)

        if verbose:
            print("Denoised %s volume: %d of %d" % (label, vol_idx + 1, data.shape[3]))
    return denoised


def adaptive_patch2self(data, bvals, model='ols', mask=None,
                        b0_threshold=50, out_dtype=None, alpha=1.0, verbose=False,
                        b0_denoising=True, clip_negative_vals=True, shift_intensity=False):
    """ Adaptive_Patch2self Denoiser

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
            Default: 'ridge'.

    mask : 3D array or None
           If not None, only denoise where mask is true.

    b0_threshold : int, optional
        Threshold for considering volumes as b0.

    out_dtype : str or dtype, optional
        The dtype for the output array. Default: output has the same dtype as
        the input.

    alpha : float, optional
        Regularization parameter, only used with ridge or lasso regression models.
        Default: 1.0

    verbose : bool, optional
        Show progress of Adaptive_Patch2self and time taken.

    b0_denoising : bool, optional
        Skips denoising b0 volumes if set to False.
        Default: True

    clip_negative_vals : bool, optional
        Sets negative values after denoising to 0 using `np.clip`.
        Default: True

    shift_intensity : bool, optional
        Shifts the distribution of intensities per volume to give
        non-negative values
        Default: False


    Returns
    --------
    denoised array : ndarray
        This is the denoised array of the same size as that of the input data,
        clipped to non-negative values.

    References
    ----------
    [Fadnavis20] S. Fadnavis, J. Batson, E. Garyfallidis, Patch2Self:
                    Denoising Diffusion MRI with Self-supervised Learning,
                    Advances in Neural Information Processing Systems 33 (2020)
    """
    if not data.ndim == 4:
        raise ValueError("Adaptive_Patch2self can only denoise on 4D arrays.",
                         data.shape)

    if data.shape[3] < 10:
        warn("The input data has less than 10 3D volumes. Adaptive_Patch2self may not",
             "give good denoising performance.")

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

    if verbose is True:
        t1 = time.time()

    # if only 1 b0 volume, skip denoising it
    if data_b0s.ndim == 3 or data_b0s.shape[-1] == 0 or not b0_denoising:
        if verbose:
            print("b0 denoising skipped...")
        denoised_b0s = data_b0s

    else:
        denoised_b0s = denoise_volumes(data_b0s, mask, model, verbose, "b0", calc_dtype)

    # Separate denoising for DWI volumes
    denoised_dwi = denoise_volumes(data_dwi, mask, model, verbose, "DWI", calc_dtype)

    if verbose is True:
        t2 = time.time()
        print('Total time taken for Adaptive_Patch2self: ', t2-t1, " seconds")

    denoised_arr = np.empty((data.shape), dtype=calc_dtype)
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
        warn('Both `clip_negative_vals` and `shift_intensity` cannot be True.')
        warn('Defaulting to `clip_negative_bvals`...')
        denoised_arr.clip(min=0, out=denoised_arr)

    return np.array(denoised_arr, dtype=out_dtype)
