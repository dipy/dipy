import numpy as np

from dipy.segment.mrf import ConstantObservationModel, IteratedConditionalModes
from dipy.sims.voxel import add_noise
from dipy.testing.decorators import warning_for_keywords
from dipy.utils.optpkg import optional_package

sklearn, has_sklearn, _ = optional_package("sklearn")
linear_model, _, _ = optional_package("sklearn.linear_model")


class TissueClassifierHMRF:
    """
    This class contains the methods for tissue classification using the
    Markov Random Fields modeling approach.
    """

    @warning_for_keywords()
    def __init__(self, *, save_history=False, verbose=True):
        self.save_history = save_history
        self.segmentations = []
        self.pves = []
        self.energies = []
        self.energies_sum = []
        self.verbose = verbose

    @warning_for_keywords()
    def classify(self, image, nclasses, beta, *, tolerance=1e-05, max_iter=100):
        """
        This method uses the Maximum a posteriori - Markov Random Field
        approach for segmentation by using the Iterative Conditional Modes
        and Expectation Maximization to estimate the parameters.

        Parameters
        ----------
        image : ndarray,
            3D structural image.
        nclasses : int,
            Number of desired classes.
        beta : float,
            Smoothing parameter, the higher this number the smoother the
            output will be.
        tolerance: float, optional
            Value that defines the percentage of change tolerated to
            prevent the ICM loop to stop. Default is 1e-05.
            If you want tolerance check to be disabled put 'tolerance = 0'.
        max_iter : int, optional
            Fixed number of desired iterations. Default is 100.
            This parameter defines the maximum number of iterations the
            algorithm will perform. The loop may terminate early if the
            change in energy sum between iterations falls below the
            threshold defined by `tolerance`. However, if `tolerance` is
            explicitly set to 0, this early stopping mechanism is disabled,
            and the algorithm will run for the specified number of
            iterations unless another stopping criterion is met.

        Returns
        -------
        initial_segmentation : ndarray,
            3D segmented image with all tissue types specified in nclasses.
        final_segmentation : ndarray,
            3D final refined segmentation containing all tissue types.
        PVE : ndarray,
            3D probability map of each tissue type.
        """
        nclasses += 1  # One extra class for the background
        energy_sum = [1e-05]

        com = ConstantObservationModel()
        icm = IteratedConditionalModes()

        if image.max() > 1:
            image = np.interp(image, [0, image.max()], [0.0, 1.0])

        mu, sigmasq = com.initialize_param_uniform(image, nclasses)
        p = np.argsort(mu)
        mu = mu[p]
        sigmasq = sigmasq[p]

        neglogl = com.negloglikelihood(image, mu, sigmasq, nclasses)
        seg_init = icm.initialize_maximum_likelihood(neglogl)

        mu, sigmasq = com.seg_stats(image, seg_init, nclasses)

        zero = np.zeros_like(image) + 0.001
        zero_noise = add_noise(zero, 10000, 1, noise_type="gaussian")
        image_gauss = np.where(image == 0, zero_noise, image)

        final_segmentation = np.empty_like(image)
        initial_segmentation = seg_init

        for i in range(max_iter):
            if self.verbose:
                print(f">> Iteration: {i}")

            PLN = icm.prob_neighborhood(seg_init, beta, nclasses)
            PVE = com.prob_image(image_gauss, nclasses, mu, sigmasq, PLN)

            mu_upd, sigmasq_upd = com.update_param(image_gauss, PVE, mu, nclasses)
            ind = np.argsort(mu_upd)
            mu_upd = mu_upd[ind]
            sigmasq_upd = sigmasq_upd[ind]

            negll = com.negloglikelihood(image_gauss, mu_upd, sigmasq_upd, nclasses)
            final_segmentation, energy = icm.icm_ising(negll, beta, seg_init)

            energy_sum.append(energy[energy > -np.inf].sum())

            if self.save_history:
                self.segmentations.append(final_segmentation)
                self.pves.append(PVE)
                self.energies.append(energy)
                self.energies_sum.append(energy_sum[-1])

            if tolerance > 0 and i > 5:
                e_sum = np.asarray(energy_sum)
                tol = tolerance * (np.amax(e_sum) - np.amin(e_sum))

                e_end = e_sum[-5:]
                test_dist = np.abs(np.amax(e_end) - np.amin(e_end))

                if test_dist < tol:
                    break

            seg_init = final_segmentation
            mu = mu_upd
            sigmasq = sigmasq_upd

        PVE = PVE[..., 1:]

        return initial_segmentation, final_segmentation, PVE


def compute_directional_average(
    data,
    bvals,
    *,
    s0_map=None,
    masks=None,
    b0_mask=None,
    b0_threshold=50,
    low_signal_threshold=50,
):
    """
    Compute the mean signal for each unique b-value shell and fit a linear model.

    Parameters
    ----------
    data : ndarray
        The diffusion MRI data.
    bvals : ndarray
        The b-values corresponding to the diffusion data.
    s0_map : ndarray, optional
        Precomputed mean signal map for b=0 images.
    masks : ndarray, optional
        Precomputed masks for each unique b-value shell.
    b0_mask : ndarray, optional
        Precomputed mask for b=0 images.
    b0_threshold : float, optional
        The intensity threshold for a b=0 image.
    low_signal_threshold : float, optional
        The threshold below which a voxel is considered to have low signal.

    Returns
    -------
    P : float
        The slope of the linear model.
    V : float
        The intercept of the linear model.
    """
    if b0_mask is None:
        b0_mask = bvals < b0_threshold
    if masks is None:
        unique_bvals = np.unique(bvals)
        masks = bvals[:, np.newaxis] == unique_bvals[np.newaxis, 1:]
    if s0_map is None:
        s0_map = data[..., b0_mask].mean(axis=-1)

    if s0_map < low_signal_threshold:
        return 0, 0

    # Calculate the mean for each mask
    means = np.sum(data[:, np.newaxis] * masks, axis=0) / np.sum(masks, axis=0)

    # Normalize by s0, avoiding division by zero by adding 0.01 for stable division
    s_bvals = means / (s0_map[..., np.newaxis] + 0.01)

    # Avoid log(0) by adding 0.001 for stable linear regression fit
    s_bvals[s_bvals == 0] = 0.001
    s_log = y = np.log(s_bvals)

    xb = -np.log(np.arange(1, s_log.shape[-1] + 1))

    # Reshape xb for linear regression
    X = xb.reshape(-1, 1)

    # Fit linear model
    model = linear_model.LinearRegression()
    model.fit(X, y)
    P = model.coef_[0]
    V = model.intercept_

    return P, V


def dam_classifier(
    data, bvals, wm_threshold, *, b0_threshold=50, low_signal_threshold=50
):
    """Computes the P-map (fitting slope) on data to extract white and grey matter.

    See :footcite:p:`Cheng2020` for further details about the method.

    Parameters
    ----------
    data : ndarray
        The diffusion MRI data.
    bvals : ndarray
        The b-values corresponding to the diffusion data.
    wm_threshold : float
        The threshold below which a voxel is considered white matter.
    b0_threshold : float, optional
        The intensity threshold for a b=0 image.
    low_signal_threshold : float, optional
        The threshold below which a voxel is considered to have low signal.

    Returns
    -------
    wm_mask : ndarray
        A binary mask for white matter.
    gm_mask : ndarray
        A binary mask for grey matter.

    References
    ----------
    .. footbibliography::

    """
    # Precompute unique b-values, masks, and b=0 mask
    unique_bvals = np.unique(bvals)
    if len(unique_bvals) <= 2:
        raise ValueError("Insufficient unique b-values for fitting.")

    b0_mask = bvals < b0_threshold
    masks = bvals[:, np.newaxis] == unique_bvals[np.newaxis, 1:]

    # Precompute s0 (mean signal for b=0)
    s0_map = data[..., b0_mask].mean(axis=-1)

    # If the mean signal for b=0 is too low, set those voxels to 0 for both P and V
    valid_voxels = s0_map >= low_signal_threshold

    P_map = np.zeros(data.shape[:-1])
    for idx in range(data.shape[0] * data.shape[1] * data.shape[2]):
        i, j, k = np.unravel_index(idx, P_map.shape)
        if valid_voxels[i, j, k]:
            P, _ = compute_directional_average(
                data[i, j, k, :],
                bvals,
                masks=masks,
                b0_mask=b0_mask,
                s0_map=s0_map[i, j, k],
                low_signal_threshold=low_signal_threshold,
            )
            P_map[i, j, k] = P

    # Adding a small slope threshold for P_map to avoid 0 sloped background voxels
    wm_mask = (P_map <= wm_threshold) & (P_map > 0.01)
    # Grey matter has a higher P value than white matter
    gm_mask = P_map > wm_threshold

    return wm_mask, gm_mask
