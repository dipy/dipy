from __future__ import division

import numpy as np
from numpy import dot
from dipy.core.geometry import sphere2cart
from dipy.core.geometry import vec2vec_rotmat


# Diffusion coefficients for white matter tracts, in mm^2/s
#
# Based roughly on values from:
#
#   Pierpaoli, Basser, "Towards a Quantitative Assessment of Diffusion
#   Anisotropy", Magnetic Resonance in Medicine, 1996; 36(6):893-906.
#
diffusion_evals = np.array([1500e-6, 400e-6, 400e-6])


def _add_gaussian(sig, noise1, noise2):
    """
    Helper function to add_noise

    This one simply adds one of the Gaussians to the sig and ignores the other
    one.
    """
    return sig + noise1


def _add_rician(sig, noise1, noise2):
    """
    Helper function to add_noise.

    This does the same as abs(sig + complex(noise1, noise2))

    """
    return np.sqrt((sig + noise1) ** 2 + noise2 ** 2)


def _add_rayleigh(sig, noise1, noise2):
    """
    Helper function to add_noise

    The Rayleigh distribution is $\sqrt\{Gauss_1^2 + Gauss_2^2}$.

    """
    return sig + np.sqrt(noise1 ** 2 + noise2 ** 2)


def add_noise(signal, snr, S0, noise_type='rician'):
    r""" Add noise of specified distribution to the signal from a single voxel.

    Parameters
    -----------
    signal : 1-d ndarray
        The signal in the voxel.
    snr : float
        The desired signal-to-noise ratio. (See notes below.)
        If `snr` is None, return the signal as-is.
    S0 : float
        Reference signal for specifying `snr`.
    noise_type : string, optional
        The distribution of noise added. Can be either 'gaussian' for Gaussian
        distributed noise, 'rician' for Rice-distributed noise (default) or
        'rayleigh' for a Rayleigh distribution.

    Returns
    --------
    signal : array, same shape as the input
        Signal with added noise.

    Notes
    -----
    SNR is defined here, following [1]_, as ``S0 / sigma``, where ``sigma`` is
    the standard deviation of the two Gaussian distributions forming the real
    and imaginary components of the Rician noise distribution (see [2]_).

    References
    ----------
    .. [1] Descoteaux, Angelino, Fitzgibbons and Deriche (2007) Regularized,
           fast and robust q-ball imaging. MRM, 58: 497-510
    .. [2] Gudbjartson and Patz (2008). The Rician distribution of noisy MRI
           data. MRM 34: 910-914.

    Examples
    --------
    >>> signal = np.arange(800).reshape(2, 2, 2, 100)
    >>> signal_w_noise = add_noise(signal, 10., 100., noise_type='rician')

    """
    if snr is None:
        return signal

    sigma = S0 / snr

    noise_adder = {'gaussian': _add_gaussian,
                   'rician': _add_rician,
                   'rayleigh': _add_rayleigh}

    noise1 = np.random.normal(0, sigma, size=signal.shape)

    if noise_type == 'gaussian':
        noise2 = None
    else:
        noise2 = np.random.normal(0, sigma, size=signal.shape)

    return noise_adder[noise_type](signal, noise1, noise2)


def sticks_and_ball(gtab, d=0.0015, S0=100, angles=[(0, 0), (90, 0)],
                    fractions=[35, 35], snr=20):
    """ Simulate the signal for a Sticks & Ball model.

    Parameters
    -----------
    gtab : GradientTable
        Signal measurement directions.
    d : float
        Diffusivity value.
    S0 : float
        Unweighted signal value.
    angles : array (K,2) or (K, 3)
        List of K polar angles (in degrees) for the sticks or array of K
        sticks as unit vectors.
    fractions : float
        Percentage of each stick.  Remainder to 100 specifies isotropic
        component.
    snr : float
        Signal to noise ratio, assuming Rician noise.  If set to None, no
        noise is added.

    Returns
    --------
    S : (N,) ndarray
        Simulated signal.
    sticks : (M,3)
        Sticks in cartesian coordinates.

    References
    ----------
    .. [1] Behrens et al., "Probabilistic diffusion
           tractography with multiple fiber orientations:  what can we gain?",
           Neuroimage, 2007.

    """
    fractions = [f / 100. for f in fractions]
    f0 = 1 - np.sum(fractions)
    S = np.zeros(len(gtab.bvals))

    angles = np.array(angles)
    if angles.shape[-1] == 3:
        sticks = angles
    else:
        sticks = [sphere2cart(1, np.deg2rad(pair[0]), np.deg2rad(pair[1]))
                  for pair in angles]
        sticks = np.array(sticks)

    for (i, g) in enumerate(gtab.bvecs[1:]):
        S[i + 1] = f0 * np.exp(-gtab.bvals[i + 1] * d) + \
            np.sum([fractions[j] * np.exp(-gtab.bvals[i + 1] * d * np.dot(s, g) ** 2)
                   for (j, s) in enumerate(sticks)])

        S[i + 1] = S0 * S[i + 1]

    S[gtab.b0s_mask] = S0
    S = add_noise(S, snr, S0)

    return S, sticks


def single_tensor(gtab, S0=1, evals=None, evecs=None, snr=None):
    """ Simulated Q-space signal with a single tensor.

    Parameters
    -----------
    gtab : GradientTable
        Measurement directions.
    S0 : double,
        Strength of signal in the presence of no diffusion gradient (also
        called the ``b=0`` value).
    evals : (3,) ndarray
        Eigenvalues of the diffusion tensor.  By default, values typical for
        prolate white matter are used.
    evecs : (3, 3) ndarray
        Eigenvectors of the tensor.  You can also think of this as a rotation
        matrix that transforms the direction of the tensor. The eigenvectors
        needs to be column wise.
    snr : float
        Signal to noise ratio, assuming Rician noise.  None implies no noise.

    Returns
    --------
    S : (N,) ndarray
        Simulated signal: ``S(q, tau) = S_0 e^(-b g^T R D R.T g)``.

    References
    ----------
    .. [1] M. Descoteaux, "High Angular Resolution Diffusion MRI: from Local
           Estimation to Segmentation and Tractography", PhD thesis,
           University of Nice-Sophia Antipolis, p. 42, 2008.
    .. [2] E. Stejskal and J. Tanner, "Spin diffusion measurements: spin echos
           in the presence of a time-dependent field gradient", Journal of
           Chemical Physics, nr. 42, pp. 288--292, 1965.

    """
    if evals is None:
        evals = diffusion_evals

    if evecs is None:
        evecs = np.eye(3)

    out_shape = gtab.bvecs.shape[:gtab.bvecs.ndim - 1]
    gradients = gtab.bvecs.reshape(-1, 3)

    R = np.asarray(evecs)
    S = np.zeros(len(gradients))
    D = dot(dot(R, np.diag(evals)), R.T)

    for (i, g) in enumerate(gradients):
        S[i] = S0 * np.exp(-gtab.bvals[i] * dot(dot(g.T, D), g))

    S = add_noise(S, snr, S0)

    return S.reshape(out_shape)


def multi_tensor(gtab, mevals, S0=100, angles=[(0, 0), (90, 0)],
                 fractions=[50, 50], snr=20):
    r"""Simulate a Multi-Tensor signal.

    Parameters
    -----------
    gtab : GradientTable
    mevals : array (K, 3)
        each tensor's eigenvalues in each row
    S0 : float
        Unweighted signal value (b0 signal).
    angles : array (K,2) or (K,3)
        List of K tensor directions in polar angles (in degrees) or unit vectors
    fractions : float
        Percentage of the contribution of each tensor. The sum of fractions
        should be equal to 100%.
    snr : float
        Signal to noise ratio, assuming Rician noise.  If set to None, no
        noise is added.

    Returns
    --------
    S : (N,) ndarray
        Simulated signal.
    sticks : (M,3)
        Sticks in cartesian coordinates.

    Examples
    --------
    >>> import numpy as np
    >>> from dipy.sims.voxel import multi_tensor
    >>> from dipy.data import get_data
    >>> from dipy.core.gradients import gradient_table
    >>> from dipy.io.gradients import read_bvals_bvecs
    >>> fimg, fbvals, fbvecs = get_data('small_101D')
    >>> bvals, bvecs = read_bvals_bvecs(fbvals, fbvecs)
    >>> gtab = gradient_table(bvals, bvecs)
    >>> mevals=np.array(([0.0015, 0.0003, 0.0003],[0.0015, 0.0003, 0.0003]))
    >>> e0 = np.array([1, 0, 0.])
    >>> e1 = np.array([0., 1, 0])
    >>> S = multi_tensor(gtab, mevals)

    """
    if np.round(np.sum(fractions), 2) != 100.0:
        raise ValueError('Fractions should sum to 100')

    fractions = [f / 100. for f in fractions]

    S = np.zeros(len(gtab.bvals))

    angles = np.array(angles)
    if angles.shape[-1] == 3:
        sticks = angles
    else:
        sticks = [sphere2cart(1, np.deg2rad(pair[0]), np.deg2rad(pair[1]))
                  for pair in angles]
        sticks = np.array(sticks)

    for i in range(len(fractions)):
            S = S + fractions[i] * single_tensor(gtab, S0=S0, evals=mevals[i],
                                                 evecs=all_tensor_evecs(
                                                     sticks[i]).T,
                                                 snr=None)

    return add_noise(S, snr, S0), sticks


def single_tensor_odf(r, evals=None, evecs=None):
    """Simulated ODF with a single tensor.

    Parameters
    ----------
    r : (N,3) or (M,N,3) ndarray
        Measurement positions in (x, y, z), either as a list or on a grid.
    evals : (3,)
        Eigenvalues of diffusion tensor.  By default, use values typical for
        prolate white matter.
    evecs : (3, 3) ndarray
        Eigenvectors of the tensor.  You can also think of these as the
        rotation matrix that determines the orientation of the diffusion
        tensor.

    Returns
    -------
    ODF : (N,) ndarray
        The diffusion probability at ``r`` after time ``tau``.

    References
    ----------
    .. [1] Aganj et al., "Reconstruction of the Orientation Distribution
           Function in Single- and Multiple-Shell q-Ball Imaging Within
           Constant Solid Angle", Magnetic Resonance in Medicine, nr. 64,
           pp. 554--566, 2010.

    """
    if evals is None:
        evals = diffusion_evals

    if evecs is None:
        evecs = np.eye(3)

    out_shape = r.shape[:r.ndim - 1]

    R = np.asarray(evecs)
    D = dot(dot(R, np.diag(evals)), R.T)
    Di = np.linalg.inv(D)
    r = r.reshape(-1, 3)
    P = np.zeros(len(r))
    for (i, u) in enumerate(r):
        P[i] = (dot(dot(u.T, Di), u)) ** (3 / 2)

    return (1 / (4 * np.pi * np.prod(evals) ** (1 / 2) * P)).reshape(out_shape)


def all_tensor_evecs(e0):
    """Given the principle tensor axis, return the array of all
    eigenvectors (or, the rotation matrix that orientates the tensor).

    Parameters
    ----------
    e0 : (3,) ndarray
        Principle tensor axis.

    Returns
    -------
    evecs : (3,3) ndarray
        Tensor eigenvectors.

    """
    axes = np.eye(3)
    mat = vec2vec_rotmat(e0, axes[0])
    e1 = np.dot(mat, axes[1])
    e2 = np.dot(mat, axes[2])
    return np.array([e0, e1, e2])


def multi_tensor_odf(odf_verts, mevals, angles, fractions):
    r'''Simulate a Multi-Tensor ODF.

    Parameters
    ----------
    odf_verts : (N,3) ndarray
        Vertices of the reconstruction sphere.
    mevals : sequence of 1D arrays,
        Eigen-values for each tensor.
    angles : sequence of 2d tuples,
        Sequence of principal directions for each tensor in polar angles
        or cartesian unit coordinates.
    fractions : sequence of floats,
        Percentages of the fractions for each tensor.

    Returns
    -------
    ODF : (N,) ndarray
        Orientation distribution function.

    Examples
    --------
    Simulate a MultiTensor ODF with two peaks and calculate its exact ODF.

    >>> import numpy as np
    >>> from dipy.sims.voxel import multi_tensor_odf, all_tensor_evecs
    >>> from dipy.data import get_sphere
    >>> sphere = get_sphere('symmetric724')
    >>> vertices, faces = sphere.vertices, sphere.faces
    >>> mevals = np.array(([0.0015, 0.0003, 0.0003],[0.0015, 0.0003, 0.0003]))
    >>> angles = [(0, 0), (90, 0)]
    >>> odf = multi_tensor_odf(vertices, mevals, angles, [50, 50])

    '''

    mf = [f / 100. for f in fractions]

    angles = np.array(angles)
    if angles.shape[-1] == 3:
        sticks = angles
    else:
        sticks = [sphere2cart(1, np.deg2rad(pair[0]), np.deg2rad(pair[1]))
                  for pair in angles]
        sticks = np.array(sticks)

    odf = np.zeros(len(odf_verts))

    mevecs = []
    for s in sticks:
        mevecs += [all_tensor_evecs(s).T]

    for (j, f) in enumerate(mf):
        odf += f * single_tensor_odf(odf_verts,
                                     evals=mevals[j], evecs=mevecs[j])
    return odf


def single_tensor_rtop(evals=None, tau=1.0 / (4 * np.pi ** 2)):
    r'''Simulate a Multi-Tensor rtop.

    Parameters
    ----------
    evals : 1D arrays,
        Eigen-values for the tensor.  By default, values typical for prolate
        white matter are used.
    tau : float,
        diffusion time. By default the value that makes q=sqrt(b).

    Returns
    -------
    rtop : float,
        Return to origin probability.

    References
    ----------
    .. [1] Cheng J., "Estimation and Processing of Ensemble Average Propagator and
           Its Features in Diffusion MRI", PhD Thesis, 2012.

    '''
    if evals is None:
        evals = diffusion_evals

    rtop = 1.0 / np.sqrt((4 * np.pi * tau) ** 3 * np.prod(evals))
    return rtop


def multi_tensor_rtop(mf, mevals=None, tau=1 / (4 * np.pi ** 2)):
    r'''Simulate a Multi-Tensor rtop.

    Parameters
    ----------
    mf : sequence of floats, bounded [0,1]
        Percentages of the fractions for each tensor.
    mevals : sequence of 1D arrays,
        Eigen-values for each tensor.  By default, values typical for prolate
        white matter are used.
    tau : float,
        diffusion time. By default the value that makes q=sqrt(b).

    Returns
    -------
    rtop : float,
        Return to origin probability.

    References
    ----------
    .. [1] Cheng J., "Estimation and Processing of Ensemble Average Propagator and
           Its Features in Diffusion MRI", PhD Thesis, 2012.

    '''
    rtop = 0

    if mevals is None:
        mevals = [None, ] * len(mf)

    for j, f in enumerate(mf):
        rtop += f * single_tensor_rtop(mevals[j], tau=tau)
    return rtop


def single_tensor_pdf(r, evals=None, evecs=None, tau=1 / (4 * np.pi ** 2)):
    """Simulated ODF with a single tensor.

    Parameters
    ----------
    r : (N,3) or (M,N,3) ndarray
        Measurement positions in (x, y, z), either as a list or on a grid.
    evals : (3,)
        Eigenvalues of diffusion tensor.  By default, use values typical for
        prolate white matter.
    evecs : (3, 3) ndarray
        Eigenvectors of the tensor.  You can also think of these as the
        rotation matrix that determines the orientation of the diffusion
        tensor.
    tau : float,
        diffusion time. By default the value that makes q=sqrt(b).


    Returns
    -------
    pdf : (N,) ndarray
        The diffusion probability at ``r`` after time ``tau``.

    References
    ----------
    .. [1] Cheng J., "Estimation and Processing of Ensemble Average Propagator and
           Its Features in Diffusion MRI", PhD Thesis, 2012.

    """
    if evals is None:
        evals = diffusion_evals

    if evecs is None:
        evecs = np.eye(3)

    out_shape = r.shape[:r.ndim - 1]

    R = np.asarray(evecs)
    D = dot(dot(R, np.diag(evals)), R.T)
    Di = np.linalg.inv(D)
    r = r.reshape(-1, 3)
    P = np.zeros(len(r))
    for (i, u) in enumerate(r):
        P[i] = (-dot(dot(u.T, Di), u)) / (4 * tau)

    pdf = (1 / np.sqrt((4 * np.pi * tau) ** 3 * np.prod(evals))) * np.exp(P)

    return pdf.reshape(out_shape)


def multi_tensor_pdf(pdf_points, mevals, angles, fractions,
                     tau=1 / (4 * np.pi ** 2)):
    r'''Simulate a Multi-Tensor ODF.

    Parameters
    ----------
    pdf_points : (N, 3) ndarray
        Points to evaluate the PDF.
    mevals : sequence of 1D arrays,
        Eigen-values for each tensor.  By default, values typical for prolate
        white matter are used.
    angles : sequence,
        Sequence of principal directions for each tensor in polar angles
        or cartesian unit coordinates.
    fractions : sequence of floats,
        Percentages of the fractions for each tensor.
    tau : float,
        diffusion time. By default the value that makes q=sqrt(b).

    Returns
    -------
    pdf : (N,) ndarray,
        Probability density function of the water displacement.

    References
    ----------
    .. [1] Cheng J., "Estimation and Processing of Ensemble Average Propagator
           and its Features in Diffusion MRI", PhD Thesis, 2012.

    '''
    mf = [f / 100. for f in fractions]

    angles = np.array(angles)
    if angles.shape[-1] == 3:
        sticks = angles
    else:
        sticks = [sphere2cart(1, np.deg2rad(pair[0]), np.deg2rad(pair[1]))
                  for pair in angles]
        sticks = np.array(sticks)

    pdf = np.zeros(len(pdf_points))

    mevecs = []
    for s in sticks:
        mevecs += [all_tensor_evecs(s).T]

    for j, f in enumerate(mf):
        pdf += f * single_tensor_pdf(pdf_points,
                                     evals=mevals[j], evecs=mevecs[j], tau=tau)
    return pdf


def single_tensor_msd(evals=None, tau=1 / (4 * np.pi ** 2)):
    r'''Simulate a Multi-Tensor rtop.

    Parameters
    ----------
    evals : 1D arrays,
        Eigen-values for the tensor.  By default, values typical for prolate
        white matter are used.
    tau : float,
        diffusion time. By default the value that makes q=sqrt(b).

    Returns
    -------
    msd : float,
        Mean square displacement.

    References
    ----------
    .. [1] Cheng J., "Estimation and Processing of Ensemble Average Propagator and
           Its Features in Diffusion MRI", PhD Thesis, 2012.

    '''
    if evals is None:
        evals = diffusion_evals

    msd = 2 * tau * np.sum(evals)
    return msd


def multi_tensor_msd(mf, mevals=None, tau=1 / (4 * np.pi ** 2)):
    r'''Simulate a Multi-Tensor rtop.

    Parameters
    ----------
    mf : sequence of floats, bounded [0,1]
        Percentages of the fractions for each tensor.
    mevals : sequence of 1D arrays,
        Eigen-values for each tensor.  By default, values typical for prolate
        white matter are used.
    tau : float,
        diffusion time. By default the value that makes q=sqrt(b).

    Returns
    -------
    msd : float,
        Mean square displacement.

    References
    ----------
    .. [1] Cheng J., "Estimation and Processing of Ensemble Average Propagator and
           Its Features in Diffusion MRI", PhD Thesis, 2012.

    '''
    msd = 0

    if mevals is None:
        mevals = [None, ] * len(mf)

    for j, f in enumerate(mf):
        msd += f * single_tensor_msd(mevals[j], tau=tau)
    return msd

# Use standard naming convention, but keep old names
# for backward compatibility
SticksAndBall = sticks_and_ball
SingleTensor = single_tensor
MultiTensor = multi_tensor
