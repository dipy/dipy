from __future__ import division

import numpy as np
from dipy.core.geometry import sphere2cart
from dipy.reconst.dti import design_matrix, lower_triangular
from dipy.core.geometry import vec2vec_rotmat


# Diffusion coefficients for white matter tracts, in mm^2/s
#
# Based roughly on values from:
#
#   Pierpaoli, Basser, "Towards a Quantitative Assessment of Diffusion
#   Anisotropy", Magnetic Resonance in Medicine, 1996; 36(6):893-906.
#
diffusion_evals = np.array([1500e-6, 400e-6, 400e-6])


def sticks_and_ball(bvals, gradients, d=0.0015, S0=100, angles=[(0,0), (90,0)],
                    fractions=[35,35], snr=20):
    """ Simulate the signal for a Sticks & Ball model.

    Parameters
    -----------
    bvals : (N,) ndarray
        B-values for measurements.
    gradients : (N,3) ndarray
        Also known as b-vectors.
    d : float
        Diffusivity value.
    S0 : float
        Unweighted signal value.
    angles : array (K,2) or (M,3)
        List of K polar angles (in degrees) for the sticks or array of M
        sticks as Cartesian unit vectors.
    fractions : float
        Percentage of each stick.
    snr : float
        Signal to noise ratio, assuming gaussian noise.  If set to None, no
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
    S = np.zeros(len(gradients))

    angles=np.array(angles)
    if angles.shape[-1] == 3:
        sticks = angles
    else:
        sticks = [sphere2cart(1, np.deg2rad(pair[0]), np.deg2rad(pair[1]))
                  for pair in angles]
        sticks = np.array(sticks)

    for (i, g) in enumerate(gradients[1:]):
        S[i + 1] = f0 * np.exp(-bvals[i+1] * d) + \
                   np.sum([
            fractions[j] * np.exp(-bvals[i + 1] * d * np.dot(s, g)**2)
            for (j,s) in enumerate(sticks)
                          ])

        S[i + 1] = S0 * S[i + 1]

    S[0] = S0
    if snr is not None:
        std = S0 / snr
        S = S + np.random.randn(len(S)) * std

    return S, sticks


def single_tensor(bvals, gradients, S0=1, evals=None, evecs=None, snr=None):
    """ Simulated Q-space signal with a single tensor.

    Parameters
    -----------
    bvals : (N,) array
        B-values for measurements.  The b-value is also ``b = \tau |q|^2``,
        where ``\tau`` is the time allowed for attenuation and ``q`` is the
        measurement position vector in Q-space (signal-space or Fourier-space).
        If b is too low, there is not enough attenuation to measure.  With b
        too high, the signal to noise ratio increases.
    gradients : (N, 3) or (M, N, 3) ndarray
        Measurement gradients / directions, also known as b-vectors, as 3D unit
        vectors (either in a list or on a grid).
    S0 : double,
        Strength of signal in the presence of no diffusion gradient (also
        called the ``b=0`` value).
    evals : (3,) ndarray
        Eigenvalues of the diffusion tensor.  By default, values typical for
        prolate white matter are used.
    evecs : (3, 3) ndarray
        Eigenvectors of the tensor.  You can also think of this as a rotation
        matrix that transforms the direction of the tensor.
    snr : float
        Signal to noise ratio, assuming gaussian noise.  None implies no noise.

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

    out_shape = gradients.shape[:gradients.ndim - 1]

    gradients = gradients.reshape(-1, 3)
    R = np.asarray(evecs)
    S = np.zeros(len(gradients))
    D = R.dot(np.diag(evals)).dot(R.T)

    for (i, g) in enumerate(gradients):
        S[i] = S0 * np.exp(-bvals[i] * g.T.dot(D).dot(g))

    """ Alternative suggestion which works with multiple b0s
    design = design_matrix(bval, gradients.T)
    S = np.exp(np.dot(design, lower_triangular(D)))
    """

    if snr is not None:
        std = S0 / snr
        S = S + np.random.randn(len(S)) * std

    return S.reshape(out_shape)


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
    D = R.dot(np.diag(evals)).dot(R.T)
    Di = np.linalg.inv(D)
    r = r.reshape(-1, 3)
    P = np.zeros(len(r))
    for (i, u) in enumerate(r):
        P[i] = (u.T.dot(Di).dot(u))**(3 / 2)

    return (1 / (4 * np.pi * np.prod(evals)**(1/2) * P)).reshape(out_shape)


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
    mat = vec2vec_rotmat(axes[2], e0)
    e1 = np.dot(mat, axes[0])
    e2 = np.dot(mat, axes[1])
    return np.array([e0, e1, e2])


def multi_tensor_odf(odf_verts, mf, mevals=None, mevecs=None):
    r'''Simulate a Multi-Tensor ODF.

    Parameters
    ----------
    odf_verts : (N,3) ndarray
        Vertices of the reconstruction sphere.
    mf : sequence of floats, bounded [0,1]
        Percentages of the fractions for each tensor.
    mevals : sequence of 1D arrays,
        Eigen-values for each tensor.  By default, values typical for prolate
        white matter are used.
    mevecs : sequence of 3D arrays,
        Eigenvectors for each tensor.  You can also think of these
        as the rotation matrices that align the different tensors.

    Returns
    -------
    ODF : (N,) ndarray
        Orientation distribution function.

    Examples
    --------
    Simulate a MultiTensor with two peaks and calculate its exact ODF.

    >>> import numpy as np
    >>> from dipy.sims.voxel import multi_tensor_odf, all_tensor_evecs
    >>> from dipy.data import get_sphere
    >>> sphere = get_sphere('symmetric724')
    >>> vertices, faces = sphere.vertices, sphere.faces
    >>> mevals=np.array(([0.0015, 0.0003, 0.0003],[0.0015, 0.0003, 0.0003]))
    >>> e0 = np.array([1, 0, 0.])
    >>> e1 = np.array([0., 1, 0])
    >>> mevecs=[all_tensor_evecs(e0), all_tensor_evecs(e1)]
    >>> odf = multi_tensor_odf(vertices, [0.5,0.5], mevals, mevecs)

    '''
    odf = np.zeros(len(odf_verts))

    if mevals is None:
        mevals = [None,] * len(mf)

    if mevecs is None:
        mevecs = [np.eye(3) for i in range(len(mf))]

    for j, f in enumerate(mf):
        odf += f * single_tensor_odf(odf_verts,
                                     evals=mevals[j], evecs=mevecs[j])
    return odf


# Use standard naming convention, but keep old names
# for backward compatibility
SticksAndBall = sticks_and_ball
SingleTensor = single_tensor
