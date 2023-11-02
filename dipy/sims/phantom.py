import numpy as np
# import scipy.stats as stats

from dipy.sims.voxel import single_tensor, diffusion_evals
import dipy.sims.voxel as vox
from dipy.core.geometry import vec2vec_rotmat
from dipy.data import get_fnames
from dipy.core.gradients import gradient_table


def add_noise(vol, snr=1.0, S0=None, noise_type='rician', rng=None):
    """ Add noise of specified distribution to a 4D array.

    Parameters
    ----------
    vol : array, shape (X,Y,Z,W)
        Diffusion measurements in `W` directions at each ``(X, Y, Z)`` voxel
        position.
    snr : float, optional
        The desired signal-to-noise ratio.  (See notes below.)
    S0 : float, optional
        Reference signal for specifying `snr` (defaults to 1).
    noise_type : string, optional
        The distribution of noise added. Can be either 'gaussian' for Gaussian
        distributed noise, 'rician' for Rice-distributed noise (default) or
        'rayleigh' for a Rayleigh distribution.
    rng : numpy.random.Generator class, optional
        Numpy's random generator for setting seed values when needed.
        Default is None.

    Returns
    -------
    vol : array, same shape as vol
        Volume with added noise.

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
    >>> signal_w_noise = add_noise(signal, snr=10, noise_type='rician')

    """
    orig_shape = vol.shape
    vol_flat = np.reshape(vol.copy(), (-1, vol.shape[-1]))

    if S0 is None:
        S0 = np.max(vol)

    for vox_idx, signal in enumerate(vol_flat):
        vol_flat[vox_idx] = vox.add_noise(signal, snr=snr, S0=S0,
                                          noise_type=noise_type, rng=rng)

    return np.reshape(vol_flat, orig_shape)


def diff2eigenvectors(dx, dy, dz):
    """ numerical derivatives 2 eigenvectors
    """
    basis = np.eye(3)
    u = np.array([dx, dy, dz])
    u = u/np.linalg.norm(u)
    R = vec2vec_rotmat(basis[:, 0], u)
    eig0 = u
    eig1 = np.dot(R, basis[:, 1])
    eig2 = np.dot(R, basis[:, 2])
    eigs = np.zeros((3, 3))
    eigs[:, 0] = eig0
    eigs[:, 1] = eig1
    eigs[:, 2] = eig2
    return eigs, R


def orbital_phantom(gtab=None,
                    evals=diffusion_evals,
                    func=None,
                    t=np.linspace(0, 2 * np.pi, 1000),
                    datashape=(64, 64, 64, 65),
                    origin=(32, 32, 32),
                    scale=(25, 25, 25),
                    angles=np.linspace(0, 2 * np.pi, 32),
                    radii=np.linspace(0.2, 2, 6),
                    S0=100.,
                    snr=None,
                    rng=None):
    """Create a phantom based on a 3-D orbit ``f(t) -> (x,y,z)``.

    Parameters
    ----------
    gtab : GradientTable
        Gradient table of measurement directions.
    evals : array, shape (3,)
        Tensor eigenvalues.
    func : user defined function f(t)->(x,y,z)
        It could be desirable for ``-1=<x,y,z <=1``.
        If None creates a circular orbit.
    t : array, shape (K,)
        Represents time for the orbit. Default is
        ``np.linspace(0, 2 * np.pi, 1000)``.
    datashape : array, shape (X,Y,Z,W)
        Size of the output simulated data
    origin : tuple, shape (3,)
        Define the center for the volume
    scale : tuple, shape (3,)
        Scale the function before applying to the grid
    angles : array, shape (L,)
        Density angle points, always perpendicular to the first eigen vector
        Default np.linspace(0, 2 * np.pi, 32).
    radii : array, shape (M,)
        Thickness radii.  Default ``np.linspace(0.2, 2, 6)``.
        angles and radii define the total thickness options
    S0 : double, optional
        Maximum simulated signal. Default 100.
    snr : float, optional
        The signal to noise ratio set to apply Rician noise to the data.
        Default is to not add noise at all.
    rng : numpy.random.Generator class, optional
        Numpy's random generator for setting seed values when needed.
        Default is None.

    Returns
    -------
    data : array, shape (datashape)

    See Also
    --------
    add_noise

    Examples
    --------

    >>> def f(t):
    ...    x = np.sin(t)
    ...    y = np.cos(t)
    ...    z = np.linspace(-1, 1, len(x))
    ...    return x, y, z

    >>> data = orbital_phantom(func=f)

    """

    if gtab is None:
        fimg, fbvals, fbvecs = get_fnames('small_64D')
        gtab = gradient_table(fbvals, fbvecs)

    if func is None:
        x = np.sin(t)
        y = np.cos(t)
        z = np.zeros(t.shape)
    else:
        x, y, z = func(t)

    dx = np.diff(x)
    dy = np.diff(y)
    dz = np.diff(z)

    x = scale[0] * x + origin[0]
    y = scale[1] * y + origin[1]
    z = scale[2] * z + origin[2]

    bx = np.zeros(len(angles))
    by = np.sin(angles)
    bz = np.cos(angles)

    # The entire volume is considered to be inside the brain.
    # Voxels without a fiber crossing through them are taken
    # to be isotropic with signal = S0.
    vol = np.zeros(datashape) + S0

    for i in range(len(dx)):
        evecs, R = diff2eigenvectors(dx[i], dy[i], dz[i])
        S = single_tensor(gtab, S0, evals, evecs, snr=None)

        vol[int(x[i]), int(y[i]), int(z[i]), :] += S

        for r in radii:
            for j in range(len(angles)):
                rb = np.dot(R, np.array([bx[j], by[j], bz[j]]))

                ix = int(x[i] + r * rb[0])
                iy = int(y[i] + r * rb[1])
                iz = int(z[i] + r * rb[2])
                vol[ix, iy, iz] = vol[ix, iy, iz] + S

    vol = vol / np.max(vol, axis=-1)[..., np.newaxis]
    vol *= S0

    if snr is not None:
        vol = add_noise(vol, snr, S0=S0, noise_type='rician', rng=rng)

    return vol


if __name__ == "__main__":

    # TODO: this can become a nice tutorial for generating phantoms

    def f(t):
        x = np.sin(t)
        y = np.cos(t)
        # z=np.zeros(t.shape)
        z = np.linspace(-1, 1, len(x))
        return x, y, z

    # helix
    vol = orbital_phantom(func=f)

    def f2(t):
        x = np.linspace(-1, 1, len(t))
        y = np.linspace(-1, 1, len(t))
        z = np.zeros(x.shape)
        return x, y, z

    # first direction
    vol2 = orbital_phantom(func=f2)

    def f3(t):
        x = np.linspace(-1, 1, len(t))
        y = -np.linspace(-1, 1, len(t))
        z = np.zeros(x.shape)
        return x, y, z

    # second direction
    vol3 = orbital_phantom(func=f3)
    # double crossing
    vol23 = vol2 + vol3

    # """
    def f4(t):
        x = np.zeros(t.shape)
        y = np.zeros(t.shape)
        z = np.linspace(-1, 1, len(t))
        return x, y, z

    # triple crossing
    vol4 = orbital_phantom(func=f4)
    vol234 = vol23 + vol4

    # unknown function
    # voln = add_rician_noise(vol234)

    # """

    # from dipy.viz import window, actor
    # scene = window.Scene()
    # scene.add(actor.volume(vol234[...,0]))
    # window.show(scene)
    # vol234n=add_rician_noise(vol234,20)
