import os
import numpy as np
import collections
from scipy.linalg import block_diag
from functools import lru_cache
from dipy.data import get_fnames


def load_J_matrix():
    fetch_Jmat_path = get_fnames('wigner_jmat')
    Jd = np.load(fetch_Jmat_path, allow_pickle=True)
    return Jd


def z_rot_mat(angle, l):
    """Create a matrix for z-axis rotation by an angle in the basis of real 
    centered spherical harmonics of degree l.

    Parameters
    ----------
    angle : float
        The angle in radians by which to rotate around the z-axis.
    l : int
        The degree of the spherical harmonics. This determines the size of the 
        rotation matrix, which will be of dimension (2 * l + 1) x (2 * l + 1).

    Returns
    -------
    M : ndarray
        The rotation matrix of size (2 * l + 1) x (2 * l + 1) representing the 
        z-axis rotation. This matrix is in the basis of real centered spherical 
        harmonics and is suitable for applying rotational transformations to 
        spherical harmonics coefficients.

    Examples
    --------
    To create a rotation matrix for a 45-degree rotation (pi/4 radians) 
    in the space of spherical harmonics of degree 2:

        >>> z_matrix = z_rot_mat(np.pi / 4, 2)
        >>> print(z_matrix)

    The output will be a 5x5 matrix representing the rotation.
    """
    M = np.zeros((2 * l + 1, 2 * l + 1))
    inds = np.arange(0, 2 * l + 1, 1)
    reversed_inds = np.arange(2 * l, -1, -1)
    frequencies = np.arange(l, -l - 1, -1)
    M[inds, reversed_inds] = np.sin(frequencies * angle)
    M[inds, inds] = np.cos(frequencies * angle)
    return M


def rot_mat(alpha, beta, gamma, l, J):
    """Compute a rotation matrix for ZYZ-Euler angles in the real spherical 
    harmonics basis.

    Parameters
    ----------
    alpha : float
        The first ZYZ-Euler angle for the rotation around the z-axis.
    beta : float
        The second ZYZ-Euler angle for the rotation around the y-axis.
    gamma : float
        The third ZYZ-Euler angle for the rotation around the z-axis.
    l : int
        The degree of the spherical harmonics. Determines the dimension of the 
        rotation matrix, which will be of size (2 * l + 1) x (2 * l + 1).
    J : ndarray
        The matrix used for conversion between the complex and real spherical 
        harmonics bases.

    Returns
    -------
    ndarray
        A rotation matrix of size (2 * l + 1) x (2 * l + 1) that represents the 
        rotation defined by the given ZYZ-Euler angles in the basis of real 
        spherical harmonics.

    References
    ----------
    Johann Goetz's notes on the Wigner-D function:
    https://sites.google.com/site/theodoregoetz/notes/wignerdfunction
    """

    Xa = z_rot_mat(alpha, l)
    Xb = z_rot_mat(beta, l)
    Xc = z_rot_mat(gamma, l)
    return Xa.dot(J).dot(Xb).dot(J).dot(Xc)


def change_of_basis_matrix(l, frm=('complex', 'seismology', 'centered', 'cs'),
                           to=('real', 'quantum', 'centered', 'cs')):
    """Compute change-of-basis matrix that takes the 'frm' basis to the 'to' 
    basis. 
    
    Each basis is identified by:
     1) A field (real or complex)
     2) A normalization / phase convention 
     3) An ordering convention ('centered', 'block')
     4) Whether to use Condon-Shortley phase (-1)^m for m > 0 ('cs', 'nocs')

    Let B = change_of_basis_matrix(l, frm, to).
    Then if Y is a vector in the frm basis, B.dot(Y) represents the same 
    vector in the to basis.

    Parameters
    ----------
    l : int or iterable of int
        The weight (non-negative integer) of the irreducible representation, 
        or an iterable of weights.
    frm : tuple of str
        A 3-tuple (field, normalization, ordering) indicating the input basis.
    to : tuple of str
        A 3-tuple (field, normalization, ordering) indicating the output basis.

    Returns
    -------
    ndarray
        a (2 * l + 1, 2 * l + 1) change of basis matrix.
    """
    from_field, from_normalization, from_ordering, from_cs = frm
    to_field, to_normalization, to_ordering, to_cs = to
    collections.Iterable = collections.abc.Iterable
    if isinstance(l, collections.Iterable):
        blocks = [change_of_basis_matrix(li, frm, to)
                  for li in l]
        return block_diag(*blocks)

    # First, bring us to the centered basis:
    if from_ordering == 'centered':
        B = np.eye(2 * l + 1)
    else:
        raise ValueError('Invalid from_order: ' + str(from_ordering))

    # Make sure we're using CS-phase (this should work for both real and 
    # complex bases)
    if from_cs == 'nocs':
        m = np.arange(-l, l + 1)
        B = ((-1.) ** (m * (m > 0)))[:, None] * B
    elif from_cs != 'cs':
        raise ValueError('Invalid from_cs: ' + str(from_cs))

    # If needed, change complex to real or real to complex
    # (we know how to do that in the centered, CS-phase bases)
    if from_field != to_field:
        if from_field == 'complex' and to_field == 'real':
            B = _cc2rc(l).dot(B)
        elif from_field == 'real' and to_field == 'complex':
            B = _cc2rc(l).conj().T.dot(B)
        else:
            raise ValueError('Invalid field:' +
                             str(from_field) + ', ' + str(to_field))

    # Set the correct CS phase
    if to_cs == 'nocs':
        # We're in CS phase now, so cancel it:
        m = np.arange(-l, l + 1)
        B = ((-1.) ** (m * (m > 0)))[:, None] * B
    elif to_cs != 'cs':
        raise ValueError('Invalid to_cs: ' + str(to_cs))

    # If needed, change the order from centered:
    if to_ordering != 'centered':
        raise ValueError('Invalid to_ordering:' + str(to_ordering))

    return B


def _cc2rc(l):
    """Compute change of basis matrix from the complex centered (cc) basis
    to the real centered (rc) basis.

    Let Y be a vector of complex spherical harmonics:
    Y = (Y^{-l}, ..., Y^0, ..., Y^l)^T
    Let S be a vector of real spherical harmonics as defined on the SH wiki page:
    S = (S^{-l}, ..., S^0, ..., S^l)^T
    Let B = cc2rc(l)
    Then S = B.dot(Y)

    B is a complex unitary matrix.

    Parameters
    ----------
    l : int
        Degree of the spherical harmonics. Determines the size of the change of 
        basis matrix, which will be of dimension (2 * l + 1) x (2 * l + 1).

    Returns
    ---------
    B : ndarray
        The (2 * l + 1) x (2 * l + 1) complex unitary matrix that represents the 
        change of basis from complex to real spherical harmonics.
    
    References
    ---------
    [1] http://en.wikipedia.org/wiki/Spherical_harmonics#Real_form_2
    """
    
    B = np.zeros((2 * l + 1, 2 * l + 1), dtype=complex)
    for m in range(-l, l + 1):
        for n in range(-l, l + 1):
            row_ind = m + l
            col_ind = n + l
            if m == 0 and n == 0:
                B[row_ind, col_ind] = np.sqrt(2)
            if m > 0 and m == n:
                B[row_ind, col_ind] = (-1.) ** m
            elif m > 0 and m == -n:
                B[row_ind, col_ind] = 1.
            elif m < 0 and m == n:
                B[row_ind, col_ind] = 1j
            elif m < 0 and m == -n:
                B[row_ind, col_ind] = -1j * ((-1.) ** m)

    return (1.0 / np.sqrt(2)) * B


def wigner_d_matrix(l, beta,
                    field='real', 
                    normalization='quantum', 
                    order='centered', 
                    condon_shortley='cs'):
    """Compute the Wigner-d matrix of degree l at beta, in the basis defined by
    (field, normalization, order, condon_shortley)

    Parameters
    ----------
    l : int
        The degree of the Wigner-d function, where l >= 0. Determines the size 
        of the resulting matrix.
    beta : float
        The rotation angle in radians for which the Wigner-d matrix is computed. 
        Must be in the range 0 <= beta <= pi.
    field : str, optional
        Specifies whether the matrix is in the 'real' or 'complex' field. 
    normalization : str, optional
        The normalization convention used, which defaults to 'quantum'
    order : str, optional
        The ordering convention of the matrix, either 'centered' or 'block'. 
    condon_shortley : str, optional
        Specifies whether to use the Condon-Shortley phase, 'cs' or 'nocs'. 

    Returns
    -------
    ndarray
        The Wigner-d matrix d^l_mn(beta) of dimensions (2l + 1) x (2l + 1) in 
        the chosen basis.
    """
    Jd = load_J_matrix()
    # This returns the d matrix in the (real, quantum-normalized, centered, cs) 
    # convention
    d = rot_mat(alpha=0., beta=beta, gamma=0., l=l, J=Jd[l])

    if (field, normalization, order, condon_shortley) != \
        ('real', 'quantum', 'centered', 'cs'):
        # TODO use change of basis function instead of matrix?
        B = change_of_basis_matrix(
            l,
            frm=('real', 'quantum', 'centered', 'cs'),
            to=(field, normalization, order, condon_shortley))
        BB = change_of_basis_matrix(
            l,
            frm=(field, normalization, order, condon_shortley),
            to=('real', 'quantum', 'centered', 'cs'))
        d = B.dot(d).dot(BB)

        # The Wigner-d matrices are always real, even in the complex basis
        # (I tested this numerically, and have seen it in several texts)
        # assert np.isclose(np.sum(np.abs(d.imag)), 0.0)
        d = d.real

    return d


def wigner_D_matrix(l, alpha, beta, gamma,
                    field='real', 
                    normalization='quantum', 
                    order='centered', 
                    condon_shortley='cs'):
    """Evaluate the Wigner-d matrix D^l_mn(alpha, beta, gamma)

    Parameters
    ----------
    l : int
        Degree of the Wigner-D function (non-negative integer). Determines the 
        dimension of the matrix, which will be (2l + 1) x (2l + 1).
    alpha : float
        First Euler angle for rotation (range: 0 <= alpha <= 2pi).
    beta : float
        Second Euler angle for rotation (range: 0 <= beta <= pi).
    gamma : float
        Third Euler angle for rotation (range: 0 <= gamma <= 2pi).
    field : str, optional
        Specifies whether the matrix is in the 'real' or 'complex' field. 
    normalization : str, optional
        The normalization convention used, defaults to 'quantum'. 
    order : str, optional
        The ordering convention of the matrix, either 'centered' or 'block'. 
    condon_shortley : str, optional
        Specifies whether to use the Condon-Shortley phase, 'cs' or 'nocs'. 

    Returns
    -------
    ndarray
        The Wigner-D matrix D^l_mn(alpha, beta, gamma) of dimensions 
        (2l + 1) x (2l + 1) in the chosen basis.
    """
    Jd = load_J_matrix()
    D = rot_mat(alpha=alpha, beta=beta, gamma=gamma, l=l, J=Jd[l])

    if (field, normalization, order, condon_shortley) != \
        ('real', 'quantum', 'centered', 'cs'):
        B = change_of_basis_matrix(
            l,
            frm=('real', 'quantum', 'centered', 'cs'),
            to=(field, normalization, order, condon_shortley))
        BB = change_of_basis_matrix(
            l,
            frm=(field, normalization, order, condon_shortley),
            to=('real', 'quantum', 'centered', 'cs'))
        D = B.dot(D).dot(BB)

        if field == 'real':
            # print('WIGNER D IMAG PART:', np.sum(np.abs(D.imag)))
            if not np.isclose(np.sum(np.abs(D.imag)), 0.0):
                raise ValueError("Imaginary part of the Wigner-D matrix in the \
                                 'real' field should be close to 0.")
            D = D.real

    return D


@lru_cache(maxsize=32)
def quadrature_weights(b, grid_type='SOFT'):
    """Compute quadrature weights for the grid used by Kostelec & Rockmore [1,2]

    This grid is:
    alpha = 2 pi i / 2b
    beta = pi (2 j + 1) / 4b
    gamma = 2 pi k / 2b
    where 0 <= i, j, k < 2b are indices
    This grid can be obtained from the function: np.meshgrid

    The quadrature weights for this grid are
    w_B(j) = 2/b * sin(pi(2j + 1) / 4b) * sum_{k=0}^{b-1} 1 
    / (2 k + 1) sin((2j + 1)(2k + 1) pi / 4b)
    This is eq. 23 in [1] and eq. 2.15 in [2].

    Parameters
    ----------
    b : int
        The bandwidth parameter. The grid will have the shape (2b x 2b x 2b).
    grid_type : str, optional
        The type of grid used for the calculation. Default is 'SOFT', referring 
        to the grid used in SO(3) Fourier Transforms as defined by 
        Kostelec & Rockmore.

    Returns
    -------
    ndarray
        An array of length 2b containing the computed quadrature weights.

    References
    ----------
    [1] Peter J. Kostelec and Daniel N. Rockmore, "SOFT: SO(3) Fourier 
    Transforms".
    [2] Peter J. Kostelec and Daniel N. Rockmore, "FFTs on the Rotation Group".
    """
    if grid_type == 'SOFT':
        k = np.arange(0, b)
        w = np.array([(2. / b) * np.sin(np.pi * (2. * j + 1.) / (4. * b)) *
                      (np.sum((1. / (2 * k + 1))
                              * np.sin((2 * j + 1) * (2 * k + 1)
                                       * np.pi / (4. * b))))
                      for j in range(2 * b)])

        # It is necessary to divide by this factor to get correct results.
        w /= 2. * ((2 * b) ** 2)
        return w
    else:
        raise NotImplementedError


@lru_cache(maxsize=32)
def _setup_wigner(b, nl, weighted):
    dss = _setup_so3_fft(b, nl, weighted)
    dss = dss.astype(np.float32)  # [beta, l * m * n] # pylint: disable=E1102
    return dss


def _setup_so3_fft(b, nl, weighted):
    """Prepare components for SO(3) Fourier transform up to a given bandwidth, 
    computing Wigner-d matrices with optional quadrature weights.

    Parameters
    ----------
    b : int
        Bandwidth parameter determining the resolution of the transform. The 
        number of grid points in beta is 2b.
    nl : int
        The maximum degree 'l' for which the Wigner-d matrices are computed. 
        This value sets the limit on the size of the matrices, with each having 
        dimensions (2l + 1) x (2l + 1).
    weighted : bool
        A flag indicating whether the quadrature weights should be applied. 
        If True, the Wigner-d matrices are weighted according to the SO(3) 
        Fourier transform quadrature weights. If False, they are scaled by a 
        factor of (2l + 1).

    Returns
    -------
    ndarray
        A 2D array of shape (2b, sum(2l + 1)^2 for l=0 to nl-1) containing the 
        flattened Wigner-d matrices for each beta and each degree l, 
        appropriately weighted or scaled.
    """
    betas = (np.arange(2 * b) + 0.5) / (2 * b) * np.pi
    w = quadrature_weights(b)
    
    if len(w) != len(betas):
        raise ValueError("The length of quadrature weights does not match the \
                         number of beta grid points.")

    dss = []
    for b, beta in enumerate(betas):
        ds = []
        for l in range(nl):
            d = wigner_d_matrix(l, beta,
                                field='complex', 
                                normalization='quantum', 
                                order='centered', 
                                condon_shortley='cs')
            d = d.reshape(((2 * l + 1) ** 2,))

            if weighted:
                d *= w[b]
            else:
                d *= 2 * l + 1

            # d # [m * n]
            ds.append(d)
        ds = np.concatenate(ds)  # [l * m * n]
        dss.append(ds)
    dss = np.stack(dss)  # [beta, l * m * n]
    return dss


def so3_rfft(x, for_grad=False, b_out=None):
    """Transform a signal on the SO(3) rotation group to its spectral 
    representation using Wigner-D functions.

    Parameters
    ----------
    x : ndarray
        The input signal array with dimensions [..., beta, alpha, gamma], 
        where 'beta', 'alpha', and 'gamma' are the Euler angles defining the 
        rotation, and '...' represents any number of leading batch dimensions.
    for_grad : bool, optional
        If True, the Wigner-D matrices are not weighted, which is required for 
        gradient computation.
    b_out : int, optional
        The bandwidth of the output. If None, it defaults to the input 
        bandwidth.

    Returns
    -------
    ndarray
        The spectral representation of the input signal with dimensions 
        [l * m * n, ..., complex], where 'l * m * n' represents the spectral 
        coefficients and '...' are the batch dimensions. 
        The output is complex-valued.
    """
    b_in = x.shape[-1] // 2

    if x.shape[-1] != 2 * b_in:
        raise ValueError(f"Expected the last dimension of input to be twice \
                         the value of b_in (2 * {b_in}), but got \
                            {x.shape[-1]}.")

    if x.shape[-2] != 2 * b_in:
        raise ValueError(f"Expected the second-to-last dimension of input to \
                         be twice the value of b_in (2 * {b_in}), but got \
                            {x.shape[-2]}.")

    if x.shape[-3] != 2 * b_in:
        raise ValueError(f"Expected the third-to-last dimension of input to \
                         be twice the value of b_in (2 * {b_in}), but got \
                            {x.shape[-3]}.")

    if b_out is None:
        b_out = b_in
    batch_size = x.shape[:-3]

    # [batch, beta, alpha, gamma] (nbatch, 2 b_in, 2 b_in, 2 b_in)
    x = np.reshape(x, (-1, 2 * b_in, 2 * b_in, 2 * b_in))

    nspec = b_out * (4 * b_out ** 2 - 1) // 3
    nbatch = x.shape[0]

    wigner = _setup_wigner(b_in, nl=b_out, weighted=not for_grad)

    output = np.zeros((nspec, nbatch, 2), dtype=x.dtype)

    fft_x = np.fft.rfft2(x)
    x = np.stack([np.real(fft_x), np.imag(fft_x)], axis=-1)

    if b_in < b_out:
        output = np.zeros_like(output)
    for l in range(b_out):
        s = slice(l * (4 * l**2 - 1) // 3, l *
                  (4 * l**2 - 1) // 3 + (2 * l + 1) ** 2)
        # if b_out > b_in, consider high frequencies as null
        l1 = min(l, b_in - 1)

        xx = np.zeros((x.shape[0], x.shape[1], 2 * l + 1, 2 * l + 1, 2))
        xx[:, :, l: l + l1 + 1, l: l + l1 + 1] = x[:, :, :l1 + 1, :l1 + 1]

        if l1 > 0:
            xx[:, :, l - l1:l, l: l + l1 + 1] = x[:, :, -l1:, :l1 + 1]
            xx[:, :, l: l + l1 + 1, l - l1:l] = x[:, :, :l1 + 1, -l1:]
            xx[:, :, l - l1:l, l - l1:l] = x[:, :, -l1:, -l1:]

        out = np.einsum("bmn,zbmnc->mnzc",
                        wigner[:, s].reshape(-1, 2 * l + 1, 2 * l + 1), xx)
        output[s] = out.reshape((2 * l + 1) ** 2, -1, 2)
     # [l * m * n, batch, complex] (b_out (4 b_out**2 - 1) // 3, nbatch, 2)
    # [l * m * n, ..., complex]
    output = np.reshape(output, (-1, *batch_size, 2))
    return output


def so3_rifft(x, for_grad=False, b_out=None):
    """Transform a spectral representation on the SO(3) rotation group back into
    its signal representation using inverse Wigner-D functions.

    Parameters
    ----------
    x : ndarray
        The input spectral representation with dimensions 
        [l * m * n, ..., complex], where 'l * m * n' represents the spectral 
        coefficients and '...' are the batch dimensions. The input must be 
        complex-valued (last dimension size 2).
    for_grad : bool, optional
        If True, the Wigner-D matrices are not weighted, which is required for 
        gradient computation.
    b_out : int, optional
        The bandwidth of the output. If None, it defaults to the input 
        bandwidth.

    Returns
    -------
    ndarray
        The signal representation of the spectral input with dimensions 
        [..., beta, alpha, gamma], where 'beta', 'alpha', and 'gamma' are the 
        Euler angles defining the rotation and '...' are the batch dimensions.
    """
    if x.shape[-1] != 2:
        raise ValueError("Input array 'x' must have a last dimension of size 2,\
                         representing complex numbers.")
    nspec = x.shape[0]
    b_in = round((3 / 4 * nspec) ** (1 / 3))

    if nspec != b_in * (4 * b_in**2 - 1) // 3:
        raise ValueError(f"The number of spectral coefficients ({nspec}) does \
                         not match the expected count for the input bandwidth\
                         ({b_in}).")
    if b_out is None:
        b_out = b_in
    batch_size = x.shape[1:-1]

    # [l * m * n, batch, complex] (nspec, nbatch, 2)
    x = np.reshape(x, (nspec, -1, 2))
    nbatch = x.shape[1]

    # [beta, l * m * n] (2 * b_out, nspec)
    wigner = _setup_wigner(b_out, nl=b_in, weighted=for_grad)

    output = np.zeros((nbatch, 2 * b_out, 2 * b_out,
                      2 * b_out, 2), dtype=x.dtype)

    for l in range(min(b_in, b_out)):
        start = l * (4 * l**2 - 1) // 3
        end = start + (2 * l + 1)**2
        s = slice(start, end)

        out = np.einsum("mnzc,bmn->zbmnc", 
                        x[s].reshape(2 * l + 1, 2 * l + 1, -1, 2), 
                        wigner[:, s].reshape(-1, 2 * l + 1, 2 * l + 1))
        l1 = min(l, b_out - 1)

        output[:, :, :l1 + 1, :l1 + 1] += \
            out[:, :, l: l + l1 + 1, l: l + l1 + 1]
        if l > 0:
            output[:, :, -l1:, :l1 + 1] += out[:, :, l - l1: l, l: l + l1 + 1]
            output[:, :, :l1 + 1, -l1:] += out[:, :, l: l + l1 + 1, l - l1: l]
            output[:, :, -l1:, -l1:] += out[:, :, l - l1: l, l - l1: l]

    ifft_output = np.fft.ifft2((output[..., 0] + 1j * output[..., 1]), 
                               axes=[2, 3]) * float(output.shape[-2]) ** 2
    output = np.real(ifft_output)

    output = np.reshape(output, [*batch_size, 2 * b_out, 2 * b_out, 2 * b_out])
    return output


def complex_mm(x, y, conj_x=False, conj_y=False):
    """Compute the product of two complex matrices, optionally conjugating 
    inputs, and returns the complex matrix result.

    Parameters
    ----------
    x : ndarray
        First input complex matrix of shape [i, k, complex] (M, K, 2), where the
        last dimension holds the real and imaginary parts.
    y : ndarray
        Second input complex matrix of shape [k, j, complex] (K, N, 2), similar 
        in structure to 'x'.
    conj_x : bool, optional
        If True, conjugate matrix 'x' before multiplication. Default is False.
    conj_y : bool, optional
        If True, conjugate matrix 'y' before multiplication. Default is False.

    Returns
    -------
    ndarray
        The resulting complex matrix product of shape [i, j, complex] (M, N, 2), 
        where the last dimension represents the complex number.
    """
    xr = x[:, :, 0]
    xi = x[:, :, 1]

    yr = y[:, :, 0]
    yi = y[:, :, 1]

    if not conj_x and not conj_y:
        zr = np.matmul(xr, yr) - np.matmul(xi, yi)
        zi = np.matmul(xr, yi) + np.matmul(xi, yr)
    if conj_x and not conj_y:
        zr = np.matmul(xr, yr) + np.matmul(xi, yi)
        zi = np.matmul(xr, yi) - np.matmul(xi, yr)
    if not conj_x and conj_y:
        zr = np.matmul(xr, yr) + np.matmul(xi, yi)
        zi = np.matmul(xi, yr) - np.matmul(xr, yi)
    if conj_x and conj_y:
        zr = np.matmul(xr, yr) - np.matmul(xi, yi)
        zi = - np.matmul(xr, yi) - np.matmul(xi, yr)

    return np.stack((zr, zi), axis=2)
