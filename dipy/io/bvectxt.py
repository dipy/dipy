
from os.path import splitext
import numpy as np

from dipy.utils.deprecator import deprecate_with_version


@deprecate_with_version("dipy.io.bvectxt module is deprecated, "
                        "Please use dipy.core.gradients module instead",
                        since='1.4', until='1.5')
def read_bvec_file(filename, atol=.001):
    """
    Read gradient table information from a pair of files with extentions
    .bvec and .bval. The bval file should have one row of values
    representing the bvalues of each volume in the dwi data set. The bvec
    file should have three rows, where the rows are the x, y, and z
    components of the normalized gradient direction for each of the
    volumes.

    Parameters
    ----------
    filename :
        The path to the either the bvec or bval file
    atol : float, optional
        The tolorance used to check all the gradient directions are
        normalized. Defult is .001

    """

    base, ext = splitext(filename)
    if ext == '':
        bvec = base+'.bvec'
        bval = base+'.bval'
    elif ext == '.bvec':
        bvec = filename
        bval = base+'.bval'
    elif ext == '.bval':
        bvec = base+'.bvec'
        bval = filename
    else:
        raise ValueError('filename must have .bvec or .bval extension')

    b_values = np.loadtxt(bval)
    grad_table = np.loadtxt(bvec)
    if grad_table.shape[0] != 3:
        raise IOError('bvec file should have three rows')
    if b_values.ndim != 1:
        raise IOError('bval file should have one row')
    if b_values.shape[0] != grad_table.shape[1]:
        raise IOError('the gradient file and b value fileshould'
                      'have the same number of columns')

    grad_norms = np.sqrt((grad_table**2).sum(0))
    if not np.allclose(grad_norms[b_values > 0], 1, atol=atol):
        raise IOError('the magnitudes of the gradient directions' +
                      'are not within ' + str(atol) + ' of 1')
    grad_table[:, b_values > 0] = (grad_table[:, b_values > 0] /
                                   grad_norms[b_values > 0])

    return (grad_table, b_values)


@deprecate_with_version("dipy.io.bvectxt module is deprecated, "
                        "Please use dipy.core.gradients module instead",
                        since='1.4', until='1.5')
def ornt_mapping(ornt1, ornt2):
    """Calculates the mapping needing to get from orn1 to orn2"""

    mapping = np.empty((len(ornt1), 2), 'int')
    mapping[:, 0] = -1
    A = ornt1[:, 0].argsort()
    B = ornt2[:, 0].argsort()
    mapping[B, 0] = A
    assert (mapping[:, 0] != -1).all()
    sign = ornt2[:, 1] * ornt1[mapping[:, 0], 1]
    mapping[:, 1] = sign
    return mapping


@deprecate_with_version("dipy.io.bvectxt module is deprecated, "
                        "Please use dipy.core.gradients module instead",
                        since='1.4', until='1.5')
def reorient_vectors(input, current_ornt, new_ornt, axis=0):
    """Changes the orientation of a gradients or other vectors

    Moves vectors, storted along axis, from current_ornt to new_ornt. For
    example the vector [x, y, z] in "RAS" will be [-x, -y, z] in "LPS".

    R: Right
    A: Anterior
    S: Superior
    L: Left
    P: Posterior
    I: Inferior

    Examples
    --------
    >>> gtab = np.array([[1, 1, 1], [1, 2, 3]])
    >>> reorient_vectors(gtab, 'ras', 'asr', axis=1)
    array([[1, 1, 1],
           [2, 3, 1]])
    >>> reorient_vectors(gtab, 'ras', 'lps', axis=1)
    array([[-1, -1,  1],
           [-1, -2,  3]])
    >>> bvec = gtab.T
    >>> reorient_vectors(bvec, 'ras', 'lps', axis=0)
    array([[-1, -1],
           [-1, -2],
           [ 1,  3]])
    >>> reorient_vectors(bvec, 'ras', 'lsp')
    array([[-1, -1],
           [ 1,  3],
           [-1, -2]])
    """
    if isinstance(current_ornt, str):
        current_ornt = orientation_from_string(current_ornt)
    if isinstance(new_ornt, str):
        new_ornt = orientation_from_string(new_ornt)

    n = input.shape[axis]
    if current_ornt.shape != (n, 2) or new_ornt.shape != (n, 2):
        raise ValueError("orientations do not match")

    input = np.asarray(input)
    mapping = ornt_mapping(current_ornt, new_ornt)
    output = input.take(mapping[:, 0], axis)
    out_view = np.rollaxis(output, axis, output.ndim)
    out_view *= mapping[:, 1]
    return output


@deprecate_with_version("dipy.io.bvectxt module is deprecated, "
                        "Please use dipy.core.gradients module instead",
                        since='1.4', until='1.5')
def reorient_on_axis(input, current_ornt, new_ornt, axis=0):
    if isinstance(current_ornt, str):
        current_ornt = orientation_from_string(current_ornt)
    if isinstance(new_ornt, str):
        new_ornt = orientation_from_string(new_ornt)

    n = input.shape[axis]
    if current_ornt.shape != (n, 2) or new_ornt.shape != (n, 2):
        raise ValueError("orientations do not match")

    mapping = ornt_mapping(current_ornt, new_ornt)
    order = [slice(None)] * input.ndim
    order[axis] = mapping[:, 0]
    shape = [1] * input.ndim
    shape[axis] = -1
    sign = mapping[:, 1]
    sign.shape = shape
    output = input[order]
    output *= sign
    return output


@deprecate_with_version("dipy.io.bvectxt module is deprecated, "
                        "Please use dipy.core.gradients module instead",
                        since='1.4', until='1.5')
def orientation_from_string(string_ornt):
    """Returns an array representation of an ornt string"""
    orientation_dict = dict(r=(0, 1), l=(0, -1), a=(1, 1),
                            p=(1, -1), s=(2, 1), i=(2, -1))
    ornt = tuple(orientation_dict[ii] for ii in string_ornt.lower())
    ornt = np.array(ornt)
    if _check_ornt(ornt):
        msg = string_ornt + " does not seem to be a valid orientation string"
        raise ValueError(msg)
    return ornt


@deprecate_with_version("dipy.io.bvectxt module is deprecated, "
                        "Please use dipy.core.gradients module instead",
                        since='1.4', until='1.5')
def orientation_to_string(ornt):
    """Returns a string representation of a 3d ornt"""
    if _check_ornt(ornt):
        msg = repr(ornt) + " does not seem to be a valid orientation"
        raise ValueError(msg)
    orientation_dict = {(0, 1): 'r', (0, -1): 'l', (1, 1): 'a',
                        (1, -1): 'p', (2, 1): 's', (2, -1): 'i'}
    ornt_string = ''
    for ii in ornt:
        ornt_string += orientation_dict[(ii[0], ii[1])]
    return ornt_string


@deprecate_with_version("dipy.io.bvectxt module is deprecated, "
                        "Please use dipy.core.gradients module instead",
                        since='1.4', until='1.5')
def _check_ornt(ornt):
    uniq = np.unique(ornt[:, 0])
    if len(uniq) != len(ornt):
        print(len(uniq))
        return True
    uniq = np.unique(ornt[:, 1])
    if tuple(uniq) not in set([(-1, 1), (-1,), (1,)]):
        print(tuple(uniq))
        return True
