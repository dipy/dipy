
from os.path import splitext
import numpy as np

from dipy.core import gradients
from dipy.utils.deprecator import deprecate_with_version


@deprecate_with_version("dipy.io.bvectxt module is deprecated, "
                        "Please use dipy.core.gradients module instead",
                        since='1.4', until='1.5')
def read_bvec_file(filename, atol=.001):
    """
    Read gradient table information from a pair of files with extensions
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
        The tolerance used to check all the gradient directions are
        normalized. Default is .001

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
        raise IOError('the gradient file and b value file should'
                      'have the same number of columns')

    grad_norms = np.sqrt((grad_table**2).sum(0))
    if not np.allclose(grad_norms[b_values > 0], 1, atol=atol):
        raise IOError('the magnitudes of the gradient directions' +
                      'are not within ' + str(atol) + ' of 1')
    grad_table[:, b_values > 0] = (grad_table[:, b_values > 0] /
                                   grad_norms[b_values > 0])

    return grad_table, b_values


@deprecate_with_version("dipy.io.bvectxt module is deprecated, "
                        "Please use dipy.core.gradients module instead",
                        since='1.4', until='1.5')
def ornt_mapping(ornt1, ornt2):
    """Calculate the mapping needing to get from orn1 to orn2."""
    return gradients.ornt_mapping(ornt1=ornt1, ornt2=ornt2)


@deprecate_with_version("dipy.io.bvectxt module is deprecated, "
                        "Please use dipy.core.gradients module instead",
                        since='1.4', until='1.5')
def reorient_vectors(bvecs, current_ornt, new_ornt, axis=0):
    """Change the orientation of gradients or other vectors.

    Moves vectors, storted along axis, from current_ornt to new_ornt. For
    example the vector [x, y, z] in "RAS" will be [-x, -y, z] in "LPS".

    R: Right
    A: Anterior
    S: Superior
    L: Left
    P: Posterior
    I: Inferior

    """
    return gradients.reorient_vectors(bvecs=bvecs, current_ornt=current_ornt,
                                      new_ornt=new_ornt, axis=axis)


@deprecate_with_version("dipy.io.bvectxt module is deprecated, "
                        "Please use dipy.core.gradients module instead",
                        since='1.4', until='1.5')
def reorient_on_axis(bvecs, current_ornt, new_ornt, axis=0):
    return gradients.reorient_on_axis(bvecs=bvecs, current_ornt=current_ornt,
                                      new_ornt=new_ornt, axis=axis)


@deprecate_with_version("dipy.io.bvectxt module is deprecated, "
                        "Please use dipy.core.gradients module instead",
                        since='1.4', until='1.5')
def orientation_from_string(string_ornt):
    """Return an array representation of an ornt string."""
    return gradients.orientation_from_string(string_ornt=string_ornt)


@deprecate_with_version("dipy.io.bvectxt module is deprecated, "
                        "Please use dipy.core.gradients module instead",
                        since='1.4', until='1.5')
def orientation_to_string(ornt):
    """Return a string representation of a 3d ornt."""
    return gradients.orientation_to_string(ornt=ornt)


@deprecate_with_version("dipy.io.bvectxt module is deprecated, "
                        "Please use dipy.core.gradients module instead",
                        since='1.4', until='1.5')
def _check_ornt(ornt):
    return gradients._check_ornt(ornt=ornt)
