import numpy as np
from nibabel.affines import apply_affine
from dipy.tracking.streamlinespeed import set_number_of_points
from dipy.tracking.streamlinespeed import length


def unlist_streamlines(streamlines):
    """ Return the streamlines not as a list but as an array and an offset

    Parameters
    ----------
    streamlines: sequence

    Returns
    -------
    points : array
    offsets : array

    """

    points = np.concatenate(streamlines, axis=0)
    offsets = np.zeros(len(streamlines), dtype='i8')

    curr_pos = 0
    prev_pos = 0
    for (i, s) in enumerate(streamlines):

            prev_pos = curr_pos
            curr_pos += s.shape[0]
            points[prev_pos:curr_pos] = s
            offsets[i] = curr_pos

    return points, offsets


def relist_streamlines(points, offsets):
    """ Given a representation of a set of streamlines as a large array and
    an offsets array return the streamlines as a list of shorter arrays.

    Parameters
    -----------
    points : array
    offsets : array

    Returns
    -------
    streamlines: sequence
    """

    streamlines = []

    streamlines.append(points[0: offsets[0]])

    for i in range(len(offsets) - 1):
        streamlines.append(points[offsets[i]: offsets[i + 1]])

    return streamlines


def center_streamlines(streamlines):
    """ Move streamlines to the origin

    Parameters
    ----------
    streamlines : list
        List of 2D ndarrays of shape[-1]==3

    Returns
    -------
    new_streamlines : list
        List of 2D ndarrays of shape[-1]==3
    inv_shift : ndarray
        Translation in x,y,z to go back in the initial position

    """
    center = np.mean(np.concatenate(streamlines, axis=0), axis=0)
    return [s - center for s in streamlines], center


def transform_streamlines(streamlines, mat):
    """ Apply affine transformation to streamlines

    Parameters
    ----------
    streamlines : list
        List of 2D ndarrays of shape[-1]==3

    Returns
    -------
    new_streamlines : list
        List of the transformed 2D ndarrays of shape[-1]==3
    """

    return [apply_affine(mat, s) for s in streamlines]


def select_random_set_of_streamlines(streamlines, select):
    """ Select a random set of streamlines

    Parameters
    ----------
    streamlines : list
        List of 2D ndarrays of shape[-1]==3

    select : int
        Number of streamlines to select. If there are less streamlines
        than ``select`` then ``select=len(streamlines)``.

    Returns
    -------
    selected_streamlines : list
    """
    len_s = len(streamlines)
    index = np.random.randint(0, len_s, min(select, len_s))
    return [streamlines[i] for i in index]
