""" Learning algorithms for tractography"""
import numpy as np
import dipy.tracking.distances as pf


def detect_corresponding_tracks(indices, tracks1, tracks2):
    """ Detect corresponding tracks from list tracks1 to list tracks2
    where tracks1 & tracks2 are lists of tracks

    Parameters
    ----------
    indices : sequence
       of indices of tracks1 that are to be detected in tracks2
    tracks1 : sequence
       of tracks as arrays, shape (N1,3) .. (Nm,3)
    tracks2 : sequence
       of tracks as arrays, shape (M1,3) .. (Mm,3)

    Returns
    -------
    track2track : array (N,2) where N is len(indices) of int
       it shows the correspondence in the following way:
       the first column is the current index in tracks1
       the second column is the corresponding index in tracks2

    Examples
    --------
    >>> import numpy as np
    >>> import dipy.tracking.learning as tl
    >>> A = np.array([[0, 0, 0], [1, 1, 1], [2, 2, 2]])
    >>> B = np.array([[1, 0, 0], [2, 0, 0], [3, 0, 0]])
    >>> C = np.array([[0, 0, -1], [0, 0, -2], [0, 0, -3]])
    >>> bundle1 = [A, B, C]
    >>> bundle2 = [B, A]
    >>> indices = [0, 1]
    >>> arr = tl.detect_corresponding_tracks(indices, bundle1, bundle2)

    Notes
    -----
    To find the corresponding tracks we use mam_distances with 'avg' option.
    Then we calculate the argmin of all the calculated distances and return it
    for every index. (See 3rd column of arr in the example given below.)


    """
    li = len(indices)

    track2track = np.zeros((li, 2))
    cnt = 0
    for i in indices:
        rt = [pf.mam_distances(tracks1[i], t, 'avg') for t in tracks2]
        rt = np.array(rt)
        track2track[cnt] = np.array([i, rt.argmin()])
        cnt += 1

    return track2track.astype(int)


def detect_corresponding_tracks_plus(indices, tracks1, indices2, tracks2):
    """ Detect corresponding tracks from 1 to 2 where tracks1 & tracks2 are
    sequences of tracks

    Parameters
    ----------
    indices : sequence
            of indices of tracks1 that are to be detected in tracks2
    tracks1 : sequence
            of tracks as arrays, shape (N1,3) .. (Nm,3)
    indices2 : sequence
            of indices of tracks2 in the initial brain
    tracks2 : sequence
            of tracks as arrays, shape (M1,3) .. (Mm,3)

    Returns
    -------
    track2track : array (N,2) where N is len(indices)
       of int showing the correspondence in th following way
       the first column is the current index of tracks1
       the second column is the corresponding index in tracks2

    Examples
    --------
    >>> import numpy as np
    >>> import dipy.tracking.learning as tl
    >>> A = np.array([[0, 0, 0], [1, 1, 1], [2, 2, 2]])
    >>> B = np.array([[1, 0, 0], [2, 0, 0], [3, 0, 0]])
    >>> C = np.array([[0, 0, -1], [0, 0, -2], [0, 0, -3]])
    >>> bundle1 = [A, B, C]
    >>> bundle2 = [B, A]
    >>> indices = [0, 1]
    >>> indices2 = indices
    >>> arr = tl.detect_corresponding_tracks_plus(indices, bundle1, indices2, bundle2)

    Notes
    -----
    To find the corresponding tracks we use mam_distances with 'avg' option.
    Then we calculate the argmin of all the calculated distances and return it
    for every index. (See 3rd column of arr in the example given below.)


    See Also
    --------
    distances.mam_distances

    """
    li = len(indices)
    track2track = np.zeros((li, 2))
    cnt = 0
    for i in indices:
        rt = [pf.mam_distances(tracks1[i], t, 'avg') for t in tracks2]
        rt = np.array(rt)
        track2track[cnt] = np.array([i, indices2[rt.argmin()]])
        cnt += 1
    return track2track.astype(int)
