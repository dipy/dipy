from __future__ import division

import numpy as np


def cc(na, nd):
    return na * np.cos(nd * np.pi/180.0)


def ss(na, nd):
    return na * np.sin(nd * np.pi/180.0)


def boys2rgb(v):
    """ boys 2 rgb cool colormap

    Maps a given field of undirected lines (line field) to rgb colors using
    Boy's Surface immersion of the real projective plane.

    Boy's Surface is one of the three possible surfaces obtained by gluing a
    Mobius strip to the edge of a disk. The other two are the crosscap and
    Roman surface, Steiner surfaces that are homeomorphic to the real
    projective plane (Pinkall 1986). The Boy's surface is the only 3D
    immersion of the projective plane without singularities.

    Visit http://hci.stanford.edu/~cagatay//rp2coloring/index.html for
    further details. Cagatay Demiralp, 9/7/2008.

    Code was initially in matlab and was rewritten in Python for dipy by
    the Dipy Team. Thank you Cagatay for putting this online.

    Parameters
    ------------

    v : array, shape (N, 3) of unit vectors (e.g., principal eigenvectors of
       tensor data) representing one of the two directions of the
       undirected lines in a line field.

    Returns
    ---------

    c : array, shape (N, 3) matrix of rgb colors corresponding to the vectors
           given in V.

    Examples
    ----------

    >>> from dipy.viz import colormap
    >>> v=np.array([[1,0,0],[0,1,0],[0,0,1]])
    >>> c=colormap.boys2rgb(v)

    """

    if v.ndim == 1:
        x = v[0]
        y = v[1]
        z = v[2]

    if v.ndim == 2:
        x = v[:, 0]
        y = v[:, 1]
        z = v[:, 2]

    x2 = x**2
    y2 = y**2
    z2 = z**2

    x3 = x*x2
    y3 = y*y2
    z3 = z*z2

    z4 = z*z2

    xy = x*y
    xz = x*z
    yz = y*z

    hh1 = .5 * (3 * z2 - 1)/1.58
    hh2 = 3 * xz/2.745
    hh3 = 3 * yz/2.745
    hh4 = 1.5 * (x2 - y2)/2.745
    hh5 = 6 * xy/5.5
    hh6 = (1/1.176) * .125 * (35 * z4 - 30 * z2 + 3)
    hh7 = 2.5 * x * (7 * z3 - 3*z)/3.737
    hh8 = 2.5 * y * (7 * z3 - 3*z)/3.737
    hh9 = ((x2 - y2) * 7.5 * (7 * z2 - 1))/15.85
    hh10 = ((2 * xy) * (7.5 * (7 * z2 - 1)))/15.85
    hh11 = 105 * (4 * x3 * z - 3 * xz * (1 - z2))/59.32
    hh12 = 105 * (-4 * y3 * z + 3 * yz * (1 - z2))/59.32

    s0 = -23.0
    s1 = 227.9
    s2 = 251.0
    s3 = 125.0

    ss23 = ss(2.71, s0)
    cc23 = cc(2.71, s0)

    ss45 = ss(2.12, s1)
    cc45 = cc(2.12, s1)

    ss67 = ss(.972, s2)
    cc67 = cc(.972, s2)

    ss89 = ss(.868, s3)
    cc89 = cc(.868, s3)

    X = 0.0

    X = X + hh2 * cc23
    X = X + hh3 * ss23
    X = X + hh5 * cc45
    X = X + hh4 * ss45

    X = X + hh7 * cc67
    X = X + hh8 * ss67
    X = X + hh10 * cc89
    X = X + hh9 * ss89

    Y = 0.0
    Y = Y + hh2 * -ss23
    Y = Y + hh3 * cc23
    Y = Y + hh5 * -ss45
    Y = Y + hh4 * cc45
    Y = Y + hh7 * -ss67
    Y = Y + hh8 * cc67
    Y = Y + hh10 * -ss89
    Y = Y + hh9 * cc89

    Z = 0.0
    Z = Z + hh1 * -2.8
    Z = Z + hh6 * -0.5
    Z = Z + hh11 * 0.3
    Z = Z + hh12 * -2.5

    # Scale and normalize to fit in the rgb space.
    w_x = 4.1925
    trl_x = -2.0425

    w_y = 4.0217
    trl_y = -1.8541

    w_z = 4.0694
    trl_z = -2.1899

    if v.ndim == 2:
        N = len(x)
        C = np.zeros((N, 3))
        C[:, 0] = 0.9 * np.abs(((X-trl_x)/w_x)) + 0.05
        C[:, 1] = 0.9 * np.abs(((Y-trl_y)/w_y)) + 0.05
        C[:, 2] = 0.9 * np.abs(((Z-trl_z)/w_z)) + 0.05

    if v.ndim == 1:
        C = np.zeros((3,))
        C[0] = 0.9 * np.abs(((X-trl_x)/w_x)) + 0.05
        C[1] = 0.9 * np.abs(((Y-trl_y)/w_y)) + 0.05
        C[2] = 0.9 * np.abs(((Z-trl_z)/w_z)) + 0.05

    return C


def orient2rgb(v):
    """ standard orientation 2 rgb colormap

    v : array, shape (N, 3) of vectors not necessarily normalized

    Returns
    ---------

    c : array, shape (N, 3) matrix of rgb colors corresponding to the vectors
           given in V.

    Examples
    ----------

    >>> from dipy.viz import colormap
    >>> v=np.array([[1,0,0],[0,1,0],[0,0,1]])
    >>> c=colormap.orient2rgb(v)

    """

    if v.ndim == 1:
        orient = v
        orient = np.abs(orient/np.linalg.norm(orient))

    if v.ndim == 2:
        orientn = np.sqrt(v[:, 0]**2 + v[:, 1]**2 + v[:, 2]**2)
        orientn.shape = orientn.shape+(1,)
        orient = np.abs(v/orientn)

    return orient


def line_colors(streamlines, cmap='rgb_standard'):
    """ Create colors for streamlines to be used in fvtk.line

    Parameters
    ----------
    streamlines : sequence of ndarrays
    cmap : ('rgb_standard', 'boys_standard')

    Returns
    -------
    colors : ndarray
    """

    if cmap == 'rgb_standard':
        col_list = [orient2rgb(streamline[-1] - streamline[0])
                    for streamline in streamlines]

    if cmap == 'boys_standard':
        col_list = [boys2rgb(streamline[-1] - streamline[0])
                    for streamline in streamlines]

    return np.vstack(col_list)


def _lab_delta(x, y):
    dL = y.l - x.l
    dA = y.a - x.a
    dB = y.b - x.b
    return np.sqrt(dL**2 + dA**2 + dB**2)


def _rgb_lab_delta(x, y):
    labX = _rgb2lab(x)
    labY = _rgb2lab(y)
    return _lab_delta(labX, labY)


def _rgb2xyz(rgb):
    var_R = rgb[:, 0] / 255  # R from 0 to 255
    var_G = rgb[:, 1] / 255  # G from 0 to 255
    var_B = rgb[:, 2] / 255  # B from 0 to 255

    idx = var_R > 0.04045
    var_R[idx] = ((var_R[idx] + 0.055) / 1.055) ** 2.4
    idx = np.logical_not(idx)
    var_R[idx] = var_R[idx] / 12.92

    idx = var_G > 0.04045
    var_G[idx] = ((var_G[idx] + 0.055) / 1.055) ** 2.4
    idx = np.logical_not(idx)
    var_G[idx] = var_G[idx] / 12.92

    idx = var_B > 0.04045
    var_B[idx] = ((var_B[idx] + 0.055) / 1.055) ** 2.4
    idx = np.logical_not(idx)
    var_B[idx] = var_B[idx] / 12.92

    var_R = var_R * 100
    var_G = var_G * 100
    var_B = var_B * 100

    #Observer. = Illuminant = D65
    X = var_R * 0.4124 + var_G * 0.3576 + var_B * 0.1805
    Y = var_R * 0.2126 + var_G * 0.7152 + var_B * 0.0722
    Z = var_R * 0.0193 + var_G * 0.1192 + var_B * 0.9505
    #xyz = XYZColor(X,Y,Z)

    return np.c_[X, Y, Z]


def _xyz2lab(xyz):
    ref_X = 095.047
    ref_Y = 100.000
    ref_Z = 108.883
    var_X = xyz[:, 0] / ref_X
    var_Y = xyz[:, 1] / ref_Y
    var_Z = xyz[:, 2] / ref_Z

    idx = var_X > 0.008856
    var_X[idx] = var_X[idx] ** (1/3)
    idx = np.logical_not(idx)
    var_X[idx] = (7.787 * var_X[idx]) + (16 / 116)

    idx = var_Y > 0.008856
    var_Y[idx] = var_Y[idx] ** (1/3)
    idx = np.logical_not(idx)
    var_Y[idx] = (7.787 * var_Y[idx]) + (16 / 116)

    idx = var_Z > 0.008856
    var_Z[idx] = var_Z[idx] ** (1/3)
    idx = np.logical_not(idx)
    var_Z[idx] = (7.787 * var_Z[idx]) + (16 / 116)

    L = (116 * var_Y) - 16
    A = 500 * (var_X - var_Y)
    B = 200 * (var_Y - var_Z)
    #lab = LabColor(L,A,B)

    return np.c_[L, A, B]


def _lab2xyz(lab):
    var_Y = (lab.l + 16) / 116
    var_X = lab.a / 500 + var_Y
    var_Z = var_Y - lab.b / 200

    if var_Y**3 > 0.008856:
        var_Y = var_Y**3
    else:
        var_Y = (var_Y - 16/116.) / 7.787

    if var_X**3 > 0.008856:
        var_X = var_X**3
    else:
        var_X = (var_X - 16/116.) / 7.787

    if var_Z**3 > 0.008856:
        var_Z = var_Z**3
    else:
        var_Z = (var_Z - 16/116.) / 7.787

    ref_X = 095.047
    ref_Y = 100.000
    ref_Z = 108.883
    X = ref_X * var_X
    Y = ref_Y * var_Y
    Z = ref_Z * var_Z
    xyz = XYZColor(X, Y, Z)

    return xyz


def _xyz2rgb(xyz):
    var_X = xyz.x / 100  # X from 0 to  95.047
    var_Y = xyz.y / 100  # Y from 0 to 100.000
    var_Z = xyz.z / 100  # Z from 0 to 108.883

    var_R = var_X * 03.2406 + var_Y * -1.5372 + var_Z * -0.4986
    var_G = var_X * -0.9689 + var_Y * 01.8758 + var_Z * 00.0415
    var_B = var_X * 00.0557 + var_Y * -0.2040 + var_Z * 01.0570

    if var_R > 0.0031308:
        var_R = 1.055 * (var_R**(1/2.4)) - 0.055
    else:
        var_R = 12.92 * var_R

    if var_G > 0.0031308:
        var_G = 1.055 * (var_G**(1/2.4)) - 0.055
    else:
        var_G = 12.92 * var_G

    if var_B > 0.0031308:
        var_B = 1.055 * (var_B**(1/2.4)) - 0.055
    else:
        var_B = 12.92 * var_B

    R = var_R * 255
    G = var_G * 255
    B = var_B * 255
    rgb = RGBColor(R, G, B)

    return rgb


def _rgb2lab(rgb):
    tmp = _rgb2xyz(rgb)
    return _xyz2lab(tmp)


def _lab2rgb(lab):
    tmp = _lab2xyz(lab)
    return _xyz2rgb(tmp)


class LabColor:
    def __init__(self, l, a, b):
        self.l = l
        self.a = a
        self.b = b


class RGBColor:
    def __init__(self, r, g, b):
        self.r = r
        self.g = g
        self.b = b


class XYZColor:
    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z


def distinguishable_colormap(bg=(0, 0, 0), exclude=[], nb_colors=None):
    """ Generates colors that are maximally perceptually distinct.

    This function generates a set of colors which are distinguishable
    by reference to the "Lab" color space, which more closely matches
    human color perception than RGB. Given an initial large list of possible
    colors, it iteratively chooses the entry in the list that is farthest (in
    Lab space) from all previously-chosen entries. While this "greedy"
    algorithm does not yield a global maximum, it is simple and efficient.
    Moreover, the sequence of colors is consistent no matter how many you
    request, which facilitates the users' ability to learn the color order
    and avoids major changes in the appearance of plots when adding or
    removing lines.

    Parameters
    ----------
    bg : tuple (optional)
        Background RGB color, to make sure that your colors are also
        distinguishable from the background. Default: (0, 0, 0).
    exclude : list of tuples (optional)
        Additional RGB colors to be distinguishable from.
    nb_colors : int (optional)
        Number of colors desired. Default: generate as many colors as needed.

    Returns
    -------
    iterable of ndarray
        If `nb_colors` is provided, returns a list of RBG colors.
        Otherwise, yields the next RBG color maximally perceptually
        distinct from previous ones.

    Examples
    --------
    >>> from dipy.viz.colormap import distinguishable_colormap
    >>> # Generate 5 colors
    >>> [c for i, c in zip(range(5), distinguishable_colormap())]
    [array([ 0.,  1.,  0.]),
     array([ 1.,  0.,  1.]),
     array([ 1.        ,  0.75862069,  0.03448276]),
     array([ 0.        ,  1.        ,  0.89655172]),
     array([ 0.        ,  0.17241379,  1.        ])]


    Notes
    -----
    Code was initially in matlab and was rewritten in Python for dipy by
    the Dipy Team. Thank you Tim Holy for putting this online. Visit
    http://www.mathworks.com/matlabcentral/fileexchange/29702 for the
    original implementation (v1.2), 14 Dec 2010 (Updated 07 Feb 2011).

    """

    NB_DIVISIONS = 30  # This constant come from the original code.

    # Generate a sizable number of RGB triples. This represents our space of
    # possible choices. By starting in RGB space, we ensure that all of the
    # colors can be generated by the monitor.
    colors_to_exclude = np.array([bg] + exclude)
    x = np.linspace(0, 1, NB_DIVISIONS)  # Divisions along each axis in RGB space.
    R, G, B = np.meshgrid(x, x, x)
    rgb = np.c_[R.flatten(), G.flatten(), B.flatten()]

    lab = _rgb2lab(rgb)
    bglab = _rgb2lab(colors_to_exclude)

    def _generate_next_color():
        lastlab = bglab[0]
        mindist2 = np.ones(len(rgb)) * np.inf
        for bglab_i in bglab[1:]:
            dist2 = np.sum((lab-bglab_i)**2, axis=1)
            mindist2 = np.minimum(dist2, mindist2)  # Dist2 to closest previously-chosen color.

        while True:
            dX = lab - lastlab  # Displacement of last from all colors on list.
            dist2 = np.sum(dX**2, axis=1)  # Square distance.

            mindist2 = np.minimum(dist2, mindist2)  # Dist2 to closest previously-chosen color.
            idx = np.argmax(mindist2)  # Find the entry farthest from all previously-chosen colors.
            yield rgb[idx]

            lastlab = lab[idx]

    if nb_colors is not None:
        return [c for i, c in zip(range(nb_colors), _generate_next_color())]

    return _generate_next_color()
