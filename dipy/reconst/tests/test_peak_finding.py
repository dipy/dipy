import numpy as np
from nose.tools import assert_true, assert_false, assert_equal, assert_almost_equal, assert_raises
from numpy.testing import assert_array_equal
from dipy.reconst.recspeed import (local_maxima, _filter_peaks,
                                   remove_similar_vertices)
from dipy.data import get_sphere, get_data
from dipy.core.sphere import unique_edges, HemiSphere
from dipy.sims.voxel import all_tensor_evecs, multi_tensor_odf

def test_local_maxima():
    sphere = get_sphere('symmetric724')
    vertices, faces = sphere.vertices, sphere.faces
    edges = unique_edges(faces)
    odf = abs(vertices.sum(-1))
    odf[1] = 10.
    odf[143] = 143.
    odf[505] = 505

    peak_values, peak_index = local_maxima(odf, edges)
    assert_array_equal(peak_values, [505, 143, 10])
    assert_array_equal(peak_index, [505, 143, 1])

    hemisphere = HemiSphere(xyz=vertices, faces=faces)
    vertices_half, edges_half = hemisphere.vertices, hemisphere.edges
    odf = abs(vertices_half.sum(-1))
    odf[1] = 10.
    odf[143] = 143.

    peak_value, peak_index = local_maxima(odf, edges_half)
    assert_array_equal(peak_value, [143, 10])
    assert_array_equal(peak_index, [143, 1])

    odf[20] = np.nan
    assert_raises(ValueError, local_maxima, odf, edges_half)


def test_remove_similar_peaks():
    vertices = np.array([[1., 0., 0.],
                         [0., 1., 0.],
                         [0., 0., 1.],
                         [1.1, 1., 0.],
                         [0., 2., 1.],
                         [2., 1., 0.],
                         [1., 0., 0.]])
    norms = np.sqrt((vertices*vertices).sum(-1))
    vertices = vertices/norms[:, None]

    uv, mapping = remove_similar_vertices(vertices, .01)
    assert_array_equal(uv, vertices[:6])
    assert_array_equal(mapping, range(6) + [0])
    uv, mapping = remove_similar_vertices(vertices, 30)
    assert_array_equal(uv, vertices[:4])
    assert_array_equal(mapping, range(4) + [1, 0, 0])
    uv, mapping = remove_similar_vertices(vertices, 60)
    assert_array_equal(uv, vertices[:3])
    assert_array_equal(mapping, range(3) + [0, 1, 0, 0])


def test_filter_peaks():
    # The setup
    peak_values = np.array([1, .9, .8, .7, .6, .2, .1])
    peak_points = np.array([[1., 0., 0.],
                            [0., 0., 1.],
                            [0., .9, .1],
                            [0., 0., 1.],
                            [0., 1., 0.],
                            [0., 0., 1.],
                            [.9, .1, 0.],
                            [0., 0., 1.],
                            [0., 0., 1.],
                            [0., 0., 1.],
                            [1., 1., 0.],
                            [0., 0., 1.],
                            [0., 1., 1.]])
    norms = np.sqrt((peak_points*peak_points).sum(-1))
    peak_points = peak_points/norms[:, None]

    # Filter above peaks down to thre peaks
    copy_peak_values = peak_values.copy()
    ind = np.arange(0, len(peak_points), 2, dtype='int')
    copy_ind = ind.copy()
    print ind
    sep_mat = abs(np.dot(peak_points, peak_points.T))
    fvalues, find = _filter_peaks(peak_values, ind, sep_mat, .5, .9)
    assert_array_equal(find, [0,2,8])
    assert_array_equal(fvalues, [1., .9, .6])
    # Check that the arguments have not been modified by _filter_peaks
    assert_array_equal(ind, copy_ind)
    assert_array_equal(peak_values, copy_peak_values)
    # Test on a larger set of peaks
    sphere = get_sphere('symmetric724')
    v, faces = sphere.vertices, sphere.faces
    sep_mat = np.dot(v, v.T)
    values = np.arange(len(v), 0., -1.)
    ind = np.arange(len(values), dtype='int')
    # Filter nothing if thresholds are 0. and 1.
    fvalues, find = _filter_peaks(values, ind, sep_mat, 0., 1.)
    assert_array_equal(find, ind)
    assert_array_equal(fvalues, values)
    # Return only the largest peak.
    fvalues, find = _filter_peaks(values, ind, sep_mat, 0., -1.1)
    assert_array_equal(find, [0])
    assert_array_equal(fvalues, [len(values)])
    fvalues, find = _filter_peaks(values, ind, sep_mat, 1., 0.)
    assert_array_equal(find, [0])
    assert_array_equal(fvalues, [len(values)])

    fvalues, find = _filter_peaks(values, ind, sep_mat, .5, 1.)
    assert_array_equal(fvalues, values[values >= .5*values[0]])
    assert_array_equal(find, ind[values >= .5*values[0]])
    values = values[1:].copy()
    ind = ind[1:].copy()
    sep_mat = sep_mat[1:, 1:].copy()
    fvalues, find = _filter_peaks(values, ind, sep_mat, .5, 1.)
    assert_array_equal(fvalues, values[values >= .5*values[0]])
    assert_array_equal(find, ind[values >= .5*values[0]])


if __name__ == '__main__':
    test_peak_finding()

