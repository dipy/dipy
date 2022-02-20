
import numpy as np
import numpy.testing as npt
from dipy.reconst.recspeed import (local_maxima, remove_similar_vertices,
                                   search_descending)
from dipy.data import default_sphere
from dipy.core.sphere import unique_edges, HemiSphere


def test_local_maxima():
    sphere = default_sphere
    vertices, faces = sphere.vertices, sphere.faces
    edges = unique_edges(faces)

    # Check that the first peak is == max(odf)
    odf = abs(vertices.sum(-1))
    peak_values, peak_index = local_maxima(odf, edges)
    npt.assert_equal(max(odf), peak_values[0])
    npt.assert_equal(max(odf), odf[peak_index[0]])

    # Create an artificial odf with a few peaks
    odf = np.zeros(len(vertices))
    odf[1] = 1.
    odf[143] = 143.
    odf[361] = 361.
    peak_values, peak_index = local_maxima(odf, edges)
    npt.assert_array_equal(peak_values, [361, 143, 1])
    npt.assert_array_equal(peak_index, [361, 143, 1])

    # Check that neighboring points can both be peaks
    odf = np.zeros(len(vertices))
    point1, point2 = edges[0]
    odf[[point1, point2]] = 1.
    peak_values, peak_index = local_maxima(odf, edges)
    npt.assert_array_equal(peak_values, [1., 1.])
    npt.assert_(point1 in peak_index)
    npt.assert_(point2 in peak_index)

    # Repeat with a hemisphere
    hemisphere = HemiSphere(xyz=vertices, faces=faces)
    vertices, edges = hemisphere.vertices, hemisphere.edges

    # Check that the first peak is == max(odf)
    odf = abs(vertices.sum(-1))
    peak_values, peak_index = local_maxima(odf, edges)
    npt.assert_equal(max(odf), peak_values[0])
    npt.assert_equal(max(odf), odf[peak_index[0]])

    # Create an artificial odf with a few peaks
    odf = np.zeros(len(vertices))
    odf[1] = 1.
    odf[143] = 143.
    odf[300] = 300.
    peak_value, peak_index = local_maxima(odf, edges)
    npt.assert_array_equal(peak_value, [300, 143, 1])
    npt.assert_array_equal(peak_index, [300, 143, 1])

    # Check that neighboring points can both be peaks
    odf = np.zeros(len(vertices))
    point1, point2 = edges[0]
    odf[[point1, point2]] = 1.
    peak_values, peak_index = local_maxima(odf, edges)
    npt.assert_array_equal(peak_values, [1., 1.])
    npt.assert_(point1 in peak_index)
    npt.assert_(point2 in peak_index)

    # Should raise an error if odf has nans
    odf[20] = np.nan
    npt.assert_raises(ValueError, local_maxima, odf, edges)

    # Should raise an error if edge values are too large to index odf
    edges[0, 0] = 9999
    odf[20] = 0
    npt.assert_raises(IndexError, local_maxima, odf, edges)


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

    # Return unique vertices
    uv = remove_similar_vertices(vertices, .01)
    npt.assert_array_equal(uv, vertices[:6])

    # Return vertices with mapping and indices
    uv, mapping, index = remove_similar_vertices(vertices, .01,
                                                 return_mapping=True,
                                                 return_index=True)
    npt.assert_array_equal(uv, vertices[:6])
    npt.assert_array_equal(mapping, list(range(6)) + [0])
    npt.assert_array_equal(index, range(6))

    # Test mapping with different angles
    uv, mapping = remove_similar_vertices(vertices, .01, return_mapping=True)
    npt.assert_array_equal(uv, vertices[:6])
    npt.assert_array_equal(mapping, list(range(6)) + [0])
    uv, mapping = remove_similar_vertices(vertices, 30, return_mapping=True)
    npt.assert_array_equal(uv, vertices[:4])
    npt.assert_array_equal(mapping, list(range(4)) + [1, 0, 0])
    uv, mapping = remove_similar_vertices(vertices, 60, return_mapping=True)
    npt.assert_array_equal(uv, vertices[:3])
    npt.assert_array_equal(mapping, list(range(3)) + [0, 1, 0, 0])

    # Test index with different angles
    uv, index = remove_similar_vertices(vertices, .01, return_index=True)
    npt.assert_array_equal(uv, vertices[:6])
    npt.assert_array_equal(index, range(6))
    uv, index = remove_similar_vertices(vertices, 30, return_index=True)
    npt.assert_array_equal(uv, vertices[:4])
    npt.assert_array_equal(index, range(4))
    uv, index = remove_similar_vertices(vertices, 60, return_index=True)
    npt.assert_array_equal(uv, vertices[:3])
    npt.assert_array_equal(index, range(3))


def test_search_descending():
    a = np.linspace(10., 1., 10)

    npt.assert_equal(search_descending(a, 1.), 1)
    npt.assert_equal(search_descending(a, .89), 2)
    npt.assert_equal(search_descending(a, .79), 3)

    # Test small array
    npt.assert_equal(search_descending(a[:1], 1.), 1)
    npt.assert_equal(search_descending(a[:1], 0.), 1)
    npt.assert_equal(search_descending(a[:1], .5), 1)

    # Test very small array
    npt.assert_equal(search_descending(a[:0], 1.), 0)
