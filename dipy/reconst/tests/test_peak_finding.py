import numpy as np
import numpy.testing as npt
from dipy.reconst.recspeed import local_maxima, remove_similar_vertices
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
    npt.assert_array_equal(peak_values, [505, 143, 10])
    npt.assert_array_equal(peak_index, [505, 143, 1])

    hemisphere = HemiSphere(xyz=vertices, faces=faces)
    vertices_half, edges_half = hemisphere.vertices, hemisphere.edges
    odf = abs(vertices_half.sum(-1))
    odf[1] = 10.
    odf[143] = 143.

    peak_value, peak_index = local_maxima(odf, edges_half)
    npt.assert_array_equal(peak_value, [143, 10])
    npt.assert_array_equal(peak_index, [143, 1])

    odf[20] = np.nan
    npt.assert_raises(ValueError, local_maxima, odf, edges_half)


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
    npt.assert_array_equal(mapping, range(6) + [0])
    npt.assert_array_equal(index, range(6))

    # Test mapping with different angles
    uv, mapping = remove_similar_vertices(vertices, .01, return_mapping=True)
    npt.assert_array_equal(uv, vertices[:6])
    npt.assert_array_equal(mapping, range(6) + [0])
    uv, mapping = remove_similar_vertices(vertices, 30, return_mapping=True)
    npt.assert_array_equal(uv, vertices[:4])
    npt.assert_array_equal(mapping, range(4) + [1, 0, 0])
    uv, mapping = remove_similar_vertices(vertices, 60, return_mapping=True)
    npt.assert_array_equal(uv, vertices[:3])
    npt.assert_array_equal(mapping, range(3) + [0, 1, 0, 0])

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


if __name__ == '__main__':
    import nose
    nose.runmodule()
