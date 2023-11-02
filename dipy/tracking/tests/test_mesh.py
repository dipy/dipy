import numpy as np
import numpy.testing as npt

from dipy.tracking.mesh import (triangles_area,
                                random_coordinates_from_surface,
                                seeds_from_surface_coordinates,
                                vertices_to_triangles_values)


def create_cube():
    cube_tri = np.array([[0, 6, 4], [0, 2, 6], [0, 3, 2], [0, 1, 3],
                         [2, 7, 6], [2, 3, 7], [4, 6, 7], [4, 7, 5],
                         [0, 4, 5], [0, 5, 1], [1, 5, 7], [1, 7, 3]])
    cube_vts = np.array([[0., 0., 0.], [0., 0., 1.], [0., 1., 0.],
                         [0., 1., 1.], [1., 0., 0.], [1., 0., 1.],
                         [1., 1., 0.], [1., 1., 1.]])
    return cube_tri, cube_vts


def test_triangles_area():
    cube_tri, cube_vts = create_cube()
    area = triangles_area(cube_tri, cube_vts)
    npt.assert_array_equal(area, 0.5)
    cube_vts *= 2.0
    area = triangles_area(cube_tri, cube_vts)
    npt.assert_array_equal(area, 2.0)


def test_surface_seeds():
    eps = 1e-12
    cube_tri, cube_vts = create_cube()
    nb_tri = len(cube_tri)
    nb_seed = 1000
    tri_idx, trilin_coord = random_coordinates_from_surface(nb_tri, nb_seed)

    # test min-max triangles indices
    npt.assert_array_less(-1, tri_idx)
    npt.assert_array_less(tri_idx, nb_tri)

    # test min-max trilin coordinates
    npt.assert_array_less(-eps, trilin_coord)
    npt.assert_array_less(trilin_coord, 1.0 + eps)

    seed_pts = seeds_from_surface_coordinates(cube_tri, cube_vts,
                                              tri_idx, trilin_coord)

    # test min-max seeds positions
    npt.assert_array_less(-eps, seed_pts)
    npt.assert_array_less(seed_pts, 1.0 + eps)

    for i in range(len(cube_tri)):
        tri_mask = np.zeros([len(cube_tri)])
        tri_mask[i] = 1.0
        tri_maskb = tri_mask.astype(bool)
        t_idx, _ = random_coordinates_from_surface(nb_tri, nb_seed,
                                                   triangles_mask=tri_maskb)
        npt.assert_array_equal(t_idx, i)
        t_idx, _ = random_coordinates_from_surface(nb_tri, nb_seed,
                                                   triangles_weight=tri_mask)
        npt.assert_array_equal(t_idx, i)


def test_vertices_to_triangles():
    cube_tri, cube_vts = create_cube()
    vts_w = np.ones([len(cube_vts)])
    tri_w = vertices_to_triangles_values(cube_tri, vts_w)
    npt.assert_array_equal(tri_w, 1.0)

    # weight each vts individually and test triangles weights
    for i in range(len(cube_vts)):
        vts_w = np.zeros([len(cube_vts)])
        vts_w[i] = 3.0
        tri_w_func = vertices_to_triangles_values(cube_tri, vts_w)
        tri_w_manual = np.zeros([len(cube_tri)])
        # if the current weighted vts is in the current triangle add 1
        for j in range(len(cube_tri)):
            if i in cube_tri[j]:
                tri_w_manual[j] += 1.0

        npt.assert_array_equal(tri_w_func, tri_w_manual)
