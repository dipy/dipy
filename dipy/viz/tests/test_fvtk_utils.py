import vtk
import numpy as np
import numpy.testing as npt

from nose.tools import assert_equal, assert_true
from numpy.testing import (assert_raises,
                           assert_almost_equal,
                           assert_array_almost_equal)

from dipy.viz import actor
from dipy.viz.utils import (map_coordinates_3d_4d,
                            construct_grid,
                            get_bounding_box_sizes)


def trilinear_interp_numpy(input_array, indices):
    """ Evaluate the input_array data at the given indices
    """

    if input_array.ndim <= 2 or input_array.ndim >= 5:
        raise ValueError("Input array can only be 3d or 4d")

    x_indices = indices[:, 0]
    y_indices = indices[:, 1]
    z_indices = indices[:, 2]

    x0 = x_indices.astype(np.integer)
    y0 = y_indices.astype(np.integer)
    z0 = z_indices.astype(np.integer)
    x1 = x0 + 1
    y1 = y0 + 1
    z1 = z0 + 1

    # Check if xyz1 is beyond array boundary:
    x1[np.where(x1 == input_array.shape[0])] = x0.max()
    y1[np.where(y1 == input_array.shape[1])] = y0.max()
    z1[np.where(z1 == input_array.shape[2])] = z0.max()

    if input_array.ndim == 3:
        x = x_indices - x0
        y = y_indices - y0
        z = z_indices - z0

    elif input_array.ndim == 4:
        x = np.expand_dims(x_indices - x0, axis=1)
        y = np.expand_dims(y_indices - y0, axis=1)
        z = np.expand_dims(z_indices - z0, axis=1)

    output = (input_array[x0, y0, z0] * (1 - x) * (1 - y) * (1 - z) +
              input_array[x1, y0, z0] * x * (1 - y) * (1 - z) +
              input_array[x0, y1, z0] * (1 - x) * y * (1-z) +
              input_array[x0, y0, z1] * (1 - x) * (1 - y) * z +
              input_array[x1, y0, z1] * x * (1 - y) * z +
              input_array[x0, y1, z1] * (1 - x) * y * z +
              input_array[x1, y1, z0] * x * y * (1 - z) +
              input_array[x1, y1, z1] * x * y * z)

    return output


def test_trilinear_interp():

    A = np.zeros((5, 5, 5))
    A[2, 2, 2] = 1

    indices = np.array([[0, 0, 0], [1, 1, 1], [2, 2, 2], [1.5, 1.5, 1.5]])

    values = trilinear_interp_numpy(A, indices)
    values2 = map_coordinates_3d_4d(A, indices)
    npt.assert_almost_equal(values, values2)

    B = np.zeros((5, 5, 5, 3))
    B[2, 2, 2] = np.array([1, 1, 1])

    values = trilinear_interp_numpy(B, indices)
    values_4d = map_coordinates_3d_4d(B, indices)
    npt.assert_almost_equal(values, values_4d)


@npt.dec.skipif(not actor.have_vtk)
def test_get_bounding_box_sizes():
    # Test bounding box sizes of a sphere.
    radius = 5
    source = vtk.vtkSphereSource()
    source.SetCenter(0, 0, 0)
    source.SetRadius(radius)
    source.SetThetaResolution(100)
    source.SetPhiResolution(100)
    source.Update()

    mapper = vtk.vtkPolyDataMapper()
    if vtk.VTK_MAJOR_VERSION <= 5:
        mapper.SetInput(source.GetOutput())
    else:
        mapper.SetInputConnection(source.GetOutputPort())

    sphere = vtk.vtkActor()
    sphere.SetMapper(mapper)
    bbox_sizes = get_bounding_box_sizes(sphere)
    assert_array_almost_equal(bbox_sizes, (2*radius,)*3, decimal=3)

    # Test bounding box sizes of a pyramid.
    pts = [[1.0, 1.0, 1.0],
           [0.0, 1.0, 1.0],
           [0.0, -1.0, 1.0],
           [1.0, -1.0, 1.0],
           [0.0, 0.0, 0.0]]

    points = vtk.vtkPoints()
    for p in pts:
        points.InsertNextPoint(p)

    pyramid = vtk.vtkPyramid()
    for i in range(len(pts)):
        pyramid.GetPointIds().SetId(i, i)

    cells = vtk.vtkCellArray()
    cells.InsertNextCell(pyramid)

    ug = vtk.vtkUnstructuredGrid()
    ug.SetPoints(points)
    ug.InsertNextCell(pyramid.GetCellType(), pyramid.GetPointIds())

    #Create an actor and mapper
    mapper = vtk.vtkDataSetMapper()
    if vtk.VTK_MAJOR_VERSION <= 5:
        mapper.SetInput(ug)
    else:
        mapper.SetInputData(ug)

    pyramid = vtk.vtkActor()
    pyramid.SetMapper(mapper)

    bbox_sizes = get_bounding_box_sizes(pyramid)
    sizes = np.max(pts, axis=0) - np.min(pts, axis=0)
    assert_array_almost_equal(bbox_sizes, sizes)


def test_construct_grid():
    count = 16*9

    # Test when every cell content has the same shape.
    shapes = [(1, 1)]*count
    coords = construct_grid(shapes)
    assert_equal(len(coords), count)
    assert_equal(np.max(coords[:, 0]), 16-1)
    assert_equal(np.min(coords[:, 1]), -(9-1))
    assert_true(np.all(coords[:, 2] == 0))

    assert_raises(ValueError, construct_grid, shapes, dim=(2, 2))

    nb_rows, nb_cols = 3, count//3
    coords = construct_grid(shapes, dim=(nb_rows, nb_cols))
    assert_equal(len(coords), count)
    assert_equal(np.max(coords[:, 0]), nb_cols-1)
    assert_equal(np.min(coords[:, 1]), -(nb_rows-1))
    assert_true(np.all(coords[:, 2] == 0))

    # Test when each cell content has a shape that is twice as long as wide.
    shapes = [(1, 2)]*(4*3*2)
    coords = construct_grid(shapes, aspect_ratio=4/3.)
    grid_size = np.abs(coords[-1] - coords[0]) + np.array(shapes[0] + (0,))
    assert_almost_equal(grid_size[0]/grid_size[1], 4/3.)

    # Test when each cell content has different shape.
    rng = np.random.RandomState(1234)
    shapes = rng.randint(1, 10, size=(count, 2))
    coords = construct_grid(shapes)
    assert_equal(len(coords), count)


if __name__ == '__main__':
    npt.run_module_suite()
