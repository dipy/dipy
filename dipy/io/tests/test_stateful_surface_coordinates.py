import nibabel as nib
import numpy as np
import numpy.testing as npt
import pytest

from dipy.io.stateful_surface import StatefulSurface
from dipy.io.utils import Origin, Space


@pytest.mark.parametrize("space", [Space.RASMM, Space.LPSMM])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("order", ["C", "F"])
@pytest.mark.parametrize("writeable", [True, False])
@pytest.mark.parametrize("n_vertices", [3, 300001])
def test_rasmm_lpsmm_coordinate_flip(space, dtype, order, writeable, n_vertices):
    # Large surfaces also exercise the matrix-product path used before #4173.
    vertices = np.resize([[1.5, -2.5, 3.5], [-4, 5, -6], [7, 8, 9]], (n_vertices, 3))
    expected = np.resize([[-1.5, 2.5, 3.5], [4, -5, -6], [-7, -8, 9]], (n_vertices, 3))
    faces = np.array([[0, 1, 2]], dtype=np.uint32)
    reference = nib.Nifti1Image(np.zeros((2, 2, 2)), np.eye(4))
    sfs = StatefulSurface(vertices, faces, reference, space, origin=Origin.TRACKVIS)
    sfs.vertices = np.array(vertices, dtype=dtype, order=order)
    sfs.vertices.setflags(write=writeable)
    target = Space.LPSMM if space == Space.RASMM else Space.RASMM

    sfs.to_space(target)
    npt.assert_array_equal(sfs.vertices, expected)
    npt.assert_equal(sfs.vertices.dtype, np.float64)
    npt.assert_equal(sfs.space, target)
    npt.assert_equal(sfs.origin, Origin.TRACKVIS)
    npt.assert_array_equal(sfs.faces, faces)

    # Conversion to the current space is a no-op; the reverse is an involution.
    sfs.to_space(target)
    npt.assert_array_equal(sfs.vertices, expected)
    sfs.to_space(space)
    npt.assert_array_equal(sfs.vertices, vertices)
    npt.assert_equal(sfs.vertices.dtype, np.float64)
    npt.assert_equal(sfs.space, space)


@pytest.mark.parametrize("space", [Space.RASMM, Space.LPSMM])
@pytest.mark.parametrize("dtype", [np.int32, np.uint32])
def test_rasmm_lpsmm_integer_vertices(space, dtype):
    minimum = np.iinfo(dtype).min
    vertices = np.array([[minimum, 1, 2], [3, 4, 5], [6, 7, 8]], dtype=dtype)
    expected = [[-int(minimum), -1, 2], [-3, -4, 5], [-6, -7, 8]]
    reference = nib.Nifti1Image(np.zeros((2, 2, 2)), np.eye(4))
    sfs = StatefulSurface(vertices, [[0, 1, 2]], reference, space)
    sfs.vertices = vertices

    sfs.to_space(Space.LPSMM if space == Space.RASMM else Space.RASMM)
    npt.assert_array_equal(sfs.vertices, expected)
    npt.assert_equal(sfs.vertices.dtype, np.int64)
