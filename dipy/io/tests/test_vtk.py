from types import SimpleNamespace

import numpy as np
import pytest

import dipy.io.vtk as io_vtk


def test_load_polydata_requires_fury(monkeypatch):
    monkeypatch.setattr(io_vtk, "have_fury", False)

    with pytest.raises(ImportError, match="fury is required"):
        io_vtk.load_polydata("tmp.vtk")


def test_get_polydata_vertices_requires_vtk(monkeypatch):
    monkeypatch.setattr(io_vtk, "have_vtk", False)
    monkeypatch.setattr(io_vtk, "have_numpy_support", False)

    with pytest.raises(ImportError, match="vtk and vtk.util.numpy_support"):
        io_vtk.get_polydata_vertices(object())


def test_get_polydata_vertices_requires_numpy_support(monkeypatch):
    monkeypatch.setattr(io_vtk, "have_vtk", True)
    monkeypatch.setattr(io_vtk, "have_numpy_support", False)

    with pytest.raises(ImportError, match="vtk and vtk.util.numpy_support"):
        io_vtk.get_polydata_vertices(object())


def test_save_vtk_streamlines_requires_fury(monkeypatch):
    monkeypatch.setattr(io_vtk, "have_fury", False)

    with pytest.raises(ImportError, match="fury is required"):
        io_vtk.save_vtk_streamlines([], "tmp.vtk")


def test_load_vtk_streamlines_requires_fury(monkeypatch):
    monkeypatch.setattr(io_vtk, "have_fury", False)

    with pytest.raises(ImportError, match="fury is required"):
        io_vtk.load_vtk_streamlines("tmp.vtk")


def test_get_polydata_triangles_requires_vtk(monkeypatch):
    monkeypatch.setattr(io_vtk, "have_vtk", False)
    monkeypatch.setattr(io_vtk, "have_numpy_support", True)

    with pytest.raises(ImportError, match="vtk and vtk.util.numpy_support"):
        io_vtk.get_polydata_triangles(object())


def test_convert_to_polydata_requires_vtk(monkeypatch):
    monkeypatch.setattr(io_vtk, "have_vtk", False)
    monkeypatch.setattr(io_vtk, "have_numpy_support", True)

    with pytest.raises(ImportError, match="vtk and vtk.util.numpy_support"):
        io_vtk.convert_to_polydata(np.array([[0.0, 0.0, 0.0]]), np.array([[0, 0, 0]]))


def test_save_polydata_legacy_kw_for_fury_ge_2(monkeypatch):
    called = {}

    def _save_polydata(**kwargs):
        called.update(kwargs)

    monkeypatch.setattr(io_vtk, "have_fury", True)
    monkeypatch.setattr(
        io_vtk,
        "fury",
        SimpleNamespace(
            __version__="2.0.0", io=SimpleNamespace(save_polydata=_save_polydata)
        ),
    )

    io_vtk.save_polydata(
        polydata="polydata", file_name="tmp.vtk", legacy_vtk_format=True
    )

    assert called["legacy_vtk_format"] is True


def test_save_polydata_no_legacy_kw_for_fury_lt_2(monkeypatch):
    called = {}

    def _save_polydata(**kwargs):
        called.update(kwargs)

    monkeypatch.setattr(io_vtk, "have_fury", True)
    monkeypatch.setattr(
        io_vtk,
        "fury",
        SimpleNamespace(
            __version__="1.0.0", io=SimpleNamespace(save_polydata=_save_polydata)
        ),
    )

    io_vtk.save_polydata(
        polydata="polydata", file_name="tmp.vtk", legacy_vtk_format=True
    )

    assert "legacy_vtk_format" not in called


def test_convert_to_polydata_data_length_mismatch(monkeypatch):
    def _vtkPoints():
        return SimpleNamespace(SetData=lambda x, deep=True: None)

    def _vtkCellArray():
        return SimpleNamespace(SetCells=lambda x, y: None)

    class MockPolyData:
        def GetNumberOfPoints(self):
            return 3

        def SetPoints(self, points):
            pass

        def SetPolys(self, polys):
            pass

    monkeypatch.setattr(io_vtk, "have_vtk", True)
    monkeypatch.setattr(io_vtk, "have_numpy_support", True)
    monkeypatch.setattr(
        io_vtk,
        "vtk",
        SimpleNamespace(
            vtkPoints=_vtkPoints, vtkCellArray=_vtkCellArray, vtkPolyData=MockPolyData
        ),
    )
    monkeypatch.setattr(
        io_vtk,
        "ns",
        SimpleNamespace(
            numpy_to_vtk=lambda x, deep=True: None,
            numpy_to_vtkIdTypeArray=lambda x, deep=True: None,
        ),
    )

    vertices = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float32)
    triangles = np.array([[0, 1, 2]], dtype=np.int32)
    # Wrong length - should be 3 points but only 2 values
    data_per_point = {"colors": np.array([[1, 0, 0], [0, 1, 0]], dtype=np.float32)}

    with pytest.raises(
        ValueError, match="Array length does not match number of points"
    ):
        io_vtk.convert_to_polydata(vertices, triangles, data_per_point)


def test_get_polydata_triangles_non_triangles(monkeypatch):
    class MockPolyData:
        def GetPolys(self):
            return SimpleNamespace(
                GetData=lambda: np.array([4, 0, 1, 2, 3])
            )  # 4 vertices, not 3

        def GetNumberOfCells(self):
            return 0

    monkeypatch.setattr(io_vtk, "have_vtk", True)
    monkeypatch.setattr(io_vtk, "have_numpy_support", True)
    monkeypatch.setattr(
        io_vtk, "ns", SimpleNamespace(vtk_to_numpy=lambda x: np.array([4, 0, 1, 2, 3]))
    )

    with pytest.raises(ValueError, match="Not all polygons are triangles"):
        io_vtk.get_polydata_triangles(MockPolyData())


def test_numpy_to_vtk_array_uses_deep_and_name(monkeypatch):
    called = {}

    class DummyVtkArray:
        def SetName(self, name):
            called["name"] = name

    def _numpy_to_vtk(arr, *, deep, array_type):
        called["deep"] = deep
        called["array_type"] = array_type
        called["shape"] = arr.shape
        return DummyVtkArray()

    monkeypatch.setattr(io_vtk, "have_vtk", True)
    monkeypatch.setattr(io_vtk, "have_numpy_support", True)
    monkeypatch.setattr(io_vtk, "ns", SimpleNamespace(numpy_to_vtk=_numpy_to_vtk))
    monkeypatch.setattr(io_vtk, "DATATYPE_DICT", {np.dtype("float32"): 99})

    io_vtk._numpy_to_vtk_array(
        np.array([[1.0, 2.0, 3.0]], dtype=np.float32), name="vals", deep=False
    )

    assert called["deep"] is False
    assert called["array_type"] == 99
    assert called["shape"] == (1, 3)
    assert called["name"] == "vals"


def test_numpy_to_vtk_array_respects_dtype_override(monkeypatch):
    called = {}

    def _numpy_to_vtk(arr, *, deep, array_type):
        called["deep"] = deep
        called["array_type"] = array_type
        return SimpleNamespace(SetName=lambda _: None)

    monkeypatch.setattr(io_vtk, "have_vtk", True)
    monkeypatch.setattr(io_vtk, "have_numpy_support", True)
    monkeypatch.setattr(io_vtk, "ns", SimpleNamespace(numpy_to_vtk=_numpy_to_vtk))
    monkeypatch.setattr(
        io_vtk,
        "DATATYPE_DICT",
        {np.dtype("float32"): 99, np.dtype("int16"): 7},
    )

    io_vtk._numpy_to_vtk_array(
        np.array([1.0, 2.0], dtype=np.float32), dtype=np.int16, deep=True
    )

    assert called["deep"] is True
    assert called["array_type"] == 7
