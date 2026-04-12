from types import SimpleNamespace

import numpy as np
import pytest

import dipy.io.vtk as io_vtk


def test_save_polydata_legacy_kw_for_fury_ge_2(monkeypatch):
    called = {}

    def _save_polydata(**kwargs):
        called.update(kwargs)

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
    def _vtk_points():
        return SimpleNamespace(SetData=lambda x, deep=True: None)

    def _vtk_cell_array():
        return SimpleNamespace(SetCells=lambda x, y: None)

    class MockPolyData:
        def GetNumberOfPoints(self):
            return 3

        def SetPoints(self, points):
            pass

        def SetPolys(self, polys):
            pass

    monkeypatch.setattr(
        io_vtk,
        "vtk",
        SimpleNamespace(
            vtkPoints=_vtk_points,
            vtkCellArray=_vtk_cell_array,
            vtkPolyData=MockPolyData,
        ),
        raising=False,
    )
    monkeypatch.setattr(
        io_vtk,
        "ns",
        SimpleNamespace(
            numpy_to_vtk=lambda x, deep=True: None,
            numpy_to_vtkIdTypeArray=lambda x, deep=True: None,
        ),
        raising=False,
    )

    vertices = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float32)
    triangles = np.array([[0, 1, 2]], dtype=np.int32)
    data_per_point = {"colors": np.array([[1, 0, 0], [0, 1, 0]], dtype=np.float32)}

    with pytest.raises(
        ValueError, match="Array length does not match number of points"
    ):
        io_vtk.convert_to_polydata(vertices, triangles, data_per_point)


def test_get_polydata_triangles_non_triangles(monkeypatch):
    class MockPolyData:
        def GetPolys(self):
            return SimpleNamespace(GetData=lambda: np.array([4, 0, 1, 2, 3]))

        def GetNumberOfCells(self):
            return 0

    monkeypatch.setattr(
        io_vtk,
        "ns",
        SimpleNamespace(vtk_to_numpy=lambda x: np.array([4, 0, 1, 2, 3])),
        raising=False,
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

    monkeypatch.setattr(
        io_vtk,
        "ns",
        SimpleNamespace(numpy_to_vtk=_numpy_to_vtk),
        raising=False,
    )
    monkeypatch.setattr(
        io_vtk, "DATATYPE_DICT", {np.dtype("float32"): 99}, raising=False
    )

    io_vtk._numpy_to_vtk_array(
        np.array([[1.0, 2.0, 3.0]], dtype=np.float32), name="vals", deep=False
    )

    assert called["deep"] is True
    assert called["array_type"] == 99
    assert called["shape"] == (1, 3)
    assert called["name"] == "vals"


def test_numpy_to_vtk_array_respects_dtype_override(monkeypatch):
    called = {}

    def _numpy_to_vtk(arr, *, deep, array_type):
        called["deep"] = deep
        called["array_type"] = array_type
        called["dtype"] = arr.dtype
        return SimpleNamespace(SetName=lambda name: None)

    monkeypatch.setattr(
        io_vtk,
        "ns",
        SimpleNamespace(numpy_to_vtk=_numpy_to_vtk),
        raising=False,
    )
    monkeypatch.setattr(
        io_vtk, "DATATYPE_DICT", {np.dtype("float64"): 101}, raising=False
    )

    io_vtk._numpy_to_vtk_array(
        np.array([1, 2, 3], dtype=np.int32), dtype=np.float64, deep=True
    )

    assert called["deep"] is True
    assert called["array_type"] == 101
    assert called["dtype"] == np.int32
