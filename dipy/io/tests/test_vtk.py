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


def test_numpy_to_vtk_array_uses_deep_argument(monkeypatch):
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
