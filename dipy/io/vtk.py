import numpy as np

from dipy.testing.decorators import warning_for_keywords
from dipy.tracking.streamline import transform_streamlines
from dipy.utils.optpkg import optional_package

fury, have_fury, setup_module = optional_package(
    "fury", min_version="0.10.0", max_version="1.0.0"
)

if have_fury:
    import fury.io
    import fury.utils
    import vtk
    import vtk.util.numpy_support as ns

    DATATYPE_DICT = {
        np.dtype("int8"): vtk.VTK_CHAR,
        np.dtype("uint8"): vtk.VTK_UNSIGNED_CHAR,
        np.dtype("int16"): vtk.VTK_SHORT,
        np.dtype("uint16"): vtk.VTK_UNSIGNED_SHORT,
        np.dtype("int32"): vtk.VTK_INT,
        np.dtype("uint32"): vtk.VTK_UNSIGNED_INT,
        np.dtype("int64"): vtk.VTK_LONG_LONG,
        np.dtype("uint64"): vtk.VTK_UNSIGNED_LONG_LONG,
        np.dtype("float32"): vtk.VTK_FLOAT,
        np.dtype("float64"): vtk.VTK_DOUBLE,
    }


def load_polydata(file_name):
    """Load a vtk polydata to a supported format file.

    Supported file formats are OBJ, VTK, VTP, FIB, PLY, STL and XML

    Parameters
    ----------
    file_name : string or Path

    Returns
    -------
    output : vtkPolyData

    """
    return fury.io.load_polydata(str(file_name))


@warning_for_keywords()
def save_polydata(
    polydata, file_name, *, binary=False, color_array_name=None, legacy_vtk_format=False
):
    """Save a vtk polydata to a supported format file.

    Save formats can be VTK, VTP, FIB, PLY, STL and XML.
    Color array can be saved as well either by being already in the polydata
    or by passing it as an argument (color_array).

    Parameters
    ----------
    polydata : vtkPolyData
    file_name : string or Path
    """
    # use kwargs for backward compatibility with fury < 2.0
    kwargs = {}
    if fury.__version__.split(".")[0] >= "2":
        kwargs.update({"legacy_vtk_format": legacy_vtk_format})

    fury.io.save_polydata(
        polydata=polydata,
        file_name=str(file_name),
        binary=binary,
        color_array_name=color_array_name,
        **kwargs,
    )


@warning_for_keywords()
def save_vtk_streamlines(streamlines, filename, *, to_lps=True, binary=False):
    """Save streamlines as vtk polydata to a supported format file.

    File formats can be OBJ, VTK, VTP, FIB, PLY, STL and XML

    Parameters
    ----------
    streamlines : list
        list of 2D arrays or ArraySequence
    filename : string or Path
        output filename (.obj, .vtk, .fib, .ply, .stl and .xml)
    to_lps : bool
        Default to True, will follow the vtk file convention for streamlines
        Will be supported by MITKDiffusion and MI-Brain
    binary : bool
        save the file as binary

    """
    if to_lps:
        # ras (mm) to lps (mm)
        to_lps = np.eye(4)
        to_lps[0, 0] = -1
        to_lps[1, 1] = -1
        streamlines = transform_streamlines(streamlines, to_lps)

    polydata, _ = fury.utils.lines_to_vtk_polydata(streamlines)
    save_polydata(polydata, file_name=filename, binary=binary)


@warning_for_keywords()
def load_vtk_streamlines(filename, *, to_lps=True):
    """Load streamlines from vtk polydata.

    Load formats can be VTK, FIB

    Parameters
    ----------
    filename : string or Path
        input filename (.vtk or .fib)
    to_lps : bool
        Default to True, will follow the vtk file convention for streamlines
        Will be supported by MITK-Diffusion and MI-Brain

    Returns
    -------
    output :  list
         list of 2D arrays

    """
    polydata = load_polydata(filename)
    lines = fury.utils.get_polydata_lines(polydata)
    if to_lps:
        to_lps = np.eye(4)
        to_lps[0, 0] = -1
        to_lps[1, 1] = -1
        return transform_streamlines(lines, to_lps)

    return lines


def get_polydata_triangles(polydata, dtype=None):
    """Get triangles from a vtkPolyData object.

    Parameters
    ----------
    polydata : vtkPolyData
        The polydata object from which to extract triangles.
    dtype : data-type, optional
        The desired data type for the output triangles. If None, the default
        data type will be used.
    Returns
    -------
    triangles : numpy.ndarray
        An array of shape (n_triangles, 3) containing the vertex indices
        of the triangles.

    """
    vtk_polys = ns.vtk_to_numpy(polydata.GetPolys().GetData())
    if len(vtk_polys) == 0:
        nbr_cells = polydata.GetNumberOfCells()
        triangles = []
        for i in range(nbr_cells):
            ids = polydata.GetCell(i).GetPointIds()
            for j in range(ids.GetNumberOfIds() - 2):
                triangles.append([ids.GetId(j), ids.GetId(j + 1), ids.GetId(j + 2)])
        triangles = np.array(triangles)
    else:
        if not (vtk_polys[::4] == 3).all():
            raise ValueError("Not all polygons are triangles")
        triangles = np.vstack([vtk_polys[1::4], vtk_polys[2::4], vtk_polys[3::4]]).T
    if dtype is not None:
        return triangles.astype(dtype)
    return triangles


def get_polydata_vertices(polydata, dtype=None):
    """Get vertices from a vtkPolyData object.

    Parameters
    ----------
    polydata : vtkPolyData
        The polydata object from which to extract vertices.
    dtype : data-type, optional
        The desired data type for the output vertices. If None, the default
        data type will be used.
    Returns
    -------
    vertices : numpy.ndarray
        An array of shape (n_vertices, 3) containing the vertex coordinates.

    """
    vertices = ns.vtk_to_numpy(polydata.GetPoints().GetData())
    if dtype is not None:
        return vertices.astype(dtype)
    return vertices


def convert_to_polydata(vertices, triangles, data_per_point=None):
    vtk_points = vtk.vtkPoints()
    vtk_points.SetData(ns.numpy_to_vtk(vertices, deep=True))

    vtk_triangles = np.hstack(np.c_[np.ones(len(triangles)).astype(int) * 3, triangles])
    vtk_triangles = ns.numpy_to_vtkIdTypeArray(vtk_triangles, deep=True)
    vtk_cells = vtk.vtkCellArray()
    vtk_cells.SetCells(len(triangles), vtk_triangles)

    polydata = vtk.vtkPolyData()
    polydata.SetPoints(vtk_points)
    polydata.SetPolys(vtk_cells)

    if data_per_point is not None:
        for name, array in data_per_point.items():
            if len(array) != polydata.GetNumberOfPoints():
                raise ValueError("Array length does not match number of points.")
            vtk_array = _numpy_to_vtk_array(np.array(array), name=name)

            if "normal" in name.lower():
                polydata.GetPointData().SetNormals(vtk_array)
            else:
                polydata.GetPointData().AddArray(vtk_array)

    return polydata


def _numpy_to_vtk_array(array, name=None, dtype=None, deep=True):
    if dtype is not None:
        vtk_dtype = DATATYPE_DICT[np.dtype(dtype)]
    else:
        vtk_dtype = DATATYPE_DICT[np.dtype(array.dtype)]
    vtk_array = ns.numpy_to_vtk(np.asarray(array), deep=True, array_type=vtk_dtype)
    if name is not None:
        vtk_array.SetName(name)
    return vtk_array
