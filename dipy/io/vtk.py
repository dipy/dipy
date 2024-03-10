import numpy as np
from dipy.tracking.streamline import transform_streamlines
from dipy.utils.optpkg import optional_package
fury, have_fury, setup_module = optional_package('fury', min_version="0.10.0")

if have_fury:
    import fury.utils
    import fury.io


def load_polydata(file_name):
    """Load a vtk polydata to a supported format file.

    Supported file formats are OBJ, VTK, VTP, FIB, PLY, STL and XML

    Parameters
    ----------
    file_name : string

    Returns
    -------
    output : vtkPolyData

    """
    return fury.io.load_polydata(file_name)


def save_polydata(polydata, file_name, binary=False, color_array_name=None):
    """Save a vtk polydata to a supported format file.

    Save formats can be VTK, VTP, FIB, PLY, STL and XML.

    Parameters
    ----------
    polydata : vtkPolyData
    file_name : string

    """
    fury.io.save_polydata(polydata=polydata, file_name=file_name,
                          binary=binary, color_array_name=color_array_name)


def save_vtk_streamlines(streamlines, filename,
                         to_lps=True, binary=False):
    """Save streamlines as vtk polydata to a supported format file.

    File formats can be OBJ, VTK, VTP, FIB, PLY, STL and XML

    Parameters
    ----------
    streamlines : list
        list of 2D arrays or ArraySequence
    filename : string
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


def load_vtk_streamlines(filename, to_lps=True):
    """Load streamlines from vtk polydata.

    Load formats can be VTK, FIB

    Parameters
    ----------
    filename : string
        input filename (.vtk or .fib)
    to_lps : bool
        Default to True, will follow the vtk file convention for streamlines
        Will be supported by MITKDiffusion and MI-Brain

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
