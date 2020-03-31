
import numpy as np

from dipy.tracking.streamline import transform_streamlines

from dipy.utils.optpkg import optional_package
fury, have_fury, setup_module = optional_package('fury')

if have_fury:
    from dipy.viz import utils, vtk
    import vtk.util.numpy_support as ns


def load_polydata(file_name):
    """Load a vtk polydata to a supported format file.

    Supported file formats are OBJ, VTK, FIB, PLY, STL and XML

    Parameters
    ----------
    file_name : string

    Returns
    -------
    output : vtkPolyData
    """
    # get file extension (type) lower case
    file_extension = file_name.split(".")[-1].lower()

    if file_extension == "vtk":
        reader = vtk.vtkPolyDataReader()
    elif file_extension == "fib":
        reader = vtk.vtkPolyDataReader()
    elif file_extension == "ply":
        reader = vtk.vtkPLYReader()
    elif file_extension == "stl":
        reader = vtk.vtkSTLReader()
    elif file_extension == "xml":
        reader = vtk.vtkXMLPolyDataReader()
    elif file_extension == "obj":
        try:  # try to read as a normal obj
            reader = vtk.vtkOBJReader()
        except Exception:  # than try load a MNI obj format
            reader = vtk.vtkMNIObjectReader()
    else:
        raise "polydata " + file_extension + " is not suported"

    reader.SetFileName(file_name)
    reader.Update()
    # print(file_name + " Mesh " + file_extension + " Loaded")
    return reader.GetOutput()


def save_polydata(polydata, file_name, binary=False, color_array_name=None):
    """Save a vtk polydata to a supported format file.

    Save formats can be VTK, FIB, PLY, STL and XML.

    Parameters
    ----------
    polydata : vtkPolyData
    file_name : string

    """
    # get file extension (type)
    file_extension = file_name.split(".")[-1].lower()

    if file_extension == "vtk":
        writer = vtk.vtkPolyDataWriter()
    elif file_extension == "fib":
        writer = vtk.vtkPolyDataWriter()
    elif file_extension == "ply":
        writer = vtk.vtkPLYWriter()
    elif file_extension == "stl":
        writer = vtk.vtkSTLWriter()
    elif file_extension == "xml":
        writer = vtk.vtkXMLPolyDataWriter()
    elif file_extension == "obj":
        raise Exception("mni obj or Wavefront obj ?")
    #    writer = utils.set_input(vtk.vtkMNIObjectWriter(), polydata)

    writer.SetFileName(file_name)
    writer = utils.set_input(writer, polydata)
    if color_array_name is not None:
        writer.SetArrayName(color_array_name)

    if binary:
        writer.SetFileTypeToBinary()
    writer.Update()
    writer.Write()


def save_vtk_streamlines(streamlines, filename,
                         to_lps=True, binary=False):
    """Save streamlines as vtk polydata to a supported format file.

    File formats can be VTK, FIB

    Parameters
    ----------
    streamlines : list
        list of 2D arrays or ArraySequence
    filename : string
        output filename (.vtk or .fib)
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

    # Get the 3d points_array
    nb_lines = len(streamlines)
    points_array = np.vstack(streamlines)

    # Get lines_array in vtk input format
    lines_array = []
    current_position = 0
    for i in range(nb_lines):
        current_len = len(streamlines[i])

        end_position = current_position + current_len
        lines_array.append(current_len)
        lines_array.extend(range(current_position, end_position))
        current_position = end_position

    # Set Points to vtk array format
    vtk_points = vtk.vtkPoints()
    vtk_points.SetData(ns.numpy_to_vtk(points_array.astype(np.float32),
                                       deep=True))

    # Set Lines to vtk array format
    vtk_lines = vtk.vtkCellArray()
    vtk_lines.SetNumberOfCells(nb_lines)
    vtk_lines.GetData().DeepCopy(ns.numpy_to_vtk(np.array(lines_array),
                                                 deep=True))

    # Create the poly_data
    polydata = vtk.vtkPolyData()
    polydata.SetPoints(vtk_points)
    polydata.SetLines(vtk_lines)

    writer = vtk.vtkPolyDataWriter()
    writer.SetFileName(filename)
    writer = utils.set_input(writer, polydata)

    if binary:
        writer.SetFileTypeToBinary()

    writer.Update()
    writer.Write()


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
    reader = vtk.vtkPolyDataReader()
    reader.SetFileName(filename)
    reader.Update()
    polydata = reader.GetOutput()

    lines_vertices = ns.vtk_to_numpy(polydata.GetPoints().GetData())
    lines_idx = ns.vtk_to_numpy(polydata.GetLines().GetData())

    lines = []
    current_idx = 0
    while current_idx < len(lines_idx):
        line_len = lines_idx[current_idx]

        next_idx = current_idx + line_len + 1
        line_range = lines_idx[current_idx + 1: next_idx]

        lines += [lines_vertices[line_range]]
        current_idx = next_idx
    if to_lps:
        to_lps = np.eye(4)
        to_lps[0, 0] = -1
        to_lps[1, 1] = -1
        return transform_streamlines(lines, to_lps)

    return lines
