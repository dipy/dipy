from __future__ import division, print_function, absolute_import

import numpy as np
import vtk.util.numpy_support as ns

from dipy.viz.colormap import colormap_lookup_table
from dipy.viz.utils import lines_to_vtk_polydata
from dipy.viz.utils import set_input

# Conditional import machinery for vtk
from dipy.utils.optpkg import optional_package

# Allow import, but disable doctests if we don't have vtk
vtk, have_vtk, setup_module = optional_package('vtk')
colors, have_vtk_colors, _ = optional_package('vtk.util.colors')
ns, have_numpy_support, _ = optional_package('vtk.util.numpy_support')

if have_vtk:

    version = vtk.vtkVersion.GetVTKSourceVersion().split(' ')[-1]
    major_version = vtk.vtkVersion.GetVTKMajorVersion()



# Input functions (load)
def load_polydata(file_name):
    """ Load a vtk polydata to a supported format file

    Parameters
    ----------
    file_name : string
    
    
    Returns
    -------
    output : vtkPolyData
    """
    # get file extension (type)
    file_extension = file_name.split(".")[-1].lower()

    if file_extension == "vtk":
        reader = vtk.vtkPolyDataReader()
    if file_extension == "fib":
        reader = vtk.vtkPolyDataReader()
    elif file_extension == "ply":
        reader = vtk.vtkPLYReader()
    elif file_extension == "stl":
        reader = vtk.vtkSTLReader()
    elif file_extension == "xml":
        reader = vtk.vtkXMLPolyDataReader()
    elif file_extension == "obj":
        #todo test
        try:  # try to read as a normal obj
            reader = vtk.vtkOBJReader()
        except:  # than try load a MNI obj format
            reader = vtk.vtkMNIObjectReader()
    else:
        raise "polydata " + file_extension + " is not suported"

    reader.SetFileName(file_name)
    reader.Update()
    print(file_name + " Mesh " + file_extension + "Loaded")
    return reader.GetOutput()


# Output functions (save)
def save_polydata(polydata, file_name, binary=False, color_array_name=None):
    """ Save a vtk polydata to a supported format file

    Parameters
    ----------
    polydata : vtkPolyData
    file_name : string
    """
    
    # get file extension (type)
    file_extension = file_name.split(".")[-1].lower()

    # todo better generic load
    # todo test all
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
        raise "mni obj or Wavefront obj ?"
    #    writer = set_input(vtk.vtkMNIObjectWriter(), polydata)

    writer.SetFileName(file_name)
    writer = set_input(writer, polydata)
    if color_array_name is not None:
        writer.SetArrayName(color_array_name);
    
    if binary :
        writer.SetFileTypeToBinary()
    writer.Update()
    writer.Write()


# PolyData to numpy
def get_polydata_lines(line_polydata):
    """ vtk polydata to a list of lines ndarrays

    Parameters
    ----------
    line_polydata : vtkPolyData
    
    
    Returns
    -------
    lines : list of N curves represented as 2D ndarrays
    """
    lines_vertices = ns.vtk_to_numpy(line_polydata.GetPoints().GetData())
    lines_idx = ns.vtk_to_numpy(line_polydata.GetLines().GetData())
    
    lines = []
    current_idx = 0
    while current_idx < len(lines_idx):
        line_len = lines_idx[current_idx]
        #print line_len
        next_idx = current_idx + line_len + 1 
        line_range = lines_idx[current_idx + 1: next_idx]
        #print line_range
        lines += [lines_vertices[line_range]]
        current_idx = next_idx
    return lines


def get_polydata_triangles(polydata):
    """ get triangles (ndarrays Nx3 int) from a vtk polydata

    Parameters
    ----------
    polydata : vtkPolyData
    
    
    Returns
    -------
    output : triangles, represented as 2D ndarrays (Nx3)
    """
    vtk_polys = ns.vtk_to_numpy(polydata.get_polydata().GetPolys().GetData())
    assert((vtk_polys[::4] == 3).all())  # test if its really triangles
    return np.vstack([vtk_polys[1::4], vtk_polys[2::4], vtk_polys[3::4]]).T

def get_polydata_vertices(polydata):
    """ get points (ndarrays Nx3 int) from a vtk polydata

    Parameters
    ----------
    polydata : vtkPolyData
    
    
    Returns
    -------
    output : points, represented as 2D ndarrays (Nx3)
    """
    return ns.vtk_to_numpy(polydata.get_polydata().GetPoints().GetData())


def get_polydata_normals(polydata):
    """ get points (ndarrays Nx3 int) from a vtk polydata

    Parameters
    ----------
    polydata : vtkPolyData
    
    
    Returns
    -------
    output : normals, represented as 2D ndarrays (Nx3)
             None (if no normals in the tk polydata)
    """
    vtk_normals = polydata.get_polydata().GetPointData().GetNormals()
    if vtk_normals is None:
        return None
    else:
        return ns.vtk_to_numpy(vtk_normals)


def get_polydata_colors(self):
    """ get points (ndarrays Nx3 int) from a vtk polydata

    Parameters
    ----------
    polydata : vtkPolyData
    
    
    Returns
    -------
    output : colors, represented as 2D ndarrays (Nx3)
             None (if no normals in the tk polydata)
    """
    vtk_colors = self.get_polydata().GetPointData().GetScalars()
    if vtk_colors is None:
        return None
    else:
        return ns.vtk_to_numpy(vtk_colors)


# PolyData to numpy
def set_polydata_triangles(polydata, triangles):
    vtk_triangles = np.hstack(np.c_[np.ones(len(triangles)).astype(np.int) * 3, triangles])
    vtk_triangles = ns.numpy_to_vtkIdTypeArray(vtk_triangles, deep=True)
    vtk_cells = vtk.vtkCellArray()
    vtk_cells.SetCells(len(triangles), vtk_triangles)
    polydata.SetPolys(vtk_cells)
    return polydata

def set_polydata_vertices(polydata, vertices):
    vtk_points = vtk.vtkPoints()
    vtk_points.SetData(ns.numpy_to_vtk(vertices, deep=True))
    polydata.SetPoints(vtk_points)
    return polydata

def set_polydata_normals(polydata, normals):
    vtk_normals = ns.numpy_to_vtk(normals, deep=True)
    polydata.GetPointData().SetNormals(vtk_normals)
    return polydata

def set_polydata_colors(polydata, colors):
    # Colors are [0,255] RGB for each points
    vtk_colors = ns.numpy_to_vtk(colors, deep=True, array_type=vtk.VTK_UNSIGNED_CHAR)
    vtk_colors.SetNumberOfComponents(3)
    vtk_colors.SetName("RGB")
    polydata.GetPointData().SetScalars(vtk_colors)
    return polydata


# update
def update_polydata_normals(polydata):
    normals_gen = set_input(polydata, vtk.vtkPolyDataNormals())
    normals_gen.ComputePointNormalsOn()
    normals_gen.ComputeCellNormalsOn()
    normals_gen.SplittingOff()
    #normals_gen.FlipNormalsOn()
    #normals_gen.ConsistencyOn()
    #normals_gen.AutoOrientNormalsOn()
    normals_gen.Update()

    vtk_normals = normals_gen.GetOutput().GetPointData().GetNormals()
    polydata.GetPointData().SetNormals(vtk_normals)
    return polydata

def update(polydata):
    if vtk.VTK_MAJOR_VERSION <= 5:
        polydata.Update()
        
    return polydata

