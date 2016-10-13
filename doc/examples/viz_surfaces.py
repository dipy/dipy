"""
========================================
Visualize surfaces; load/save, get/set and update vtkPolyData.
========================================

import usefull functions and dipy utils
"""
from __future__ import division, print_function, absolute_import
import numpy as np

# and dipy tools
import dipy.io.vtk as io_vtk
import dipy.viz.utils as ut_vtk
from dipy.viz import window

# Conditional import machinery for vtk
# Allow import, but disable doctests if we don't have vtk
from dipy.utils.optpkg import optional_package
vtk, have_vtk, setup_module = optional_package('vtk')

"""
generate a empty vtkPolyData
"""
my_polydata = vtk.vtkPolyData()

"""
generate a cube with vertices and triangles numpy array
"""
my_vetices = np.array([[0.0,  0.0,  0.0],
                       [0.0,  0.0,  1.0],
                       [0.0,  1.0,  0.0],
                       [0.0,  1.0,  1.0],
                       [1.0,  0.0,  0.0],
                       [1.0,  0.0,  1.0],
                       [1.0,  1.0,  0.0],
                       [1.0,  1.0,  1.0]])

my_triangles = np.array([[0,  6,  4],
                         [0,  2,  6],
                         [0,  3,  2],
                         [0,  1,  3],
                         [2,  7,  6],
                         [2,  3,  7],
                         [4,  6,  7],
                         [4,  7,  5],
                         [0,  4,  5],
                         [0,  5,  1],
                         [1,  5,  7],
                         [1,  7,  3]])


"""
set vertices and triangles in poly data
"""
ut_vtk.set_polydata_vertices(my_polydata, my_vetices)
ut_vtk.set_polydata_triangles(my_polydata, my_triangles)

"""
save polydata
"""
file_name = "my_cube.vtk"
io_vtk.save_polydata(my_polydata, file_name)
print("save surface :", file_name)

"""
load polydata
"""
cube_polydata = io_vtk.load_polydata(file_name)

"""
add color based on vertices position
"""
cube_vertices = ut_vtk.get_polydata_vertices(cube_polydata)
colors = cube_vertices * 255
ut_vtk.set_polydata_colors(cube_polydata, colors)

print("new surface colors")
print(ut_vtk.get_polydata_colors(cube_polydata))

"""
Visualize surfaces
"""
# get vtkActor
cube_actor = ut_vtk.get_actor_from_polydata(cube_polydata)

# renderer and scene
renderer = window.Renderer()
renderer.add(cube_actor)
renderer.set_camera(position=(10, 5, 7), focal_point=(0.5, 0.5, 0.5))
renderer.zoom(3)

# display
# window.show(renderer, size=(600, 600), reset_camera=False)
window.record(renderer, out_path='cube.png', size=(600, 600))
