
from __future__ import division, print_function, absolute_import

import numpy as np

from dipy.viz.colormap import line_colors
from dipy.viz.fvtk.util import numpy_to_vtk_points, numpy_to_vtk_colors
from dipy.viz.fvtk.util import set_input, trilinear_interp

# Conditional import machinery for vtk
from dipy.utils.optpkg import optional_package

#import vtk
# Allow import, but disable doctests if we don't have vtk
vtk, have_vtk, setup_module = optional_package('vtk')
colors, have_vtk_colors, _ = optional_package('vtk.util.colors')
numpy_support, have_ns, _ = optional_package('vtk.util.numpy_support')



def streamtube(lines, colors=None, opacity=1, linewidth=0.01, tube_sides=9,
               lod=True, lod_points=10 ** 4, lod_points_size=3,
               spline_subdiv=None, lookup_colormap=None):
    """ Uses streamtubes to visualize polylines

    Parameters
    ----------
    lines : list
        list of N curves represented as 2D ndarrays
    colors : array (N, 3), tuple (3,) or colormap
    opacity : float
    linewidth : float
    tube_sides : int
    lod : bool
        use vtkLODActor rather than vtkActor
    lod_points : int
        number of points to be used when LOD is in effect
    lod_points_size : int
        size of points when lod is in effect
    spline_subdiv : int
        number of splines subdivision to smooth streamtubes
    lookup_colormap : bool
        add a default lookup table to the colormap

    Examples
    --------
    >>> from dipy.viz import fvtk
    >>> r=fvtk.ren()
    >>> lines=[np.random.rand(10, 3), np.random.rand(20, 3)]
    >>> colors=np.random.rand(2, 3)
    >>> c=fvtk.streamtube(lines, colors)
    >>> fvtk.add(r,c)
    >>> #fvtk.show(r)

    Notes
    -----
    Streamtubes can be heavy on GPU when loading many streamlines and therefore,
    you may experience slow rendering time depending on system GPU. A solution
    to this problem is to reduce the number of points in each streamline. In Dipy
    we provide an algorithm that will reduce the number of points on the straighter
    parts of the streamline but keep more points on the curvier parts. This can
    be used in the following way

    from dipy.tracking.distances import approx_polygon_track
    lines = [approx_polygon_track(line, 0.2) for line in lines]
    """
    # Poly data with lines and colors
    poly_data, is_colormap = lines_to_vtk_polydata(lines, colors)
    next_input = poly_data

    # Set Normals
    poly_normals = set_input(vtk.vtkPolyDataNormals(), next_input)
    poly_normals.ComputeCellNormalsOn()
    poly_normals.ComputePointNormalsOn()
    poly_normals.ConsistencyOn()
    poly_normals.AutoOrientNormalsOn()
    poly_normals.Update()
    next_input = poly_normals.GetOutputPort()

    # Spline interpolation
    if (spline_subdiv is not None) and (spline_subdiv > 0) :
        spline_filter = set_input(vtk.vtkSplineFilter(), next_input)
        spline_filter.SetSubdivideToSpecified()
        spline_filter.SetNumberOfSubdivisions(spline_subdiv)
        spline_filter.Update()
        next_input = spline_filter.GetOutputPort()

    # Add thickness to the resulting lines
    tube_filter = set_input(vtk.vtkTubeFilter(), next_input)
    tube_filter.SetNumberOfSides(tube_sides)
    tube_filter.SetRadius(linewidth)
    #tube_filter.SetVaryRadiusToVaryRadiusByScalar()
    tube_filter.CappingOn()
    tube_filter.Update()
    next_input = tube_filter.GetOutputPort()

    # Poly mapper
    poly_mapper = set_input(vtk.vtkPolyDataMapper(), next_input)
    poly_mapper.ScalarVisibilityOn()
    poly_mapper.SetScalarModeToUsePointFieldData()
    poly_mapper.SelectColorArray("Colors")
    poly_mapper.GlobalImmediateModeRenderingOn()
    #poly_mapper.SetColorModeToMapScalars()
    poly_mapper.Update()

    # Color Scale with a lookup table
    if is_colormap:
        if lookup_colormap is None:
            lookup_colormap = colormap_lookup_table()
        poly_mapper.SetLookupTable(lookup_colormap)
        poly_mapper.UseLookupTableScalarRangeOn()
        poly_mapper.Update()

    # Set Actor
    if lod:
        actor = vtk.vtkLODActor()
        actor.SetNumberOfCloudPoints(lod_points)
        actor.GetProperty().SetPointSize(lod_points_size)
    else:
        actor = vtk.vtkActor()

    actor.SetMapper(poly_mapper)
    actor.GetProperty().SetAmbient(0.1)  # .3
    actor.GetProperty().SetDiffuse(0.15)  # .3
    actor.GetProperty().SetSpecular(0.05)  # .3
    actor.GetProperty().SetSpecularPower(6)
    #actor.GetProperty().SetInterpolationToGouraud()
    actor.GetProperty().SetInterpolationToPhong()
    actor.GetProperty().BackfaceCullingOn()
    actor.GetProperty().SetOpacity(opacity)

    return actor


def line(lines, colors=None, opacity=1, linewidth=1,
         spline_subdiv=None, lookup_colormap=None):
    """ Create an actor for one or more lines.

    Parameters
    ------------
    lines :  list of arrays representing lines as 3d points  for example
            lines = [np.random.rand(10,3),np.random.rand(20,3)]
            represents 2 lines the first with 10 points and the second with
            20 points in x,y,z coordinates.
    colors : array, shape (N,3)
            Colormap where every triplet is encoding red, green and blue e.g.

            ::
              r1,g1,b1
              r2,g2,b2
              ...
              rN,gN,bN

            where

            ::
              0=<r<=1,
              0=<g<=1,
              0=<b<=1

    opacity : float, optional
        ``0 <= transparency <= 1``
    linewidth : float, optional
        Line thickness.
    spline_subdiv : int, optional
        number of splines subdivision to smooth streamtubes
    lookup_colormap : bool, optional
        add a default lookup table to the colormap

    Returns
    ----------
    v : vtkActor object
        Line.

    Examples
    ----------
    >>> from dipy.viz import fvtk
    >>> r=fvtk.ren()
    >>> lines=[np.random.rand(10,3), np.random.rand(20,3)]
    >>> colors=np.random.rand(2,3)
    >>> c=fvtk.line(lines, colors)
    >>> fvtk.add(r,c)
    >>> #fvtk.show(r)
    """
    # Poly data with lines and colors
    poly_data, is_colormap = lines_to_vtk_polydata(lines, colors)
    next_input = poly_data

    # use spline interpolation
    if (spline_subdiv is not None) and (spline_subdiv > 0) :
        spline_filter = set_input(vtk.vtkSplineFilter(), next_input)
        spline_filter.SetSubdivideToSpecified()
        spline_filter.SetNumberOfSubdivisions(spline_subdiv)
        spline_filter.Update()
        next_input = spline_filter.GetOutputPort()

    poly_mapper = set_input(vtk.vtkPolyDataMapper(), next_input)
    poly_mapper.ScalarVisibilityOn()
    poly_mapper.SetScalarModeToUsePointFieldData()
    poly_mapper.SelectColorArray("Colors")
    #poly_mapper.SetColorModeToMapScalars()
    poly_mapper.Update()

    # Color Scale with a lookup table
    if is_colormap:

        if lookup_colormap is None:
            lookup_colormap = colormap_lookup_table()

        poly_mapper.SetLookupTable(lookup_colormap)
        poly_mapper.UseLookupTableScalarRangeOn()
        poly_mapper.Update()

    # Set Actor
    actor = vtk.vtkActor()
    actor.SetMapper(poly_mapper)
    actor.GetProperty().SetLineWidth(linewidth)
    actor.GetProperty().SetOpacity(opacity)

    return actor


def lines_to_vtk_polydata(lines, colors=None):
    """ Create a vtkPolyData with lines and colors

    Parameters
    ----------
    lines : list
        list of N curves represented as 2D ndarrays
    colors : array (N, 3), tuple (3,) or colormap


    Returns
    ----------
    poly_data :  VTK polydata
    is_colormap : bool, true if the input color array was a colormap
    """

    #Get the 3d points_array
    points_array = np.vstack(lines)

    nb_lines = len(lines)
    nb_points = len(points_array)

    lines_range = range(nb_lines)

    # Get lines_array in vtk input format
    lines_array = []
    points_per_line = np.zeros([nb_lines],np.int64)
    current_position = 0
    for i in lines_range:
        current_len = len(lines[i])
        points_per_line[i] = current_len

        end_position = current_position + current_len
        lines_array += [current_len]
        lines_array += range(current_position,end_position)
        current_position = end_position

    lines_array = np.array(lines_array)

    # Set Points to vtk array format
    vtk_points = numpy_to_vtk_points(points_array)

    # Set Lines to vtk array format
    vtk_lines = vtk.vtkCellArray()
    vtk_lines.GetData().DeepCopy(numpy_support.numpy_to_vtk(lines_array))
    vtk_lines.SetNumberOfCells(len(lines))

    is_colormap = False
    # Get colors_array (reformat to have colors for each points)
    #           - if/else tested and work in normal simple case
    if colors is None: #set automatic rgb colors
        cols_arr = line_colors(lines)
        colors_mapper = np.repeat(lines_range, points_per_line, axis=0)
        vtk_colors = numpy_to_vtk_colors(255 * cols_arr[colors_mapper])
    else:
        cols_arr = np.asarray(colors)
        if cols_arr.dtype == np.object: # colors is a list of colors
            vtk_colors = numpy_to_vtk_colors(255 * np.vstack(colors))
        else:
            if len(cols_arr) == nb_points: # one colors per points / colormap way
                vtk_colors = numpy_support.numpy_to_vtk(cols_arr, deep=True)
                is_colormap = True

            elif cols_arr.ndim == 1: # the same colors for all points
                vtk_colors = numpy_to_vtk_colors(
                                np.tile(255 * cols_arr, (nb_points, 1)) )


            elif cols_arr.ndim == 2: # map color to each line
                colors_mapper = np.repeat(lines_range, points_per_line, axis=0)
                vtk_colors = numpy_to_vtk_colors(255 * cols_arr[colors_mapper])
            else: # colormap
                # get colors for each vertex
                cols_arr = trilinear_interp(cols_arr, points_array)
                vtk_colors = numpy_support.numpy_to_vtk(cols_arr, deep=True)
                is_colormap = True

    vtk_colors.SetName("Colors")

    #Create the poly_data
    poly_data = vtk.vtkPolyData()
    poly_data.SetPoints(vtk_points)
    poly_data.SetLines(vtk_lines)
    poly_data.GetPointData().SetScalars(vtk_colors)
    return poly_data, is_colormap


def colormap_lookup_table(scale_range=(0,1), hue_range=(0.8,0),
                          saturation_range=(1,1),  value_range=(0.8,0.8)):
    """ Default Lookup table for the colormap
    """
    vtk_lookup_table = vtk.vtkLookupTable()
    vtk_lookup_table.SetRange(scale_range)
    vtk_lookup_table.SetTableRange(scale_range)

    vtk_lookup_table.SetHueRange(hue_range)
    vtk_lookup_table.SetSaturationRange(saturation_range)
    vtk_lookup_table.SetValueRange(value_range)

    vtk_lookup_table.Build()
    return vtk_lookup_table


def scalar_bar(lookup_table):
    """ Default Scalar bar actor for the colormap

    Deepcopy the lookup_table because sometime vtkPolyDataMapper delete it
    """
    lookup_table_copy = vtk.vtkLookupTable()
    lookup_table_copy.DeepCopy(lookup_table)
    scalar_bar = vtk.vtkScalarBarActor()
    scalar_bar.SetLookupTable(lookup_table_copy)
    scalar_bar.SetNumberOfLabels(6)

    return scalar_bar



def plot_stats( values, names=None, colors=np.array([0,0,0,255]),
                lookup_table=None, scalar_bar=None):
    """ Plots statstics in a new windows
    """

    nb_lines = len(values)
    if colors.ndim == 1:
        colors = np.repeat(colors.reshape((1,-1)),nb_lines,axis=0)

    chart =  vtk.vtkChartXY()
    chart.SetShowLegend(True)

    table = vtk.vtkTable()
    table.SetNumberOfRows(values.shape[1])

    for i in range(nb_lines):
        value = values[i]
        vtk_array= numpy_support.numpy_to_vtk(value,deep=1)
        vtk_array.SetName(names[i])
        table.AddColumn(vtk_array)

        color = colors[i] * 255
        if i>0:
            graph_line = chart.AddPlot(vtk.vtkChart.LINE)
            graph_line.SetInput(table, 0, i)
            graph_line.SetColor(color[0],color[1],color[2],color[3])
            #graph_line.GetPen().SetLineType(vtk.vtkPen.SOLID_LINE)
            graph_line.SetWidth(2.0)

            if lookup_table is not None :
                graph_line.SetMarkerStyle(vtk.vtkPlotPoints.CIRCLE)
                graph_line.ScalarVisibilityOn()
                graph_line.SetLookupTable(lookup_table)
                graph_line.SelectColorArray(i)

    #render plot
    view = vtk.vtkContextView()
    view.GetRenderer().SetBackground(colors[0][0], colors[0][1], colors[0][2])
    view.GetRenderWindow().SetSize(400, 300)
    view.GetScene().AddItem(chart)

    if scalar_bar is not None :
        view.GetRenderer().AddActor2D(scalar_bar)

    return view
