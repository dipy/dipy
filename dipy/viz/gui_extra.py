# Conditional import machinery for vtk.
import math

from dipy.utils.optpkg import optional_package

from ipdb import set_trace

from dipy.viz.gui import UI, TextActor2D

# Allow import, but disable doctests if we don't have vtk.
vtk, have_vtk, setup_module = optional_package('vtk')

if have_vtk:
    vtkInteractorStyleUser = vtk.vtkInteractorStyleUser
    version = vtk.vtkVersion.GetVTKSourceVersion().split(' ')[-1]
    major_version = vtk.vtkVersion.GetVTKMajorVersion()
else:
    vtkInteractorStyleUser = object

numpy_support, have_ns, _ = optional_package('vtk.util.numpy_support')


class Button3D(UI):
    """ A 3D button

    """

    def __init__(self, icon_fnames):
        super(Button3D, self).__init__()
        self.icons = self.build_icons(icon_fnames)
        self.icon_names = list(self.icons.keys())
        self.current_icon_id = 0
        self.current_icon_name = self.icon_names[self.current_icon_id]
        self.actor = self.build_actor(self.icons[self.current_icon_name], (100, 100))

        self.ui_list.append(self)

    def build_icons(self, icon_fnames):
        """ Converts filenames to vtkImageDataGeometryFilters
        A peprocessing step to prevent re-read of filenames during every state change

        Parameters
        ----------
        icon_fnames : A list of filenames

        Returns
        -------
        icons : A list of corresponding vtkImageDataGeometryFilters
        """
        icons = {}
        for icon_name, icon_fname in icon_fnames.items():
            png = vtk.vtkPNGReader()
            png.SetFileName(icon_fname)
            # png.Update()

            # Convert the image to a polydata
            # imageDataGeometryFilter = vtk.vtkImageDataGeometryFilter()
            # imageDataGeometryFilter.SetInputConnection(png.GetOutputPort())
            # imageDataGeometryFilter.Update()

            icons[icon_name] = png

        return icons

    def build_actor(self, icon, size):
        """ Return an image as a 2D actor with a specific position

        Parameters
        ----------
        icon : png
        size : a two tuple

        Returns
        -------
        button : vtkTexturedActor2D
        """

        points = vtk.vtkPoints()
        points.InsertNextPoint(0, 0, 0)
        points.InsertNextPoint(size[0], 0, 0)
        points.InsertNextPoint(size[0], size[1], 0)
        points.InsertNextPoint(0, size[1], 0)

        # Create the polygon
        polygon = vtk.vtkPolygon()
        polygon.GetPointIds().SetNumberOfIds(4)  # make a quad
        polygon.GetPointIds().SetId(0, 0)
        polygon.GetPointIds().SetId(1, 1)
        polygon.GetPointIds().SetId(2, 2)
        polygon.GetPointIds().SetId(3, 3)

        # Add the polygon to a list of polygons
        polygons = vtk.vtkCellArray()
        polygons.InsertNextCell(polygon)

        # Create a PolyData
        polygonPolyData = vtk.vtkPolyData()
        polygonPolyData.SetPoints(points)
        polygonPolyData.SetPolys(polygons)

        texture_coordinates = vtk.vtkFloatArray()
        texture_coordinates.SetNumberOfComponents(3)
        texture_coordinates.SetName("TextureCoordinates")

        texture_coordinates.InsertNextTuple((0.0, 0.0, 0.0))
        texture_coordinates.InsertNextTuple((1.0, 0.0, 0.0))
        texture_coordinates.InsertNextTuple((1.0, 1.0, 0.0))
        texture_coordinates.InsertNextTuple((0.0, 1.0, 0.0))

        polygonPolyData.GetPointData().SetTCoords(texture_coordinates)

        texture = vtk.vtkTexture()
        if vtk.VTK_MAJOR_VERSION <= 5:
            texture.SetInput(icon.GetOutput())
        else:
            texture.SetInputConnection(icon.GetOutputPort())

        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputData(polygonPolyData)

        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        actor.SetTexture(texture)

        return actor

    def add_to_renderer(self, ren):
        ren.add(self.actor)

    def add_callback(self, event_type, callback):
        """ Adds events to button actor

        Parameters
        ----------
        event_type: event code
        callback: callback function
        """
        super(Button3D, self).add_callback(self.actor, event_type, callback)

    def set_icon(self, icon):
        """ Modifies the icon used by the vtkTexturedActor2D

        Parameters
        ----------
        icon : imageDataGeometryFilter
        """
        texture = vtk.vtkTexture()
        if vtk.VTK_MAJOR_VERSION <= 5:
            texture.SetInput(icon.GetOutput())
        else:
            texture.SetInputConnection(icon.GetOutputPort())
        self.actor.SetTexture(texture)

    def next_icon_name(self):
        self.current_icon_id += 1
        if self.current_icon_id == len(self.icons):
            self.current_icon_id = 0
        self.current_icon_name = self.icon_names[self.current_icon_id]

    def next_icon(self):
        """ Increments the state of the Button
            Also changes the icon
        """
        self.next_icon_name()
        self.set_icon(self.icons[self.current_icon_name])
