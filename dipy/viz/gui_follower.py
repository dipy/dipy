# Conditional import machinery for vtk.
import math

from dipy.utils.optpkg import optional_package

from ipdb import set_trace

# Allow import, but disable doctests if we don't have vtk.
from dipy.viz.gui import UI, TextActor2D

vtk, have_vtk, setup_module = optional_package('vtk')

if have_vtk:
    vtkInteractorStyleUser = vtk.vtkInteractorStyleUser
    version = vtk.vtkVersion.GetVTKSourceVersion().split(' ')[-1]
    major_version = vtk.vtkVersion.GetVTKMajorVersion()
else:
    vtkInteractorStyleUser = object

numpy_support, have_ns, _ = optional_package('vtk.util.numpy_support')


class FollowerMenu(UI):
    def __init__(self, position, diameter, camera, elements):
        super(FollowerMenu, self).__init__()

        self.diameter = diameter
        self.position = position
        self.camera = camera
        self.actor = self.build_assembly()

        self.menu_orbit = FollowerMenuOrbit(diameter=self.diameter, position=self.position)
        self.orbit = self.menu_orbit
        self.actor.AddPart(self.orbit.actor)

        self.add_parts(parts=elements)

        self.ui_list.append(self)

    def add_to_renderer(self, ren):
        ren.add(self.actor)

    @staticmethod
    def find_total_dist(coordinate, coordinate_list):
        distance_aggregate = 0
        for coordinate_element in coordinate_list:
            distance_aggregate += math.sqrt((coordinate_element[0]-coordinate[0])**2 +
                                            (coordinate_element[1]-coordinate[1])**2)
        return distance_aggregate

    def add_parts(self, parts):
        number_of_parts = len(parts)
        angular_difference = 360/number_of_parts
        allotted_coordinates = []
        for i in range(number_of_parts):
            theta = math.radians(angular_difference*(i+1))
            x1 = self.position[0] + ((self.diameter/2)/math.sqrt(1 + math.tan(theta)*math.tan(theta)))
            y1 = self.position[1] + math.tan(theta)*(x1-self.position[0])
            x2 = self.position[0] - ((self.diameter/2)/math.sqrt(1 + math.tan(theta)*math.tan(theta)))
            y2 = self.position[1] + math.tan(theta)*(x2-self.position[0])
            if (self.find_total_dist((x1, y1), allotted_coordinates) >
                    self.find_total_dist((x2, y2), allotted_coordinates)):
                x = x1
                y = y2
            else:
                x = x2
                y = y2
            allotted_coordinates.append((int(x), int(y)))
            parts[i].actor.AddPosition(x, y, self.position[2]+1)
            element_center = parts[i].actor.GetCenter()
            parts[i].actor.AddPosition(x-element_center[0], y-element_center[1],
                                       self.position[2]+1-element_center[2])
            self.actor.AddPart(parts[i].actor)

    def build_assembly(self):
        assembly = vtk.vtkAssembly()
        dummy_follower = vtk.vtkFollower()
        assembly.AddPart(dummy_follower)
        dummy_follower.SetCamera(self.camera)

        # Get assembly transformation matrix.
        M = vtk.vtkTransform()
        M.SetMatrix(assembly.GetMatrix())

        # Get the inverse of the assembly transformation matrix.
        M_inv = vtk.vtkTransform()
        M_inv.SetMatrix(assembly.GetMatrix())
        M_inv.Inverse()

        # Create a transform object that gets updated whenever the input matrix
        # is updated, which is whenever the camera moves.
        dummy_follower_transform = vtk.vtkMatrixToLinearTransform()
        dummy_follower_transform.SetInput(dummy_follower.GetMatrix())

        T = vtk.vtkTransform()
        T.PostMultiply()
        # Bring the assembly to the origin.
        T.Concatenate(M_inv)
        # Change orientation of the assembly.
        T.Concatenate(dummy_follower_transform)
        # Bring the assembly back where it was.
        T.Concatenate(M)

        assembly.SetUserTransform(T)
        return assembly


class FollowerMenuOrbit(UI):
    def __init__(self, position, diameter):
        super(FollowerMenuOrbit, self).__init__()
        self.actor = self.build_actor(center=position, diameter=diameter)

        self.ui_list.append(self)

    def build_actor(self, center, diameter):
        disk = vtk.vtkDiskSource()
        disk.SetInnerRadius(diameter/2)
        disk.SetOuterRadius(diameter/2 + 1)
        disk.SetRadialResolution(10)
        disk.SetCircumferentialResolution(50)
        disk.Update()

        # mapper
        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputConnection(disk.GetOutputPort())

        # actor
        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        # actor.SetCamera(self.camera)

        actor.SetPosition(center[0], center[1], center[2])

        return actor


class CubeButtonFollower(UI):
    def __init__(self, size, color):
        super(CubeButtonFollower, self).__init__()
        self.actor = self.build_actor(size=size, color=color)
        self.element_type = "cube"

        self.ui_list.append(self)

    def build_actor(self, size, color):
        cube = vtk.vtkCubeSource()
        cube.SetXLength(size[0])
        cube.SetYLength(size[1])
        cube.SetZLength(size[2])
        cubeMapper = vtk.vtkPolyDataMapper()
        cubeMapper.SetInputConnection(cube.GetOutputPort())
        cubeActor = vtk.vtkActor()
        cubeActor.SetMapper(cubeMapper)
        if color is not None:
            cubeActor.GetProperty().SetColor(color)
        cubeActor.SetPosition(0, 0, 0)
        return cubeActor

    def add_callback(self, event_type, callback):
        """ Adds events to the actor

        Parameters
        ----------
        event_type: event code
        callback: callback function
        """
        self.actor.AddObserver(event_type, callback)


class ButtonFollower(UI):
    """ Currently implements a 2D overlay button and is of type vtkTexturedActor2D.

    """

    def __init__(self, icon_fnames):
        super(ButtonFollower, self).__init__()
        self.icons = self.build_icons(icon_fnames)
        self.icon_names = list(self.icons.keys())
        self.current_icon_id = 0
        self.current_icon_name = self.icon_names[self.current_icon_id]
        self.actor = self.build_actor(self.icons[self.current_icon_name])
        self.element_type = "button"

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
            png.Update()

            # Convert the image to a polydata
            imageDataGeometryFilter = vtk.vtkImageDataGeometryFilter()
            imageDataGeometryFilter.SetInputConnection(png.GetOutputPort())
            imageDataGeometryFilter.Update()

            icons[icon_name] = imageDataGeometryFilter

        return icons

    def build_actor(self, icon, center=None):
        """ Return an image as a 2D actor with a specific position

        Parameters
        ----------
        icon : imageDataGeometryFilter
        center : a two tuple

        Returns
        -------
        button : vtkActor
        """

        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputConnection(icon.GetOutputPort())

        button = vtk.vtkActor()
        button.SetMapper(mapper)
        # button.SetCamera(self.camera)

        button.SetPosition((0, 0, 0))

        if center is not None:
            button.SetCenter(*center)

        return button

    def add_callback(self, event_type, callback):
        """ Adds events to the actor

        Parameters
        ----------
        event_type: event code
        callback: callback function
        """
        self.actor.AddObserver(event_type, callback)

    def set_icon(self, icon):
        """ Modifies the icon used by the vtkTexturedActor2D

        Parameters
        ----------
        icon : imageDataGeometryFilter
        """
        self.actor.GetMapper().SetInputConnection(icon.GetOutputPort())

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


class TextFollower(UI):

    def __init__(self, text, color):
        super(TextFollower, self).__init__()

        self.actor = self.build_actor(text=text, color=color)

        self.ui_list.append(self)

    def build_actor(self, text, color):
        actor_text = vtk.vtkVectorText()
        actor_text.SetText(text)

        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputConnection(actor_text.GetOutputPort())
        actor = vtk.vtkFollower()
        actor.SetMapper(mapper)

        actor.SetScale(5, 5, 5)

        actor.GetProperty().SetColor(color)

        actor.SetPosition(0, 0, 0)

        return actor

    def add_callback(self, event_type, callback):
        """ Adds events to the actor

        Parameters
        ----------
        event_type: event code
        callback: callback function
        """
        self.actor.AddObserver(event_type, callback)
