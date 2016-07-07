import vtk
import numpy as np

from dipy.viz import actor, window


def make_cube(edge=1):
    cube_src = vtk.vtkCubeSource()
    cube_src.SetXLength(edge)
    cube_src.SetYLength(edge)
    cube_src.SetZLength(edge)

    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputConnection(cube_src.GetOutputPort())
    cube = vtk.vtkActor()
    cube.SetMapper(mapper)
    return cube


def send_into_orbit(obj, satellites, r=1., show_orbit=True):
    radius = r * (obj.GetLength() / 2.)
    t = np.linspace(0, 2*np.pi)

    orbit_pts = np.array([np.cos(t) * radius,
                          np.sin(t) * radius,
                          np.zeros_like(t)]).astype(np.float32)
    orbit = actor.streamtube([np.ascontiguousarray(orbit_pts.T)],
                             np.array((0., 0., 1.), dtype=np.float32))

    # Disperse satellites evenly on the orbit.
    # Create an assembly to group multiple actors together.
    # This might cause some issues with the picking though.
    orbital_system = vtk.vtkAssembly()
    if show_orbit:
        orbital_system.AddPart(orbit)

    t = np.linspace(0, 2*np.pi, num=len(satellites), endpoint=False)
    satellites_coords = np.array([np.cos(t) * radius,
                                  np.sin(t) * radius,
                                  np.zeros_like(t)]
                                 ).astype(np.float32)

    for coord, satellite in zip(satellites_coords.T, satellites):
        satellite.SetPosition(coord)
        orbital_system.AddPart(satellite)

    return orbital_system


# Create the "Earth" (i.e. object to snap a circular menu onto).
earth = make_cube()

# Create "satellites" (i.e. buttons of the circular menu).
s1 = make_cube(edge=0.25)
s2 = make_cube(edge=0.25)
s3 = make_cube(edge=0.25)

# Position the statellites around the Earth.
orbital_system = send_into_orbit(earth, [s1, s2, s3],  r=1.1)

ren = window.ren()

# Make the orbit always faces the camera.
# Usually vtkFollower works by using
# SetMapper(my_object.GetMapper()) but since our orbital_system
# is a vtkAssembly there is no Mapper. So, we have to manually
# update the transformation matrix of our orbital_system according to
# an empty vtkFollower actor that we explictly add into the assembly.
# By adding the vtkFollower into the assembly its transformation matrix
# get automatically updated so it always faces the camera. Using that
# trasnformation matrix and we can transform our orbital_system accordingly.
dummy_follower = vtk.vtkFollower()
orbital_system.AddPart(dummy_follower)
dummy_follower.SetCamera(ren.GetActiveCamera())
orbital_system.SetUserMatrix(dummy_follower.GetMatrix())

ren.add(earth, orbital_system)

show_m = window.ShowManager(ren, size=(800, 600))
show_m.start()
