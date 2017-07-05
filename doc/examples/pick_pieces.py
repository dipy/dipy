#!/usr/bin/python

# This example creates a board with a flat surface at z=0
# and a number of pieces that may interactively be moved
# around the board by the mouse.
#
# Dov Grobgeld <dov.grobgeld@gmail.com>
# This example is released under the same BSD licence as vtk.

import vtk

# Inherit an interactor and override the button events in order
# to be able to pick up pieces from the board.
class MouseInteractor(vtk.vtkInteractorStyleTrackballCamera):
  def __init__(self, renderer, renWin, pieces):
    # The following three events are involved in the pieces interaction.
    self.AddObserver('RightButtonPressEvent', self.OnRightButtonDown)
    self.AddObserver('RightButtonReleaseEvent', self.OnRightButtonRelease)
    self.AddObserver('MouseMoveEvent', self.OnMouseMove)

    # Remember data we need for the interaction
    self.renderer = renderer
    self.chosenPiece = None
    self.renWin = renWin
    self.pieces = pieces

  def DisplayToWorld(self, XYZ):
    """Translate a display XYZ coordinate to a world XYZ coordinate"""
    worldPt = [0, 0, 0, 0]
    vtk.vtkInteractorObserver.ComputeDisplayToWorld(self.renderer,
                                                    XYZ[0], XYZ[1], XYZ[2],
                                                    worldPt)
    return worldPt[0]/worldPt[3], worldPt[1]/worldPt[3], worldPt[2]/worldPt[3]

  def WorldZToDisplayZ(self, displayXY, worldZ=0):
    """Given a display coordinate displayXY and a worldZ coordinate,
    return the corresponding displayZ coordinate"""
    wzNear = self.DisplayToWorld(list(displayXY) + [0])[2]
    wzFar = self.DisplayToWorld(list(displayXY) + [1])[2]
    return (worldZ-wzNear)/(wzFar-wzNear)

  def OnRightButtonRelease(self, obj, eventType):
    # When the right button is released, we stop the interaction
    self.chosenPiece = None

    # Call parent interaction
    vtk.vtkInteractorStyleTrackballCamera.OnRightButtonUp(self)

  def OnRightButtonDown(self, obj, eventType):
    # The rightbutton is used to pick up the piece.

    # Get the display mouse event position
    clickPos = self.GetInteractor().GetEventPosition()

    # Use a picker to see which actor is under the mouse
    picker = vtk.vtkPropPicker()
    picker.Pick(clickPos[0], clickPos[1], 0, self.renderer)
    actor = picker.GetActor()

    # Is this a piece that we should interact with?
    if actor in self.pieces:
      # Yes! Remember it.
      self.chosenPiece = actor

      # Get the intersection of the click pos in our board plane.
      mouseDisplayZ = self.WorldZToDisplayZ(clickPos, worldZ=0)

      # Get the board xy coordinate of the picked point
      self.worldPickXY = self.DisplayToWorld(list(clickPos) + [mouseDisplayZ])[0:2]
    # Call parent interaction
    vtk.vtkInteractorStyleTrackballCamera.OnRightButtonDown(self)

  def OnMouseMove(self, obj, eventType):
    # Translate a choosen piece
    if self.chosenPiece is not None:
      # Redo the same calculation as during OnRightButtonDown
      mousePos = self.GetInteractor().GetEventPosition()
      mouseDisplayZ = self.WorldZToDisplayZ(mousePos)
      worldMouseXY = self.DisplayToWorld(list(mousePos) + [mouseDisplayZ])[0:2]

      # Calculate the xy movement
      dx = worldMouseXY[0]-self.worldPickXY[0]
      dy = worldMouseXY[1]-self.worldPickXY[1]

      # Remember the new reference coordinate
      self.worldPickXY = worldMouseXY

      # Shift the choosen piece in the xy plane
      x, y, z = self.chosenPiece.GetPosition()
      self.chosenPiece.SetPosition(x+dx, y+dy, z)

      # Request a redraw
      self.renWin.Render()
    else:
      vtk.vtkInteractorStyleTrackballCamera.OnMouseMove(self)

# Some pieces that we'll interact with.
def createConeActor(color=None,
                    center=None,
                    height=0.3,
                    radius=0.15):
  cone = vtk.vtkConeSource()
  cone.SetResolution(128)
  cone.SetDirection(0,0,1)
  cone.SetRadius(radius)
  cone.SetHeight(height)
  if not center is None:
    cone.SetCenter(*center)
  else:
    cone.SetCenter(0,0,height/2)
  coneMapper = vtk.vtkPolyDataMapper()
  coneMapper.SetInputConnection(cone.GetOutputPort())
  coneActor = vtk.vtkActor()
  coneActor.SetMapper(coneMapper)
  if color is not None:
    coneActor.GetProperty().SetColor(color)
  return coneActor

def createCubeActor(color=None,
                    size=(0.2,0.2,0.2),
                    center=None):
  cube = vtk.vtkCubeSource()
  cube.SetXLength(size[0])
  cube.SetYLength(size[1])
  cube.SetZLength(size[2])
  if center is not None:
    cube.SetCenter(*center)
  cubeMapper = vtk.vtkPolyDataMapper()
  cubeMapper.SetInputConnection(cube.GetOutputPort())
  cubeActor = vtk.vtkActor()
  cubeActor.SetMapper(cubeMapper)
  if color is not None:
    cubeActor.GetProperty().SetColor(color)
  return cubeActor

def createCylinderActor(color=None,
                        radius=0.25,
                        height=0.3,
                        center=None,
                        ):
  cylinder = vtk.vtkCylinderSource()
  cylinder.SetResolution(128)
  cylinder.SetRadius(radius)
  cylinder.SetHeight(height)
  cylinderMapper = vtk.vtkPolyDataMapper()
  polyDataFilter = vtk.vtkTransformPolyDataFilter()
  polyDataFilter.SetInputConnection(cylinder.GetOutputPort())

  transform = vtk.vtkTransform()
  if center is not None:
    transform.Translate(*center)
  else:
    transform.Translate(0,0,height/2)
  transform.RotateWXYZ(90, 1,0,0)
  polyDataFilter.SetTransform(transform)

  cylinderMapper.SetInputConnection(polyDataFilter.GetOutputPort())
  cylinderActor = vtk.vtkActor()
  cylinderActor.SetMapper(cylinderMapper)
  if color is not None:
    cylinderActor.GetProperty().SetColor(color)
  return cylinderActor

# Create the scene
ren = vtk.vtkRenderer()
renWin = vtk.vtkRenderWindow()
renWin.AddRenderer(ren)
renWin.SetSize(600, 600)

ren.AddActor(createCubeActor(size=(1,1,0.1),
                             center=(0,0,-0.05),
                             color=(0.5,0.5,0.5)))


# Our pieces
pieces = [
  createCubeActor(size=(0.2,0.2,0.2),
                  center=(-0.33,-0.33,0.1),
                  color=(1,0,0)),
  createConeActor(color=(0,1,0),
                  center=(0.33,0.33,0.2),
                  height=0.4),
  createCylinderActor(center=(0,0,0.125),
                      height=0.25,
                      radius=0.125,
                      color=(0,0,1))
]

# assign our actor to the renderer
for actor in pieces:
  ren.AddActor(actor)

# enable user interface interactor
iren = vtk.vtkRenderWindowInteractor()
iren.SetRenderWindow(renWin)
inStyle = MouseInteractor(ren, renWin, pieces)
iren.SetInteractorStyle(inStyle)

ren.SetBackground(0.1, 0.2, 0.4)
renWin.Render()
iren.Start()
