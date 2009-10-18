''' Simple visualization functions using VTK. Fos means light in Greek.
    
'''

try:
    import vtk       
except ImportError:
    raise ImportError('VTK is not installed.')
    
try:
    import numpy as np
except ImportError:
    raise ImportError('Numpy is not installed.')

import types    

def ren():
    ''' Create a renderer
    
    Returns
    --------
    a vtkRenderer() object    
    
    Examples
    ---------
    >>> r=ren()    
    >>> lines=[np.random.rand(10,3)]        
    >>> c=line(lines)    
    >>> add(r,c)
    >>> show(r)    
    '''
    return vtk.vtkRenderer()


def add(ren,a):
    ''' Add a specific actor    
    '''
    ren.AddActor(a)

def rm(ren,a):
    ''' Remove a specific actor    
    '''
    ren.RemoveActor(a)

def clear(ren):
    ''' Remove all actors from the renderer 
    '''
    ren.RemoveAllViewProps()

def rm_all(ren):
    ''' Remove all actors from the renderer 
    '''
    clear(ren)


def _arrow(pos=(0,0,0),color=(1,0,0),scale=(1,1,1),opacity=1):
    
    arrow = vtk.vtkArrowSource()
    #arrow.SetTipLength(length)
    
    arrowm = vtk.vtkPolyDataMapper()
    arrowm.SetInput(arrow.GetOutput())
    
    arrowa= vtk.vtkActor()
    arrowa.SetMapper(arrowm)
    
    arrowa.GetProperty().SetColor(color)
    arrowa.GetProperty().SetOpacity(opacity)
    arrowa.SetScale(scale)
    
    return arrowa
    
def axes(scale=(1,1,1),colorx=(1,0,0),colory=(0,1,0),colorz=(0,0,1),opacity=1):
    ''' Create an actor with the coordinate system axes where  red = x, green = y, blue =z.
    '''
    
    arrowx=_arrow(color=colorx,scale=scale,opacity=opacity)
    arrowy=_arrow(color=colory,scale=scale,opacity=opacity)
    arrowz=_arrow(color=colorz,scale=scale,opacity=opacity)
    
    arrowy.RotateZ(90)
    arrowz.RotateY(-90)
   
    #ass=vtk.vtkPropAssembly()
    ass=vtk.vtkAssembly()
    ass.AddPart(arrowx)
    ass.AddPart(arrowy)
    ass.AddPart(arrowz)
           
    return ass

def line(lines,colors=None,opacity=1,linewidth=1):
    ''' Create an actor for one or more lines.    
    
    Parameters
    ----------
    ren : list of numpy arrays representing lines as 3d points
    colors : one dimensional array or list whith the color of every line. 0<= color <=1
    opacity : 0<=transparency <=1
    linewidth : (r,g,b) and RGB tuple
    
    Returns
    ----------
    vtkActor object
    
    Examples
    --------    
    >>> r=ren()    
    >>> lines=[np.random.rand(10,3),np.random.rand(20,3)]    
    >>> colors=[0.2,0.8]
    >>> c=line(lines,colors)    
    >>> add(r,c)
    >>> l=label(r)
    >>> add(r,l)
    >>> show(r)
    '''    
    if not isinstance(lines,types.ListType):
        lines=[lines]    
        
    points= vtk.vtkPoints()
    lines_=vtk.vtkCellArray()
    linescalars=vtk.vtkFloatArray()
   
    lookuptable=vtk.vtkLookupTable()
    
    scalar=1.0
    curPointID=0
    scalarmin=0.0
    scalarmax=1.0
           
    m=(0.0,0.0,0.0)
    n=(1.0,0.0,0.0)
    
    if colors!=None:
        colors=colors*np.ones(len(lines))
        lit=iter(colors)
        
    else:
        colors=np.random.rand(len(lines))
        lit=iter(colors)
    
    for Line in lines:
        
        inw=True
        mit=iter(Line)
        nit=iter(Line)
        nit.next()
        
        scalar=lit.next()
        
        while(inw):
            
            try:
                m=mit.next() 
                n=nit.next()
                
                #scalar=sp.rand(1)
                
                linescalars.SetNumberOfComponents(1)
                points.InsertNextPoint(m)
                linescalars.InsertNextTuple1(scalar)
            
                points.InsertNextPoint(n)
                linescalars.InsertNextTuple1(scalar)
                
                lines_.InsertNextCell(2)
                lines_.InsertCellPoint(curPointID)
                lines_.InsertCellPoint(curPointID+1)
                
                curPointID+=2
            except StopIteration:
                break

    polydata = vtk.vtkPolyData()
    polydata.SetPoints(points)
    polydata.SetLines(lines_)
    polydata.GetPointData().SetScalars(linescalars)
    
    
    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInput(polydata)
    mapper.SetLookupTable(lookuptable)
    
    mapper.SetColorModeToMapScalars()
    mapper.SetScalarRange(scalarmin,scalarmax)
    mapper.SetScalarModeToUsePointData()
    
    actor=vtk.vtkActor()
    actor.SetMapper(mapper)
    actor.GetProperty().SetLineWidth(linewidth)
    actor.GetProperty().SetOpacity(opacity)
    
    return actor

def dots(points,color=(1,0,0),opacity=1):
  '''
  Adds one or more 3d dots(points) returns one actor handling all the points
  '''

  if points.ndim==2:
    points_no=points.shape[0]
  else:
    points_no=1
    
  polyVertexPoints = vtk.vtkPoints()
  polyVertexPoints.SetNumberOfPoints(points_no)
  aPolyVertex = vtk.vtkPolyVertex()
  aPolyVertex.GetPointIds().SetNumberOfIds(points_no)
  
  cnt=0
  if points.ndim>1:
        for point in points:
            polyVertexPoints.InsertPoint(cnt, point[0], point[1], point[2])
            aPolyVertex.GetPointIds().SetId(cnt, cnt)
            cnt+=1
  else:
        polyVertexPoints.InsertPoint(cnt, points[0], points[1], points[2])
        aPolyVertex.GetPointIds().SetId(cnt, cnt)
        cnt+=1
    

  aPolyVertexGrid = vtk.vtkUnstructuredGrid()
  aPolyVertexGrid.Allocate(1, 1)
  aPolyVertexGrid.InsertNextCell(aPolyVertex.GetCellType(), aPolyVertex.GetPointIds())

  aPolyVertexGrid.SetPoints(polyVertexPoints)
  aPolyVertexMapper = vtk.vtkDataSetMapper()
  aPolyVertexMapper.SetInput(aPolyVertexGrid)
  aPolyVertexActor = vtk.vtkActor()
  aPolyVertexActor.SetMapper(aPolyVertexMapper)

  aPolyVertexActor.GetProperty().SetColor(color)
  aPolyVertexActor.GetProperty().SetOpacity(opacity)

  return aPolyVertexActor

def point(points,color=(1,0,0),opacity=1):
    return dots(points,color=(1,0,0),opacity=1)

def sphere(position=(0,0,0),radius=0.5,thetares=8,phires=8,color=(0,0,1),opacity=1,tessel=0):
    
    sphere = vtk.vtkSphereSource()
    sphere.SetRadius(radius)
    sphere.SetLatLongTessellation(tessel)
   
    sphere.SetThetaResolution(thetares)
    sphere.SetPhiResolution(phires)
    
    spherem = vtk.vtkPolyDataMapper()
    spherem.SetInput(sphere.GetOutput())
    spherea = vtk.vtkActor()
    spherea.SetMapper(spherem)
    spherea.SetPosition(position)
    spherea.GetProperty().SetColor(color)
    spherea.GetProperty().SetOpacity(opacity)
        
    return spherea

def ellipsoid(R=np.array([[2, 0, 0],[0, 1, 0],[0, 0, 1] ]),position=(0,0,0),thetares=20,phires=20,color=(0,0,1),opacity=1,tessel=0):

    '''
    Stretch a unit sphere to make it an ellipsoid under a 3x3 translation matrix R 
    
    R=sp.array([[2, 0, 0],
                         [0, 1, 0],
                         [0, 0, 1] ])
    '''
    
    Mat=sp.identity(4)
    Mat[0:3,0:3]=R
       
    '''
    Mat=sp.array([[2, 0, 0, 0],
                             [0, 1, 0, 0],
                             [0, 0, 1, 0],
                             [0, 0, 0,  1]  ])
    '''
    mat=vtk.vtkMatrix4x4()
    
    for i in sp.ndindex(4,4):
        
        mat.SetElement(i[0],i[1],Mat[i])
    
    radius=1
    sphere = vtk.vtkSphereSource()
    sphere.SetRadius(radius)
    sphere.SetLatLongTessellation(tessel)
   
    sphere.SetThetaResolution(thetares)
    sphere.SetPhiResolution(phires)
    
    trans=vtk.vtkTransform()
    
    trans.Identity()
    #trans.Scale(0.3,0.9,0.2)
    trans.SetMatrix(mat)
    trans.Update()
    
    transf=vtk.vtkTransformPolyDataFilter()
    transf.SetTransform(trans)
    transf.SetInput(sphere.GetOutput())
    transf.Update()
    
    spherem = vtk.vtkPolyDataMapper()
    spherem.SetInput(transf.GetOutput())
    
    spherea = vtk.vtkActor()
    spherea.SetMapper(spherem)
    spherea.SetPosition(position)
    spherea.GetProperty().SetColor(color)
    spherea.GetProperty().SetOpacity(opacity)
    #spherea.GetProperty().SetRepresentationToWireframe()
    
    return spherea

    
def label(ren,text='Origin',pos=(0,0,0),scale=(0.2,0.2,0.2),color=(1,1,1)):
    
    ''' Create a label actor 
    This actor will always face the camera
    
    Parameters
    ----------
    ren : vtkRenderer() object as returned from ren()
    text : a text for the label
    pos : left down position of the label
    scale : change the size of the label 
    color : (r,g,b) and RGB tuple
    
    Returns
    ----------
    vtkActor object
    
    Examples
    --------    
    >>> r=ren()    
    >>> lines=[np.random.rand(10,3),np.random.rand(20,3)]    
    >>> colors=[0.2,0.8]
    >>> c=line(lines,colors)    
    >>> add(r,c)
    >>> l=label(r)
    >>> add(r,l)
    >>> show(r)
    '''
    atext=vtk.vtkVectorText()
    atext.SetText(text)
    
    textm=vtk.vtkPolyDataMapper()
    textm.SetInput(atext.GetOutput())
    
    texta=vtk.vtkFollower()
    texta.SetMapper(textm)
    texta.SetScale(scale)    

    texta.GetProperty().SetColor(color)
    texta.SetPosition(pos)
    
    ren.AddActor(texta)
    texta.SetCamera(ren.GetActiveCamera())
        
    return texta

def show(ren,title='Fos',size=(300,300)):
    ''' Show window 
    
    Parameters
    ----------
    ren : vtkRenderer() object as returned from ren()
    title : a string for the window title bar
    size : (width,height) of the window
    
    Examples
    --------    
    >>> r=ren()    
    >>> lines=[np.random.rand(10,3),np.random.rand(20,3)]    
    >>> colors=[0.2,0.8]
    >>> c=line(lines,colors)    
    >>> add(r,c)
    >>> l=label(r)
    >>> add(r,l)
    >>> show(r)
    '''
        
    window = vtk.vtkRenderWindow()
    window.AddRenderer(ren)
    window.SetWindowName(title) 
    window.SetSize(size)
    style=vtk.vtkInteractorStyleTrackballCamera()        
    iren = vtk.vtkRenderWindowInteractor()
    iren.SetRenderWindow(window)
    iren.SetInteractorStyle(style)
    iren.Start()
    
    
if __name__ == "__main__":
   
    r=ren()    
    lines=[np.random.rand(10,3),np.random.rand(20,3)]    
    colors=[0.2,0.8]
    c=line(lines,colors)    
    add(r,c)
    l=label(r)
    add(r,l)
    show(r)
