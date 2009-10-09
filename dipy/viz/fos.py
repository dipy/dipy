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

def line(lines,colors=None,opacity=1,linewidth=1):
    
    ''' Create a line actor     
    
    Parameters
    ----------
    ren : list of numpy arrays representing lines a 3d points
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
        
        lit=iter(colors)
    else:
        colors=np.random.rand(len(lines))
        lit=iter(colors)
    
    for Line in lines:
        
        inw=True
        mit=iter(Line)
        nit=iter(Line)
        nit.next()
        
        if colors==None:
            scalar=colors[0]        
        else:
            scalar=lit.next()
            if scalar==0:
                inw=False   
        
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