''' Fos module implements simple visualization functions using VTK. Fos means light in Greek.    
   
    The main idea is the following:
    A window can have one or more renderers. A renderer can have none, one or more actors. Examples of actors are a sphere, line, point etc.
    You basically add actors in a renderer and in that way you can visualize the forementioned objects e.g. sphere, line ...

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
    >>> from dipy.viz import fos
    >>> import numpy as np
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
    if isinstance(a,vtk.vtkVolume):
        ren.AddVolume(a)
    else:    
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
    ''' Internal function for generating arrow actors.    
    '''
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

    ass=vtk.vtkAssembly()
    ass.AddPart(arrowx)
    ass.AddPart(arrowy)
    ass.AddPart(arrowz)
           
    return ass

def _lookup(colors):
    ''' Internal function
    Creates a lookup table with given colors.
    
    Parameters
    ------------
    colors : array, shape (N,3)
            Colormap where every triplet is encoding red, green and blue e.g. 
            r1,g1,b1
            r2,g2,b2
            ...
            rN,gN,bN        
            
            where
            0=<r<=1,
            0=<g<=1,
            0=<b<=1,
    
    Returns
    ----------
    vtkLookupTable
    
    '''
        
    colors=np.asarray(colors,dtype=np.float32)
    
    if colors.ndim>2:
        raise ValueError('Incorrect shape of array in colors')
    
    if colors.ndim==1:
        N=1
        
    if colors.ndim==2:
        
        N=colors.shape[0]    
    
    
    lut=vtk.vtkLookupTable()
    lut.SetNumberOfColors(N)
    lut.Build()
    
    if colors.ndim==2:
        scalar=0
        for (r,g,b) in colors:
            
            lut.SetTableValue(scalar,r,g,b,1.0)
            scalar+=1
    if colors.ndim==1:
        
        lut.SetTableValue(0,colors[0],colors[1],colors[2],1.0)
            
    return lut

def line(lines,colors,opacity=1,linewidth=1):
    ''' Create an actor for one or more lines.    
    
    Parameters
    ----------
    lines :  list of arrays representing lines as 3d points  for example            
            lines=[np.random.rand(10,3),np.random.rand(20,3)]   
            represents 2 lines the first with 10 points and the second with 20 points in x,y,z coordinates.
    colors : array, shape (N,3)
            Colormap where every triplet is encoding red, green and blue e.g. 
            r1,g1,b1
            r2,g2,b2
            ...
            rN,gN,bN        
            
            where
            0=<r<=1,
            0=<g<=1,
            0=<b<=1
            
    opacity : float, default 1
                    0<=transparency <=1
    linewidth : float, default is 1
                    line thickness
                    
    
    Returns
    ----------
    vtkActor object
    
    Examples
    --------    
    >>> from dipy.viz import fos
    >>> r=fos.ren()    
    >>> lines=[np.random.rand(10,3),np.random.rand(20,3)]    
    >>> colors=np.random.rand(2,3)
    >>> c=fos.line(lines,colors)    
    >>> fos.add(r,c)
    >>> fos.show(r)
    '''    
    if not isinstance(lines,types.ListType):
        lines=[lines]    
        
    points= vtk.vtkPoints()
    lines_=vtk.vtkCellArray()
    linescalars=vtk.vtkFloatArray()
   
    #lookuptable=vtk.vtkLookupTable()
    lookuptable=_lookup(colors)

    scalarmin=0
    if colors.ndim==2:            
        scalarmax=colors.shape[0]-1
    if colors.ndim==1:        
        scalarmax=0
   
    curPointID=0
          
    m=(0.0,0.0,0.0)
    n=(1.0,0.0,0.0)
    
    scalar=0
    #many colors
    if colors.ndim==2:
        for Line in lines:
            
            inw=True
            mit=iter(Line)
            nit=iter(Line)
            nit.next()        
            
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
             
            scalar+=1
    #one color only
    if colors.ndim==1:
        for Line in lines:
            
            inw=True
            mit=iter(Line)
            nit=iter(Line)
            nit.next()        
            
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
  Create one or more 3d dots(points) returns one actor handling all the points
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
    ''' Create 3d points and generate only one actor for all points. Same as dots.
    '''
    return dots(points,color=(1,0,0),opacity=1)

def sphere(position=(0,0,0),radius=0.5,thetares=8,phires=8,color=(0,0,1),opacity=1,tessel=0):
    ''' Create a sphere actor
    '''
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

    ''' Create a ellipsoid actor.    
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
    >>> from dipy.viz import fos  
    >>> r=fos.ren()    
    >>> l=fos.label(r)
    >>> fos.add(r,l)
    >>> fos.show(r)
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

def volume(vol,voxsz=(1.0,1.0,1.0),affine=None,center_origin=1,final_volume_info=1,maptype=0,iso=0,iso_thr=100,opacitymap=None,colormap=None):    
    ''' Create a volume and return a volumetric actor using volumetric rendering. 
    This function has many different interesting capabilities. The maptype, opacitymap and colormap are the most crucial parameters here.
    
    Parameters:
    ----------------
    vol : array, shape (N, M, K), dtype uint8
         an array representing the volumetric dataset that we want to visualize using volumetric rendering            
        
    voxsz : sequence of 3 floats
            default (1., 1., 1.)
            
    affine : array, shape (4,4), default None
            as given by volumeimages             
            
    center_origin : int {0,1}, default 1
             it considers that the center of the volume is the 
            point (-vol.shape[0]/2.0+0.5,-vol.shape[1]/2.0+0.5,-vol.shape[2]/2.0+0.5)
            
    final_volume_info : int {0,1}, default 1
            if 1 it prints out some info about the volume.
            
    maptype : int {0,1}, default 0,        
            The maptype is a very important parameter which affects the raycasting algorithm in use for the rendering. 
            The options are:
            If 0 then vtkVolumeTextureMapper2D is used.
            If 1 then vtkVolumeRayCastFunction is used.
            
    iso : int {0,1} default 0,
            If iso is 1 and maptype is 1 then  we use vtkVolumeRayCastIsosurfaceFunction which generates an isosurface at 
            the predefined iso_thr value. If iso is 0 and maptype is 1 vtkVolumeRayCastCompositeFunction is used.
            
    iso_thr : int, default 100,
            if iso is 1 then then this threshold in the volume defines the value which will be used to create the isosurface.
            
    opacitymap : array, shape (N,2) default None.
            The opacity map assigns a transparency coefficient to every point in the volume.
            
    colormap : array, shape (N,4), default None.
            The color map assigns a color value to every point in the volume.
                
    Returns:
    ----------
    vtkVolume    
    
    Notes:
    --------
    What is the difference between TextureMapper2D and RayCastFunction? 
    See VTK user's guide [book] & The Visualization Toolkit [book] and VTK's online documentation & online docs.
    What is the difference between RayCastIsosurfaceFunction and RayCastCompositeFunction?
    See VTK user's guide [book] & The Visualization Toolkit [book] and VTK's online documentation & online docs.
    
    Examples:
    ------------
    First example random points    
    
    >>> from dipy.viz import fos
    >>> import numpy as np
    >>> vol=100*np.random.rand(100,100,100)
    >>> vol=vol.astype('uint8')
    >>> print vol.min(), vol.max()
    >>> r = fos.ren()
    >>> v = fos.volume(vol)
    >>> fos.add(r,v)
    >>> fos.show(r)
    
    Second example with a more complicated function
        
    >>> from dipy.viz import fos
    >>> import numpy as np
    >>> x, y, z = np.ogrid[-10:10:20j, -10:10:20j, -10:10:20j]
    >>> s = np.sin(x*y*z)/(x*y*z)
    >>> r = fos.ren()
    >>> v = fos.volume(vol)
    >>> fos.add(r,v)
    >>> fos.show(r)
    
    '''
        
    print('Datatype',vol.dtype)
    vol=vol.astype('uint16')

    if opacitymap==None:
        
        #'''
        bin,res=np.histogram(vol.ravel())
        res2=np.interp(res,[vol.min(),vol.max()],[0,1])
        opacitymap=np.vstack((res,res2)).T
        opacitymap=opacitymap.astype('float32')
        #'''
                
        #'''
        opacitymap=np.array([[ 0.0, 0.0],
                          [50.0, 0.9]])
        #''' 
        '''       
        opacitymap=np.array( [[  0.00000000e+00,   0],
                [  7.22400024e+02  , 0.8],
                [  9.03000000e+02  , 0.9],
                [1.80600000e+03  , 1]])
        '''
        print opacitymap
        
    if colormap==None:
        
        #'''
        bin,res=np.histogram(vol.ravel())
        res2=np.interp(res,[vol.min(),vol.max()],[0,1])
        
        colormap=np.vstack((res,res2,res2,res2)).T
        colormap=colormap.astype('float32')
        
        #'''
        #'''
        colormap=np.array([[0.0, 0.5, 0.0, 0.0],
                                        [64.0, 1.0, 0.5, 0.5],
                                        [128.0, 0.9, 0.2, 0.3],
                                        [196.0, 0.81, 0.27, 0.1],
                                        [255.0, 0.5, 0.5, 0.5]])
        #'''
        '''
        colormap=np.array([[  0.00000000e+00,   0.00000000e+00 ,  0.00000000e+00,   0.00000000e+00],                 
                 [  7.22400024e+02,   4.00000006e-01 ,  0 , 0],
                 [  9.03000000e+02,   8.00000000e-01 ,  0 ,  0],
                 [  1.08359998e+03,   8.00000024e-01 ,  0 ,  0],
                 [  1.26419995e+03,   9.0000008e-01 ,  0 ,  0],
                 [  1.44480005e+03,   9.00000012e-01 ,  0 ,  0],
                 [  1.62540002e+03,   9.99999976e-01 ,  0 ,  0],
                 [  1.80600000e+03,   1.00000000e+00,   0 ,  0]])
        '''
        print colormap                        
    
    im = vtk.vtkImageData()
    im.SetScalarTypeToUnsignedChar()
    im.SetDimensions(vol.shape[0],vol.shape[1],vol.shape[2])
    #im.SetOrigin(0,0,0)
    #im.SetSpacing(voxsz[2],voxsz[0],voxsz[1])
    im.AllocateScalars()        
    
    for i in range(vol.shape[0]):
        for j in range(vol.shape[1]):
            for k in range(vol.shape[2]):
                
                im.SetScalarComponentFromFloat(i,j,k,0,vol[i,j,k])
    
    if affine != None:

        aff = vtk.vtkMatrix4x4()
        aff.DeepCopy((affine[0,0],affine[0,1],affine[0,2],affine[0,3],affine[1,0],affine[1,1],affine[1,2],affine[1,3],affine[2,0],affine[2,1],affine[2,2],affine[2,3],affine[3,0],affine[3,1],affine[3,2],affine[3,3]))
        #aff.DeepCopy((affine[0,0],affine[0,1],affine[0,2],0,affine[1,0],affine[1,1],affine[1,2],0,affine[2,0],affine[2,1],affine[2,2],0,affine[3,0],affine[3,1],affine[3,2],1))
        #aff.DeepCopy((affine[0,0],affine[0,1],affine[0,2],127.5,affine[1,0],affine[1,1],affine[1,2],-127.5,affine[2,0],affine[2,1],affine[2,2],-127.5,affine[3,0],affine[3,1],affine[3,2],1))
        
        reslice = vtk.vtkImageReslice()
        reslice.SetInput(im)
        #reslice.SetOutputDimensionality(2)
        #reslice.SetOutputOrigin(127,-145,147)    
        
        reslice.SetResliceAxes(aff)
        #reslice.SetOutputOrigin(-127,-127,-127)    
        #reslice.SetOutputExtent(-127,128,-127,128,-127,128)
        #reslice.SetResliceAxesOrigin(0,0,0)
        #print 'Get Reslice Axes Origin ', reslice.GetResliceAxesOrigin()
        #reslice.SetOutputSpacing(1.0,1.0,1.0)
        
        reslice.SetInterpolationModeToLinear()    
        #reslice.UpdateWholeExtent()
        
        #print 'reslice GetOutputOrigin', reslice.GetOutputOrigin()
        #print 'reslice GetOutputExtent',reslice.GetOutputExtent()
        #print 'reslice GetOutputSpacing',reslice.GetOutputSpacing()
    
        changeFilter=vtk.vtkImageChangeInformation() 
        changeFilter.SetInput(reslice.GetOutput())
        #changeFilter.SetInput(im)
        if center_origin:
            changeFilter.SetOutputOrigin(-vol.shape[0]/2.0+0.5,-vol.shape[1]/2.0+0.5,-vol.shape[2]/2.0+0.5)
            print 'ChangeFilter ', changeFilter.GetOutputOrigin()
        
    opacity = vtk.vtkPiecewiseFunction()
    for i in range(opacitymap.shape[0]):
        opacity.AddPoint(opacitymap[i,0],opacitymap[i,1])

    color = vtk.vtkColorTransferFunction()
    for i in range(colormap.shape[0]):
        color.AddRGBPoint(colormap[i,0],colormap[i,1],colormap[i,2],colormap[i,3])
        
    if(maptype==0): 
    
        property = vtk.vtkVolumeProperty()
        property.SetColor(color)
        property.SetScalarOpacity(opacity)
        
        mapper = vtk.vtkVolumeTextureMapper2D()
        if affine == None:
            mapper.SetInput(im)
        else:
            #mapper.SetInput(reslice.GetOutput())
            mapper.SetInput(changeFilter.GetOutput())
        
    
    if (maptype==1):

        property = vtk.vtkVolumeProperty()
        property.SetColor(color)
        property.SetScalarOpacity(opacity)
        property.ShadeOn()
        property.SetInterpolationTypeToLinear()

        if iso:
            isofunc=vtk.vtkVolumeRayCastIsosurfaceFunction()
            isofunc.SetIsoValue(iso_thr)
        else:
            compositeFunction = vtk.vtkVolumeRayCastCompositeFunction()
        
        mapper = vtk.vtkVolumeRayCastMapper()
        if iso:
            mapper.SetVolumeRayCastFunction(isofunc)
        else:
            mapper.SetVolumeRayCastFunction(compositeFunction)   
            #mapper.SetMinimumImageSampleDistance(0.2)
             
        if affine == None:
            mapper.SetInput(im)
        else:
            #mapper.SetInput(reslice.GetOutput())    
            mapper.SetInput(changeFilter.GetOutput())
            #Return mid position in world space    
            #im2=reslice.GetOutput()
            #index=im2.FindPoint(vol.shape[0]/2.0,vol.shape[1]/2.0,vol.shape[2]/2.0)
            #print 'Image Getpoint ' , im2.GetPoint(index)
           
        
    volum = vtk.vtkVolume()
    volum.SetMapper(mapper)
    volum.SetProperty(property)

    if final_volume_info :  
         
        print 'Origin',   volum.GetOrigin()
        print 'Orientation',   volum.GetOrientation()
        print 'OrientationW',    volum.GetOrientationWXYZ()
        print 'Position',    volum.GetPosition()
        print 'Center',    volum.GetCenter()  
        print 'Get XRange', volum.GetXRange()
        print 'Get YRange', volum.GetYRange()
        print 'Get ZRange', volum.GetZRange()  
        
    return volum


def show(ren,title='Fos',size=(300,300)):
    ''' Show window 
    
    Parameters
    ----------
    ren : vtkRenderer() object 
            as returned from function ren()
    title : string 
            a string for the window title bar
    size : (int, int) 
            (width,height) of the window
    
    Examples
    --------    
    >>> from dipy.viz import fos
    >>> r=fos.ren()    
    >>> lines=[np.random.rand(10,3),np.random.rand(20,3)]    
    >>> colors=[0.2,0.8]
    >>> c=fos.line(lines,colors)    
    >>> fos.add(r,c)
    >>> l=fos.label(r)
    >>> fos.add(r,l)
    >>> fos.show(r)
    '''
    ren.ResetCamera()        
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

    pass
    
    