'''  Phos is a faster 3d engine than using only wxPython and OpenGL. 
    At the moment it is working only for line plotting.
    When loaded use Arrow, Home and End keys to navigate.
    
    Examples:
    -------------
    >>> from dipy.viz import phos
    >>> phos.trajs=[100*np.random.rand(1000,3)]
    >>> phos.show([100*np.random.rand(1000,3)])
'''
import numpy as np    

try:  
    import wx
    from wx import glcanvas
except ImportError:
    ImportError('wxPython is not installed')

try:
    import OpenGL.GL as GL
    import OpenGL.GLUT as GLUT
    import OpenGL.GLU as GLU
except ImportError:
    ImportError('PyOpenGL is not installed')


def axes():
    GL.glNewList(1, GL.GL_COMPILE)        
    GL.glBegin(GL.GL_LINES)
    
    GL.glColor3f(1.0,0.0,0.0)		# Red
    GL.glVertex3f(0.0, 0.0, 0.0) # origin of the line
    GL.glVertex3f(100.0, 0.0, 0.0) # ending point of the line

    GL.glColor3f(0.0,1.0,0.0)			# Green
    GL.glVertex3f(0.0, 0.0, 0.0) # origin of the line
    GL.glVertex3f(0.0, 100.0, 0.0) # ending point of the line

    GL.glColor3f(0.0,0.0,1.0)			# Blue
    GL.glVertex3f(0.0, 0.0, 0.0) # origin of the line
    GL.glVertex3f(0.0, 0.0, 100.0) # ending point of the line

    GL.glEnd()
    GL.glEndList() 
    

def line(lines,colors=None,opacity=1,linewidth=1):
    # we may have been given a single line instead of a series
    if isinstance(lines, np.ndarray) and lines.ndim > 1:
        raise ValueError('Need sequence of lines for lines argument')
    
    scalar=1.0

    if colors!=None:        
        lit=iter(colors)
    else:
        colors=np.random.rand(len(lines),3)
        lit=iter(colors)
    
    GL.glNewList(2, GL.GL_COMPILE)        
    nol=0
    for Line in lines:
        
        inw=True
        mit=iter(Line)
        nit=iter(Line)
        nit.next()
        
        scalar=lit.next()
        GL.glBegin(GL.GL_LINE_STRIP)    
        GL.glColor3f(scalar[0],scalar[1],scalar[2])
        while(inw):            
            try:
                m=mit.next()                                        
                GL.glVertex3f(m[0], m[1], m[2]) # point                                                
            except StopIteration:
                break

        GL.glEnd()                                
        
        nol+=1
        if nol%1000==0:            
            print(nol,'Lines Loaded')
        
    GL.glEndList() 
    

class Interactor(glcanvas.GLCanvas):  

    def __init__(self, parent):   

        glcanvas.GLCanvas.__init__(self, parent, -1)   
        self.init = False  
        
        # initial mouse position   
        self.lastx = self.x = 30   
        self.lasty = self.y = 30           
        
        # world coordinates
        self.xw=0.0
        self.yw=0.0
        self.zw=0.0

        self.size = None   
        self.Bind(wx.EVT_ERASE_BACKGROUND, self.OnEraseBackground)   
        self.Bind(wx.EVT_SIZE, self.OnSize)   
        self.Bind(wx.EVT_PAINT, self.OnPaint)   
        self.Bind(wx.EVT_LEFT_DOWN, self.OnMouseDown)   
        self.Bind(wx.EVT_LEFT_UP, self.OnMouseUp)   
        self.Bind(wx.EVT_MOTION, self.OnMouseMotion)  
        self.Bind(wx.EVT_KEY_DOWN, self.OnKeyDown)
  
    def OnEraseBackground(self, event):   
        pass # Do nothing, to avoid flashing on MSW.   
  
    def OnSize(self, event):   
        size = self.size = self.GetClientSize()   
        if self.GetContext():   
            self.SetCurrent()   
            GL.glViewport(0, 0, size.width, size.height)   
        event.Skip()   
  
    def OnPaint(self, event):   
        dc = wx.PaintDC(self)   
        self.SetCurrent()   
        if not self.init:   
            self.InitGL()   
            self.init = True  
        self.OnDraw()   
  
    def OnMouseDown(self, evt):   
        self.CaptureMouse()   
        self.x, self.y = self.lastx, self.lasty = evt.GetPosition()   
  
    def OnMouseUp(self, evt):   
        self.ReleaseMouse()   
  
    def OnMouseMotion(self, evt):   
        if evt.Dragging() and evt.LeftIsDown():   
            self.lastx, self.lasty = self.x, self.y   
            self.x, self.y = evt.GetPosition()   
            self.Refresh(False)   

    def OnKeyDown(self,evt):
        if evt.KeyCode()==wx.WXK_UP:
            print 'Up'
            self.zw-=1
            print (self.xw,self.yw,self.zw)
            self.OnDraw()            
        elif evt.KeyCode()==wx.WXK_DOWN:
            print 'Down'
            self.zw+=1
            print (self.xw,self.yw,self.zw)
            self.OnDraw()
        elif evt.KeyCode()==wx.WXK_LEFT:
            print 'Left'
            self.xw-=1
            print (self.xw,self.yw,self.zw)
            self.OnDraw()
        elif evt.KeyCode()==wx.WXK_RIGHT:
            print 'Right'
            self.xw+=1
            print (self.xw,self.yw,self.zw)
            self.OnDraw()
        elif evt.KeyCode()==wx.WXK_HOME:
            print 'Home'
            self.yw-=1
            print (self.xw,self.yw,self.zw)
            self.OnDraw()
        elif evt.KeyCode()==wx.WXK_END:
            print 'End'
            self.yw+=1
            print (self.xw,self.yw,self.zw)
            self.OnDraw()
      
        print evt.GetPosition()


class Renderer(Interactor):
    
    def __init__(self, parent, trajs=None, colors=None):
        super(Renderer, self).__init__(parent)
        self.trajs = trajs
        self.colors = colors

    def InitGL(self):

        GL.glEnable(GL.GL_CULL_FACE)
        GL.glEnable(GL.GL_DEPTH_TEST)
        GL.glDepthMask(GL.GL_TRUE)

        GL.glClearColor( 1, 0.9,0.5, 0 )
        #glViewport( 0, 0,1024, 800 )
        GL.glMatrixMode( GL.GL_PROJECTION )
        GL.glLoadIdentity()
        GLU.gluPerspective( 60.0, float(1024)/float(800), 0.1, 300.0 )
        #glDepthMask(1) 
        GL.glMatrixMode(GL.GL_MODELVIEW)   
        GL.glLoadIdentity()
                
        self.xw=-91
        self.yw=-105
        self.zw=-220
        #'''
        self.on=False
        GLUT.glutInit()        
        
    def LoadActors(self):
        axes()
        #line([100*np.random.rand(1000,3)])        
        line(self.trajs,self.colors)
        
    def OnDraw(self):

        GL.glClear(GL.GL_COLOR_BUFFER_BIT | GL.GL_DEPTH_BUFFER_BIT)   
        GL.glLoadIdentity()
        GL.glTranslatef(self.xw, self.yw, self.zw)       
                
        if not self.on:
            #LoadObjects()           
            self.LoadActors()
            self.on=True
        else:
            #glutWireTeapot(15)
            GL.glCallList(1)  
            GL.glCallList(2)
            #glPopMatrix()
        
        self.SwapBuffers()


class Window(wx.Frame):
    default_render_maker = Renderer
    
    def __init__(self,
                 render_maker=None,
                 parent = None,
                 id = -1,
                 title = "Phos"):   
        # Init   
        wx.Frame.__init__(   
            self, parent, id, title, size = (1024,800),   
            style = wx.DEFAULT_FRAME_STYLE | wx.NO_FULL_REPAINT_ON_RESIZE   
            )
        if render_maker is None:
            render_maker = self.default_render_maker
        box = wx.BoxSizer(wx.HORIZONTAL)
        box.Add(render_maker(self), 1, wx.EXPAND)
        #box.Add(Renderer2(self), 1, wx.EXPAND)
        self.SetAutoLayout(True)
        self.SetSizer(box)
        self.Layout()     
        # StatusBar   
        self.CreateStatusBar()     
        # Filemenu   
        filemenu = wx.Menu()     
        # Filemenu - About   
        menuitem = filemenu.Append(-1, "&About", "Information about this program")   
        self.Bind(wx.EVT_MENU, self.OnAbout, menuitem) # here comes the event-handler   
        # Filemenu - Separator   
        filemenu.AppendSeparator()     
        # Filemenu - Exit
        menuitem = filemenu.Append(-1, "E&xit", "Terminate the program")   
        self.Bind(wx.EVT_MENU, self.OnExit, menuitem) # here comes the event-handler     
        # Menubar   
        menubar = wx.MenuBar()   
        menubar.Append(filemenu,"&File")   
        self.SetMenuBar(menubar)     
        # Show
        self.Show(True)   
  
    def OnAbout(self,event):   
        message = "Phos 3D engine"   
        caption = "Dipy Team"   
        wx.MessageBox(message, caption, wx.OK)   
  
    def OnExit(self,event):   
        self.Close(True)  # Close the frame.


def make_window_maker(trajs=None, colors=None):
    ''' Make window maker that uses useful renderer '''
    class MyWindow(Window): pass
    class MyRenderer(Renderer):
        def __init__(self, parent):
            Renderer.__init__(self, parent, trajs, colors)
    MyWindow.default_render_maker = MyRenderer
    return MyWindow
        

def show(trajs=None, colors=None):
    ''' Create wx application to show OpenGL output
    '''
    app = wx.PySimpleApp()
    frame = make_window_maker(trajs, colors)()
    app.MainLoop()

    del frame   
    del app  
    

if __name__ == "__main__":

    show()

