import numpy as np
import dipy as dp
import pyglet
from pyglet.gl import *
#from delaunay.core import Triangulation #http://flub.stuffwillmade.org/delny/

try:
    # Try and create a window with multisampling (antialiasing)
    config = Config(sample_buffers=1, samples=4, 
                    depth_size=24, double_buffer=True,vsync=False)
    window = pyglet.window.Window(resizable=True, config=config)
except pyglet.window.NoSuchConfigException:
    # Fall back to no multisampling for old hardware
    window = pyglet.window.Window(resizable=True)


#fps_display = pyglet.clock.ClockDisplay()
    
@window.event
def on_resize(width, height):
    # Override the default on_resize handler to create a 3D projection
    print('%d width, %d height' % (width,height))
    glViewport(0, 0, width, height)
    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()
    gluPerspective(60., width / float(height), .1, 1000.)
    glMatrixMode(GL_MODELVIEW)
    #window.flip()
    return pyglet.event.EVENT_HANDLED


def update(dt):
    global rx, ry, rz

    #rx += dt * 5
    #ry += dt * 80
    #rz += dt * 30
    #rx %= 360
    #ry %= 360
    #rz %= 360
   
    pass

pyglet.clock.schedule(update) 
#pyglet.clock.schedule_interval(update,1/100.)

@window.event
def on_draw():

    global surf

    for i in range(0,900,3):

        if np.random.rand()>0.5:

            surf.vertex_list.vertices[i]+=0.001*np.random.rand()
            surf.vertex_list.vertices[i+1]+=0.001*np.random.rand()
            surf.vertex_list.vertices[i+2]+=0.001*np.random.rand()
            
        else:
            
            surf.vertex_list.vertices[i]-=0.001*np.random.rand()
            surf.vertex_list.vertices[i+1]-=0.001*np.random.rand()
            surf.vertex_list.vertices[i+2]-=0.001*np.random.rand()

    
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
    glLoadIdentity()

    #fps_display.draw()

    #glScalef(3,1,1)
    glTranslatef(0, 0, -4)
    glRotatef(rx, 0, 0, 1)
    glRotatef(ry, 0, 1, 0)
    glRotatef(rx, 1, 0, 0)

    batch.draw()

    #pyglet.image.get_buffer_manager().get_color_buffer().save('/tmp/test.png')

    print pyglet.clock.get_fps()

    #window.clear()

    #fps_display.draw()

def setup():
    # One-time GL setup
    glClearColor(1, 1, 1, 1)
    #glClearColor(0,0,0,0)
    glColor3f(1, 0, 0)
    glEnable(GL_DEPTH_TEST)
    #glEnable(GL_CULL_FACE)

    # Uncomment this line for a wireframe view
    glPolygonMode(GL_FRONT_AND_BACK, GL_LINE)
    glLineWidth(3.)

    # Simple light setup.  On Windows GL_LIGHT0 is enabled by default,
    # but this is not the case on Linux or Mac, so remember to always 
    # include it.
    glEnable(GL_LIGHTING)
    glEnable(GL_LIGHT0)
    glEnable(GL_LIGHT1)

    # Define a simple function to create ctypes arrays of floats:
    def vec(*args):
        return (GLfloat * len(args))(*args)

    glLightfv(GL_LIGHT0, GL_POSITION, vec(.5, .5, 1, 0))
    glLightfv(GL_LIGHT0, GL_SPECULAR, vec(.5, .5, 1, 1))
    glLightfv(GL_LIGHT0, GL_DIFFUSE, vec(1, 1, 1, 1))

    glLightfv(GL_LIGHT1, GL_POSITION, vec(1, 0, .5, 0))
    glLightfv(GL_LIGHT1, GL_DIFFUSE, vec(.5, .0, 0, 1))
    glLightfv(GL_LIGHT1, GL_SPECULAR, vec(1, 0, 0, 1))

    glMaterialfv(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE, vec(0.5, 0, 0.3, 0.5))
    glMaterialfv(GL_FRONT_AND_BACK, GL_SPECULAR, vec(1, 1, 1, 0.5))
    glMaterialf(GL_FRONT_AND_BACK, GL_SHININESS, 50)

class Surface(object):

    def __init__(self, vertices,faces,batch,group=None):

            
        inds=faces.ravel().tolist()
        verx=vertices.ravel().tolist()

        normals=np.zeros((len(vertices),3))
        p=vertices
        l=faces
            
        trinormals=np.cross(p[l[:,0]]-p[l[:,1]],p[l[:,1]]-p[l[:,2]],axisa=1,axisb=1)
        for (i,lp) in enumerate(faces):
            normals[lp]+=trinormals[i]

        div=np.sqrt(np.sum(normals**2,axis=1))        
        div=div.reshape(len(div),1)        
        normals=(normals/div)
            
        #normals=vertices/np.linalg.norm(vertices)
        norms=np.array(normals).ravel().tolist()
        
        self.vertex_list = batch.add_indexed(len(vertices),\
                                                 GL_TRIANGLES,\
                                                 group,\
                                                 inds,\
                                                 ('v3d/static',verx),\
                                                 ('n3d/static',norms))
    def delete(self):
        self.vertex_list.delete()
                                             


fname='/home/eg01/Data_Backup/Data/Marta/DSI/SimData/results_SNR030_1fibre'
#fname='/home/eg01/Data_Backup/Data/Marta/DSI/SimData/results_SNR030_isotropic'
marta_table_fname='/home/eg01/Data_Backup/Data/Marta/DSI/SimData/Dir_and_bvals_DSI_marta.txt'
sim_data=np.loadtxt(fname)
#bvalsf='/home/eg01/Data_Backup/Data/Marta/DSI/SimData/bvals101D_float.txt'
dname =  '/home/eg01/Data_Backup/Data/Frank_Eleftherios/frank/20100511_m030y_cbu100624/08_ep2d_advdiff_101dir_DSI'

#real_data,affine,bvals,gradients=dp.load_dcm_dir(dname)

b_vals_dirs=np.loadtxt(marta_table_fname)

bvals=b_vals_dirs[:,0]*1000
gradients=b_vals_dirs[:,1:]

sim_data=sim_data

gq = dp.GeneralizedQSampling(sim_data,bvals,gradients)
tn = dp.Tensor(sim_data,bvals,gradients)

evals=tn.evals[0]
evecs=tn.evecs[0]

setup()
batch = pyglet.graphics.Batch()

eds=np.load('/home/eg01/Devel/dipy/dipy/core/matrices/evenly_distributed_sphere_362.npz')

vertices=eds['vertices']
faces=eds['faces']

surf = Surface(vertices,faces, batch=batch)
rx = ry = rz = 0

print('Application Starting Now...')
pyglet.app.run()

 



    
    
