import numpy as np
import dipy as dp
import pyglet
from pyglet.gl import *
from delaunay.core import Triangulation #http://flub.stuffwillmade.org/delny/
from math import pi, sin, cos
'''
fname='/home/eg01/Data_Backup/Data/Marta/DSI/SimData/results_SNR030_1fibre'
sim_data=np.loadtxt(fname)
#bvalsf='/home/eg01/Data_Backup/Data/Marta/DSI/SimData/bvals101D_float.txt'
dname =  '/home/eg01/Data_Backup/Data/Frank_Eleftherios/frank/20100511_m030y_cbu100624/08_ep2d_advdiff_101dir_DSI'

real_data,affine,bvals,gradients=dp.load_dcm_dir(dname)

gq = dp.GeneralizedQSampling(sim_data,bvals,gradients)
tn = dp.Tensor(sim_data,bvals,gradients)
'''

try:
    # Try and create a window with multisampling (antialiasing)
    config = Config(sample_buffers=1, samples=4, 
                    depth_size=16, double_buffer=True,)
    window = pyglet.window.Window(resizable=True, config=config)
except pyglet.window.NoSuchConfigException:
    # Fall back to no multisampling for old hardware
    window = pyglet.window.Window(resizable=True)


@window.event
def on_resize(width, height):
    # Override the default on_resize handler to create a 3D projection
    print('%d width, %d height' % (width,height))
    glViewport(0, 0, width, height)
    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()
    gluPerspective(60., width / float(height), .1, 1000.)
    glMatrixMode(GL_MODELVIEW)
    return pyglet.event.EVENT_HANDLED


def update(dt):
    global rx, ry, rz

    rx += dt * 30
    #ry += dt * 80
    #rz += dt * 30
    #rx %= 360
    #ry %= 360
    #rz %= 360
    
    pass
    
pyglet.clock.schedule(update)


@window.event
def on_draw():
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
    glLoadIdentity()
    glTranslatef(0, 0, -4)
    glRotatef(rx, 0, 0, 1)
    glRotatef(ry, 0, 1, 0)
    glRotatef(rx, 1, 0, 0)
    batch.draw()

def setup():
    # One-time GL setup
    glClearColor(1, 1, 1, 1)
    glColor3f(1, 0, 0)
    #glEnable(GL_DEPTH_TEST)
    #glEnable(GL_CULL_FACE)

    # Uncomment this line for a wireframe view
    #glPolygonMode(GL_FRONT_AND_BACK, GL_LINE)

    # Simple light setup.  On Windows GL_LIGHT0 is enabled by default,
    # but this is not the case on Linux or Mac, so remember to always 
    # include it.
    glEnable(GL_LIGHTING)
    glEnable(GL_LIGHT0)
    #glEnable(GL_LIGHT1)

    glEnable(GL_NORMALIZE)
    glLineWidth(3.)

    # Define a simple function to create ctypes arrays of floats:
    def vec(*args):
        return (GLfloat * len(args))(*args)

    glLightfv(GL_LIGHT0, GL_POSITION, vec(.5, .5, 1, 0))
    glLightfv(GL_LIGHT0, GL_SPECULAR, vec(.5, .5, 1, 1))
    glLightfv(GL_LIGHT0, GL_DIFFUSE, vec(1, 1, 1, 1))

    '''
    glLightfv(GL_LIGHT1, GL_POSITION, vec(1, 0, .5, 0))
    glLightfv(GL_LIGHT1, GL_DIFFUSE, vec(.5, .5, .5, 1))
    glLightfv(GL_LIGHT1, GL_SPECULAR, vec(1, 1, 1, 1))
    '''

    glMaterialfv(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE, vec(0.5, 0, 0.3, 0.5))
    glMaterialfv(GL_FRONT_AND_BACK, GL_SPECULAR, vec(1, 1, 1, 0.5))
    glMaterialf(GL_FRONT_AND_BACK, GL_SHININESS, 50)

class Surface(object):

    def __init__(self, vertices,faces,batch,group=None):

        if faces==None:

            tria=Triangulation(vertices,dim=2)
            indices=tria.indices
            normals=vertices/np.linalg.norm(vertices)
            #group=None

            verx=vertices.ravel().tolist()
            norms=np.array(normals).ravel().tolist()
            inds=np.array(indices).ravel().tolist()

            print('%d verx %d norms %d inds %d indices' %(len(verx),\
                                                              len(norms),\
                                                              len(inds),\
                                                              len(indices)))

        else:

            inds=faces.ravel().tolist()
            verx=vertices.ravel().tolist()
            normals=vertices/np.linalg.norm(vertices)
            norms=np.array(normals).ravel().tolist()
            

        #1/0

        '''
        self.vertex_list = batch.add_indexed(len(vertices),\
                                                 GL_TRIANGLES,\
                                                 group,\
                                                 inds,\
                                                 ('v3d/static',verx),\
                                                 ('n3d/static',norms))
        '''
        self.vertex_list = batch.add_indexed(len(vertices),\
                                                 GL_TRIANGLES,\
                                                 group,\
                                                 inds,\
                                                 ('v3d/static',verx))

        

    def delete(self):
        self.vertex_list.delete()
                                             

class Torus(object):
    list = None
    def __init__(self, radius, inner_radius, slices, inner_slices, 
                 batch, group=None):
        # Create the vertex and normal arrays.
        vertices = []
        normals = []

        u_step = 2 * pi / (slices - 1)
        v_step = 2 * pi / (inner_slices - 1)
        u = 0.
        for i in range(slices):
            cos_u = cos(u)
            sin_u = sin(u)
            v = 0.
            for j in range(inner_slices):
                cos_v = cos(v)
                sin_v = sin(v)

                d = (radius + inner_radius * cos_v)
                x = d * cos_u
                y = d * sin_u
                z = inner_radius * sin_v

                nx = cos_u * cos_v
                ny = sin_u * cos_v
                nz = sin_v

                vertices.extend([x, y, z])
                normals.extend([nx, ny, nz])
                v += v_step
            u += u_step

        # Create a list of triangle indices.
        indices = []
        for i in range(slices - 1):
            for j in range(inner_slices - 1):
                p = i * inner_slices + j
                indices.extend([p, p + inner_slices, p + inner_slices + 1])
                indices.extend([p, p + inner_slices + 1, p + 1])


        print('%d vertices %d indices %d normals ' % (len(vertices),len(indices),len(normals)))
                
                
        self.vertex_list = batch.add_indexed(len(vertices)//3, 
                                             GL_TRIANGLES,
                                             group,
                                             indices,
                                             ('v3f/static', vertices),
                                             ('n3f/static', normals))


        

setup()
batch = pyglet.graphics.Batch()

#sphere=np.load('/home/eg01/Desktop/200.npy')
#sphere=np.concatenate([sphere,-sphere])

eds=np.load('/home/eg01/Devel/dipy/dipy/core/matrices/evenly_distributed_sphere_362.npz')

vertices=eds['vertices']
faces=eds['faces']

#surf = Surface(vertices,faces, batch=batch)
rx = ry = rz = 0
torus = Torus(1, 0.3, 50, 30, batch=batch)

print('Application Starting Now...')
pyglet.app.run()

    



    
    
