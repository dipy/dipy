import numpy as np
import dipy.core.meshes as meshes
import get_vertices as gv
from dipy.core.triangle_subdivide import create_unit_sphere
#from dipy.viz import fos
#from dipy.io import dicomreaders as dcm
#import dipy.core.geometry as geometry
#import matplotlib.pyplot as mplp
import dipy.core.sphere_plots as splot

# set up a dictionary of sphere points that are in use EITHER as a set
# directions for diffusion weighted acquisitions OR as a set of
# evaluation points for an ODF (orientation distribution function.
sphere_dic = {'fy362': {'filepath' : '/home/ian/Devel/dipy/dipy/core/data/evenly_distributed_sphere_362.npz', 'object': 'npz', 'vertices': 'vertices', 'omit': 0, 'hemi': False},
              'fy642': {'filepath' : '/home/ian/Devel/dipy/dipy/core/data/evenly_distributed_sphere_642.npz', 'object': 'npz', 'vertices': 'odf_vertices', 'omit': 0, 'hemi': False},
              'siem64': {'filepath':'/home/ian/Devel/dipy/dipy/core/tests/data/small_64D.gradients.npy', 'object': 'npy', 'omit': 1, 'hemi': True},
              'create2': {},
              'create3': {},
              'create4': {},
              'create5': {},
              'create6': {},
              'create7': {},
              'create8': {},
              'create9': {},
              'marta200': {'filepath': '/home/ian/Data/Spheres/200.npy', 'object': 'npy', 'omit': 0, 'hemi': True},
              'dsi101': {'filepath': '/home/ian/Data/Frank_Eleftherios/frank/20100511_m030y_cbu100624/08_ep2d_advdiff_101dir_DSI', 'object': 'dicom', 'omit': 0, 'hemi': True}}

def plot_sphere(v,key):
    r = fos.ren()
    fos.add(r,fos.point(v,fos.green, point_radius= 0.01))
    fos.show(r, title=key, size=(1000,1000))

def plot_lambert(v,key):
    lamb = geometry.lambert_equal_area_projection_cart(*v.T).T
    (y1,y2) = lamb
    radius = np.sum(lamb**2,axis=0) < 1
    #print inner
    #print y1[inner]
    #print y1[-inner]
    figure = mplp.figure(facecolor='w')
    current = figure.add_subplot(111)
    current.patch.set_color('k')
    current.plot(y1[radius],y2[radius],'.g')
    current.plot(y1[-radius],y2[-radius],'.r')
    current.axes.set_aspect(aspect = 'equal', adjustable = 'box')
    figure.show()
    figure.waitforbuttonpress()
    mplp.close()
    
def get_vertex_set(key):
    if key[:6] == 'create':
        number = eval(key[6:])
        vertices, edges, faces = create_unit_sphere(number) 
        omit = 0
    else:
        entry = sphere_dic[key]
        #print entry
        if entry.has_key('omit'):
            omit = entry['omit']
        else:
            omit = 0
        filepath = entry['filepath']
        if entry['object'] == 'npz':
            filearray  = np.load(filepath)
            vertices = filearray[entry['vertices']]
        elif sphere_dic[key]['object'] == 'npy':
            vertices = np.load(filepath)
        elif entry['object'] == 'dicom':
            data,affine,bvals,gradients=dcm.read_mosaic_dir(filepath)
            #print (bvals.shape, gradients.shape)
            # grad3 = np.vstack((bvals,bvals,bvals)).transpose()
            #print grad3.shape
            #vertices = grad3*gradients
            vertices = gradients
        if omit > 0:
            vertices = vertices[omit:,:]
        if entry['hemi']:
            vertices = np.vstack([vertices, -vertices])
    print key, ': number of vertices = ', vertices.shape[0], '(drop ',omit,')'
    return vertices[omit:,:]


xup=np.array([ 1,0,0])
xdn=np.array([-1,0,0])
yup=np.array([0, 1,0])
ydn=np.array([0,-1,0])
zup=np.array([0,0, 1])
zdn=np.array([0,0,-1])

#for key in sphere_dic:
#for key in ['siem64']:
for key in ['fy642']:
    v = gv.get_vertex_set(key)
    splot.plot_sphere(v,key)
    splot.plot_lambert(v,key,centre=np.array([0.,0.]))
    equat, polar = meshes.spherical_statistics(v,north=xup,width=0.2)
    l = 2.*len(v)
    equat = equat/l
    polar = polar/l
    print '%6.3f %6.3f %6.3f %6.3f' % (equat.min(), equat.mean(), equat.max(), np.sqrt(equat.var()))
    print '%6.3f %6.3f %6.3f %6.3f' % (polar.min(), polar.mean(), polar.max(), np.sqrt(polar.var()))

def spherical_statistics(vertices, north=np.array([0,0,1]), width=0.02):
    '''
    function to evaluate a spherical triangulation by looking at the
    variability of numbers of vertices in 'vertices' in equatorial bands
    of width 'width' orthogonal to each point in 'vertices'
    ''' 

    equatorial_counts = np.array([len(equatorial_zone_vertices(vertices, pole, width=width)) for pole in vertices if np.dot(pole,north) >= 0])

    #equatorial_counts = np.bincount(equatorial_counts)
    
    #args = np.where(equatorial_counts>0)

    #print zip(list(args[0]), equatorial_counts[args])

    polar_counts = np.array([len(polar_zone_vertices(vertices, pole, width=width)) for pole in vertices if np.dot(pole,north) >= 0])

    #unique_counts = np.sort(np.array(list(set(equatorial_counts))))
    #polar_counts = np.bincount(polar_counts)
    
    #counts_tokens = [(uc,  bin_counts[uc]) for uc in bin_counts if ]

    #args = np.where(polar_counts>0)

    #print '(number, frequency):', zip(unique_counts,tokens)
    #print '(number, frequency):', counts_tokens

    #print zip(args, bin_counts[args])
    #print zip(list(args[0]), polar_counts[args])

    return equatorial_counts, polar_counts

def spherical_proportion(zone_width):
    # assuming radius is 1: (2*np.pi*zone_width)/(4*np.pi)
    # 0 <= zone_width <= 2 
    return zone_width/2.

def angle_for_zone(zone_width):
    return np.arcsin(zone_width/2.)

# unresolved reference to geom
# def coarseness(faces):
#     faces = np.asarray(faces)
#     # coarseness = 0.0
#     for face in faces:
#         a, b, c = face
#         coarse = np.max(coarse, geom.circumradius(a,b,c))
#     return coarse



