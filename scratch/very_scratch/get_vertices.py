sphere_dic = {'fy362': {'filepath' : '/home/ian/Devel/dipy/dipy/data/evenly_distributed_sphere_362.npz', 'object': 'npz', 'vertices': 'vertices', 'omit': 0, 'hemi': False},
              'fy642': {'filepath' : '/home/ian/Devel/dipy/dipy/data/evenly_distributed_sphere_642.npz', 'object': 'npz', 'vertices': 'odf_vertices', 'omit': 0, 'hemi': False},
              'siem64': {'filepath':'/home/ian/Devel/dipy/dipy/data/small_64D.gradients.npy', 'object': 'npy', 'omit': 1, 'hemi': True},
              'create2': {},
              'create3': {},
              'create4': {},
              'create5': {},
              'create6': {},
              'create7': {},
              'create8': {},
              'create9': {},
              'marta200': {'filepath': '/home/ian/Data/Spheres/200.npy', 'object': 'npy', 'omit': 0, 'hemi': True},
              'dsi102': {'filepath': '/home/ian/Data/Frank_Eleftherios/frank/20100511_m030y_cbu100624/08_ep2d_advdiff_101dir_DSI', 'object': 'dicom', 'omit': 1, 'hemi': True}}

import numpy as np
from dipy.core.triangle_subdivide import create_unit_sphere
#from dipy.io import dicomreaders as dcm

def get_vertex_set(key):

    if key[:6] == 'create':
        number = eval(key[6:])
        vertices, edges, faces = create_unit_sphere(number) 
        # omit = 0
        return vertices
    else:
        entry = sphere_dic[key]

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
            data,affine,bvals,gradients = dcm.read_mosaic_dir(filepath)
            # print (bvals.shape, gradients.shape)
            # grad3 = np.vstack((bvals, bvals, bvals)).transpose()
            # print grad3.shape
            # vertices = grad3*gradients
            vertices = gradients
        if omit > 0:
            vertices = vertices[omit:, :]
        if entry['hemi']:
            vertices = np.vstack([vertices, -vertices])

        return vertices[omit:, :]

print(sphere_dic.keys())

# vertices = get_vertex_set('create5')
# vertices = get_vertex_set('siem64')
# vertices = get_vertex_set('dsi102')

vertices = get_vertex_set('fy362')
gradients = get_vertex_set('siem64')
gradients = gradients[:gradients.shape[0]/2]
print(gradients.shape)

from dipy.viz import window, actor
sph = -np.sinc(np.dot(gradients[1], vertices.T))
ren = window.Renderer()
# sph = np.arange(vertices.shape[0])
print(sph.shape)
cols = window.colors(sph, 'jet')
ren.add(actor.point(vertices, cols, point_radius=.1, theta=10, phi=10))
window.show(ren)

