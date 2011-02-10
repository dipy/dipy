import numpy as np
from nose.tools import assert_true, assert_false, assert_equal, assert_raises
from numpy.testing import assert_array_equal, assert_array_almost_equal

import nibabel as nib
from dipy.data import get_data, get_sphere
from dipy.reconst.dandelion import SphericalDandelion
from dipy.reconst.recspeed import peak_finding
from dipy.reconst.gqi import GeneralizedQSampling

def test_dandelion():
    
    fimg,fbvals,fbvecs=get_data('small_101D')    
    bvals=np.loadtxt(fbvals)
    gradients=np.loadtxt(fbvecs).T
    data=nib.load(fimg).get_data()    
    
    print(bvals.shape, gradients.shape, data.shape)    
    sd=SphericalDandelion(data,bvals,gradients)    
    
    sdf=sd.spherical_diffusivity(data[5,5,5])    
    
    XA=sd.xa()
    np.set_printoptions(2)
    print XA.min(),XA.max(),XA.mean()
    print sdf*10**4
    
    
    
    """
    print(sdf.shape)
    gq=GeneralizedQSampling(data,bvals,gradients)
    sodf=gq.odf(data[5,5,5])
    vertices, faces = get_sphere('symmetric362')
    print(faces.shape)    
    peaks,inds=peak_finding(np.squeeze(sdf),faces)
    print(peaks, inds)    
    peaks2,inds2=peak_finding(np.squeeze(sodf),faces)
    print(peaks2, inds2)
    """

    '''
    from fos.data import get_sphere
    from fos.comp_geom.gen_normals import auto_normals

    vertices=100*vertices.astype(np.float32)#make a bigger sphere
    faces=faces.astype(np.uint32)

    colors=np.ones((len(vertices),4))
    colors[:,0]=np.interp(sdf,[sdf.min(),sdf.max()],[0,1])
    colors[inds[0]]=np.array([1,0,0,1])
    colors[inds2[0]]=np.array([0,1,0,1])
    colors=colors.astype('f4')

    len(vertices)
    print 'vertices.shape', vertices.shape, vertices.dtype
    print 'faces.shape', faces.shape,faces.dtype

    normals=auto_normals(vertices,faces)

    print vertices.min(),vertices.max(),vertices.mean()
    print normals.min(),normals.max(), normals.mean()

    print vertices.dtype,faces.dtype, colors.dtype, normals.dtype    
    
    from fos.actor.surf import Surface
    from fos import Window, World, DefaultCamera
    
    aff = np.eye(4, dtype = np.float32)
    #aff[0,3] = 30
    s=Surface(vertices,faces,colors,normals=normals, affine = aff,)
    
    w=World()
    w.add(s)
    #w.add(s2)
    
    cam=DefaultCamera()
    w.add(cam)
    
    wi=Window()
    wi.attach(w)
    '''









