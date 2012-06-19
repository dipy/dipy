import numpy as np
from enthought.mayavi import mlab

def show_blobs(blobs, v, faces,fa_slice=None,colormap='jet',size=1.5,scale=2.2,norm=True):
    """Mayavi gets really slow when triangular_mesh is called too many times
    so this function stacks blobs and calls triangular_mesh once
    """
 
    print blobs.shape
    xcen = blobs.shape[0]/2.
    ycen = blobs.shape[1]/2.
    zcen = blobs.shape[2]/2.
    faces = np.asarray(faces, 'int')
    xx = []
    yy = []
    zz = []
    count = 0
    ff = []
    mm = []
    for ii in xrange(blobs.shape[0]):
        for jj in xrange(blobs.shape[1]):
            for kk in xrange(blobs.shape[2]):
                m = blobs[ii,jj,kk]
                if norm==True:
                    m /= (size*abs(m).max())                
                x, y, z = v.T*m/size                
                x += scale*(ii - xcen)
                y += scale*(jj - ycen)
                z += scale*(kk - zcen)                
                ff.append(count+faces)
                count += len(x)
                xx.append(x)
                yy.append(y)
                zz.append(z)
                mm.append(m)
    ff = np.concatenate(ff)
    xx = np.concatenate(xx)
    yy = np.concatenate(yy)
    zz = np.concatenate(zz)
    mm = np.concatenate(mm)
    mlab.triangular_mesh(xx, yy, zz, ff, scalars=mm, colormap=colormap)
    if fa_slice!=None:        
        mlab.imshow(fa_slice, colormap='gray', interpolate=False)
    mlab.colorbar()
    mlab.show()
 
