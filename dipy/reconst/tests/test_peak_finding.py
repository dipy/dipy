import numpy as np
from nose.tools import assert_true, assert_false, assert_equal, assert_almost_equal, assert_raises
from dipy.reconst.recspeed import peak_finding, peak_finding_onedge
from dipy.data import get_sphere, get_data
from dipy.core.geometry import vec2vec_rotmat, reduce_antipodal, unique_edges

def all_evecs(e0):
    axes=np.array([[1.,0,0],[0,1.,0],[0,0,1.]])
    mat=vec2vec_rotmat(axes[2],e0)
    e1=np.dot(mat,axes[0])
    e2=np.dot(mat,axes[1])
    return np.array([e0,e1,e2])

def ODF(odf_verts,mf,mevals,mevecs):
    odf=np.zeros(len(odf_verts))
    m=len(mf)
    for (i,v) in enumerate(odf_verts):
        for (j,f) in enumerate(mf):
            evals=mevals[j]
            evecs=mevecs[j]
            D=np.dot(np.dot(evecs,np.diag(evals)),evecs.T)
            iD=np.linalg.inv(D)
            nD=np.linalg.det(D)
            upper=(np.dot(np.dot(v.T,iD),v))**(-3/2.)
            lower=4*np.pi*np.sqrt(nD)
            odf[i]+=f*upper/lower
    return odf

def test_peak_finding():

    vertices,faces=get_sphere('symmetric724')
    odf=np.zeros(len(vertices))
    odf = np.abs(vertices.sum(-1))

    odf[1] = 10.
    odf[505] = 505.
    odf[143] = 143.

    peaks,inds=peak_finding(odf.astype('f8'),faces.astype('uint16'))
    print peaks, inds
    edges = unique_edges(faces)
    peaks, inds = peak_finding_onedge(odf, edges)
    print peaks, inds
    vertices_half, edges_half, faces_half = reduce_antipodal(vertices, faces)
    n = len(vertices_half)
    peaks, inds = peak_finding_onedge(odf[:n], edges_half)
    print peaks, inds
    mevals=np.array(([0.0015,0.0003,0.0003],
                    [0.0015,0.0003,0.0003]))
    e0=np.array([1,0,0.])
    e1=np.array([0.,1,0])
    mevecs=[all_evecs(e0),all_evecs(e1)]
    odf = ODF(vertices,[0.5,0.5],mevals,mevecs)
    peaks,inds=peak_finding(odf,faces)
    print peaks, inds
    peaks2, inds2 = peak_finding_onedge(odf[:n], edges_half)
    print peaks2, inds2
    assert_equal(len(peaks),2)
    assert_equal(len(peaks2),2)
    


    
 
if __name__ == '__main__':
    test_peak_finding()

