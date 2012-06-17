import numpy as np
from nose.tools import assert_true, assert_false, assert_equal, assert_almost_equal, assert_raises
from dipy.reconst.recspeed import peak_finding, peak_finding_onedge
from dipy.data import get_sphere, get_data
from dipy.core.geometry import reduce_antipodal, unique_edges

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
    odf = multi_tensor_odf(vertices,[0.5,0.5],mevals,mevecs)
    peaks,inds=peak_finding(odf,faces)
    print peaks, inds
    peaks2, inds2 = peak_finding_onedge(odf[:n], edges_half)
    print peaks2, inds2
    assert_equal(len(peaks),2)
    assert_equal(len(peaks2),2)

 
if __name__ == '__main__':
    test_peak_finding()

