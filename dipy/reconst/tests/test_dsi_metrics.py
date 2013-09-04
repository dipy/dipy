import nibabel as nib
import numpy as np
from dipy.align.aniso2iso import resample
from dipy.core.ndindex import ndindex
from dipy.data import get_sphere
import dipy.reconst.dti as dti
from dipy.data import get_data, dsi_voxels
from dipy.core.sphere import Sphere
from dipy.core.gradients import gradient_table
from dipy.io.gradients import read_bvals_bvecs
from numpy.testing import (assert_equal,
                           assert_almost_equal,
                           run_module_suite,
                           assert_array_equal,
                           assert_raises)
from dipy.data import get_data, dsi_voxels
from dipy.reconst.dsi import DiffusionSpectrumModel
from dipy.reconst.odf import gfa, peak_directions
from dipy.sims.voxel import (SticksAndBall, MultiTensor, multi_tensor_odf, add_noise, )
from dipy.core.sphere import Sphere
from dipy.core.gradients import gradient_table
from dipy.data import get_sphere
from numpy.testing import assert_equal
from dipy.core.subdivide_octahedron import create_unit_sphere
from dipy.core.sphere_stats import angular_similarity

def hanning_width(attenuation,radius):
  return 2*np.pi*radius/(np.arccos(attenuation))

def create_table(radius, bmax):
    N = 0
    for i in range(-radius, radius + 1):
        for j in range(-radius, radius + 1):
            for k in range(-radius, radius + 1):
                if (i ** 2 + j ** 2 + k ** 2) <= radius ** 2:
                  N += 1
    bvecs = np.zeros((N, 3))
    bvals = np.zeros(N)
    N = 0
    for i in range(-radius, radius + 1):
        for j in range(-radius, radius + 1):
            for k in range(-radius, radius + 1):
                if (i ** 2 + j ** 2 + k ** 2) <= radius ** 2:
                  bvecs[N, :] = i, j, k
                  N += 1
    rays = np.sqrt(bvecs[:, 0] ** 2 + bvecs[:, 1] ** 2 + bvecs[:, 2] ** 2)
    I = np.argsort(rays)
    rays = rays[I]
    bvecs = bvecs[I, :]
    bvals = (rays**2) * bmax / (float(rays.max())**2)
    bvecs[1:] = bvecs[1:] / rays[1:, None].astype(np.float)
    return bvecs, bvals

def test_dsi():
    # load symmetric 724 sphere
    sphere = get_sphere('symmetric724')
    # load icosahedron sphere
    sphere2 = create_unit_sphere(5)
    btable = np.loadtxt(get_data('dsi515btable'))
    gtab = gradient_table(btable[:, 0], btable[:, 1:])
    data, golden_directions = SticksAndBall(gtab, d=0.0015,
                                            S0=100, angles=[(0, 0), (90, 0)],
                                            fractions=[50, 50], snr=None)
    data = data / float(data[0])
    dsmodel = DiffusionSpectrumModel(gtab)
    rtop_signal_norm = dsmodel.fit(data).rtop_signal()
    rtop_pdf_norm = dsmodel.fit(data).rtop_pdf()
    rtop_pdf = dsmodel.fit(data).rtop_pdf(normalized=False)
    assert_almost_equal(rtop_signal_norm, rtop_pdf, 10)
    # We need a test for MSD!!!
    mevals0 = np.array(([0.0015, 0.0003, 0.0003],[0.0015, 0.0003, 0.0003]))
    mevals1 = np.array(([0.0030, 0.0006, 0.0006],[0.0030, 0.0006, 0.0006]))
    vecs,bvals=create_table(40,40.0*11538.0/5.0)
    gtab = gradient_table(bvals, bvecs)
    gridsize = 81
    angl=[(0,0),(60,0)]

    S_0, sticks_0 = MultiTensor(gtab, mevals0, S0=100, angles=angl, fractions=[50, 50], snr=None)
    S_1, sticks_0 = MultiTensor(gtab, mevals1, S0=100, angles=angl, fractions=[50, 50], snr=None)
    MSD_norm_0 = dsmodel.fit(S_0).msd_discrete(normalized=True)
    MSD_norm_1 = dsmodel.fit(S_1).msd_discrete(normalized=True)
    assert_almost_equal(MSD_norm_0, 0.5*MSD_norm_1, 10)
    print(MSD_norm_0)
    print(MSD_norm_1)
    
if __name__ == '__main__':
    run_module_suite()
