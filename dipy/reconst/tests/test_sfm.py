import numpy as np
import numpy.testing as npt
import dipy.reconst.sfm as sfm
import dipy.data as dpd


def test_design_matrix():
    data, gtab = dpd.dsi_voxels()
    sphere = dpd.get_sphere()
    sparse_fascicle_model = sfm.SparseFascicleModel(gtab, sphere)

    npt.assert_equal(sparse_fascicle_model.design_matrix.shape,
                     (np.sum(~gtab.b0s_mask), sphere.vertices.shape[0]))
