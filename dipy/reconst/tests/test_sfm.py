import numpy as np
import numpy.testing as npt
import nibabel as nib
import dipy.reconst.sfm as sfm
import dipy.data as dpd
import dipy.core.gradients as grad


def test_design_matrix():
    data, gtab = dpd.dsi_voxels()
    sphere = dpd.get_sphere()
    sparse_fascicle_model = sfm.SparseFascicleModel(gtab, sphere)

    npt.assert_equal(sparse_fascicle_model.design_matrix.shape,
                     (np.sum(~gtab.b0s_mask), sphere.vertices.shape[0]))


def test_SparseFascicleModel():
    fdata, fbvals, fbvecs = dpd.get_data()
    data = nib.load(fdata).get_data()
    gtab = grad.gradient_table(fbvals, fbvecs)
    sfmodel = sfm.SparseFascicleModel(gtab)
    sffit1 = sfmodel.fit(data[0, 0, 0])
    pred1 = sffit1.predict(gtab)
    mask = np.ones(data.shape[:-1])
    sffit2 = sfmodel.fit(data, mask)
    pred2 = sffit2.predict(gtab)
    sffit3 = sfmodel.fit(data)
    pred3 = sffit3.predict(gtab)
