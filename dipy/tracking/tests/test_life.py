import os

import numpy as np
import numpy.testing as npt
import numpy.testing.decorators as dec
import scipy.sparse as sps
import scipy.linalg as la

import nibabel as nib

import dipy.tracking.life as life
import dipy.tracking.eudx as edx
import dipy.core.sphere as dps
import dipy.core.gradients as dpg
import dipy.data as dpd
import dipy.core.optimize as opt
import dipy.core.ndindex as nd
import dipy.core.gradients as grad
import dipy.reconst.dti as dti


def test_fit_data():
    fdata, fbval, fbvec = dpd.get_data('small_25')
    gtab = grad.gradient_table(fbval, fbvec)
    ni_data = nib.load(fdata)
    data = ni_data.get_data()
    dtmodel = dti.TensorModel(gtab)
    dtfit = dtmodel.fit(data)
    sphere = dpd.get_sphere()
    peak_idx = dti.quantize_evecs(dtfit.evecs, sphere.vertices)
    eu = edx.EuDX(dtfit.fa.astype('f8'), peak_idx,
                  seeds=list(nd.ndindex(data.shape[:-1])),
                  odf_vertices=sphere.vertices, a_low=0)
    tensor_streamlines = [streamline for streamline in eu]
    life_model = life.FiberModel(gtab)
    life_fit = life_model.fit(data, tensor_streamlines)
    model_error = life_fit.predict() - life_fit.data
    model_rmse = np.sqrt(np.mean(model_error ** 2, -1))
    matlab_rmse, matlab_weights = dpd.matlab_life_results()
    # Lower error than the matlab implementation for these data:
    npt.assert_(np.median(model_rmse) < np.median(matlab_rmse))
    # And a moderate correlation with the Matlab implementation weights:
    npt.assert_(np.corrcoef(matlab_weights, life_fit.beta)[0, 1] > 0.6)
