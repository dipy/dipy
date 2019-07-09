
import numpy as np
import nibabel as nib
import dipy.reconst.dti as dti
import dipy.data as dpd
import dipy.core.gradients as grad

b0 = 1000.
bvecs, bval = dpd.read_bvec_file(dpd.get_fnames('55dir_grad.bvec'))
gtab = grad.gradient_table(bval, bvecs)
B = bval[1]

D = np.array([1., 1., 1., 0., 0., 1., -np.log(b0) * B]) / B
evals = np.array([2., 1., 0.]) / B
md = evals.mean()
tensor = dti.from_lower_triangular(D)

X = dti.design_matrix(bvecs, bval)

data = np.exp(np.dot(X,D))
data.shape = (-1,) + data.shape

dti_wls = dti.TensorModel(gtab)
fit_wls = dti_wls.fit(data)
fa1 = fit_wls.fa

noisy_data = np.copy(data)
noisy_data[..., -1] = 1.0

fit_wls_noisy = dti_wls.fit(noisy_data)
fa2 = fit_wls_noisy.fa

dti_restore = dti.TensorModel(gtab,  fit_method='RESTORE', sigma=67.)
fit_restore_noisy = dti_restore.fit(noisy_data)
fa3 = fit_restore_noisy.fa

print("FA for noiseless data: %s"%fa1)
print("FA for noise-introduced data: %s"%fa2)
print("FA for noise-introduced data, analyzed with RESTORE: %s"%fa3)
