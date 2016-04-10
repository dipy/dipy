"""
To use:

import profile_dti as p
import dipy.reconst.dti as dti
lprun -f dti.restore_fit_tensor -f p.tm.fit_method p.func()

"""

import nibabel as nib
import dipy.core.gradients as grad
import dipy.data as dpd
import dipy.reconst.dti as dti

data, bvals, bvecs = dpd.get_data('small_25')
dd = nib.load(data).get_data()
gtab = grad.gradient_table(bvals, bvecs)


fit_method = 'restore' # 'NLLS'
jac = True # False

# To profile RESTORE, set some of the signals to be outliers (otherwise comment
# out the following line):
dd[..., 5] = 1.0

tm = dti.TensorModel(gtab, fit_method=fit_method, jac=True, sigma=10)

def func():
    tf = tm.fit(dd)

if __name__=="__main__":
    func()
