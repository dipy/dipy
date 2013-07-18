import nibabel as nib
import dipy.core.gradients as grad
import dipy.data as dpd
import dipy.reconst.dti as dti

data, bvals, bvecs = dpd.get_data('small_25')
dd = nib.load(data).get_data()
gtab = grad.gradient_table(bvals, bvecs)

def func_jac():
    tm = dti.TensorModel(gtab, fit_method='NLLS', jac=True)
    tf = tm.fit(dd)

def func_no_jac():
    tm = dti.TensorModel(gtab, fit_method='NLLS', jac=False)
    tf = tm.fit(dd)

if __name__=="__main__":
    func()
