"""
To use:

import profile_dti as p
import dipy.reconst.dti as dti
lprun -f dti.restore_fit_tensor -f p.tm.fit_method p.func()

"""

import dipy.core.gradients as grad
import dipy.data as dpd
import dipy.reconst.dti as dti

img, gtab = dpd.read_stanford_hardi()
dd = img.get_data()

tm = dti.TensorModel(gtab)
tf = tm.fit(dd)

def func():
    tf.odf(dpd.default_sphere)

if __name__=="__main__":
    func()
