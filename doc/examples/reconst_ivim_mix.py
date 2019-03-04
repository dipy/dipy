import matplotlib.pyplot as plt
import dipy.reconst.ivim as ivim
from time import time
import numpy as np
import nibabel as nib
from dipy.data.fetcher import read_ivim


img, gtab = read_ivim()
data = img.get_data()

z = 33
b = 0

plt.imshow(data[:, :, z, b].T, origin='lower', cmap='gray',
           interpolation='nearest')
plt.axhline(y=100)
plt.axvline(x=170)
plt.savefig("ivim_data_slice.png")
plt.close()

x1, x2 = 90, 155
y1, y2 = 90, 170
data_slice = data[x1:x2, y1:y2, z:z + 1, :]

ivim_model = ivim.IvimModel(gtab, fit_method='VarPro')
bvals = ivim_model.bvals
t1 = time()

ivim_fit = ivim_model.fit(data_slice)

t2 = time()
fast_time = t2 - t1
print(fast_time)


def ivim_mix_prediction(params, gtab, S0=1):

        """
        The Intravoxel incoherent motion (IVIM) model function.

        Parameters
        ----------
        params : array
            An array of IVIM parameters - [S0, f, D_star, D].

        gtab : GradientTable class instance
            Gradient directions and bvalues.

        S0 : float, optional
            This has been added just for consistency with the existing
            API. Unlike other models, IVIM predicts S0 and this is over written
            by the S0 value in params.

        Returns
        -------
        S : array
        An array containing the IVIM signal estimated using given parameters.
        """
        f, D_star, D = params
        b = gtab.bvals
        S0 = 1
        S = S0 * (f * np.exp(-b * D_star) + (1 - f) * np.exp(-b * D))
        return S


i, j = 10, 10
ivimx_predict = ivim_mix_prediction(ivim_fit.model_params[i, j, :][0], gtab)
plt.scatter(gtab.bvals, data_slice[i, j, :],
            color="green", label="Actual signal")
plt.plot(gtab.bvals, ivimx_predict*4000, color="red",
         label="Estimated Signal")
plt.xlabel("bvalues")
plt.ylabel("Signals")

f_est = ivim_fit.model_params[i, j, :][0][0]
D_star_est = ivim_fit.model_params[i, j, :][0][1]
D_est = ivim_fit.model_params[i, j, :][0][2]

text_fit = """Estimated \n f={:06.4f}\n
            D*={:06.5f} D={:06.5f}""".format(f_est, D_star_est, D_est)

plt.text(0.65, 0.50, text_fit, horizontalalignment='center',
         verticalalignment='center', transform=plt.gca().transAxes)
plt.legend(loc='upper right')
plt.savefig("ivim_voxel_plot.png")

affine = img.affine.copy()
nib.save(nib.Nifti1Image(ivim_fit[:, :, 0], affine), 'f_ivim005.nii.gz')
nib.save(nib.Nifti1Image(ivim_fit[:, :, 1], affine), 'D_star_ivim005.nii.gz')
nib.save(nib.Nifti1Image(ivim_fit[:, :, 2], affine), 'D_ivim005.nii.gz')
nib.save(nib.Nifti1Image(ivim_fit[:, :, 0] * ivim_fit[:, :, 1], affine),
         'fD_star_ivim005.nii.gz')
