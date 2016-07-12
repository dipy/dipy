"""
============================================================
Intravoxel incoherent motion
============================================================
The intravoxel incoherent motion (IVIM) model describes diffusion
and perfusion in the signal acquired with diffusion. The IVIM model
can be understood as an adaptation of the work of Stejskals and Tanners
[Stejskal65]_ in biologic tissue, and was proposed by Le Bihan [LeBihan84]_
The model posits that two compartments exist: a slow moving compartment,
where particles diffuse in a Brownian fashion as a consequence of thermal
energy, and a fast moving compartment (the vascular compartment), where
blood moves as a consequence of pressure gradient. In this second
compartment, a pseudo diffusion term (D* ) is introduced that describes
the displacement of the blood elements in an assumed randomly laid out
vascular network, at the macroscopic level. For the perfusion to have a
physiological meaning, one expects that D* is greater than D.

The IVIM model expresses the diffusion signal as follows:

 .. math: :

    S(b) = S_{0}[fe ^ {-b*D_star} + (1 - f)e^{-b*D}]

where $\mathbf{b}$ is the gradient value (which is dependent on
the measurement parameters), $S_0$ is the signal in the absence
of diffusion gradient sensitization, $\mathbf{f}$ is the perfusion
fraction, $\mathbf{D}$ is the diffusion coefficient,
 $\mathbf{D * }$ is the pseudo-diffusion diffusion constant,
 due to vascular contributions

In the following example we show how to fit the IVIM model
on a diffusion weighted dataset and visualize the perfusion
and diffusion. First, we import all relevant modules:
"""

import matplotlib.pyplot as plt

from dipy.reconst.ivim import IvimModel, ivim_function
from dipy.data.fetcher import read_ivim

"""
We get an IVIM dataset using Dipy's data fetcher `read_ivim`.
This dataset was acquired with 21 b-values in 3 different directions.
Volumes corresponding to different directions were registered to each
other, and averaged across directions. Thus, this dataset has 4 dimensions,
with the length of the last dimension corresponding to the number
of b-values.
"""

img, gtab = read_ivim()

"""
img contains a nibabel Nifti1Image object(with the data)
and gtab contains a GradientTable object(information about
the gradients e.g. b-values and b-vectors). We get the
data from img using `read_data`.
"""

data = img.get_data()
print('data.shape (%d, %d, %d, %d)' % data.shape)

"""
The data has 54 slices, with 256-by-256 voxels in each slice.
The fourth dimension corresponds to the b-values in the gtab.
Let us visualize the data by taking a slice midway(z=27) at bvalue = 0.
"""

z = 33
b = 20

plt.imshow(data[:, :, z, b].T, origin='lower', cmap='gray')
plt.axhline(y=100)
plt.axvline(x=170)
plt.savefig("ivim_data_slice.png")
plt.close()

"""
The marked point in the image shows a section containing cerebral spinal fluid (CSF)
so it should have a very high f and D*, the area between the right and left is white
matter so that should be lower, and the region on the right is gray matter and CSF.
That should give us some contrast to see the values varying across the regions.
"""

x1, x2 = 160, 180
y1, y2 = 90, 110
data_slice = data[x1:x2, y1:y2, z, :]

plt.imshow(data[x1:x2, y1:y2, z, b].T, origin='lower', cmap="gray")
plt.savefig("CSF_slice.png")
plt.close()

"""
Now that we have prepared the datasets we can go forward with
the ivim fit. Instead of fitting the entire volume, we focus on a
small section of the slice, to fit the IVIM model.
First, we instantiate the Ivim model in the following way.
"""

ivimmodel = IvimModel(gtab)

"""
To fit the model, call the `fit_method`, specifying the fit_method.
Here, we use a two-stage approach: first, a tensor is fit to the data,
and then this tensor as the initial starting point for the non-linear
fit of IVIM parameters.
"""

ivimfit = ivimmodel.fit(data_slice, fit_method="two_stage")

"""
The fit method creates a IvimFit object which contains the
fitting parameters of the model the fit parameters of the model.
These are accessible through the model_parameters attribute of the
IvimFit object. Parameters are arranged as a 4D array, corresponding
to the spatial dimensions of the data, and the last dimension (of length 4)
corresponding to the model parameters according to the following
order: S0, f, D* and D.
"""

ivimparams = ivimfit.model_params
print("ivimparams.shape : {}".format(ivimparams.shape))

"""
As we see, we have a 20x20 slice at the height z = 33. Thus we
have 400 voxels. We will now plot the values of S0, f, D* and D
for some voxels and also the various maps for the entire slice.
This will give us an idea about the diffusion and perfusion in
the section. Let(i, j) denote the coordinate of the voxel.
"""

i, j = 10, 10
estimated_params = ivimfit.model_params[i, j, :]
print(estimated_params)

"""
Let us define a plotting our results. For this we will use the
ivim_function defined in the module ivim which takes bvalues
and ivim parameters and returns the estimated signal.
"""

estimated_signal = ivim_function(estimated_params, gtab.bvals)

plt.scatter(gtab.bvals, data_slice[i, j, :], color="green",
            label="Actual signal")
plt.scatter(gtab.bvals, estimated_signal,
            color="red", label="Estimated Signal")
plt.xlabel("bvalues")
plt.ylabel("Signals")

S0_est, f_est, D_star_est, D_est = estimated_params
text_fit = """Estimated \n S0={:06.3f} f={:06.4f}\n
            D*={:06.5f} D={:06.5f}""".format(S0_est, f_est, D_star_est, D_est)

plt.text(0.65, 0.50, text_fit, horizontalalignment='center',
         verticalalignment='center', transform=plt.gca().transAxes)
plt.legend(loc='upper left')
plt.savefig("ivim_voxel_plot.png")
plt.close()

"""
Now we can map the perfusion and diffusion maps for the slice. We
will plot a heatmap showing the values using a colormap. It will be
useful to define a plotting function for the heatmap here since we
will use it to plot for all the IVIM parameters. We will need to specify
the lower and upper limits for our data. For example, the perfusion
fractions should be in the range (0,1). Similarly, the diffusion and
pseudo-diffusion constants are much smaller than 1. We pass an argument
called "variable" to out plotting function which gives the label for
the plot.
"""

def plot_map(raw_data, variable, limits):
    lower, upper = limits
    plt.title('Map for {}'.format(variable))
    plt.imshow(raw_data.T, origin='lower', clim=(lower, upper), cmap="gray")
    plt.colorbar()
    plt.savefig(variable + ".png")
    plt.close()

"""
Let us get the various plots so that we can visualize them in one page
"""

plot_map(ivimparams[:, :, 0], "S0", (0, 10000))
plot_map(data_slice[:, :, 0], "S at b = 0", (0, 10000))

plot_map(ivimparams[:, :, 1], "f", (0, 1))
plot_map(ivimparams[:, :, 2], "D*", (0, 0.01))
plot_map(ivimparams[:, :, 3], "D", (0, 0.001))

"""
References:

.. [Stejskal65] Stejskal, E. O.; Tanner, J. E. (1 January 1965).
                "Spin Diffusion Measurements: Spin Echoes in the Presence
                of a Time-Dependent Field Gradient". The Journal of Chemical
                Physics 42 (1): 288. Bibcode:1965JChPh..42..288S.
                doi:10.1063/1.1695690.

.. [LeBihan84] Le Bihan, Denis, et al. "Separation of diffusion
               and perfusion in intravoxel incoherent motion MR
               imaging." Radiology 168.2 (1988): 497-505.

.. include:: ../links_names.inc
"""
