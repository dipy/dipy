"""
============================================================
Intravoxel incoherent motion
============================================================
The intravoxel incoherent motion (IVIM) model describes diffusion
and perfusion in the signal acquired with a diffusion MRI sequence
that contains multiple low b-values. The IVIM model can be understood
as an adaptation of the work of Stejskal and Tanner [Stejskal65]_
in biological tissue, and was proposed by Le Bihan [LeBihan84]_.
The model assumes two compartments: a slow moving compartment,
where particles diffuse in a Brownian fashion as a consequence of thermal
energy, and a fast moving compartment (the vascular compartment), where
blood moves as a consequence of a pressure gradient. In the first compartment,
the diffusion coefficient is $\mathbf{D}$ while in the second compartment, a
pseudo diffusion term $\mathbf{D^*}$ is introduced that describes the
displacement of the blood elements in an assumed randomly laid out vascular
network, at the macroscopic level. According to [LeBihan84]_,
$\mathbf{D^*}$ is greater than $\mathbf{D}$.

The IVIM model expresses the MRI signal as follows:

 .. math::
    S(b)=S_0(fe^{-bD^*}+(1-f)e^{-bD})

where $\mathbf{b}$ is the diffusion gradient weighing value (which is dependent
on the measurement parameters), $\mathbf{S_{0}}$ is the signal in the absence
of diffusion gradient sensitization, $\mathbf{f}$ is the perfusion
fraction, $\mathbf{D}$ is the diffusion coefficient and $\mathbf{D^*}$ is
the pseudo-diffusion constant, due to vascular contributions.

In the following example we show how to fit the IVIM model on a
diffusion-weighteddataset and visualize the diffusion and pseudo
diffusion coefficients. First, we import all relevant modules:
"""

import matplotlib.pyplot as plt

from dipy.reconst.ivim import IvimModel
from dipy.data.fetcher import read_ivim

"""
We get an IVIM dataset using DIPY_'s data fetcher ``read_ivim``.
This dataset was acquired with 21 b-values in 3 different directions.
Volumes corresponding to different directions were registered to each
other, and averaged across directions. Thus, this dataset has 4 dimensions,
with the length of the last dimension corresponding to the number
of b-values. In order to use this model the data should contain signals
measured at 0 bvalue.
"""

img, gtab = read_ivim()

"""
The variable ``img`` contains a nibabel NIfTI image object (with the data)
and gtab contains a GradientTable object (information about the gradients e.g.
b-values and b-vectors). We get the data from img using ``read_data``.
"""

data = img.get_data()
print('data.shape (%d, %d, %d, %d)' % data.shape)

"""
The data has 54 slices, with 256-by-256 voxels in each slice. The fourth
dimension corresponds to the b-values in the gtab. Let us visualize the data
by taking a slice midway(z=33) at $\mathbf{b} = 0$.
"""

z = 33
b = 0

plt.imshow(data[:, :, z, b].T, origin='lower', cmap='gray',
           interpolation='nearest')
plt.axhline(y=100)
plt.axvline(x=170)
plt.savefig("ivim_data_slice.png")
plt.close()

"""
.. figure:: ivim_data_slice.png
   :align: center

   Heat map of a slice of data

The region around the intersection of the cross-hairs in the figure
contains cerebral spinal fluid (CSF), so it so it should have a very high
$\mathbf{f}$ and $\mathbf{D^*}$, the area between the right and left is white
matter so that should be lower, and the region on the right is gray matter
and CSF. That should give us some contrast to see the values varying across
the regions.
"""

x1, x2 = 90, 155
y1, y2 = 90, 170
data_slice = data[x1:x2, y1:y2, z, :]

plt.imshow(data[x1:x2, y1:y2, z, b].T, origin='lower',
           cmap="gray", interpolation='nearest')
plt.savefig("CSF_slice.png")
plt.close()

"""
.. figure:: CSF_slice.png
   :align: center

   Heat map of the CSF slice selected.

Now that we have prepared the datasets we can go forward with
the ivim fit. Instead of fitting the entire volume, we focus on a
small section of the slice as selected aboove, to fit the IVIM model.
First, we instantiate the Ivim model. Using a two-stage approach: first,
a linear fit used to get quick initial guesses for the parameters
$\mathbf{S_{0}}$ and $\mathbf{D}$ by considering b-values greater than
``split_b_D`` (default: 400))and assuming a mono-exponential signal. This is
based on the assumption that at high b-values the signal can be approximated
as a mono exponential decay and by taking the logarithm of the signal values
a linear fit can be obtained. Another linear fit for ``S0`` (bvals <
``split_b_S0`` (default: 200)) follows and ``f`` is estimated using $1 -
S0_{prime}/S0$. Then a non-linear least squares fitting is performed to fit
``D_star`` and ``f``. If the ``two_stage`` flag is set to ``True`` while
initializing the model, a final non-linear least squares fitting is performed
for all the parameters using Scipy's ``leastsq`` or ``least_square`` function
depending on which Scipy version you are using. All initializations for the
model such as ``split_b_D`` are passed while creating the ``IvimModel``. If you
are using Scipy 0.17, you can also set bounds by setting ``bounds=([0., 0., 0.,
0.], [np.inf, 1., 1., 1.]))`` while initializing the ``IvimModel``. It is
recommeded that you upgrade to Scipy 0.17 since the fitting results might at
times return values which do not make sense physically (for example, a negative
$\mathbf{f}$).
"""

ivimmodel = IvimModel(gtab)

"""
To fit the model, call the `fit` method and pass the data for fitting.
"""

ivimfit = ivimmodel.fit(data_slice)

"""
The fit method creates a IvimFit object which contains the
parameters of the model obtained after fitting. These are accessible
through the `model_params` attribute of the IvimFit object.
The parameters are arranged as a 4D array, corresponding to the spatial
dimensions of the data, and the last dimension (of length 4)
corresponding to the model parameters according to the following
order : $\mathbf{S_{0}, f, D^*, D}$.
"""

ivimparams = ivimfit.model_params
print("ivimparams.shape : {}".format(ivimparams.shape))

"""
As we see, we have a 20x20 slice at the height z = 33. Thus we
have 400 voxels. We will now plot the parameters obtained from the
fit for a voxel and also various maps for the entire slice.
This will give us an idea about the diffusion and perfusion in
that section. Let(i, j) denote the coordinate of the voxel. We have
already fixed the z component as 33 and hence we will get a slice
which is 33 units above.

"""

i, j = 10, 10
estimated_params = ivimfit.model_params[i, j, :]
print(estimated_params)

"""
Next, we plot the results relative to the model fit.
For this we will use the `predict` method of the IvimFit object
to get the estimated signal.
"""

estimated_signal = ivimfit.predict(gtab)[i, j, :]

plt.scatter(gtab.bvals, data_slice[i, j, :],
            color="green", label="Actual signal")
plt.plot(gtab.bvals, estimated_signal, color="red", label="Estimated Signal")
plt.xlabel("bvalues")
plt.ylabel("Signals")

S0_est, f_est, D_star_est, D_est = estimated_params
text_fit = """Estimated \n S0={:06.3f} f={:06.4f}\n
            D*={:06.5f} D={:06.5f}""".format(S0_est, f_est, D_star_est, D_est)

plt.text(0.65, 0.50, text_fit, horizontalalignment='center',
         verticalalignment='center', transform=plt.gca().transAxes)
plt.legend(loc='upper right')
plt.savefig("ivim_voxel_plot.png")
plt.close()
"""
.. figure:: ivim_voxel_plot.png
   :align: center

   Plot of the signal from one voxel.

Now we can map the perfusion and diffusion maps for the slice. We
will plot a heatmap showing the values using a colormap. It will be
useful to define a plotting function for the heatmap here since we
will use it to plot for all the IVIM parameters. We will need to specify
the lower and upper limits for our data. For example, the perfusion
fractions should be in the range (0,1). Similarly, the diffusion and
pseudo-diffusion constants are much smaller than 1. We pass an argument
called ``variable`` to out plotting function which gives the label for
the plot.
"""


def plot_map(raw_data, variable, limits, filename):
    lower, upper = limits
    plt.title('Map for {}'.format(variable))
    plt.imshow(raw_data.T, origin='lower', clim=(lower, upper),
               cmap="gray", interpolation='nearest')
    plt.colorbar()
    plt.savefig(filename)
    plt.close()

"""
Let us get the various plots so that we can visualize them in one page
"""

plot_map(ivimfit.S0_predicted, "Predicted S0", (0, 10000), "predicted_S0.png")
plot_map(data_slice[:, :, 0], "Measured S0", (0, 10000), "measured_S0.png")
plot_map(ivimfit.perfusion_fraction, "f", (0, 1), "perfusion_fraction.png")
plot_map(ivimfit.D_star, "D*", (0, 0.01), "perfusion_coeff.png")
plot_map(ivimfit.D, "D", (0, 0.001), "diffusion_coeff.png")

"""
.. figure:: predicted_S0.png
   :align: center

   Heatmap of S0 predicted from the fit

.. figure:: measured_S0.png
   :align: center

   Heatmap of measured signal at bvalue = 0.

.. figure:: perfusion_fraction.png
   :align: center

   Heatmap of perfusion fraction values predicted from the fit

.. figure:: perfusion_coeff.png
   :align: center

   Heatmap of perfusion coefficients predicted from the fit.

.. figure:: diffusion_coeff.png
   :align: center

   Heatmap of diffusion coefficients predicted from the fit

References:

.. [Stejskal65] Stejskal, E. O.; Tanner, J. E. (1 January 1965).
                "Spin Diffusion Measurements: Spin Echoes in the Presence
                of a Time-Dependent Field Gradient". The Journal of Chemical
                Physics 42 (1): 288. Bibcode: 1965JChPh..42..288S.
                doi:10.1063/1.1695690.

.. [LeBihan84] Le Bihan, Denis, et al. "Separation of diffusion
               and perfusion in intravoxel incoherent motion MR
               imaging." Radiology 168.2 (1988): 497-505.

.. include:: ../links_names.inc
"""
