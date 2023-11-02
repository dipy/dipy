"""
=========================
Getting started with DIPY
=========================

In diffusion MRI (dMRI) usually we use three types of files, a Nifti file with
the diffusion weighted data, and two text files one with b-values and
one with the b-vectors.

In DIPY_ we provide tools to load and process these files and we also provide
access to publicly available datasets for those who haven't acquired yet
their own datasets.

Let's start with some necessary imports.
"""

from os.path import expanduser, join

import matplotlib.pyplot as plt

from dipy.core.gradients import gradient_table
from dipy.data import fetch_sherbrooke_3shell
from dipy.io import read_bvals_bvecs
from dipy.io.image import load_nifti, save_nifti

###############################################################################
# With the following commands we can download a dMRI dataset

fetch_sherbrooke_3shell()

###############################################################################
# By default these datasets will go in the ``.dipy`` folder inside your home
# directory. Here is how you can access them.

home = expanduser('~')

###############################################################################
# ``dname`` holds the directory name where the 3 files are in.

dname = join(home, '.dipy', 'sherbrooke_3shell')

###############################################################################
# Here, we show the complete filenames of the 3 files

fdwi = join(dname, 'HARDI193.nii.gz')

print(fdwi)

fbval = join(dname, 'HARDI193.bval')

print(fbval)

fbvec = join(dname, 'HARDI193.bvec')

print(fbvec)

###############################################################################
# Now, that we have their filenames we can start checking what these look like.
#
# Let's start first by loading the dMRI datasets. For this purpose, we
# use a python library called nibabel_ which enables us to read and write
# neuroimaging-specific file formats.

data, affine, img = load_nifti(fdwi, return_img=True)

###############################################################################
# ``data`` is a 4D array where the first 3 dimensions are the i, j, k voxel
# coordinates and the last dimension is the number of non-weighted (S0s) and
# diffusion-weighted volumes.
#
# We can very easily check the size of ``data`` in the following way:

print(data.shape)

###############################################################################
# We can also check the dimensions of each voxel in the following way:

print(img.header.get_zooms()[:3])

###############################################################################
# We can quickly visualize the results using matplotlib_. For example,
# let's show here the middle axial slices of volume 0 and volume 10.

axial_middle = data.shape[2] // 2
plt.figure('Showing the datasets')
plt.subplot(1, 2, 1).set_axis_off()
plt.imshow(data[:, :, axial_middle, 0].T, cmap='gray', origin='lower')
plt.subplot(1, 2, 2).set_axis_off()
plt.imshow(data[:, :, axial_middle, 10].T, cmap='gray', origin='lower')
plt.show()
plt.savefig('data.png', bbox_inches='tight')

###############################################################################
# .. rst-class:: centered small fst-italic fw-semibold
#
# Showing the middle axial slice without (left) and with (right) diffusion
# weighting.
#
#
#
# The next step is to load the b-values and b-vectors from the disk using
# the function ``read_bvals_bvecs``.

bvals, bvecs = read_bvals_bvecs(fbval, fbvec)

###############################################################################
# In DIPY, we use an object called ``GradientTable`` which holds all the
# acquisition specific parameters, e.g. b-values, b-vectors, timings and
# others. To create this object you can use the function ``gradient_table``.

gtab = gradient_table(bvals, bvecs)

###############################################################################
# Finally, you can use ``gtab`` (the GradientTable object) to show some
# information about the acquisition parameters

print(gtab.info)

###############################################################################
# You can also see the b-values using:

print(gtab.bvals)

###############################################################################
# Or, for example the 10 first b-vectors using:

print(gtab.bvecs[:10, :])

###############################################################################
# ``gtab``  can be used to tell what part of the data is the S0 volumes
# (volumes which correspond to b-values of 0).

S0s = data[:, :, :, gtab.b0s_mask]

###############################################################################
# Here, we had only 1 S0 as we can verify by looking at the dimensions of S0s

print(S0s.shape)

###############################################################################
# Just, for fun let's save this in a new Nifti file.

save_nifti('HARDI193_S0.nii.gz', S0s, affine)

###############################################################################
# Now, that we learned how to load dMRI datasets we can start the analysis.
# See example :ref:`sphx_glr_examples_built_reconstruction_reconst_dti.py` to
# learn how to create FA maps.
