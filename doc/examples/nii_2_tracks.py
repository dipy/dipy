""" 

============================
From raw data to streamlines
============================

Overview
========

First import the necessary modules
----------------------------------

``numpy`` is for numerical computation

"""

import numpy as np

"""
``nibabel`` is for data formats
"""

import nibabel as nib

"""
``dipy.reconst`` is for the reconstruction algorithms which we use to create directionality models 
for a voxel from the raw data. 
"""

import dipy.reconst.dti as dti

"""
``dipy.tracking`` is for tractography algorithms which create sets of tracks by integrating 
  directionality models across voxels.
"""

from dipy.tracking.eudx import EuDX

"""
``dipy.data`` is used for small datasets that we use in tests and examples.
"""

from dipy.data import fetch_beijing_dti, read_beijing_dti

"""
Fetch will download the raw dMRI dataset of a single subject. The size of the dataset is 51 MBytes.
"""

fetch_beijing_dti()

"""
Next, we read the saved dataset
"""

img, gtab = read_beijing_dti()

data=img.get_data()
print('data.shape (%d,%d,%d,%d)' % data.shape)

affine=img.get_affine()

