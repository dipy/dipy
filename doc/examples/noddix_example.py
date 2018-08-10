from __future__ import division
import time as time
import numpy as np
import nibabel as nib
from dipy.data import get_data
from dipy.reconst.shore import ShoreModel
from dipy.core.gradients import gradient_table
import dipy.reconst.NODDIx as noddix
from scipy.linalg import get_blas_funcs
from dipy.data import get_sphere
# from dipy.io import read_bvals_bvecs
sphere = get_sphere('repulsion724')
gemm = get_blas_funcs("gemm")

fname, fscanner = get_data('small_NODDIx_data')
params = np.loadtxt(fscanner)