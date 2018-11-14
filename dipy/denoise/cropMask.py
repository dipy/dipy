import numpy as np
import SimpleITK as sitk
import nibabel as nib
import multiprocessing
import sys
import ctypes
from contextlib import redirect_stdout
# import itk
import os
import matplotlib.pyplot as plt
# import scipy.ndimage.measurements as imageMeasure

file_name = 'subj3_Philips_s0_all_exps_64d_AP_A_b1000_3_1.nii'

arr = (nib.load(file_name)).get_data()
size1 = arr.shape[0] * arr.shape[1] * arr.shape[2] * arr.shape[3]
mp_arr = multiprocessing.RawArray(ctypes.c_double, size1)
shared_arr = np.frombuffer(mp_arr)
shared_input = shared_arr.reshape(arr.shape)
shared_input[:] = arr[:]

image = sitk.ReadImage(file_name)
Origin = np.asarray(image.GetOrigin())
Size = np.asarray(image.GetSize())
Spacing = np.asarray(image.GetSpacing())
Direction = np.array(image.GetDirection())
Direction = Direction.reshape(4, 4)
old_FOV = Size * Spacing

# sys.stdout = open('draft.txt', 'w')
# print(size1, Origin)
# sys.stdout = sys.__stdout__
direction = Direction[:3,:3]
spacing = Spacing[:3]
origin = Origin[:3]
pixel0 = [0,0,0]
physCoor = direction*spacing.T*pixel0 + np.eye(3) * origin
oldfov = np.eye(3)*old_FOV[:3]
# Assume that  newfov = [150,150,100]
new_FOV = [150,150,100]
# Checking conditions for croping image
new_size = np.zeros(3)
for f, fov in enumerate(old_FOV[:3]):
    if (abs(new_FOV[f] - old_FOV[f]) >= 2 * Spacing[f]):
        new_size[f] = np.ceil(new_FOV[f]/spacing[f])
        print("Croping data, new size: ", new_size[f])

new_center_index = (new_size - 1) // 2
scl = spacing * np.eye(3)
scl = scl * direction

new_origin = np.zeros(3)
for r in range(0,3):
    sm = 0
    for c in range (0,3):
        sm = sm + scl[r,c] * new_center_index[c]
        print(sm)
    # new_origin[r] =


"""with open('draft.txt', 'w') as f:
    with redirect_stdout(f):
        print("Printing data into result file")
        print("Origin: ", Origin)
        print("Spacing: ", Spacing)
        print("original Fov: ", old_FOV )"""

