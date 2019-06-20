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

filename = 'subj3_Philips_s0_all_exps_64d_AP_A_b1000_3_1.nii'
originalData = (nib.load(filename)).get_data()

image = sitk.ReadImage(filename)
Origin = np.asarray(image.GetOrigin())
Size = np.asarray(image.GetSize())
Spacing = np.asarray(image.GetSpacing())
Direction = np.array(image.GetDirection())
Direction = Direction.reshape(4, 4)
# old_FOV = Size * Spacing

# Changing size into 3D image
origin = Origin[:3]
size = Size[:3]
spacing = Spacing[:3]
direction = Direction[:3,:3]
oldFov = size*spacing

print(oldFov)
newFOV_1 = [150,150,100]
newFOV_2 = [250,250,200]


#
# validate_fov(oldFov, newFOV_1, spacing)
# validate_fov(oldFov, newFOV_2, spacing)


class FOV:
    def __init__(self, newfov, spacing, size, origin):
        self.oldfov = spacing * size
        self.newfov = newfov
        self.spacing = spacing
        self.size = size
        self.origin = origin

    def validate_fov(self):
        if all(abs(self.oldfov - self.newfov) >= 2*self.spacing):
            return True
        else:
            raise ValueError ('Invalid New FOV')
            # return False

    def calculate_add(self):

        sizefov = len(self.spacing)
        total_add_3D = np.zeros(sizefov).astype(int)
        for f, fov in enumerate(self.oldfov):
            if self.newfov[f] > fov:
                total_add_3D[f] = int(np.ceil((self.newfov[f] - fov)/self.spacing[f]))
                if ((total_add_3D[f] %2) == 1):
                    total_add_3D[f] +=1
            else:
                total_add_3D[f] = int(np.floor((fov - self.newfov[f])/self.spacing[f]))
                if ((total_add_3D[f] %2) == 1):
                    total_add_3D[f] -= 1
                total_add_3D[f] = - total_add_3D[f]
        return total_add_3D

    def calculate_newsize(self):
        return self.size + self.calculate_add()

    def calculate_newOriginIndex(self):
        return self.calculate_add() // 2

    def transformIndexToPhysiscalPoint(self, newIndex = None):
        if newIndex is None:
            point = self.calculate_newOriginIndex()
        else:
            point = newIndex

        oldOrigin = self.origin
        physicaladded = spacing * self.calculate_newOriginIndex()
        newPhysPoint = oldOrigin - physicaladded

        return newPhysPoint
    def SetActualOrigin(self):
        return - ((self.calculate_newsize()-1)/2) * self.spacing




fov1 = FOV( newFOV_1, spacing, size, origin)
fov2 = FOV( newFOV_2, spacing, size, origin)
print(fov1.newfov)
print(fov1.calculate_newsize())
