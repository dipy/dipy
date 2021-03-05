'''
Created on Feb 18, 2021

@author: patrykfi
'''

import os.path
import numpy as np

from dipy.data import Sphere
from dipy.reconst.gqi import GeneralizedQSamplingModel
from dipy.reconst.dsi import DiffusionSpectrumModel

from dipy.direction import peaks_from_model
from scipy.io import loadmat


def dsiSphere8Fold():
    dsi_sphere = loadmat(os.path.join(
        os.path.dirname(__file__), "../data/files/dsi_sphere_8fold.mat"
    ))
    return Sphere(
        x=dsi_sphere['odf_vertices'][0,:],
        y=dsi_sphere['odf_vertices'][1,:],
        z=dsi_sphere['odf_vertices'][2,:],
        faces=dsi_sphere['odf_faces'].T
    )


class OdffpModel(object):

    def __init__(self, gtab):
        self.gtab = gtab
    
    
    def _get_rotation_of_vector(self, A, B):
        cAB = np.cross(A,B)
        sAB = np.array([[0,-cAB[2],cAB[1]], [cAB[2],0,-cAB[0]], [-cAB[1],cAB[0],0]])
        R = np.eye(3) + sAB + np.dot(sAB,sAB) * (1-np.dot(A,B)) / np.sum(cAB**2)
    
        if np.all(np.isnan(R)):
            R = np.eye(3)

        return R
    
    
    def _get_rotations_to_main_peak(self, peak_dirs):
        rotations = np.zeros(peak_dirs.shape + (3,3))
        
        for x in range(peak_dirs.shape[0]):
            for y in range(peak_dirs.shape[1]):
                for z in range(peak_dirs.shape[2]):

                    rotations[x,y,z,:,:] = self._get_rotation_of_vector(
                        peak_dirs[x,y,z,0,:], [0,0,1]
                    )
        
        return rotations
        
        
    def fit(self, data):
        tessellation = dsiSphere8Fold()
        
        diff_model = GeneralizedQSamplingModel(self.gtab)
        data_peaks = peaks_from_model(
            model=diff_model, data=data, sphere=tessellation, return_odf=True,
            relative_peak_threshold=.5, min_separation_angle=25, npeaks=1
        )
        
        self._rotations = self._get_rotations_to_main_peak(data_peaks.peak_dirs)
                
        return data_peaks.odf

        
class OdffpFit(object):
    
    def __init__(self):
        pass