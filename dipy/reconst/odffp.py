'''
Created on Feb 18, 2021

@author: patrykfi
'''

import os.path
import numpy as np
import copy

from dipy.data import Sphere
from dipy.reconst.gqi import GeneralizedQSamplingModel
from dipy.reconst.dsi import DiffusionSpectrumModel

from dipy.core.ndindex import ndindex
from dipy.direction import peak_directions
from scipy.io import loadmat

def dsiSphere8Fold():
    dsi_sphere = loadmat(os.path.join(
        os.path.dirname(__file__), "../data/files/dsi_sphere_8fold.mat"
    ))
    return Sphere(
        xyz=dsi_sphere['odf_vertices'].T,
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
    
    
    def _get_rotated_sphere(self, sphere, rotation):
        return Sphere(
            xyz=np.dot(sphere.vertices, rotation),
            faces=sphere.faces
        )
    
    
    def fit(self, data):
        diff_model = GeneralizedQSamplingModel(self.gtab)

        data_shape = data.shape[:-1]
        tessellation = dsiSphere8Fold()

        self._rotations = np.zeros(data_shape + (3,3))
        rotated_odf = np.zeros(data_shape + (len(tessellation.vertices),))

        for idx in ndindex(data_shape):
            model_fit = diff_model.fit(data[idx])
            
            peak_dirs,_,_ = peak_directions(
                model_fit.odf(tessellation), tessellation
            )

            self._rotations[idx] = self._get_rotation_of_vector(
                np.squeeze(peak_dirs[:1]), [0,0,1]
            )

            rotated_odf[idx] = model_fit.odf(
                self._get_rotated_sphere(tessellation, self._rotations[idx])
            )

        return rotated_odf

        
class OdffpFit(object):
    
    def __init__(self):
        pass