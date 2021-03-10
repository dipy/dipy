'''
Created on Feb 18, 2021

@author: patrykfi
'''

import h5py
import numpy as np
import os.path

from dipy.data import Sphere
from dipy.reconst.gqi import GeneralizedQSamplingModel
from dipy.reconst.dsi import DiffusionSpectrumModel

from dipy.core.ndindex import ndindex
from dipy.direction import peak_directions
from dipy.reconst.shm import sf_to_sh, sh_to_sf
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

    def __init__(self, gtab, dict_file):
        self.gtab = gtab
        
        with h5py.File(dict_file, 'r') as mat_file:
            self.dict_odf = np.array(mat_file['odfrot'])    
        
        # normalize the ODF dictionary
        self.dict_odf -= np.min(self.dict_odf, axis=0)
        self.dict_odf /= np.sqrt(np.sum(self.dict_odf**2, axis=0))
    
    
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
        tessellation_size = int(len(tessellation.vertices) / 2)

        self._rotations = np.zeros(data_shape + (3,3))
        output_odf = np.zeros(data_shape + (tessellation_size,))

        for idx in ndindex(data_shape):
            model_fit = diff_model.fit(data[idx])
            
            peak_dirs,_,_ = peak_directions(
                model_fit.odf(tessellation), tessellation
            )

            self._rotations[idx] = self._get_rotation_of_vector(
                np.squeeze(peak_dirs[:1]), [0,0,1]
            )

            rotated_tessellation = self._get_rotated_sphere(tessellation, self._rotations[idx])

            rotated_odf = model_fit.odf(rotated_tessellation)[:tessellation_size]
            rotated_odf -= np.min(rotated_odf)
            rotated_odf /= np.sqrt(np.sum(rotated_odf**2))
            
            dict_idx = np.argmax(np.dot(rotated_odf, self.dict_odf))
            
            sh = sf_to_sh(np.concatenate((self.dict_odf[:,dict_idx],self.dict_odf[:,dict_idx])), rotated_tessellation, sh_order=14, basis_type='tournier07')
            sf = sh_to_sf(sh, tessellation, sh_order=14, basis_type='tournier07')
             
            output_odf[idx] = sf[:tessellation_size]

        return output_odf

        
class OdffpFit(object):
    
    def __init__(self):
        pass