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
    
    
    def _rotate_tessellation(self, tessellation, in_direction, out_direction=[0,0,1]):
        cAB = np.cross(in_direction, out_direction)
        sAB = np.array([[0,-cAB[2],cAB[1]], [cAB[2],0,-cAB[0]], [-cAB[1],cAB[0],0]])
        rotation = np.eye(3) + sAB + np.dot(sAB,sAB) * (1-np.dot(in_direction,out_direction)) / np.sum(cAB**2)
    
        if np.all(np.isnan(rotation)):
            rotation = np.eye(3)

        return Sphere(
            xyz=np.dot(tessellation.vertices, rotation),
            faces=tessellation.faces
        )


    def _rotate_odf(self, odf, in_tessellation, out_tessellation):
        sh = sf_to_sh(odf, in_tessellation) # , sh_order=14, basis_type='tournier07')
        sf = sh_to_sf(sh, out_tessellation) #, sh_order=14, basis_type='tournier07')
        return sf
    
    
    def fit(self, data):
        diff_model = GeneralizedQSamplingModel(self.gtab)

        data_shape = data.shape[:-1]
        tessellation = dsiSphere8Fold()
        tessellation_size = len(tessellation.vertices)

        self._rotations = np.zeros(data_shape + (3,3))
        output_odf = np.zeros(data_shape + (tessellation_size,))

        for idx in ndindex(data_shape):
            model_fit = diff_model.fit(data[idx])
            
            input_odf = model_fit.odf(tessellation)
            peak_dirs,_,_ = peak_directions(input_odf, tessellation)

            rotated_tessellation = self._rotate_tessellation(tessellation, np.squeeze(peak_dirs[:1]))

            odf_trace = self._rotate_odf(input_odf, tessellation, rotated_tessellation)[:int(tessellation_size/2)]
            
            odf_trace -= np.min(odf_trace)
            odf_trace /= np.sqrt(np.sum(odf_trace**2))
            
            dict_idx = np.argmax(np.dot(odf_trace, self.dict_odf))
           
            output_odf[idx] = self._rotate_odf(
                np.concatenate((self.dict_odf[:,dict_idx],self.dict_odf[:,dict_idx])), 
                rotated_tessellation, tessellation
            )

        return output_odf

        
class OdffpFit(object):
    
    def __init__(self):
        pass