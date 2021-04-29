'''
Created on Feb 18, 2021

@author: patrykfi
'''

import h5py
import numpy as np
import os.path

from dipy.data import Sphere, HemiSphere
from dipy.reconst.gqi import GeneralizedQSamplingModel
from dipy.reconst.dsi import DiffusionSpectrumModel

from dipy.core.geometry import sphere2cart
from dipy.core.ndindex import ndindex
from dipy.direction import peak_directions
from dipy.reconst.shm import sf_to_sh, sh_to_sf

from scipy.io import loadmat, savemat


class _DsiSphere8Fold(Sphere):
    _instance = None

    @staticmethod 
    def get_instance():
        if _DsiSphere8Fold._instance == None:
            _DsiSphere8Fold()
        return _DsiSphere8Fold._instance
    
    
    def __init__(self):
        if _DsiSphere8Fold._instance != None:
            raise Exception("The class _DsiSphere8Fold is a singleton. Call dsiSphere8Fold() function instead.")
        else:
            dsi_sphere = loadmat(os.path.join(
                os.path.dirname(__file__), "../data/files/dsi_sphere_8fold.mat"
            ))
            Sphere.__init__(self,
                xyz=dsi_sphere['odf_vertices'].T,
                faces=dsi_sphere['odf_faces'].T
            )
            _DsiSphere8Fold._instance = self


def dsiSphere8Fold():
    return _DsiSphere8Fold.get_instance()


class OdffpModel(object):
 
    @staticmethod 
    def resample_odf(odf, in_sphere, out_sphere):
        return sh_to_sf(
            sf_to_sh(odf, in_sphere), # sh_order=14, basis_type='tournier07') 
            out_sphere
        ) #, sh_order=14, basis_type='tournier07')
     

    def __init__(self, gtab, dict_file, drop_odf_amplitude=True):
        self.gtab = gtab
         
        with h5py.File(dict_file, 'r') as mat_file:
            self._dict_odf = np.array(mat_file['odfrot'])    
            self._dict_peak_dirs = np.array(mat_file['dirrot'])
         
        self._drop_odf_amplitude = drop_odf_amplitude
        self._dict_odf = self._normalize_odf(self._dict_odf)

     
    def _normalize_odf(self, odf):
        if self._drop_odf_amplitude:
            odf -= np.min(odf, axis=0)

        return odf / np.maximum(1e-8, np.sqrt(np.sum(odf**2, axis=0)))

     
    def _get_rotation(self, in_direction, out_direction=[0,0,1]):
        cAB = np.cross(in_direction, out_direction)
        sAB = np.array([[0,-cAB[2],cAB[1]], [cAB[2],0,-cAB[0]], [-cAB[1],cAB[0],0]])
        rotation = np.eye(3) + sAB + np.dot(sAB,sAB) * (1-np.dot(in_direction,out_direction)) / np.sum(cAB**2)
     
        if np.all(np.isnan(rotation)):
            rotation = np.eye(3)

        return rotation
     
     
    def _rotate_tessellation(self, tessellation, rotation):
        return Sphere(
            xyz=np.dot(tessellation.vertices, rotation),
            faces=tessellation.faces
        )
        
        
    def _rotate_peak_dirs(self, peak_dirs, rotation):
        return np.dot(
            np.array(sphere2cart(1, np.pi/2 + peak_dirs[1,:], peak_dirs[0,:])).T, 
            rotation
        )
 
   
    def fit(self, data):
        diff_model = GeneralizedQSamplingModel(self.gtab)
 
        data_shape = data.shape[:-1]
        max_peaks_num = self._dict_peak_dirs.shape[1]

        tessellation = dsiSphere8Fold()
        tessellation_size = len(tessellation.vertices)
 
        self._rotations = np.zeros(data_shape + (3,3))
        output_odf = np.zeros(data_shape + (tessellation_size,))
        output_peak_dirs = np.zeros(data_shape + (max_peaks_num,3))
 
        for idx in ndindex(data_shape):
            
            model_fit = diff_model.fit(data[idx])
             
            input_odf = model_fit.odf(tessellation)
            input_peak_dirs,_,_ = peak_directions(input_odf, tessellation)

            if len(input_peak_dirs) > 0:
                rotation = self._get_rotation(np.squeeze(input_peak_dirs[:1]))
            else:
                rotation = np.eye(3)
            
            rotated_tessellation = self._rotate_tessellation(tessellation, rotation)
            rotated_input_odf = OdffpModel.resample_odf(input_odf, tessellation, rotated_tessellation)
             
            input_odf_trace = self._normalize_odf(rotated_input_odf[:int(tessellation_size/2)])
             
            dict_idx = np.argmax(np.dot(input_odf_trace, self._dict_odf))
            
            output_odf[idx] = OdffpModel.resample_odf(
                np.concatenate((self._dict_odf[:,dict_idx],self._dict_odf[:,dict_idx])), 
                rotated_tessellation, tessellation
            )
            
            output_peak_dirs[idx] = self._rotate_peak_dirs(self._dict_peak_dirs[:,:,dict_idx], rotation.T)
 
        return OdffpFit(data, output_odf, output_peak_dirs, tessellation)

        
class OdffpFit(object):
    
    def __init__(self, data, odf, peak_dirs, tessellation):
        self._data = data
        self._odf = odf
        self._peak_dirs = peak_dirs
        self._tessellation = tessellation
    
        
    def odf(self):
        return self._odf
    
    
    def peak_dirs(self):
        return self._peak_dirs
    
    
    def saveToFib(self, file_name = 'output.fib'):
        fib = {}
        
        fib['dimension'] = self._data.shape[:-1]
#         fib['voxel_size'] = np.array([2,2,2])
        fib['odf_vertices'] = self._tessellation.vertices.T
        fib['odf_faces'] = self._tessellation.faces.T
        
        map_size = [fib['dimension'][0] * fib['dimension'][1], fib['dimension'][2]]
        flat_size = [1, np.prod(fib['dimension'])]
        
        max_peaks_num = self._peak_dirs.shape[3]
        
        for i in range(max_peaks_num):
            fib['fa%d' % i] = 0.1 * np.ones(map_size)
            fib['nqa%d' % i] = 0.1 * np.ones(map_size)
            fib['index%d' % i] = np.zeros(fib['dimension'])
        
            for idx in ndindex(fib['dimension']):
                fib['index%d' % i][idx] = np.argmax(np.dot(self._peak_dirs[idx][i], fib['odf_vertices']))
                
            fib['index%d' % i] = fib['index%d' % i].reshape(flat_size, order='F')
       
        savemat(file_name, fib, format='4')
        
        
    
    