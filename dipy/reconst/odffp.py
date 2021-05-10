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
from dipy.stats.analysis import peak_values
from phantomas.utils.tessellation import tessellation

from datetime import datetime
from click.core import batch
from conda.common._logic import FALSE


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
     

    def __init__(self, gtab, dict_file, drop_odf_baseline=True, output_dict_odf=True):
        self.gtab = gtab
         
        with h5py.File(dict_file, 'r') as mat_file:
            self._dict_odf = np.array(mat_file['odfrot'])    
            self._dict_peak_dirs = np.array(mat_file['dirrot'])
         
        self._drop_odf_baseline = drop_odf_baseline
        self._output_dict_odf = output_dict_odf
        self._normalized_dict_odf,_ = self._normalize_odf(self._dict_odf)

     
    def _normalize_odf(self, odf):
        if self._drop_odf_baseline:
            odf -= np.min(odf, axis=0)
            
        odf_norm = np.maximum(1e-8, np.sqrt(np.sum(odf**2, axis=0)))
        return odf / odf_norm, odf_norm

     
    def _find_highest_peak_rotation(self, input_odf, tessellation, target_direction=[0,0,1]):
        rotation = np.eye(3)
        input_peak_dirs,_,_ = peak_directions(input_odf, tessellation)

        if len(input_peak_dirs) > 0:
            highest_peak_direction = np.squeeze(input_peak_dirs[:1])
        
            cr = np.cross(highest_peak_direction, target_direction)
            sum_sqr_cr = np.sum(cr**2)
            
            if sum_sqr_cr != 0:
                s = np.array([[0,-cr[2],cr[1]], [cr[2],0,-cr[0]], [-cr[1],cr[0],0]])
                rotation += s + np.dot(s,s) * (1-np.dot(highest_peak_direction,target_direction)) / sum_sqr_cr

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
 
   
#     def fit(self, data):
#         diff_model = GeneralizedQSamplingModel(self.gtab)
#  
#         data_shape = data.shape[:-1]
#         max_peaks_num = self._dict_peak_dirs.shape[1]
# 
#         tessellation = dsiSphere8Fold()
#         tessellation_size = len(tessellation.vertices)
#  
#         output_odf = np.zeros(data_shape + (tessellation_size,))
#         output_peak_dirs = np.zeros(data_shape + (max_peaks_num, 3))
#  
#         for idx in ndindex(data_shape):
#             model_fit = diff_model.fit(data[idx])
#             input_odf = model_fit.odf(tessellation)
#             
#             rotation = self._find_highest_peak_rotation(input_odf, tessellation)
#             rotated_tessellation = self._rotate_tessellation(tessellation, rotation)
#             rotated_input_odf = OdffpModel.resample_odf(input_odf, tessellation, rotated_tessellation)
#              
#             input_odf_trace, input_odf_norm = self._normalize_odf(rotated_input_odf[:int(tessellation_size/2)])
#              
#             dict_idx = np.argmax(np.dot(input_odf_trace, self._normalized_dict_odf))
#             
#             if self._output_dict_odf:
#                 output_odf[idx] = OdffpModel.resample_odf(
#                     input_odf_norm * np.concatenate((self._dict_odf[:,dict_idx], self._dict_odf[:,dict_idx])), 
#                     rotated_tessellation, tessellation
#                 )
#             else:
#                 output_odf[idx] = input_odf
#             
#             output_peak_dirs[idx] = self._rotate_peak_dirs(self._dict_peak_dirs[:,:,dict_idx], rotation.T)
#  
#         return OdffpFit(data, output_odf, output_peak_dirs, tessellation)


    def fit(self, data, mask=None):
        diff_model = GeneralizedQSamplingModel(self.gtab)
 
        data_shape = data.shape[:-1]
        max_peaks_num = self._dict_peak_dirs.shape[1]

        tessellation = dsiSphere8Fold()
        tessellation_size = len(tessellation.vertices)
 
        if mask is None:
            mask = np.ones(data_shape, dtype=bool)
 
        masked_data = data[mask]
                
        batch_size = 1000        
        voxel_num = masked_data.shape[0]
        
        output_odf = []
        output_peak_dirs = []
        
        j = 0
        
        for data_chunk in np.split(masked_data, range(batch_size, voxel_num, batch_size)):
        
            print(j*batch_size, voxel_num)
            j += 1
        
            model_fit = diff_model.fit(data_chunk)
            input_odf = model_fit.odf(tessellation)
 
            rotation = {}
            rotated_tessellation = {}
            rotated_input_odf = {}
            input_odf_trace = np.zeros((data_chunk.shape[0],int(tessellation_size/2)))
            input_odf_norm = {}
            
            for i in range(data_chunk.shape[0]):
 
                rotation[i] = self._find_highest_peak_rotation(input_odf[i], tessellation)
                rotated_tessellation[i] = self._rotate_tessellation(tessellation, rotation[i])
                rotated_input_odf[i] = OdffpModel.resample_odf(input_odf[i], tessellation, rotated_tessellation[i])
            
                input_odf_trace[i], input_odf_norm[i] = self._normalize_odf(rotated_input_odf[i][:int(tessellation_size/2)])
              
            dict_idx = np.argmax(np.dot(input_odf_trace, self._normalized_dict_odf), axis=1)
        
            output_odf_chunk = np.zeros((data_chunk.shape[0], tessellation_size))
            output_peak_dirs_chunk = np.zeros((data_chunk.shape[0], max_peaks_num, 3))

            for i in range(data_chunk.shape[0]):
             
                if self._output_dict_odf:
                    output_odf_chunk[i] = OdffpModel.resample_odf(
                        input_odf_norm[i] * np.concatenate((self._dict_odf[:,dict_idx[i]], self._dict_odf[:,dict_idx[i]])), 
                        rotated_tessellation[i], tessellation
                    )
                else:
                    output_odf_chunk[i] = input_odf[i]
                  
                output_peak_dirs_chunk[i] = self._rotate_peak_dirs(self._dict_peak_dirs[:,:,dict_idx[i]], rotation[i].T)
   
            output_odf.append(output_odf_chunk)
            output_peak_dirs.append(output_peak_dirs_chunk)
   
        output_odf = np.concatenate(output_odf)
        output_peak_dirs = np.concatenate(output_peak_dirs)
   
        output_odf_3d = np.zeros(data_shape + (tessellation_size,))
        output_peak_dirs_3d = np.zeros(data_shape + (max_peaks_num, 3))
   
        output_odf_3d[mask] = output_odf
        output_peak_dirs_3d[mask] = output_peak_dirs   
   
        return OdffpFit(data, output_odf_3d, output_peak_dirs_3d, tessellation)
            
 
#  
#         ###
#  
#         return OdffpFit(None, None, None, None)

#         output_odf = np.zeros(data_shape + (tessellation_size,))
#         output_peak_dirs = np.zeros(data_shape + (max_peaks_num, 3))
#  
#         for idx in ndindex(data_shape):
#             model_fit = diff_model.fit(data[idx])
#             input_odf = model_fit.odf(tessellation)
#             
#             rotation = self._find_highest_peak_rotation(input_odf, tessellation)
#             rotated_tessellation = self._rotate_tessellation(tessellation, rotation)
#             rotated_input_odf = OdffpModel.resample_odf(input_odf, tessellation, rotated_tessellation)
#              
#             input_odf_trace, input_odf_norm = self._normalize_odf(rotated_input_odf[:int(tessellation_size/2)])
#              
#             dict_idx = np.argmax(np.dot(input_odf_trace, self._normalized_dict_odf))
#             
#             if self._output_dict_odf:
#                 output_odf[idx] = OdffpModel.resample_odf(
#                     input_odf_norm * np.concatenate((self._dict_odf[:,dict_idx], self._dict_odf[:,dict_idx])), 
#                     rotated_tessellation, tessellation
#                 )
#             else:
#                 output_odf[idx] = input_odf
#             
#             output_peak_dirs[idx] = self._rotate_peak_dirs(self._dict_peak_dirs[:,:,dict_idx], rotation.T)
#  
#         return OdffpFit(data, output_odf, output_peak_dirs, tessellation)

        
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
        fib['voxel_size'] = np.array([2,2,2])
        fib['odf_vertices'] = self._tessellation.vertices.T
        fib['odf_faces'] = self._tessellation.faces.T
        
        map_size = [fib['dimension'][0] * fib['dimension'][1], fib['dimension'][2]]
        voxel_num = np.prod(fib['dimension'])
        tessellation_half_size = len(self._tessellation.vertices) // 2
        
        max_peaks_num = self._peak_dirs.shape[3]

        self._odf = np.flip(self._odf, 1).reshape((np.prod(fib['dimension']), len(self._tessellation.vertices)), order='F')
        self._peak_dirs = np.flip(self._peak_dirs, 1).reshape((np.prod(fib['dimension']), max_peaks_num, 3), order='F')
  
        for i in range(max_peaks_num):
            fib['fa%d' % i] = np.zeros(voxel_num)
            fib['nqa%d' % i] = 0.1 * np.ones(voxel_num)
            fib['index%d' % i] = np.zeros(voxel_num)
 
        for j in range(self._odf.shape[0]):
            peak_vertex_idx = np.zeros(max_peaks_num, dtype=int)
            peak_vertex_values = np.zeros(max_peaks_num)
 
            for i in range(max_peaks_num):
                peak_vertex_idx[i] = np.argmax(np.dot(self._peak_dirs[j][i], fib['odf_vertices']))
                peak_vertex_values[i] = self._odf[j][peak_vertex_idx[i]]
                 
            sorted_i = np.argsort(-peak_vertex_values)
                 
            for i in range(max_peaks_num):
                fib['index%d' % i][j] = np.mod(peak_vertex_idx[sorted_i[i]], tessellation_half_size)
                fib['fa%d' % i][j] = peak_vertex_values[sorted_i[i]] - np.min(self._odf[j])

        for i in range(max_peaks_num):
            fib['fa%d' % i] -= np.min(fib['fa%d' % i])
            fib['fa%d' % i] /= np.maximum(1e-8, np.max(fib['fa%d' % i]))

        fib['odf0'] = self._odf[fib['fa0'] > 0]
        fib['odf0'] /= np.maximum(1e-8, np.max(fib['odf0']))
#         fib['odf0'] = np.flip(fib['odf0'], 1)
#         fib['odf0'] = fib['odf0'].reshape((np.prod(fib['dimension']), len(self._tessellation.vertices)), order='F')
#         fib['odf0'] = fib['odf0'][fib['fa0'] > 0]
        fib['odf0'] = fib['odf0'].T
        fib['odf0'] = fib['odf0'][:tessellation_half_size,:]
             
        for i in range(max_peaks_num):
#             fib['fa%d' % i] = np.flip(fib['fa%d' % i], 1)
#             fib['index%d' % i] = np.flip(fib['index%d' % i], 1)
            fib['fa%d' % i] = fib['fa%d' % i].reshape(map_size, order='F')
            fib['index%d' % i] = fib['index%d' % i].reshape((1, voxel_num), order='F')
         

         
        savemat(file_name, fib, format='4')        
    
    