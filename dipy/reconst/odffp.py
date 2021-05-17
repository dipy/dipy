'''
Created on Feb 18, 2021

@author: patrykfi
'''

import h5py
import numpy as np
import nibabel as nib
import os.path

from dipy.data import Sphere
from dipy.reconst.gqi import GeneralizedQSamplingModel
# from dipy.reconst.dsi import DiffusionSpectrumModel

from dipy.core.geometry import sphere2cart
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
     

    def __init__(self, gtab, dict_file, 
                 drop_negative_odf=True, zero_baseline_odf=True, output_dict_odf=True, 
                 max_chunk_size=1000):
        """ODF-Fingerprinting reconstruction"""
        
        self.gtab = gtab
         
        with h5py.File(dict_file, 'r') as mat_file:
            self._dict_odf = np.array(mat_file['odfrot'])    
            self._dict_peak_dirs = np.array(mat_file['dirrot'])
        
        self._drop_negative_odf = drop_negative_odf 
        self._zero_baseline_odf = zero_baseline_odf
        self._output_dict_odf = output_dict_odf
        self._max_chunk_size = np.maximum(1, max_chunk_size)
        
        self._normalized_dict_odf,_ = self._normalize_odf(self._dict_odf)

     
    def _normalize_odf(self, odf):
        if self._drop_negative_odf:
            odf = np.maximum(0, odf)
        
        if self._zero_baseline_odf:
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
 

    def fit(self, data, mask=None):
        diff_model = GeneralizedQSamplingModel(self.gtab)
 
        data_shape = data.shape[:-1]
        max_peaks_num = self._dict_peak_dirs.shape[1]

        tessellation = dsiSphere8Fold()
        tessellation_size = len(tessellation.vertices)
 
        if mask is None:
            mask = np.ones(data_shape, dtype=bool)
        else:
            mask = mask.astype(bool)
 
        masked_data = data[mask]
        voxels_num = masked_data.shape[0]
        
        output_odf = np.zeros(data_shape + (tessellation_size,))
        output_peak_dirs = np.zeros(data_shape + (max_peaks_num, 3))
   
        masked_output_odf = np.zeros((voxels_num, tessellation_size))
        masked_output_peak_dirs = np.zeros((voxels_num, max_peaks_num, 3))
        
        for chunk_idx in np.split(range(voxels_num), range(self._max_chunk_size, voxels_num, self._max_chunk_size)):
        
            print("%.1f%%" % (100 * (np.max(chunk_idx) + 1) / voxels_num))
        
            chunk_size = len(chunk_idx)

            input_odf = diff_model.fit(masked_data[chunk_idx]).odf(tessellation)
            input_odf_trace = np.zeros((chunk_size, int(tessellation_size/2)))
            input_odf_norm = np.zeros(chunk_size)
             
            rotation = np.zeros((chunk_size, 3, 3))
            rotated_tessellation = {}
            rotated_input_odf = np.zeros_like(input_odf)
            
            for i in range(chunk_size):

                rotation[i] = self._find_highest_peak_rotation(input_odf[i], tessellation)
                rotated_tessellation[i] = self._rotate_tessellation(tessellation, rotation[i])
                rotated_input_odf[i] = OdffpModel.resample_odf(input_odf[i], tessellation, rotated_tessellation[i])
            
                input_odf_trace[i], input_odf_norm[i] = self._normalize_odf(rotated_input_odf[i][:int(tessellation_size/2)])
              
            dict_idx = np.argmax(np.dot(input_odf_trace, self._normalized_dict_odf), axis=1)
        
            for i in range(chunk_size):
             
                if self._output_dict_odf:
                    masked_output_odf[chunk_idx[i]] = OdffpModel.resample_odf(
                        input_odf_norm[i] * np.concatenate((self._dict_odf[:,dict_idx[i]], self._dict_odf[:,dict_idx[i]])), 
                        rotated_tessellation[i], tessellation
                    )
                else:
                    masked_output_odf[chunk_idx[i]] = input_odf[i]
                  
                masked_output_peak_dirs[chunk_idx[i]] = self._rotate_peak_dirs(self._dict_peak_dirs[:,:,dict_idx[i]], rotation[i].T)
   
        output_odf[mask] = masked_output_odf
        output_peak_dirs[mask] = masked_output_peak_dirs
   
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
    
    
    def saveToFib(self, affine, file_name = 'output.fib'):
        fib = {}
        
        orientation_agreement = np.array(nib.aff2axcodes(affine)) == np.array(('L', 'P', 'S'))
        orientation_sign = 2 * (orientation_agreement).astype(int) - 1
        orientation_flip = np.arange(3)[~orientation_agreement] 
                
        fib['dimension'] = self._data.shape[:-1]
        fib['voxel_size'] = np.array([2,2,2])
        fib['odf_vertices'] = self._tessellation.vertices.T
        fib['odf_faces'] = self._tessellation.faces.T
        
        voxels_num = np.prod(fib['dimension'])
        slice_size = [fib['dimension'][0] * fib['dimension'][1], fib['dimension'][2]]

        max_peaks_num = self._peak_dirs.shape[3]
        tessellation_half_size = len(self._tessellation.vertices) // 2

        output_odf = np.flip(self._odf, orientation_flip).reshape(
            (voxels_num, len(self._tessellation.vertices)), order='F'
        )

        if not np.all(orientation_agreement):
            output_odf = OdffpModel.resample_odf(
                output_odf, self._tessellation, Sphere(
                    xyz=orientation_sign * self._tessellation.vertices,
                    faces=self._tessellation.faces
                )
            )
        
        output_peak_dirs = orientation_sign * np.flip(self._peak_dirs, orientation_flip).reshape(
            (voxels_num, max_peaks_num, 3), order='F'
        )
  
        for i in range(max_peaks_num):
            fib['fa%d' % i] = np.zeros(voxels_num)
            fib['index%d' % i] = np.zeros(voxels_num)
        
        for j in range(voxels_num):
            peak_vertex_idx = np.zeros(max_peaks_num, dtype=int)
            peak_vertex_values = np.zeros(max_peaks_num)
 
            for i in range(max_peaks_num):
                peak_vertex_idx[i] = np.argmax(np.dot(output_peak_dirs[j][i], fib['odf_vertices']))
                peak_vertex_values[i] = output_odf[j][peak_vertex_idx[i]]
                 
            sorted_i = np.argsort(-peak_vertex_values)
                 
            for i in range(max_peaks_num):
                fib['index%d' % i][j] = np.mod(peak_vertex_idx[sorted_i[i]], tessellation_half_size)
                fib['fa%d' % i][j] = peak_vertex_values[sorted_i[i]] - np.min(output_odf[j])

        for i in range(max_peaks_num):
            fib['fa%d' % i] -= np.min(fib['fa%d' % i])
            fib['nqa%d' % i] = fib['fa%d' % i] / np.maximum(1e-8, np.max(fib['fa%d' % i]))

        fib['odf0'] = output_odf[fib['fa0'] > 0,:tessellation_half_size].T
        fib['odf0'] /= np.maximum(1e-8, np.max(fib['odf0']))

        for i in range(max_peaks_num):
            fib['fa%d' % i] = fib['fa%d' % i].reshape(slice_size, order='F')
            fib['nqa%d' % i] = fib['nqa%d' % i].reshape(slice_size, order='F')
            fib['index%d' % i] = fib['index%d' % i].reshape((1, voxels_num), order='F')
         
        savemat(file_name, fib, format='4')        
    
        