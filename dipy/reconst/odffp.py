'''
Created on Feb 18, 2021

@author: patrykfi
'''

import h5py
import os, sys, gzip
import numpy as np
import nibabel as nib
import warnings

import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.art3d as art3d

from scipy.io import loadmat, savemat

from dipy.data import Sphere
from dipy.reconst.gqi import GeneralizedQSamplingModel
# from dipy.reconst.dsi import DiffusionSpectrumModel

from dipy.core.geometry import sphere2cart
from dipy.direction import peak_directions
from dipy.reconst.shm import sf_to_sh, sh_to_sf
from dipy.reconst.odf import OdfFit
from phantomas.utils.tessellation import tessellation
from nibabel import orientations
from _warnings import warn
from _pylief import NONE


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


def plot_odf(odf, filename='odf.png', tessellation=dsiSphere8Fold()):
    
    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")
    norm = plt.Normalize(odf.min(), odf.max())
    v = (tessellation.vertices.T * norm(odf)).T
    f = tessellation.faces

    colors = np.mean(abs(v[f]),axis=1)
    norm = plt.Normalize(colors.min(), colors.max())
    colors = norm(colors)

    pc = art3d.Poly3DCollection(v[f], facecolors=colors, edgecolor=None)

    ax.add_collection(pc)
    ax.set_xlim3d([-1, 1])
    ax.set_ylim3d([-1, 1])
    ax.set_zlim3d([-1, 1])
 
    elev = 0 # 90
    azim = -90
    ax.view_init(elev, azim)
    
    fig.savefig(filename, bbox_inches='tight')
    
    plt.close(fig)    


class OdffpDictionary(object):
    odf = None
    peak_dirs = None
    max_peaks_num = 0    
        
        
    def __init__(self, dict_file=None, is_sorted=False, tessellation=dsiSphere8Fold()):
        self.tessellation = tessellation
        self._is_sorted = is_sorted

        if dict_file is not None:
            self.load(dict_file)
    
           
    def load(self, dict_file):
        with h5py.File(dict_file, 'r') as mat_file:
            
            # Load ODF fingerprints
            self.odf = np.array(mat_file['odfrot'])
            
            # Load tentatively all peak directions
            peak_dirs = np.array(mat_file['dirrot'])
            
            # Disregard peak directions that are always empty   
            self.max_peaks_num = np.sum(np.any(~np.isnan(peak_dirs[0]),axis=1))
            self.peak_dirs = peak_dirs[:,:self.max_peaks_num]
            
            # Free water fraction has 0th index, hence max_peaks_num+1 in total
            self.micro = np.array(mat_file['micro'])[:,:self.max_peaks_num+1]
            self.ratio = np.array(mat_file['rat'])[:self.max_peaks_num+1]

        # Arrange peaks in the descending order
        if not self._is_sorted:
            self._sort_peaks()

   
    def _sort_peaks(self):
        for j in range(self.peak_dirs.shape[2]):

            # First, convert spherical coordinates from Matlab (azim, elev, radius) to Python (radius, phi, theta) 
            # and then to Cartesian coordinates
            peak_dirs = np.array(sphere2cart(1, np.pi/2 + self.peak_dirs[1,:,j], self.peak_dirs[0,:,j]))
            peaks_filter = np.any(~np.isnan(peak_dirs), axis=0)

            if ~np.any(peaks_filter):
                continue

            peak_vertex_idx = np.argmax(np.dot(self.tessellation.vertices, peak_dirs[:,peaks_filter]), axis=0)     
            peak_vertex_values = self.odf[np.mod(peak_vertex_idx,self.odf.shape[0]),j]     
                  
            idx = np.arange(len(peak_vertex_values))
            
            # Sort in the descending order, hence -peak_vertex_values
            sorted_idx = np.argsort(-peak_vertex_values)
            
            if np.any(idx != sorted_idx):
                self.peak_dirs[:,idx,j] = self.peak_dirs[:,sorted_idx,j]
                self.micro[:,idx+1,j] = self.micro[:,sorted_idx+1,j]
                self.ratio[idx+1,j] = self.ratio[sorted_idx+1,j]
   
    
    
    def save(self, dict_file):
        pass
   
    
    def generate(self, gtab):
        pass
     

class OdffpModel(object):
 
    @staticmethod 
    def resample_odf(odf, in_sphere, out_sphere):
        return sh_to_sf(
            sf_to_sh(odf, in_sphere), # sh_order=14, basis_type='tournier07') 
            out_sphere
        ) #, sh_order=14, basis_type='tournier07')
     

    def __init__(self, gtab, odf_dict, 
                 drop_negative_odf=True, zero_baseline_odf=True, output_dict_odf=True, 
                 max_chunk_size=1000):
        """ODF-Fingerprinting reconstruction"""
        
        self.gtab = gtab

        self._drop_negative_odf = drop_negative_odf 
        self._zero_baseline_odf = zero_baseline_odf
        self._output_dict_odf = output_dict_odf
        
        self.max_chunk_size = np.maximum(1, max_chunk_size)
         
        if not hasattr(odf_dict, 'odf') or odf_dict.odf is None:
            raise Exception('Specified ODF-dictionary is corrupted or empty.')
         
        self._dict = odf_dict 
         
     
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

 
    def _unmask(self, vector, mask):
        output_matrix = np.zeros(mask.shape + vector.shape[1:], dtype=vector.dtype)
        output_matrix[mask] = vector
        return output_matrix


    def fit(self, data, mask=None):
        diff_model = GeneralizedQSamplingModel(self.gtab)

        tessellation_size = len(self._dict.tessellation.vertices)
 
        if mask is None:
            mask = np.ones(data.shape[:-1], dtype=bool)
        else:
            mask = mask.astype(bool)
 
        masked_data = data[mask]
        voxels_num = masked_data.shape[0]
        
        dict_idx = np.zeros(voxels_num, dtype=int)
        dict_odf_trace,_ = self._normalize_odf(self._dict.odf)
        
        output_odf = np.zeros((voxels_num, tessellation_size))
        output_peak_dirs = np.zeros((voxels_num, self._dict.max_peaks_num, 3))
        
        for chunk_idx in np.split(range(voxels_num), range(self.max_chunk_size, voxels_num, self.max_chunk_size)):
        
            print("%.1f%%" % (100 * (np.max(chunk_idx) + 1) / voxels_num))
        
            chunk_size = len(chunk_idx)

            input_odf = diff_model.fit(masked_data[chunk_idx]).odf(self._dict.tessellation)
            input_odf_trace = np.zeros((chunk_size, int(tessellation_size/2)))
            input_odf_norm = np.zeros(chunk_size)
             
            rotation = np.zeros((chunk_size, 3, 3))
            rotated_tessellation = {}
            rotated_input_odf = np.zeros_like(input_odf)
            
            for i in range(chunk_size):

                rotation[i] = self._find_highest_peak_rotation(input_odf[i], self._dict.tessellation)
                rotated_tessellation[i] = self._rotate_tessellation(self._dict.tessellation, rotation[i])
                rotated_input_odf[i] = OdffpModel.resample_odf(input_odf[i], self._dict.tessellation, rotated_tessellation[i])
            
                input_odf_trace[i], input_odf_norm[i] = self._normalize_odf(rotated_input_odf[i][:int(tessellation_size/2)])

            dict_idx[chunk_idx] = np.argmax(np.dot(input_odf_trace, dict_odf_trace), axis=1)
        
            for i, j in zip(range(chunk_size), chunk_idx):
                if self._output_dict_odf:
                    output_odf[j] = OdffpModel.resample_odf(
                        input_odf_norm[i] * np.concatenate((self._dict.odf[:,dict_idx[j]], self._dict.odf[:,dict_idx[j]])), 
                        rotated_tessellation[i], self._dict.tessellation
                    )
                else:
                    output_odf[j] = input_odf[i]
                  
                output_peak_dirs[j] = self._rotate_peak_dirs(self._dict.peak_dirs[:,:,dict_idx[j]], rotation[i])
   
        return OdffpFit(
            data, self._dict, 
            self._unmask(output_odf, mask), 
            self._unmask(output_peak_dirs, mask),
            self._unmask(dict_idx, mask)
        )
            
        
class OdffpFit(OdfFit):
    
    def __init__(self, data, odf_dict, odf, peak_dirs, dict_idx):
        self._data = data
        self._dict = odf_dict
        self._odf = odf
        self._peak_dirs = peak_dirs
        self._dict_idx = dict_idx
    
        
    def odf(self, sphere=None):
        if sphere is None or sphere == self._dict.tessellation:
            output_odf = self._odf
        else:
            output_odf = OdffpModel.resample_odf(self._odf, self._dict.tessellation, sphere)
        
        return output_odf / np.maximum(1e-8, np.max(output_odf))
    
    
    def peak_dirs(self):
        return self._peak_dirs
    
    
    def _fib_reshape(self, matrix, new_size, orientation_agreement=True):
        if not np.all(orientation_agreement):
            matrix = np.flip(matrix, np.arange(3)[~orientation_agreement])
        
        return matrix.reshape(new_size, order='F')


    def _fib_index_map(self, var_data, dict_idx, slice_size, fa_filter):
        index_map = np.zeros(dict_idx.shape)
        index_map[fa_filter] = var_data[dict_idx[fa_filter]]
        return self._fib_reshape(index_map, slice_size)

    
    def save_as_fib(self, affine, voxel_size, output_file_name='output.fib'):
        fib = {}
        fib['dimension'] = self._data.shape[:-1]
        fib['voxel_size'] = voxel_size
        fib['odf_vertices'] = self._dict.tessellation.vertices.T
        fib['odf_faces'] = self._dict.tessellation.faces.T
        
        voxels_num = np.prod(fib['dimension'])
        slice_size = [fib['dimension'][0] * fib['dimension'][1], fib['dimension'][2]]

        max_peaks_num = self._peak_dirs.shape[3]
        tessellation_half_size = len(self._dict.tessellation.vertices) // 2

        try:
            orientation_agreement = np.array(nib.aff2axcodes(affine)) == np.array(('L', 'P', 'S'))
        except:
            warnings.warn("Couldn't determine the orientation of coordinates from the affine.")
            orientation_agreement = np.ones(3, dtype=bool)
                
        output_odf_vertices = (2 * orientation_agreement.astype(int) - 1) * self._dict.tessellation.vertices
        
        # Reorient maps to LPS and flatten to voxels_num x N
        output_odf = self._fib_reshape(self._odf, (voxels_num, len(output_odf_vertices)), orientation_agreement)
        output_peak_dirs = self._fib_reshape(self._peak_dirs, (voxels_num, max_peaks_num, 3), orientation_agreement)
        dict_idx = self._fib_reshape(self._dict_idx, voxels_num, orientation_agreement)

        if not np.all(orientation_agreement):
            output_odf = OdffpModel.resample_odf(
                output_odf, self._dict.tessellation, Sphere(
                    xyz=output_odf_vertices,
                    faces=self._dict.tessellation.faces
                )
            )
  
        for i in range(max_peaks_num):
            fib['fa%d' % i] = np.zeros(voxels_num)
            fib['index%d' % i] = np.zeros(voxels_num)
        
        for j in range(voxels_num):
            peaks_filter = np.any(output_peak_dirs[j] != 0, axis=1)
            peak_vertex_idx = np.argmax(np.dot(output_peak_dirs[j,peaks_filter], output_odf_vertices.T), axis=1)     
            peak_vertex_values = output_odf[j][peak_vertex_idx]     
                 
            sorted_i = np.argsort(-peak_vertex_values)
            
            if np.any(sorted_i != np.arange(len(sorted_i))):
                warnings.warn('Reorientation needed for dict_idx=%d, peak_vertex_values=%s' % (dict_idx[j], peak_vertex_values))
            

            for i in range(len(peak_vertex_idx)):
                fib['index%d' % i][j] = np.mod(peak_vertex_idx[sorted_i[i]], tessellation_half_size)
                fib['fa%d' % i][j] = peak_vertex_values[sorted_i[i]] - np.min(output_odf[j])

        for i in range(max_peaks_num):
            fib['fa%d' % i] -= np.min(fib['fa%d' % i])
            fib['nqa%d' % i] = fib['fa%d' % i] / np.maximum(1e-8, np.max(fib['fa%d' % i]))

        fa_filter = fib['fa0'] > 0

        if np.any(fa_filter):
            fib['odf0'] = output_odf[fa_filter,:tessellation_half_size].T
            fib['odf0'] /= np.maximum(1e-8, np.max(fib['odf0']))
        else:
            warnings.warn("The output is empty.")

        # microstructure parameters
        fib['D_iso'] = self._fib_index_map(self._dict.micro[1,0], dict_idx, slice_size, fa_filter)
        fib['p_iso'] = self._fib_index_map(self._dict.ratio[0], dict_idx, slice_size, fa_filter)

        for i in range(1, max_peaks_num+1):
            index_maps = {
                'fib%d_Da' % i: self._dict.micro[0,i], 'fib%d_De' % i: self._dict.micro[1,i],
                'fib%d_Dr' % i: self._dict.micro[2,i], 'fib%d_fin' % i: self._dict.micro[3,i],
                'fib%d_p' % i: self._dict.ratio[i]
            }
            for index_name in index_maps:
                fib[index_name] = self._fib_index_map(index_maps[index_name], dict_idx, slice_size, fa_filter)

        for i in range(max_peaks_num):
            fib['fa%d' % i] = self._fib_reshape(fib['fa%d' % i], slice_size)
            fib['nqa%d' % i] = self._fib_reshape(fib['nqa%d' % i], slice_size)
            fib['index%d' % i] = self._fib_reshape(fib['index%d' % i], (1, voxels_num))
        
        output_file_prefix = output_file_name.lower().replace(".fib.gz", "").replace(".fib", "") 
        savemat("%s.fib" % output_file_prefix, fib, format='4')        

        with open("%s.fib" % output_file_prefix, 'rb') as fib_file:
            with gzip.open("%s.fib.gz" % output_file_prefix, 'wb') as fib_gz_file:
                fib_gz_file.writelines(fib_file)
                
        os.remove("%s.fib" % output_file_prefix)
        
        
        
