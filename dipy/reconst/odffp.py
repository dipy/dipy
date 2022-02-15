'''
Created on Feb 18, 2021

@author: patrykfi
'''

import os, sys, gzip
import h5py, hdf5storage
import warnings

import numpy as np
import nibabel as nib

import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.art3d as art3d
import scipy.stats

from dipy.core.sphere import Sphere
from dipy.core.dsi_sphere import dsiSphere8Fold
from dipy.core.geometry import sphere2cart

from dipy.direction import peak_directions

from dipy.reconst.gqi import GeneralizedQSamplingModel
from dipy.reconst.odf import OdfFit
from dipy.reconst.shm import sf_to_sh, sh_to_sf

from scipy.io import savemat


DEFAULT_RECON_EDGE = 1.2
DEFAULT_DICT_EDGE = 1.2

DEFAULT_FIT_PENALTY = 0.001
MAX_FIT_PENALTY = 0.25


def plot_odf(odf, filename='odf.png', tessellation=dsiSphere8Fold()):
    
    fig = plt.figure(figsize=(30, 10))
    
    elev = [90, 90, 0]
    azim = [-90, 0, -90]
    
    for i in range(3):
        ax = fig.add_subplot(1, 3, i+1, projection="3d")
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
     
        ax.view_init(elev[i], azim[i])
    
    fig.savefig(filename, bbox_inches='tight')
    
    plt.close(fig)    


class OdffpDictionary(object):
    odf = None
    peak_dirs = None
    max_peaks_num = 0    
        
    MICRO_DA  = 0
    MICRO_DE  = 1
    MICRO_DR  = 2
    MICRO_FIN = 3    
    
    MICRO_PARAMS_NUM = len((MICRO_DA, MICRO_DE, MICRO_DR, MICRO_FIN)) 

        
    def __init__(self, dict_file=None, is_sorted=False, tessellation=dsiSphere8Fold()):
        
        self.tessellation = tessellation
        self._is_sorted = is_sorted

        if dict_file is not None:
            self.load(dict_file)
   
           
    def _sort_peaks(self):
        for j in range(self.peak_dirs.shape[2]):

            # First, convert spherical coordinates from Matlab (azim, elev, radius) to Python (radius, phi, theta) 
            # and then to Cartesian coordinates
            peak_dirs = np.array(sphere2cart(1, np.pi/2 + self.peak_dirs[1,:,j], self.peak_dirs[0,:,j]))
            peaks_filter = np.any(~np.isnan(peak_dirs), axis=0)

            if ~np.any(peaks_filter):
                continue

            peak_vertex_idx = np.argmax(np.dot(self.tessellation.vertices, peak_dirs[:,peaks_filter]), axis=0)     
            peak_vertex_values = self.odf[np.mod(peak_vertex_idx, self.odf.shape[0]),j]     
                  
            idx = np.arange(len(peak_vertex_values))
            
            # Sort in the descending order, hence -peak_vertex_values
            sorted_idx = np.argsort(-peak_vertex_values)
            
            if np.any(idx != sorted_idx):
                self.peak_dirs[:,idx,j] = self.peak_dirs[:,sorted_idx,j]
                self.micro[:,idx+1,j] = self.micro[:,sorted_idx+1,j]
                self.ratio[idx+1,j] = self.ratio[sorted_idx+1,j]

   
    def _peaks_per_voxel_cdf(self, total_dirs_num):
        """Cummulative Distribution Function (CDF) of a random variable: peaks_per_voxel"""
        
        # Numbers of directions are in the proportion 1 : 1*(k-1) : 1*(k-1)*(k-2) : ...
        # Thus, for k=321 (total_dirs_num) the cumulative number of directions is [1,321,102401,...] 
        cumulative_dirs_num = np.ones(self.max_peaks_num)
        
        # One fiber can have only one orientation [0,0,1]
        dirs_per_peak = 1

        # Compute the cumulative number of directions for fibers other than [0,0,1]
        for i in range(1,self.max_peaks_num):
            dirs_per_peak *= total_dirs_num-i 
            cumulative_dirs_num[i] = cumulative_dirs_num[i-1] + dirs_per_peak
        
        return cumulative_dirs_num[:-1] / cumulative_dirs_num[-1]
    
    
    def _validate_interval_parameter(self, parm):
        return np.array([np.min(parm), np.max(parm)])
    
    
    def _validate_fraction_volumes(self, p_iso, p_fib):
        
        # Convert constants or mismatched arrays to intervals
        p_iso = self._validate_interval_parameter(p_iso)
        p_fib = self._validate_interval_parameter(p_fib)
        
        # Lower bounds are hard limits, so they must sum up to less than 1
        if p_iso[0] + self.max_peaks_num * p_fib[0] >= 1:
            raise Exception(
                "Lower boundaries of fraction volumes are too high for max_peaks_num=%d" % self.max_peaks_num
            )
            
        return p_iso, p_fib


    def _validate_micro_parameters(self, f_in, D_iso, D_a, D_e, D_r):
        f_in = self._validate_interval_parameter(f_in)
        D_iso = self._validate_interval_parameter(D_iso)
        D_a = self._validate_interval_parameter(D_a)
        D_e = self._validate_interval_parameter(D_e)
        D_r = self._validate_interval_parameter(D_r)
        
        return f_in, D_iso, D_a, D_e, D_r 


    def _random_fraction_volumes(self, p_iso, p_fib, peaks_per_voxel):        
        fraction_volumes = np.zeros(peaks_per_voxel+1)
        
        # Lower bounds are hard limits, so the variability remains between 0 and p_random_max 
        p_random_max = 1 - (p_iso[0] + peaks_per_voxel * p_fib[0])

        # Draw fraction volumes randomly 
        p_random = np.hstack((
            np.random.uniform(0, p_iso[1] - p_iso[0]),
            np.random.uniform(0, p_fib[1] - p_fib[0], size=peaks_per_voxel)
        ))
        
        # Apply soft limits on upper bounds
        p_random /= np.maximum(1e-8, np.sum(p_random))
        
        # Set the fraction volumes of fibers
        fraction_volumes[1:] = p_fib[0] + p_random_max * p_random[1:]
        
        # Set the fraction volume of free water 
        fraction_volumes[0] = 1 - np.sum(fraction_volumes[1:])
       
        return fraction_volumes
    
    
    def _random_micro_parameters(self, f_in, D_iso, D_a, D_e, D_r, peaks_per_voxel, 
                                 equal_fibers, assert_faster_D_a, tortuosity_approximation):
        
        micro_params = np.zeros((self.MICRO_PARAMS_NUM, peaks_per_voxel+1))
        
        # Free water compartment has D_a=0, f_in=0, and D_a=D_e
        micro_params[1:3,0] = np.random.uniform(D_iso[0], D_iso[1])
        
        # Repeat until the microstructure parameters are valid
        while True:

            # Equal fibers means that all fibers have the same diffusivities and f_in
            if equal_fibers:
                micro_params[:,1:] = np.tile(
                    [[np.random.uniform(D_a[0], D_a[1])], [np.random.uniform(D_e[0], D_e[1])], 
                    [np.random.uniform(D_r[0], D_r[1])], [np.random.uniform(f_in[0], f_in[1])]], 
                    peaks_per_voxel
                )
            else:
                micro_params[:,1:] = np.array([
                    np.random.uniform(D_a[0], D_a[1], size=peaks_per_voxel), 
                    np.random.uniform(D_e[0], D_e[1], size=peaks_per_voxel), 
                    np.random.uniform(D_r[0], D_r[1], size=peaks_per_voxel), 
                    np.random.uniform(f_in[0], f_in[1], size=peaks_per_voxel)
                ])
                
            if assert_faster_D_a and np.any(micro_params[self.MICRO_DA,1:] < micro_params[self.MICRO_DE,1:]):
                continue

            if tortuosity_approximation:
                micro_params[self.MICRO_DR,1:] = (1 - micro_params[self.MICRO_FIN,1:]) * micro_params[self.MICRO_DA,1:]
                if np.any(micro_params[self.MICRO_DR,1:] < D_r[0]) or np.any(micro_params[self.MICRO_DR,1:] > D_r[1]):
                    continue
                
            break
        
        return micro_params
    
    
    def _compute_dwi(self, gtab, ratio, micro, peak_dirs_idx):
   
        ratio[np.isnan(ratio)] = 0
        micro[np.isnan(micro)] = 0
           
        # Convert the b-values from s/mm^2 to ms/um^2 
        bvals = np.vstack(1e-3 * gtab.bvals)
           
        # First, compute the diffusion signal of free water
        dwi = ratio[0] * np.exp(-bvals * micro[self.MICRO_DE,0])
           
        # Then, add the diffusion signal of fibers
        for j in range(len(peak_dirs_idx)):
               
            # Squared dot product of the b-vectors and the j-th peak directions
            dir_prod_sqr = np.dot(gtab.bvecs, self.tessellation.vertices[peak_dirs_idx[j]].T) ** 2
               
            dwi_intra = np.exp(-bvals * micro[self.MICRO_DA,j+1] * dir_prod_sqr)
            dwi_extra = np.exp(-bvals * (micro[self.MICRO_DE,j+1] * dir_prod_sqr + micro[self.MICRO_DR,j+1] * (1 - dir_prod_sqr)))
               
            dwi += ratio[j+1] * (micro[self.MICRO_FIN,j+1] * dwi_intra + (1 - micro[self.MICRO_FIN,j+1]) * dwi_extra)
           
        return dwi.T

    
    def _compute_odf_trace(self, gtab, odf_recon_model, ratio, micro, peak_dirs_idx):
        dwi = self._compute_dwi(gtab, ratio, micro, peak_dirs_idx)        
      
        # Compute the ODF for the generated DWI
        odf = odf_recon_model.fit(dwi).odf(self.tessellation).T
           
        return np.squeeze(odf[:len(self.tessellation.vertices)//2])

    
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

        # Compute the number of peaks per voxel as the number of non-NaN fiber directions per voxel            
        self.peaks_per_voxel = np.sum(~np.isnan(self._dict.peak_dirs[0,:,:]),axis=0)            


    def save(self, dict_file = 'odf_dict.mat'):
        odf_dict = {
            'odfrot': self.odf.T,
            'dirrot': self.peak_dirs.T,
            'micro' : self.micro.T,
            'rat'   : self.ratio.T,
            'adc'   : np.zeros_like(self.ratio.T)
        }
        hdf5storage.write(odf_dict, '.', dict_file, matlab_compatible=True)
 
    
    def generate(self, gtab, dict_size=1000000, max_peaks_num=3, equal_fibers=False,
                 p_iso=[0.0,1.0], p_fib=[0.0,1.0], f_in=[0.0,1.0], 
                 D_iso=[2.0,3.0], D_a=[0.5,2.5], D_e=[0.5,2.5], D_r=[0.0,2.0],
                 max_chunk_size=10000, odf_recon_model=None, 
                 assert_faster_D_a=False, tortuosity_approximation=False):
           
        if odf_recon_model is None:
            odf_recon_model = GeneralizedQSamplingModel(gtab, sampling_length=DEFAULT_DICT_EDGE)
           
        dict_size = np.maximum(1, dict_size)
        self.max_peaks_num = np.maximum(1, max_peaks_num)
        self.peaks_per_voxel = np.zeros(dict_size, dtype=int)
        p_iso, p_fib = self._validate_fraction_volumes(p_iso, p_fib)
        f_in, D_iso, D_a, D_e, D_r = self._validate_micro_parameters(f_in, D_iso, D_a, D_e, D_r)
   
        # Total number of directions allowed by the tessellation (k), by default k=321 
        total_dirs_num = len(self.tessellation.vertices) // 2
                   
        # Not used elements will be kept as NaNs for backward compatibility
        self.peak_dirs = np.nan * np.zeros((2, self.max_peaks_num, dict_size))        
        self.ratio = np.nan * np.zeros((self.max_peaks_num+1, dict_size))
        self.micro = np.nan * np.zeros((4, self.max_peaks_num+1, dict_size))
           
        self.odf = np.zeros((total_dirs_num, dict_size))
   
        # Generate the 0th element of the ODF-dictionary representing the 0 fibers case
        self.ratio[0,0] = 1    # no fibers, hence p_iso=1 
        self.micro[1:3,0] = 3  # diffusivity of free water at 37C
        self.peaks_per_voxel[0] = 0
        self.odf[:,0] = self._compute_odf_trace(gtab, odf_recon_model, self.ratio[:,0], self.micro[:,:,0], [])
   
        for chunk_idx in np.split(range(1, dict_size), range(max_chunk_size, dict_size, max_chunk_size)):
           
            print("%.1f%%" % (100 * (np.max(chunk_idx) + 1) / dict_size))
   
            chunk_size = len(chunk_idx)
            peak_dirs_idx = np.zeros((self.max_peaks_num, chunk_size), dtype=int)
   
            # Draw the numbers of peaks per voxel randomly. The direction [0,0,1] is obligatory, hence 1 + np.sum(...)
            self.peaks_per_voxel[chunk_idx] = 1 + np.sum(
                np.random.uniform(size=(chunk_size,1)) > self._peaks_per_voxel_cdf(total_dirs_num), axis=1
            )
           
            for i, j in zip(range(chunk_size), chunk_idx):
        
                # Obligatory direction [0,0,1] has index 0 in the tesselation, 
                # hence 1:peaks_per_voxel[j] and later: peaks_per_voxel[j]-1
                peak_dirs_idx[1:self.peaks_per_voxel[j],i] = np.random.choice(
                    range(1,total_dirs_num), self.peaks_per_voxel[j]-1, replace=False
                )
                   
                # Store spherical coordinates of the directions in the Matlab format (azim,elev) for backward compatibility 
                self.peak_dirs[:,:self.peaks_per_voxel[j],j] = np.array([
                    self.tessellation.phi[peak_dirs_idx[:self.peaks_per_voxel[j],i]], 
                    self.tessellation.theta[peak_dirs_idx[:self.peaks_per_voxel[j],i]] - np.pi/2
                ])
                   
                # Draw fraction volumes randomly
                self.ratio[:self.peaks_per_voxel[j]+1,j] = self._random_fraction_volumes(p_iso, p_fib, self.peaks_per_voxel[j])
              
                # Draw microstructure parameters randomly
                self.micro[:,:self.peaks_per_voxel[j]+1,j] = self._random_micro_parameters(
                    f_in, D_iso, D_a, D_e, D_r, self.peaks_per_voxel[j], 
                    equal_fibers, assert_faster_D_a, tortuosity_approximation
                )
               
            self.odf[:,chunk_idx] = self._compute_odf_trace(
                gtab, odf_recon_model, self.ratio[:,chunk_idx], self.micro[:,:,chunk_idx], peak_dirs_idx
            )
               
            recompute_filter = np.zeros(chunk_size, dtype=bool)
                 
            for i, j in zip(range(chunk_size), chunk_idx):
          
                if self.peaks_per_voxel[j] < 2:
                    continue
                        
                # Sort the peaks in the descending order, hence -self.odf
                sorted_idx = np.argsort(-self.odf[peak_dirs_idx[:self.peaks_per_voxel[j],i],j])
                        
                # If peaks were not sorted, reorder the microstructure parameters accordingly
                seq_idx = np.arange(self.peaks_per_voxel[j])
                if np.any(sorted_idx != seq_idx):
                    self.micro[:,seq_idx+1,j] = self.micro[:,sorted_idx+1,j]
                    self.ratio[seq_idx+1,j] = self.ratio[sorted_idx+1,j]
                        
                # If the highest peak was not at [0,0,1], recompute the ODF with the reordered parameters. 
                # Note that peak_dirs_idx[0] = 0, so it's sufficient to test if sorted_idx[0] != 0
                if sorted_idx[0] != 0:
                    recompute_filter[i] = True
                             
            self.odf[:,chunk_idx[recompute_filter]] = self._compute_odf_trace(
                gtab, odf_recon_model, self.ratio[:,chunk_idx[recompute_filter]], 
                self.micro[:,:,chunk_idx[recompute_filter]], peak_dirs_idx[:,recompute_filter]
            )


class PosteriorOdffpDictionary(OdffpDictionary):
    
    def __init__(self, odffp_fit, dict_file=None, is_sorted=False, tessellation=dsiSphere8Fold()):
        self._micro_pdf = {}
        self._ratio_pdf = {} 

        for peak_id in range(odffp_fit.get_max_peaks_num()+1):

            self._ratio_pdf[peak_id] = self._estimate_pdf(
                odffp_fit.get_compartment_volume(peak_id)
            )

            self._micro_pdf[peak_id] = {}

            # Diffusivity parameters of the free water compartment aren't estimated
            if peak_id < 1:
                continue
            
            for parameter_id in range(self.MICRO_PARAMS_NUM):
                self._micro_pdf[peak_id][parameter_id] = self._estimate_pdf(
                    odffp_fit.get_micro_parameter(parameter_id, peak_id)
                )

        OdffpDictionary.__init__(self, dict_file, is_sorted, tessellation)
    

    def _estimate_pdf(self, samples):
        return scipy.stats.gaussian_kde(samples[np.isfinite(samples)])


    def _random_from_pdf(self, pdf, min_value, max_value):
        while True:
            value = float(pdf.resample(1))
            if value < min_value or value > max_value:
                continue

            break
        
        return value
                

    def _random_fraction_volumes(self, p_iso, p_fib, peaks_per_voxel):        
        fraction_volumes = np.zeros(peaks_per_voxel+1)

        fraction_volumes[0] = self._random_from_pdf(self._ratio_pdf[0], p_iso[0], p_iso[1])

        for peak_id in range(1,peaks_per_voxel+1):
            fraction_volumes[peak_id] = self._random_from_pdf(self._ratio_pdf[peak_id], p_fib[0], p_fib[1])
                     
        return fraction_volumes / np.maximum(1e-8, np.sum(fraction_volumes))


    def _random_micro_parameters(self, f_in, D_iso, D_a, D_e, D_r, peaks_per_voxel, 
                                 equal_fibers, assert_faster_D_a, tortuosity_approximation):
        
        micro_params = np.zeros((self.MICRO_PARAMS_NUM, peaks_per_voxel+1))
        
        # Free water compartment has D_a=0, f_in=0, and D_a=D_e
        micro_params[1:3,0] = np.random.uniform(D_iso[0], D_iso[1])
        
        # Repeat until the microstructure parameters are valid
        while True:

            # Equal fibers means that all fibers have the same diffusivities and f_in
            if equal_fibers:
                micro_params[:,1:] = np.tile(
                    [
                        [self._random_from_pdf(self._micro_pdf[1][self.MICRO_DA], D_a[0], D_a[1])],
                        [self._random_from_pdf(self._micro_pdf[1][self.MICRO_DE], D_e[0], D_e[1])],
                        [self._random_from_pdf(self._micro_pdf[1][self.MICRO_DR], D_r[0], D_r[1])],
                        [self._random_from_pdf(self._micro_pdf[1][self.MICRO_FIN], f_in[0], f_in[1])]
                    ], 
                    peaks_per_voxel
                )
            else:
                for peak_id in range(1,peaks_per_voxel+1):
                    micro_params[:,peak_id] = np.array([
                        self._random_from_pdf(self._micro_pdf[peak_id][self.MICRO_DA], D_a[0], D_a[1]),
                        self._random_from_pdf(self._micro_pdf[peak_id][self.MICRO_DE], D_e[0], D_e[1]),
                        self._random_from_pdf(self._micro_pdf[peak_id][self.MICRO_DR], D_r[0], D_r[1]),
                        self._random_from_pdf(self._micro_pdf[peak_id][self.MICRO_FIN], f_in[0], f_in[1])
                    ])
                
            if assert_faster_D_a and np.any(micro_params[self.MICRO_DA,1:] < micro_params[self.MICRO_DE,1:]):
                continue

            if tortuosity_approximation:
                micro_params[self.MICRO_DR,1:] = (1 - micro_params[self.MICRO_FIN,1:]) * micro_params[self.MICRO_DA,1:]
                if np.any(micro_params[self.MICRO_DR,1:] < D_r[0]) or np.any(micro_params[self.MICRO_DR,1:] > D_r[1]):
                    continue
                
            break
        
        return micro_params


class OdffpModel(object):
 
    def __init__(self, gtab, odf_dict, 
                 drop_negative_odf=True, zero_baseline_odf=True, output_dict_odf=True,
                 odf_recon_model=None):
        """ODF-Fingerprinting reconstruction"""
        
        self.gtab = gtab

        self._drop_negative_odf = drop_negative_odf 
        self._zero_baseline_odf = zero_baseline_odf
        self._output_dict_odf = output_dict_odf
        
        if not hasattr(odf_dict, 'odf') or odf_dict.odf is None:
            raise Exception('Specified ODF-dictionary is corrupted or empty.')
         
        self._dict = odf_dict 
        
        if odf_recon_model is None:
            self._odf_recon_model = GeneralizedQSamplingModel(
                self.gtab, sampling_length=DEFAULT_RECON_EDGE
            )
        else:
            self._odf_recon_model = odf_recon_model
         
     
    @staticmethod 
    def resample_odf(odf, in_sphere, out_sphere):
        """Resamples ODF from the full sphere `in_sphere` to the full sphere `out_sphere`.
           The argument `odf` is either a single ODF vector or a matrix of ODF row-vectors.
           Returns a half-sphere ODF trace."""
           
        sphere_half_size = len(in_sphere.vertices) // 2
        
        # If `odf` is a single ODF vector, convert it to (1xODF) matrix.
        odf = np.atleast_2d(odf)
        
        if odf.shape[1] == sphere_half_size:
            odf = np.hstack((odf, odf))
        
        return np.squeeze(
            sh_to_sf(sf_to_sh(odf, in_sphere), out_sphere)[:,:sphere_half_size] 
        )
        

    def _normalize_odf(self, odf):
        """Normalizes a single ODF or multiple ODFs. 
           The argument `odf` is either a single ODF vector or a matrix of ODF column-vectors."""
        
        if self._drop_negative_odf:
            odf = np.maximum(0, odf)
        
        if self._zero_baseline_odf:
            odf -= np.min(odf, axis=0)
            
        odf_norm = np.maximum(1e-8, np.sqrt(np.sum(odf**2, axis=0)))
        return odf / odf_norm, odf_norm


    def _find_highest_peak_rotation(self, input_odf, target_dir=[0,0,1]):
        rotation = np.eye(3)
        input_peak_dirs,_,_ = peak_directions(input_odf, self._dict.tessellation)

        if len(input_peak_dirs) > 0:
            highest_peak_dir = np.squeeze(input_peak_dirs[0])
            
            cr = np.cross(highest_peak_dir, target_dir)
            sum_sqr_cr = np.sum(cr**2)
            
            if sum_sqr_cr != 0:
                s = np.array([[0,-cr[2],cr[1]], [cr[2],0,-cr[0]], [-cr[1],cr[0],0]])
                rotation += s + np.dot(s, s) * (1 - np.dot(highest_peak_dir, target_dir)) / sum_sqr_cr
    
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


    def _find_matching_odf_trace(self, input_odf_trace, dict_odf_trace, penalty, max_chunk_size):
        dot_product = np.dot(input_odf_trace, dict_odf_trace)
        
        if penalty > 0.0:
            dict_size = len(self._dict.peaks_per_voxel)
            for chunk_idx in np.split(range(dict_size), range(max_chunk_size, dict_size, max_chunk_size)):
                dot_product[:,chunk_idx] = np.log(dot_product[:,chunk_idx]) - 2 * penalty * self._dict.peaks_per_voxel[chunk_idx]
        
        return np.argmax(dot_product, axis=1) 

   
    def fit(self, data, mask=None, max_chunk_size=1000, penalty = DEFAULT_FIT_PENALTY):
        max_chunk_size = np.maximum(1, max_chunk_size)
        penalty = np.maximum(0.0, np.minimum(MAX_FIT_PENALTY, penalty))
 
        tessellation_size = len(self._dict.tessellation.vertices)
        tessellation_half_size = tessellation_size // 2
  
        if mask is None:
            mask = np.ones(data.shape[:-1], dtype=bool)
        else:
            mask = mask.astype(bool)
  
        masked_data = data[mask]
        voxels_num = masked_data.shape[0]

        dict_idx = np.zeros(voxels_num, dtype=int)
        dict_odf_trace,_ = self._normalize_odf(self._dict.odf)
         
        output_odf = np.zeros((voxels_num, tessellation_half_size))
        output_peak_dirs = np.zeros((voxels_num, self._dict.max_peaks_num, 3))
         
        for chunk_idx in np.split(range(voxels_num), range(max_chunk_size, voxels_num, max_chunk_size)):
         
            print("%.1f%%" % (100 * (np.max(chunk_idx) + 1) / voxels_num))
         
            chunk_size = len(chunk_idx)
 
            input_odf = self._odf_recon_model.fit(masked_data[chunk_idx]).odf(self._dict.tessellation)
            input_odf_trace = np.zeros((chunk_size, tessellation_half_size))
            input_odf_norm = np.zeros(chunk_size)
              
            rotation = np.zeros((chunk_size, 3, 3))
            rotated_tessellation = {}
             
            for i in range(chunk_size):
 
                rotation[i] = self._find_highest_peak_rotation(input_odf[i])
                rotated_tessellation[i] = self._rotate_tessellation(self._dict.tessellation, rotation[i])
             
                input_odf_trace[i], input_odf_norm[i] = self._normalize_odf(
                    self.resample_odf(input_odf[i], self._dict.tessellation, rotated_tessellation[i])
                )
 
            dict_idx[chunk_idx] = self._find_matching_odf_trace(input_odf_trace, dict_odf_trace, penalty, max_chunk_size) 
         
            for i, j in zip(range(chunk_size), chunk_idx):
                if self._output_dict_odf:
                    output_odf[j] = self.resample_odf(
                        input_odf_norm[i] * self._dict.odf[:,dict_idx[j]], 
                        rotated_tessellation[i], self._dict.tessellation
                    )
                else:
                    output_odf[j] = input_odf[i][:tessellation_half_size]
                   
                output_peak_dirs[j] = self._rotate_peak_dirs(self._dict.peak_dirs[:,:,dict_idx[j]], rotation[i])
                 
#                 ## DEBUG:
#                 
#                 plot_odf(input_odf[i], 'odf/%08d_in.png' % j)
#                 plot_odf(output_odf[j], 'odf/%08d_out.png' % j)
#                 
#                 ##
    
        return OdffpFit(
            data, self._dict, 
            self._unmask(output_odf, mask), 
            self._unmask(output_peak_dirs, mask),
            self._unmask(dict_idx, mask)
        )
            
        
class OdffpFit(OdfFit):
    
    # Constants imposed by the .FIB format (DSI Studio)
    FIB_COORDS_ORIENTATION = np.array(('L', 'P', 'S'))
    FIB_ODF_MAX_CHUNK_SIZE = 20000
    
    
    def __init__(self, data, odf_dict, odf, peak_dirs, dict_idx):
        self._data = data
        self._dict = odf_dict
        self._odf = odf
        self._peak_dirs = peak_dirs
        self._dict_idx = dict_idx
    
        
    def _nonzero_dict_idx(self):
        return self._dict_idx[self._dict_idx > 0]

    
    def _export_dict_var(self, dict_var, peak_id):
        if peak_id > self._dict.max_peaks_num:
            raise Exception("Argument peak_id=%d exceeds the maximum number of peaks." % peak_id)
            
        return dict_var[peak_id,self._nonzero_dict_idx()]

    
    def _fib_reshape(self, matrix, new_size, orientation_agreement=True):
        if not np.all(orientation_agreement):
            matrix = np.flip(matrix, np.arange(3)[~orientation_agreement])
        
        return matrix.reshape(new_size, order='F')


    def _fib_index_map(self, var_data, dict_idx, slice_size, fa_filter):
        index_map = np.zeros(dict_idx.shape)
        index_map[fa_filter] = var_data[dict_idx[fa_filter]]
        return self._fib_reshape(index_map, slice_size)


    def odf(self, sphere=None):
        if sphere is None or sphere == self._dict.tessellation:
            output_odf = self._odf
        else:
            output_odf = OdffpModel.resample_odf(self._odf, self._dict.tessellation, sphere)
        
        return output_odf / np.maximum(1e-8, np.max(output_odf))
    
    
    def peak_dirs(self):
        return self._peak_dirs
    
    
    def get_max_peaks_num(self):
        return self._dict.max_peaks_num
    
    
    def get_micro_parameter(self, parameter_id, peak_id):
        return self._export_dict_var(self._dict.micro[parameter_id], peak_id)
    
    
    def get_compartment_volume(self, peak_id):
        return self._export_dict_var(self._dict.ratio, peak_id)
    
    
    def save_as_fib(self, affine, voxel_size, output_file_name='output.fib'):
        fib = {}
        fib['dimension'] = self._data.shape[:-1]
        fib['voxel_size'] = voxel_size
        fib['odf_vertices'] = self._dict.tessellation.vertices.T
        fib['odf_faces'] = self._dict.tessellation.faces.T
        
        voxels_num = np.prod(fib['dimension'])
        slice_size = [fib['dimension'][0] * fib['dimension'][1], fib['dimension'][2]]

        tessellation_half_size = len(self._dict.tessellation.vertices) // 2

        try:
            orientation_agreement = np.array(nib.aff2axcodes(affine)) == self.FIB_COORDS_ORIENTATION
        except:
            warnings.warn("Couldn't determine the orientation of coordinates from the affine.")
            orientation_agreement = np.ones(3, dtype=bool)
                
        # Reorient maps to LPS and reshape them to voxels_num x N (requirements of the FIB format)
        output_odf = self._fib_reshape(self._odf, (voxels_num, tessellation_half_size), orientation_agreement)
        output_peak_dirs = self._fib_reshape(self._peak_dirs, (voxels_num, self._dict.max_peaks_num, 3), orientation_agreement)
        dict_idx = self._fib_reshape(self._dict_idx, voxels_num, orientation_agreement)

        # Resample output ODFs to the LPS coordinates     
        if not np.all(orientation_agreement):
            fib_tessellation = Sphere(
                xyz=(2 * orientation_agreement.astype(int) - 1) * self._dict.tessellation.vertices,
                faces=self._dict.tessellation.faces
            )
            for odf_chunk_idx in np.split(range(voxels_num), range(self.FIB_ODF_MAX_CHUNK_SIZE, voxels_num, self.FIB_ODF_MAX_CHUNK_SIZE)):
                output_odf[odf_chunk_idx] = OdffpModel.resample_odf(
                    output_odf[odf_chunk_idx], self._dict.tessellation, fib_tessellation
                )
        else:
            fib_tessellation = self._dict.tessellation
  
        for i in range(self._dict.max_peaks_num):
            fib['fa%d' % i] = np.zeros(voxels_num)
            fib['index%d' % i] = np.zeros(voxels_num)
        
        for j in range(voxels_num):
            peaks_filter = np.any(output_peak_dirs[j] != 0, axis=1)
            peak_vertex_idx = np.mod(
                np.argmax(np.dot(output_peak_dirs[j,peaks_filter], fib_tessellation.vertices.T), axis=1), 
                tessellation_half_size
            )     
            peak_vertex_values = output_odf[j][peak_vertex_idx]     
                 
            for i in range(len(peak_vertex_idx)):
                fib['index%d' % i][j] = np.mod(peak_vertex_idx[i], tessellation_half_size)
                fib['fa%d' % i][j] = peak_vertex_values[i] - np.min(output_odf[j])

        for i in range(self._dict.max_peaks_num):
            fib['fa%d' % i] -= np.min(fib['fa%d' % i])
            fib['nqa%d' % i] = fib['fa%d' % i] / np.maximum(1e-8, np.max(fib['fa%d' % i]))

        fa_filter = fib['fa0'] > 0

        if np.any(fa_filter):
            odf_map = output_odf[fa_filter,:tessellation_half_size]
            odf_map /= np.maximum(1e-8, np.max(odf_map))
            
            odf_chunk_idx = 0
            for odf_chunk in np.split(odf_map, range(self.FIB_ODF_MAX_CHUNK_SIZE, len(odf_map), self.FIB_ODF_MAX_CHUNK_SIZE)):
                fib['odf%d' % odf_chunk_idx] = odf_chunk.T
                odf_chunk_idx += 1
        else:
            warnings.warn("The output is empty.")

        # Fetch microstructure parameters
        fib['D_iso'] = self._fib_index_map(self._dict.micro[1,0], dict_idx, slice_size, fa_filter)
        fib['p_iso'] = self._fib_index_map(self._dict.ratio[0], dict_idx, slice_size, fa_filter)

        for i in range(1, self._dict.max_peaks_num+1):
            index_maps = {
                'fib%d_Da' % i: self._dict.micro[0,i], 'fib%d_De' % i: self._dict.micro[1,i],
                'fib%d_Dr' % i: self._dict.micro[2,i], 'fib%d_fin' % i: self._dict.micro[3,i],
                'fib%d_p' % i: self._dict.ratio[i]
            }
            for index_name in index_maps:
                fib[index_name] = self._fib_index_map(index_maps[index_name], dict_idx, slice_size, fa_filter)

        for i in range(self._dict.max_peaks_num):
            fib['fa%d' % i] = self._fib_reshape(fib['fa%d' % i], slice_size)
            fib['nqa%d' % i] = self._fib_reshape(fib['nqa%d' % i], slice_size)
            fib['index%d' % i] = self._fib_reshape(fib['index%d' % i], (1, voxels_num))
        
        output_file_prefix = output_file_name.replace(".fib.gz", "").replace(".fib", "") 
        savemat("%s.fib" % output_file_prefix, fib, format='4')        
 
        with open("%s.fib" % output_file_prefix, 'rb') as fib_file:
            with gzip.open("%s.fib.gz" % output_file_prefix, 'wb') as fib_gz_file:
                fib_gz_file.writelines(fib_file)
                 
        os.remove("%s.fib" % output_file_prefix)
        
