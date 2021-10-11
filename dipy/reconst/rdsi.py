import numpy as np

# from scipy.ndimage import map_coordinates
# from scipy.fftpack import fftn, fftshift, ifftshift
from dipy.reconst.odf import OdfModel, OdfFit
from dipy.reconst.cache import Cache
from dipy.reconst.multi_voxel import multi_voxel_fit


class RadialDsiModel(OdfModel, Cache):

    def __init__(self, gtab, sampling_length=1.2):
        self.gtab = gtab

        if self.gtab.big_delta is None:
            self.gtab.big_delta = 1
        
        if self.gtab.small_delta is None:
            self.gtab.small_delta = 0
        
        # Multiply by 2*np.pi for backward compatibility with the Matlab version
        self.qtable = np.vstack(self.gtab.qvals) * self.gtab.bvecs * 2 * np.pi
            
        bshells = []
        sorted_bvals = np.sort(self.gtab.bvals)
      
        for bvals_chunk in np.split(sorted_bvals, 1 + np.nonzero(np.diff(sorted_bvals) > 50)[0]):
            bshells.append(np.mean(bvals_chunk))
            
        self.max_displacement = sampling_length / (2 * np.max(np.diff(np.sqrt(bshells))))
            
    
    def fit(self, data):
        return RadialDsiFit(self, data)
    
    
class RadialDsiFit(OdfFit):

    def _sinc_second_derivative(self, x):
        result = np.zeros_like(x)
        
        near_zero_filter = np.abs(x) <= 1e-3
        
        x0 = x[near_zero_filter]
        x1 = x[~near_zero_filter]
        
        result[near_zero_filter] = -1/3 + x0 * x0 / 10
        result[~near_zero_filter] = 2 * np.sin(x1) / x1 / x1 / x1 - 2 * np.cos(x1) / x1 / x1 - np.sin(x1) / x1
        
        return result


    def __init__(self, model, data):
        self._model = model
        self._data = data
    
    
    def odf(self, sphere):
        E = np.dot(sphere.vertices, self._model.qtable.T)
        F = -self._sinc_second_derivative(2 * np.pi * E * self._model.max_displacement)
        
        odf = np.matmul(self._data, F.T)
#         clear dwi;
#         % odf = odf(1:(end/2),:);
#         odf = odf';
#         
#         odf(isnan(odf)) = 0;
        
        return odf