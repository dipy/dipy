import os
import numpy as np
import numpy.matlib
import pandas as pd

import torch
import torchkbnufft as tkbn

from dipy.core.sphere import Sphere
from dipy.core.dsi_sphere import dsiSphere8Fold

from dipy.reconst.odf import OdfModel, OdfFit
from dipy.reconst.cache import Cache
from dipy.reconst.multi_voxel import multi_voxel_fit

from scipy.spatial import Voronoi
from scipy.spatial import ConvexHull


class RadialDsiModel(OdfModel, Cache):

    @staticmethod
    def extract_shells(values, diff_threshold = 50):
        shells = []
        sorted_values = np.sort(values)
      
        for values_chunk in np.split(sorted_values, 1 + np.nonzero(np.diff(sorted_values) > diff_threshold)[0]):
            shells.append(np.mean(values_chunk))
        
        return np.array(shells)


    def __init__(self, gtab, sampling_length=1.2, tessellation=dsiSphere8Fold(), density_compensation=True):
        self.gtab = gtab
        self.tessellation = tessellation

        if self.gtab.big_delta is None:
            self.gtab.big_delta = 1
        
        if self.gtab.small_delta is None:
            self.gtab.small_delta = 0
        
        self.dir_filter = ~gtab.b0s_mask

        # Choose if a density compensation function (DCF) should be computed 
        self._density_compensation = density_compensation

        # Take only the first b0 image
        if (np.sum(gtab.b0s_mask)):
            self.dir_filter[np.nonzero(gtab.b0s_mask)[0][0]] = True
        
        # Multiply q-values by 2*np.pi for backward compatibility with the Matlab version
        self.qtable = np.vstack(self.gtab.qvals[self.dir_filter]) * self.gtab.bvecs[self.dir_filter] * 2 * np.pi

        # Set the sampling length to determine max_displacement, then compute the transition matrix.
        self.set_sampling_length(sampling_length)
        
        
    def set_sampling_length(self, sampling_length):
        self.max_displacement = sampling_length / (2 * np.max(np.diff(np.sqrt(
            self.extract_shells(self.gtab.bvals)
        ))))
        self.compute_transition_matrix()
        
        
    def compute_transition_matrix(self):
        E = np.dot(self.tessellation.vertices, self.qtable.T)
        F = -self._sinc_second_derivative(2 * np.pi * E * self.max_displacement)

        if self._density_compensation:        
            F = np.multiply(F, np.matlib.repmat(self._dcf_calc(), E.shape[0], 1))
    
        self.transition_matrix = F.T

    
    def _sinc_second_derivative(self, x):
        result = np.zeros_like(x)
        
        near_zero_filter = np.abs(x) <= 1e-3
        
        x0 = x[near_zero_filter]
        x1 = x[~near_zero_filter]
        
        result[near_zero_filter] = -1/3 + x0 * x0 / 10
        result[~near_zero_filter] = 2 * np.sin(x1) / x1 / x1 / x1 - 2 * np.cos(x1) / x1 / x1 - np.sin(x1) / x1

        return result


    def _ir_mri_dcf_voronoi0(self, kspace):
    
        # Find the points at the origin
        i0 = np.sum(np.abs(kspace), 1) == 0
        
        # if multiple zero points found, keep the first one only
        if np.sum(i0) > 1:
        
            # Find the first zero point
            i0f = np.nonzero(i0)[0][0]
            i0[i0f] = False
        
            wi = np.zeros((kspace.shape[0], 1))
            wi[~i0] = self._ir_mri_dcf_voronoi(kspace[~i0,:])
        
            i0[i0f] = True
        
            # Distribute DCF equally
            wi[i0] = wi[i0f] / np.sum(i0)

        else:
            wi = self._ir_mri_dcf_voronoi(kspace)
        
        return wi


    def _ir_mri_dcf_voronoi(self, kspace):
        wi = np.zeros((kspace.shape[0], 1))
        
        vor = Voronoi(kspace, qhull_options='Qbb')
        v = vor.vertices
        c = vor.regions
        
        for mm in range(len(wi)):

            # Disregard infinite cells                    
            if -1 in c[vor.point_region[mm]]:
                continue
            
            try:
                conv_hull = ConvexHull(v[c[vor.point_region[mm]],:])
                wi[mm] = conv_hull.volume
            except:
                print("Couldn't compute convex hull of the cell %d." % mm)
            
        return wi

    
    def dcf_calc(self):
        normalize = True
        dimi = self.qtable.shape[1]
        qval = self.gtab.qvals * 2 * np.pi
        qshells = RadialDsiModel.extract_shells(qval, 5)        
        steps = qshells
        
        # k-space trajectory and FoV (for normalization)
        nz = 2
        dq = np.mean(np.diff(qshells * 1e3))
        Rmax = 1 / dq
        FoV = 2 * Rmax * np.ones(3)
        res = (2 * len(qshells) + 1) * np.ones(3)
        kspace = (self.qtable * FoV[0] * np.pi) / (2 * res[0])
        kspace = np.vstack((kspace,-kspace))
        kspace = (kspace * np.pi) / np.max(kspace)
        
        qvalextra = max(qshells) + np.mean(np.diff(qshells[1:]))

        dirextra = pd.read_csv(
            os.path.join(os.path.dirname(__file__), "../data/files/rdsi_dir1000.txt"), sep=' ', header=None
        ).values

        kspacet = np.vstack((kspace,dirextra*1/2*2*np.pi*qvalextra/qshells[-1])) * qshells[-1]/qvalextra

        Dest = self._ir_mri_dcf_voronoi0(kspacet)
        Dest = Dest[:-dirextra.shape[0]]

        dcf = Dest[:int(Dest.shape[0]/2)]

        if normalize:
            
            om = kspace
            Nd = tuple(res.astype(int)) 
            Kd = tuple(2 * res.astype(int))

            spmatrix = tkbn.calc_tensor_spmatrix(torch.from_numpy(kspace.T), im_size=Nd, grid_size=Kd, numpoints=5)
            stp = spmatrix[0] + 1j*spmatrix[1]
            dcf2 = np.matlib.repmat(dcf, 2, 1)
            
            o = torch.complex(
                torch.from_numpy(np.ones((int(np.prod(2*res)), 1))), 
                torch.from_numpy(np.zeros((int(np.prod(2*res)), 1)))
            )
            
            out = torch.matmul(stp.to_dense().T, torch.from_numpy(dcf2) * torch.matmul(stp, o))
            dcf = dcf / torch.mean(torch.abs(out)).item()
        
        return dcf
    
    
    # Density compensation function
    def _dcf_calc(self, normalize = True):
        qshells = RadialDsiModel.extract_shells(self.gtab.qvals * 2 * np.pi, 5)        

        dq = np.mean(np.diff(qshells * 1e3))
        rmax = 1 / dq
        fov = 2 * rmax * np.ones(3)
        res = (2 * len(qshells) + 1) * np.ones(3)
        kspace = (self.qtable * fov[0] * np.pi) / (2 * res[0])
        kspace = np.vstack((kspace,-kspace))
        kspace = (kspace * np.pi) / np.max(kspace)
        
        # Add extra outside Q-shell
        qvalextra = max(qshells) + np.mean(np.diff(qshells[1:]))
        dirextra = pd.read_csv(
            os.path.join(os.path.dirname(__file__), "../data/files/rdsi_dir1000.txt"), sep=' ', header=None
        ).values

        kspacet = np.vstack((kspace,dirextra*1/2*2*np.pi*qvalextra/qshells[-1])) * qshells[-1]/qvalextra

        Dest = self._ir_mri_dcf_voronoi0(kspacet)
        Dest = Dest[:-dirextra.shape[0]]

        dcf = Dest[:int(Dest.shape[0]/2)]
        
        if normalize:
        
            # Compute sparse Kaiser-Bessel interpolation matrix            
            spmatrix = tkbn.calc_tensor_spmatrix(
                torch.from_numpy(kspace.T), 
                im_size=tuple(res.astype(int)), grid_size=tuple(2 * res.astype(int)), numpoints=5
            )
            stp = spmatrix[0] + 1j*spmatrix[1]
            
            ones_complex = torch.complex(
                torch.from_numpy(np.ones((int(np.prod(2*res)), 1))), 
                torch.from_numpy(np.zeros((int(np.prod(2*res)), 1)))
            )
            
            out = torch.matmul(
                stp.to_dense().T, 
                torch.from_numpy(np.matlib.repmat(dcf, 2, 1)) * torch.matmul(stp, ones_complex)
            )
            dcf = dcf / torch.mean(torch.abs(out)).item()
        
        return dcf.T


    def fit(self, data):
        return RadialDsiFit(self, data)
    
    
class RadialDsiFit(OdfFit):

    def __init__(self, model, data):
        self._model = model
        self._data = data
    
    
    def odf(self, sphere=None):
        
        if sphere is not None and sphere != self._model.tessellation:
            self._model.tessellation = sphere
            self._model.compute_transition_matrix()
        
        return np.matmul(self._data[...,self._model.dir_filter], self._model.transition_matrix)

