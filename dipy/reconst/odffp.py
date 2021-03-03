'''
Created on Feb 18, 2021

@author: patrykfi
'''

import os.path

from dipy.data import Sphere
from dipy.reconst.gqi import GeneralizedQSamplingModel
from dipy.reconst.dsi import DiffusionSpectrumModel

from scipy.io import loadmat


def dsiSphere8Fold():
    dsi_sphere = loadmat(os.path.join(
        os.path.dirname(__file__), "../data/files/dsi_sphere_8fold.mat"
    ))
    return Sphere(
        x=dsi_sphere['odf_vertices'][0,:],
        y=dsi_sphere['odf_vertices'][1,:],
        z=dsi_sphere['odf_vertices'][2,:],
        faces=dsi_sphere['odf_faces'].T
    )


class OdffpModel(object):

    def __init__(self, gtab):
        self.gtab = gtab
        
    def fit(self, data):
        tessellation = dsiSphere8Fold()
        
        diff_model = GeneralizedQSamplingModel(self.gtab)
#         diff_model = DiffusionSpectrumModel(self.gtab)
        diff_fit = diff_model.fit(data)
        diff_odf = diff_fit.odf(tessellation)
        
        # This function can return ODFs, checkout at:
        # https://dipy.org/documentation/1.1.1./reference/dipy.direction/#id23
        diff_peaks = peaks_from_model(
            model=diff_model, data=data, sphere=tessellation, 
            relative_peak_threshold=.5, min_separation_angle=25, npeaks=1
        )
        
        return diff_odf

        
class OdffpFit(object):
    
    def __init__(self):
        pass