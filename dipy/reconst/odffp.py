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
        diff_model = GeneralizedQSamplingModel(self.gtab)
#         diff_model = DiffusionSpectrumModel(self.gtab)
        diff_fit = diff_model.fit(data)
        diff_odf = diff_fit.odf(dsiSphere8Fold())
        return diff_odf

        
class OdffpFit(object):
    
    def __init__(self):
        pass