'''
Created on Feb 18, 2021

@author: patrykfi
'''

import os.path

from dipy.data import Sphere
from dipy.reconst.shm import CsaOdfModel

from scipy.io import loadmat


def dsiSphere8Fold():
    dsi_sphere = loadmat(os.path.join(
        os.path.dirname(__file__), "../data/files/dsi_sphere_8fold.mat"
    ))
    return Sphere(
        x=dsi_sphere['odf_vertices'][0,:],
        y=dsi_sphere['odf_vertices'][1,:],
        z=dsi_sphere['odf_vertices'][2,:],
        faces=dsi_sphere['odf_faces']
    )


class OdffpModel(object):

    def __init__(self, gtab):
        self.gtab = gtab
        
    def fit(self, data):
        csa_model = CsaOdfModel(self.gtab, sh_order=6)
        csa_fit = csa_model.fit(data)
        csa_odf = csa_fit.odf(dsiSphere8Fold())
        return csa_odf
        
 
        
class OdffpFit(object):
    
    def __init__(self):
        pass