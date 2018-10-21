import scipy as sc
import nibabel as nib
import numpy as np
from dipy.io import read_bvals_bvecs

# loading the example data
fbvals = '/home/shreyasfadnavis/Downloads/baydiff.minimal/example_data/data_hex28.bval'
fbvecs = '/home/shreyasfadnavis/Downloads/baydiff.minimal/example_data/data_hex28.bvec'
fdata = '/home/shreyasfadnavis/Downloads/baydiff.minimal/example_data/data_hex28.nii'

bvals, bvecs = read_bvals_bvecs(fbvals, fbvecs)
img = nib.load(fdata)
data = img.get_data()

def loadModel(data, varargin):
    bd_model = bd_trainmodel(data, varargin)
    return bd_model

def baydiff(data, varargin):
    noise = 'estglobal'
    sigma_snrestimation = 1.75
    sigma_smooth = 0
    ten = data_tensor
    b = np.squeeze(ten[0, 0, :] + ten[1, 1, :] + ten[2, 2, :])
    sz = np.size(data_dwi)
    if sigma_smooth == 0:
        preproc = lambda x: x
    else:
        gau = makeGau(sigma_smooth, sz)
        sm2 = lambda x: np.real(sc.ifft(sc.fft(x) * gau))
        col = lambda x: x[:]
        preproc = lambda x: sm2[np.reshape(x, sz[1:3])]
    
    b0idx = round(b / 100) == 0
    
    if sum(b0idx) < 6 and not noise.isnumeric():
        print("not enough b=0 images ofr noise estimation (10)")
        noise = 10
    

def makeGau(sigma, sz):
    ng = np.ceil(sigma * 5) 
    ng = ng + np.mod(ng, 2) + 1
    gau = 