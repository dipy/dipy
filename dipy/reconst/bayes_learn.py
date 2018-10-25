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

    data_dwi = data_dwi
    gau = makeGau(sigma_snrestimation, sz)
    sm = lambda x: np.real(sc.ifft(sc.fft(x) * gau))
    b0 = np.mean(data_dwi[:, :, :, b0idx], 4)
    if isinstance(noise, str):
            if noise == 'estlocal':
                err = np.std(data.dwi[:, :, :, b0idx], [], 4)
                SNR = b0 / (eps + err)
            elif noise == 'estglobal':
                _, idx = np.sort(mb0[:])
                idx = idx[round(len(idx) * 0.6):end]
                err = np.std(data.dwi[:, :, :, b0idx], [], 4)
                err = np.mean(err(idx))
                SNR = b0 / (eps + err)
    else:
        if len(np.size(noise)) == 3:
            SNR = noise
        else:
            SNR = b0 / noise
    
    SNR[SNR > 100] = 100
    SNR = sm[SNR]

def fgaussian(size, sigma):
    m, n = size
    h, k = m // 2, n // 2
    x, y = np.mgrid[-h:h, -k:k]
    return x, y


def makeGau(sigma, sz):
    ng = np.ceil(sigma * 5)
    ng = ng + np.mod(ng, 2) + 1
    gau = fgaussian([ng, 1], sigma)
    gx, gy, gz = np.meshgrid(gau)
    gau = gx * gy * gz
    sz = sz[0:2]
    gau = np.pad(gau, (np.floor((sz - ng) / 2), np.floor((sz - ng) / 2) -
                       np.mod(sz, 2) + 1), 'constant')
    gau = sc.fft(sc.ifft(gau))
    gau = gau/ gau[0]
    