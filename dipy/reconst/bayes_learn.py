import scipy as sc
import nibabel as nib
import numpy as np
from dipy.io import read_bvals_bvecs
from scipy.io import loadmat
import dipy.reconst.dti as dti
from dipy.core.gradients import gradient_table


# loading the example data
fbvals = '/home/shreyasfadnavis/Downloads/baydiff.minimal/example_data/data_hex28.bval'
fbvecs = '/home/shreyasfadnavis/Downloads/baydiff.minimal/example_data/data_hex28.bvec'
fdata = '/home/shreyasfadnavis/Downloads/baydiff.minimal/example_data/data_hex28.nii'

# loading the data and creating the mask as per Matlab
bvals, bvecs = read_bvals_bvecs(fbvals, fbvecs)
img = nib.load(fdata)
dwi = img.get_data()
mask = dwi[:, :, :, -1]
mask[mask > 150] = 0
# mat = loadmat('baydif_model.mat')
gtab = gradient_table(bvals, bvecs)
noise = 'estglobal'
sigma_snrestimation = 1.75


def fspecial_gauss(shape=(3, 3), sigma=0.5):
    """
    2D gaussian mask - should give the same result as MATLAB's
    fspecial('gaussian',[shape],[sigma])
    """
    m, n = [(ss - 1.) / 2. for ss in shape]
    y, x = np.ogrid[-m: m + 1, -n: n + 1]
    h = np.exp(-(x * x + y * y) / (2. * sigma * sigma))
    h[h < np.finfo(h.dtype).eps*h.max()] = 0
    sumh = h.sum()
    if sumh != 0:
        h /= sumh
    return h


def makeGau(sigma, sz):
    ng = np.ceil(sigma * 5)
    ng = ng + np.mod(ng, 2) + 1
    gau = fspecial_gauss([ng, 1], sigma)
    g = np.tile(gau, (11, 11, 11))
    gx = g[]
    gau = gx * gy * gz
    sz = sz[0:2]
    gau = np.pad(gau, (np.floor((sz - ng) / 2), np.floor((sz - ng) / 2) -
                       np.mod(sz, 2) + 1), 'constant')
    gau = sc.fft(sc.ifft(gau))
    gau = gau / gau[0]


# generate the tensors
T = 0
ten = np.zeros((3, 3, 28))
for i in range(len(bvals)):
    gdir = T * bvecs.T[:, i]
    gdir = gdir / (np.spacing(1) + np.linalg.norm(gdir))
    ten[:, :, i] = gdir * gdir.T * bvals[i]

sigma_smooth = 0
b = np.squeeze(ten[0, 0, :] + ten[1, 1, :] + ten[2, 2, :])
sz = np.asarray(np.shape(dwi))

if sigma_smooth == 0:
    preproc = lambda x: x
else:
    gau = makeGau(sigma_smooth, sz)
    sm2 = lambda x: np.real(sc.ifft(sc.fft(x) * gau))
    col = lambda x: x[:]
    preproc = lambda x: sm2[np.reshape(x, sz[1:3])]  

b0idx = np.round(b / 100) == 0

if sum(b0idx) < 6 and not noise.isnumeric():
    print("not enough b=0 images of noise estimation (10)")
    noise = 10

gau = makeGau(sigma_snrestimation, sz)
sm = lambda x: np.real(sc.ifft(sc.fft(x) * gau))
b0 = np.mean(dwi[:, :, :, b0idx], 4)
if isinstance(noise, str):
    if noise == 'estlocal':
        err = np.std(dwi[:, :, :, b0idx], [], 4)
        SNR = b0 / (np.spacing(1) + err)
    elif noise == 'estglobal':
        mb0 = np.mean(dwi[:, :, :, b0idx], 4)
        _, idx = np.sort(mb0[:])
        idx = idx[round(len(idx) * 0.6):]
        err = np.std(dwi[:, :, :, b0idx], [], 4)
        err = np.mean(err(idx))
        SNR = b0 / (np.spacing(1) + err)
else:
    if len(np.size(noise)) == 3:
        SNR = noise
    else:
        SNR = b0 / noise
SNR[SNR > 100] = 100
SNR = sm[SNR]
