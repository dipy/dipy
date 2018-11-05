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


def iso_gaussian_3d(mu=5, sigma=0.2, dim=11):
    out = np.zeros((dim, dim, dim))
    cov = np.eye(3) * sigma
    icov = np.linalg.inv(cov)
    det = np.linalg.det(cov)

    for i in range(dim):
        for j in range(dim):
            for k in range(dim):
                x = np.array([i, j, k])
                numer = (-1/2) * ((x - mu).T.dot(icov)).dot((x - mu))
                denom = 1 / ((2 * np.pi) * (det**(1/2)))
                out[i, j, k] = (denom * np.exp(numer))
    return out

def makeGau(sigma, sz):
    ng = np.ceil(sigma * 5)
    ng = ng + np.mod(ng, 2) + 1
    gau = fspecial_gauss([ng, 1], sigma)
    gx, gy, gz = np.meshgrid(gau, gau, gau)
    gau = iso_gaussian_3d(mu=5, sigma=1.75, dim=11)
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


def shMatrix(n, lmax):
    n = n / np.tile(sum(pow(n, 2)), (3, 1))
    M = []
    for i in range(0, lmax, 2):
        m = sc.special.lpn(i, n[2, :])
        n2 = n[0, :] + i * n[1, :] + np.spacing(1)
        n2 = n2 / abs(n2)
        m = np.flipud(m) * np.tile(n2, (np.size(m), 1)) ** \
            np.tile((np.matrix([j for j in range(i)])).T, (1, np.size(n2)))

        idx1 = np.size(M)
        M = np.transpose([M, m[0: -1] * np.sqrt(2), m[-1, :]])
        idx2 = np.size(M)
        idx[i/2 + 1] = idx[idx1+1:idx2]
    M = M / np.sqrt(np.size(n))
    return M, idx


def compPowerSpec(b, scheme, lmax, S, pp, qspace, nmax, D0):
    M = []
    buni = np.unique(round(b * 10)) / 10
    S[S > 2] = 0
    dirs = bvals
    proj_tmp, idx_sh = shMatrix(dirs+np.spacing(1), lmax)    
    