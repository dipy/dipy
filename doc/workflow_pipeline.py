import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from time import time
from dipy.denoise.nlmeans import nlmeans
from dipy.denoise.noise_estimate import estimate_sigma
from dipy.data import fetch_sherbrooke_3shell, read_sherbrooke_3shell

fetch_sherbrooke_3shell()
img, gtab = read_sherbrooke_3shell()

data = img.get_data()
affine = img.affine

mask = data[..., 0] > 80

# We select only one volume for the example to run quickly.
data = data[..., 1]

print("vol size", data.shape)

sigma = estimate_sigma(data, N=4)

t = time()

t = time()

den = nlmeans(data, sigma=sigma, mask=mask, patch_radius=1, block_radius=1,
              rician=True)

print("total time", time()-t)
