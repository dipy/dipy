import numpy as np
from dipy.data import fetch_stanford_hardi
from dipy.data import read_stanford_hardi
from dipy.denoise.aonlm import aonlm
import matplotlib.pyplot as plt
from dipy.denoise.mixingsubband import mixingsubband


if __name__ == '__main__':
    fetch_stanford_hardi()
    img, gtab = read_stanford_hardi()
    data = img.get_data()
    S0 = data[..., 0].astype(np.float64)
    print('Filtering with aonlm, parameters (3,1,1)')
    f1 = np.array(aonlm(S0, 3, 1, 1))
    print('Filtering with aonlm, parameters (3,2,1)')
    f2 = np.array(aonlm(S0, 3, 2, 1))
    print('Mixing wavelet coefficients')
    filtered = mixingsubband(f1, f2)
    mid_coronal = S0.shape[1] // 2
    plt.figure()
    plt.subplot(1, 2, 1)
    plt.imshow(S0[:, mid_coronal, :])
    plt.title('Input')
    plt.subplot(1, 2, 2)
    plt.imshow(filtered[:, mid_coronal, :])
    plt.title('Output')
