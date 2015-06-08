
import numpy as np
from dipy.segment.mask import multi_median
from dipy.segment.threshold import otsu

def otsu_param(input_volume, median_radius=4, numpass=4):

    input_filtered = multi_median(input_volume, median_radius, numpass)
    thresh = otsu(input_filtered)
    tissue1 = input_filtered > thresh
    tissue2 = input_filtered < thresh
    
    mu_tissue1 = np.mean(tissue1)
    sig_tissue1 = np.std(tissue1)
    mu_tissue2 = np.mean(tissue2)
    sig_tissue2 = np.std(tissue2)
    
    return mu_tissue1, sig_tissue1, mu_tissue2, sig_tissue2

