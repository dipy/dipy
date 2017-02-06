#ssss
import numpy as np
import math
import scipy
import scipy.stats
from scipy.fftpack import fft, ifft
import scipy.interpolate
import nibabel as nib
import matplotlib.pyplot as plt
from dipy.io.gradients import read_bvals_bvecs
from dipy.segment.mask import median_otsu
import scipy.ndimage as ndimage
from scipy.ndimage import zoom
from scipy import interpolate
from dipy.core.ndindex import ndindex


def sharpen_image(data):
    r""" This function implement first step of field estimation, and compute the
    expected value E[u|v], then assign values to the new estimate image based
    on v in the old image according to this expected value, as discribed in [1]
    The input should be the log form of medical image dataset

    :math:
        E[\overset\Lambda {u}|\overset\Lambda {v}] = \int_{\infty}^{\infty}
        \overset \Lambda {u} p(\overset \Lambda {u}|\overset \Lambda {v})d\overset
        \Lambda {u}

    Parameters
    ----------
    data : ndarray
        3 dimensional medical data, has to be log form of the original input

    Returns
    -------
    sharpend_img : ndarray


    References
    ----------
    [1] https://www.nitrc.org/docman/view.php/6/880/sled.pdf

    """
    oringin_data = data

    sharpened_img = np.zeros(shape=(data.shape[0],
                                    data.shape[1],
                                    data.shape[2]),)

    num_of_bins = 200
    fwhm = 0.15

    g, h = np.histogram(data, bins=num_of_bins)

    histogramslop = (h[200] - h[0]) / num_of_bins

    hist = np.zeros(num_of_bins,)

    for index in ndindex(data.shape):
        k, j, i = index
        cidx = (data[k, j, i] - h[0]) / histogramslop
        idx = int(math.floor(cidx))
        offset = cidx - idx
        if offset == 0 and idx < num_of_bins - 1:
            hist[idx] = hist[idx] + 1.0
        elif idx < num_of_bins - 1:
            hist[idx] = hist[idx] + 1.0 - offset
            hist[idx+1] = hist[idx+1] + offset
    pi = 3.1415926

    exponent = int(math.ceil(np.log2(num_of_bins)) + 1)

    paddedhistogramsize = int(np.power(2, exponent))

    histogram_offset = int(0.5 * (paddedhistogramsize - num_of_bins))

    sample_v = np.zeros(paddedhistogramsize,)

    for n in np.array(range(num_of_bins)):
        sample_v[n + histogram_offset] = hist[n]

    vf = fft(sample_v)

    """
    Create the Gaussian filter
    """

    scaledfwhm = fwhm / histogramslop
    expfactor = 4.0 * np.log(2.0) / (scaledfwhm * scaledfwhm)
    scalefactor = 2.0 * math.sqrt(np.log(2.0)/pi)/scaledfwhm

    sample_f = np.zeros(paddedhistogramsize, dtype=complex)

    halfsize = int(0.5 * paddedhistogramsize)

    for i in range(halfsize):
        n = i + 1
        sample_f[n] = sample_f[paddedhistogramsize - n] = \
            scalefactor * np.exp(-(n * n) * expfactor)

    sample_f[0] = scalefactor

    if paddedhistogramsize % 2 == 0:
        sample_f[halfsize] = scalefactor * math.exp(-0.25 *
                                                    paddedhistogramsize *
                                                    paddedhistogramsize *
                                                    expfactor)

    ff = np.zeros(paddedhistogramsize,)

    ff = fft(sample_f)

    """
    Create the Wiener deconvolution filter
    """
    gf = np.zeros(paddedhistogramsize, dtype=complex)

    for n in np.array(range(paddedhistogramsize)):
        c = np.conjugate(ff[n])
        gf[n] = c / ((c * ff[n]).real + 0.01)

    uf = np.zeros(paddedhistogramsize, dtype=complex)

    for n in np.array(range(paddedhistogramsize)):
        uf[n] = vf[n] * gf[n].real

    uf = np.conjugate(uf)
    u = np.fft.ifft(uf)
    u = u * paddedhistogramsize

    for n in np.array(range(paddedhistogramsize)):
        u[n] = np.max(u[n].real, 0.0)

    """
    Compute mapping E(u|v)
    """
    numerator = np.zeros(paddedhistogramsize, dtype=complex)

    for n in np.array(range(paddedhistogramsize)):
        numerator[n] = \
            (h[0] + (n - histogram_offset) * histogramslop) * u[n].real

    f_numerator = fft(numerator)

    for n in np.array(range(paddedhistogramsize)):
        numerator[n] = ff[n] * f_numerator[n]

    numerator = ifft(numerator)
    numerator = numerator * paddedhistogramsize
    denominator = fft(u)

    for n in np.array(range(paddedhistogramsize)):
        denominator[n] = denominator[n] * ff[n]
    denominator = ifft(denominator)
    denominator = denominator * paddedhistogramsize

    expect = np.zeros(paddedhistogramsize,)

    for n in np.array(range(paddedhistogramsize)):
        if denominator[n].real != 0:
            expect[n] = numerator[n].real / denominator[n].real
        else:
            expect[n] = 0.0

    expect_extract = \
        expect[histogram_offset: histogram_offset + num_of_bins]

    print(expect_extract)

    sample_scale = num_of_bins / 20
    sample_scale = int(sample_scale)
    expect_interp = np.array(np.zeros(20))
    for i in np.array(range(20)):
        expect_interp[i] = \
            np.median(expect_extract[i * sample_scale:(i+1) * sample_scale])

    x = np.linspace(0, 1, 20)
    tck = interpolate.splrep(x, expect_interp)
    x2 = np.linspace(0, 1, 200)
    expect_interp = interpolate.splev(x2, tck)

    print(expect_interp)

    for index in ndindex(data.shape):
        i, j, k = index
        cidx = (oringin_data[i, j, k] - h[0]) / histogramslop
        idx = int(np.round(cidx))
        corrected_pixel = 0
        if idx < (expect_extract.size - 1):
            corrected_pixel = \
                expect_interp[idx] + (expect_interp[idx + 1] -
                                      expect_interp[idx]) * (cidx - idx)
        else:
            corrected_pixel = expect_interp[expect_extract.size - 1]

        sharpened_img[i, j, k] = corrected_pixel

    return sharpened_img


def GenerateData(t1):

#    data = np.squeeze(t1_slic)
    loginput = np.log(t1)

#    b0_mask, mask = median_otsu(t1_slic, 2, 1)
#    region_coordinate = np.where(mask == True)

#    region_rowmin = np.min(region_coordinate[0])
#    region_rowmax = np.max(region_coordinate[0])
#    region_colmin = np.min(region_coordinate[1])
#    region_colmax = np.max(region_coordinate[1])

#    t1_region = t1_slic[region_rowmin: region_rowmax, region_colmin: region_colmax]

    logUncorrectedImage = np.log(t1)

    print(t1.shape)

#    for i in np.array(range(8)):
#        print("iteration:")
#        print(i)
    logSharpenedImage = SharpenImage(logUncorrectedImage)

    #    return logSharpenedImage

    #    plt.imshow(logSharpenedImage[:, :, 10])

    residualbiasfield = logUncorrectedImage - logSharpenedImage
    newlogbiasfield = residualbiasfield
#    for index in ndindex(newlogbiasfield.shape):
#        i, j, k = index
#        if(newlogbiasfield[i, j, k] < 0):
#            newlogbiasfield[i, j, k] = 0
#    for j in np.array(range(5)):
#        newlogbiasfield = ndimage.gaussian_filter(newlogbiasfield, sigma=1)

    logUncorrectedImage = loginput - newlogbiasfield

#    logSharpenedImage2 = SharpenImage(logUncorrectedImage)


#    f_real = np.zeros(shape=(t1_slic.shape[0], t1_slic.shape[1]))
#    u_real = np.zeros(shape=(t1_slic.shape[0], t1_slic.shape[1]))

    return logSharpenedImage, logUncorrectedImage, newlogbiasfield


def Setfieldpoints(inputimg, xscale, yscale, zscale):
    xsize = inputimg.shape[0]
    ysize = inputimg.shape[1]
    zsize = inputimg.shape[2]
    output = np.zeros(shape=(xscale, yscale, zscale))
    xsubsize = np.floor(xsize / xscale)
    ysubsize = np.floor(ysize / yscale)
    zsubsize = np.floor(zsize / zscale)
    xsubsize = xsubsize.astype(int)
    ysubsize = ysubsize.astype(int)
    zsubsize = zsubsize.astype(int)
    for i in np.array(range(zscale)):
        for j in np.array(range(xscale)):
            for k in np.array(range(yscale)):
                output[j, k, i] = np.mean(inputimg[j*xsubsize:(j+1)*xsubsize,k*ysubsize:(k+1)*ysubsize,i*zsubsize:(i+1)*zsubsize])
    return output


def Smoothfield_3D(inputimg, xscale, yscale, zscale):
    output = zoom(inputimg, (xscale, yscale, zscale))
    return output


def Smoothfield_2D(inputimg, xrescale, yrescale):
    xsize = inputimg.shape[0]
    ysize = inputimg.shape[1]
    zsize = inputimg.shape[2]
    output = np.zeros(shape=(xrescale, yrescale, zsize))
    xv, yv = np.mgrid[-1:1:4j, -1:1:4j]
    xnew, ynew = np.mgrid[-1:1:160j, -1:1:239j]
    for i in np.array(range(zsize)):
        tck = interpolate.bisplrep(xv, yv, inputimg[:, :, i], s=0)
        znew = interpolate.bisplev(xnew[:, 0], ynew[0, :], tck)
        output[:, :, i] = znew
    return output


def self_test(t1, logfield):
    loginput = np.log(t1)
    logsub = loginput - logfield
    sharpenedimg = SharpenImage(logsub)
    newlogfield = loginput - sharpenedimg
    logfield_subsample = Setfieldpoints(newlogfield, 5, 5, 5)
    newlogfield = zoom(logfield_subsample, (t1.shape[0]/5, t1.shape[1]/5, t1.shape[2]/5))
    error = np.abs(logfield - newlogfield)
    plt.figure()
    plt.imshow(newlogfield[:, :, 10], "gray")

    return newlogfield, error



dname = "/Users/tiwanyan/ANTs/Images"
t1_input = "/Raw/Q_0001_T1.nii.gz"

ft1 = dname + t1_input

t1 = nib.load(ft1).get_data()
#t1 = t1[:,:,0:155]
#unsharpenedImage = nib.load(ft1).get_data()
#data = np.log(unsharpenedImage.reshape((unsharpenedImage.shape[0] * unsharpenedImage.shape[1] * unsharpenedImage.shape[2], 1)))

#g, h = np.histogram(data, bins=200)

#t1_slic = t1[:, :, 10]

logSharpenedImage, logUncorrectedImage, newlogbiasfield = GenerateData(t1)
