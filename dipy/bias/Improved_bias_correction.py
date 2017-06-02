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
from scipy import signal
from sklearn.neighbors import KernelDensity
from bspline_fit import cubic_bspline
import matplotlib.pyplot as plt
from scipy.ndimage import zoom


class n3_correction(object):
    """Define a class for N3 bias correciton
    """
    def __init__(self, data, kernel="triangle", init_u="deconv"):

        self.data = data
        self.sharpened_data = np.zeros((self.data.shape))
        self.field = np.zeros((self.data.shape))
        self.estimate_img = np.zeros((self.data.shape))
        self.kernel = kernel
        self.data_shape = data.shape
        self.num_of_bins = 200
        self.h_scale = 2
        self.hist, self.bins = np.histogram(data, bins=self.num_of_bins)
        self.bin_lengh = self.bins[1] - self.bins[0]
        self.h = self.h_scale * self.bin_lengh
        self.sample_f = self.gaussian_init(5, self.num_of_bins)
        self.kernel_dhist = self.kernel_density_estimate(self.hist)
        if init_u == "deconv":
            self.estimate_u = self.wiener_decon_filter()
        if init_u == "init_v":
            self.estimate_u = np.zeros((self.num_of_bins,))
            self.estimate_u = self.kernel_dhist[:]

    def gaussian_init(self, phi, histogramsize):
        """This function initialize a normal distribution field
        """
        pi = 3.1415926
        expfactor = 1 / (2.0 * phi * phi)
        scalefactor = 1 / math.sqrt(2 * pi * phi * phi)
        sample_f = np.zeros(histogramsize, dtype=complex)

        halfsize = int(0.5 * histogramsize)

        for n in range(halfsize):
            sample_f[n] = sample_f[histogramsize - n - 1] = \
                scalefactor * \
                np.exp(-((n-halfsize) * (n-halfsize)) * expfactor)

        return sample_f

    def wiener_decon_filter(self):
        """This function perform a wiener deconvolution filter
        """
        paddel_sample_f = np.zeros((len(self.sample_f) + \
                                    len(self.kernel_dhist) - 1),)
        paddel_sample_v = np.zeros((len(self.sample_f) + \
                                    len(self.kernel_dhist) - 1),)
        paddel_sample_f[:self.num_of_bins] = self.sample_f
        paddel_sample_v[:self.num_of_bins] = self.kernel_dhist
        ff = fft(paddel_sample_f)
        vf = fft(paddel_sample_v)
        gf = np.zeros(len(paddel_sample_f), dtype=complex)
        for n in np.array(range(self.num_of_bins)):
            c = np.conjugate(ff[n])
            gf[n] = c / ((ff[n] * ff[n]).real + 0.1)

        uf = np.zeros(self.num_of_bins, dtype=complex)

        uf = vf * gf

        uf = np.conjugate(uf)
        u = ifft(uf)

        for n in np.array(range(len(u))):
            u[n] = np.max(u[n].real, 0.0)

        estimate_u = u[int(round((self.num_of_bins/2))):\
                       int(round((self.num_of_bins/2))) + self.num_of_bins]
        return estimate_u

    def update_stratergy(self):
        """This function perform the update rule
        """
        paddel_sample_f = np.zeros((len(self.sample_f) + \
                                    len(self.estimate_u) - 1),)
        paddel_sample_f[:self.num_of_bins] = self.sample_f
        paddel_estimate_u = np.zeros((len(self.sample_f) + \
                                      len(self.estimate_u) - 1),)
        paddel_estimate_u[:self.num_of_bins] = self.estimate_u
        numerator = np.zeros(len(paddel_sample_f),)
#        expect = np.zeros((self.num_of_bins,))
        ff = fft(paddel_sample_f)
        for n in range(self.num_of_bins):
            numerator[n] = self.bins[n] * paddel_estimate_u[n]

        f_numerator = ff * fft(numerator)
        inumerator = ifft(f_numerator)

        denominator = fft(paddel_estimate_u)
        f_denominator = ff * denominator
        idenominator = ifft(f_denominator)

        paddel_expect = inumerator.real / idenominator.real

        expect = paddel_expect[int(round((self.num_of_bins/2))):\
                               int(round((self.num_of_bins/2))) + \
                                          self.num_of_bins]

        expect_p = expect[0:self.num_of_bins:4]
        expect_obj = cubic_bspline(expect_p, self.num_of_bins + 15,
                                   spacing="uniform")
        expect_smooth = expect_obj.cubicbspline_2d()[7:self.num_of_bins+7]
#        kernel_estimate_expect = self.kernel_density_estimate(expect)

        for index in ndindex(self.data.shape):
            i, j, k = index
            idx = int(np.floor((self.data[i, j, k] - self.bins[0]) / \
                                self.bin_lengh))
            if idx > 200 or idx == 200:
                idx = 199
            self.sharpened_data[i, j, k] = expect_smooth[idx]

        return expect_smooth

    def optimization_converge(self):
        expect_smooth = self.update_stratergy()
        field = self.data - self.sharpened_data
        field_extract = Setfieldpoints(field, 4, 6, 5)
        #np.exp(field[0:160:10, 0:239:10, 0:200:10])
        self.field = Smoothfield_3D(field_extract, 160/4, 239/6, 200/5)
        hist, h = np.histogram(self.field,
                               bins=self.num_of_bins)
        self.estimate_img = self.data - self.field
        hist_u, h_u = np.histogram(self.estimate_img, bins=self.num_of_bins)
        new_f = self.kernel_density_estimate(hist)
        plt.plot(np.array(range(200)), self.sample_f)
        plt.figure()
        plt.plot(np.array(range(200)), new_f)
        plt.figure()
        plt.plot(np.array(range(200)), hist_u)
        print(np.sum(np.abs(new_f-self.sample_f)))
        self.sample_f = new_f
        self.estimate_u = self.kernel_density_estimate(hist_u)
        #self.estimate_u = self.wiener_decon_filter()

    def kernel_density_estimate(self, dist):
        """Doing a kernel density estimation
        """
        density = np.zeros((self.num_of_bins,))
        if self.kernel == "triangle":
            for i in range(self.num_of_bins):
                scale_factor = 0
                left = i - self.h_scale
                right = i + self.h_scale
                if(left < 0):
                    left = i
                if(right > self.num_of_bins - 1):
                    right = self.num_of_bins - 1
                for j in range(self.h_scale - 1):
                    left_bin = i - j - 1
                    right_bin = i + j + 1
                    if(left_bin < 0):
                        left_bin_num = 0
                    else:
                        left_bin_num = dist[left_bin]
                    if(right_bin > self.num_of_bins - 1):
                        right_bin_num = 0
                    else:
                        right_bin_num = dist[right_bin]
                    scale_factor += (j + 1) * self.bin_lengh * \
                                    (left_bin_num + right_bin_num)
                density[i] += (np.sum(dist[left:right]) / self.h - \
                                     (1 / self.h * self.h) * scale_factor) / \
                                      np.sum(dist)
        return density


def Smoothfield_3D(inputimg, xscale, yscale, zscale):
    output = zoom(inputimg, (xscale, yscale, zscale))
    return output


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
                output[j, k, i] = np.mean(inputimg[j*xsubsize:(j+1)*xsubsize,
                                                   k*ysubsize:(k+1)*ysubsize,
                                                   i*zsubsize:(i+1)*zsubsize])
    return output

#dname = "/Users/tiwanyan/ANTs/Images"
dname = "/home/elef/Dropbox/Tingyi/Images/Raw/"
#t1_input = "/Raw/Q_0001_T1.nii.gz"
ft1 = dname + "Q_0001_T1.nii.gz"

t1 = nib.load(ft1).get_data()

N3_correct = n3_correction(np.log(t1))
N3_correct.optimization_converge()


