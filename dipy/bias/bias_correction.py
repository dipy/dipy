"""
This code implements a purly statistical based Non-parametric learning on bias
field correction using a N3 stratergy

Tingyi Wanyan
"""

from __future__ import division, print_function
import numpy as np
import math
import scipy
import scipy.stats
from scipy.fftpack import fft, ifft
import scipy.interpolate
import nibabel as nib
import matplotlib.pyplot as plt
import scipy.ndimage as ndimage
from scipy.ndimage import zoom
from dipy.core.ndindex import ndindex
from bspline_fit import cubic_bspline
from dipy.segment.threshold import otsu
from Goal0_1 import show_volume


class NP_correction(object):
    """Define a class for Non parametric bias correciton
    """
    def __init__(self, data, num_of_bins=200, h_scale=2, kernel="triangle",
                 init_u="init_v"):
        if data.min() == 0:
            data[data == 0] = 10
            self.data = np.log(data + 10)
        else:
            self.data = np.log(data)
        self.sharpened_data = np.zeros((self.data.shape))
        self.field = np.zeros((self.data.shape))
        self.estimate_img = np.zeros((self.data.shape))
        self.estimate_img[:] = self.data
        self.kernel = kernel
        self.error = 0
        self.median = 0
        self.data_shape = data.shape
        self.num_of_bins = 200
        self.h_scale = h_scale
        self.exp_stat = np.zeros((self.num_of_bins))
        self.time_delay = int(round(self.num_of_bins / 2))
        self.threshold = otsu(self.data)
        self.bins_u = np.zeros((self.num_of_bins+1))
        self.hist, self.bins = np.histogram(self.data,
                                            bins=self.num_of_bins,
                                            range=(self.threshold,
                                                   self.data.max()))
        self.bin_lengh = self.bins[1] - self.bins[0]
        self.bins_u[:] = self.bins[:]
        self.h = self.h_scale * self.bin_lengh
        self.sample_f = self.gaussian_init(50, self.num_of_bins)
        #self.sample_f = self.uniform_init(4, self.num_of_bins)
        self.kernel_dhist = self.kernel_density_estimate(self.hist)
        self.hist_median = self.finding_mid_point(self.kernel_dhist)
        if init_u == "deconv":
            self.estimate_u = self.wiener_decon_filter()
        if init_u == "init_v":
            self.estimate_u = np.zeros((self.num_of_bins,))
            self.estimate_u = self.kernel_dhist[:]

    def gaussian_init(self, phi, histogramsize):
        """This function initialize a normal distribution field
        """
        pi = np.pi
        expfactor = 1 / (2.0 * phi * phi)
        scalefactor = 1 / math.sqrt(2 * pi * phi * phi)
        sample_f = np.zeros(histogramsize)

        halfsize = int(0.5 * histogramsize)

        for n in range(halfsize):
            sample_f[n] = sample_f[histogramsize - n - 1] = \
                scalefactor * \
                np.exp(-((n-halfsize) * (n-halfsize)) * expfactor)

        return sample_f

    def uniform_init(self, size, histogramsize):
        """This function initialize a uniform distribution field
        """
        sample_f = np.zeros(histogramsize)
        halfsize = int(0.5 * histogramsize)
        p = 1/size

        for n in range(size):
            sample_f[halfsize - size/2 + n] = p

        return sample_f

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
        ff = fft(paddel_sample_f)
        for n in range(self.num_of_bins):
            numerator[n] = self.bins_u[n+1] * paddel_estimate_u[n]

        f_numerator = ff * fft(numerator)
        inumerator = ifft(f_numerator)

        denominator = fft(paddel_estimate_u)
        f_denominator = ff * denominator
        idenominator = ifft(f_denominator)

        paddel_expect = inumerator.real / idenominator.real

        expect = paddel_expect[self.time_delay:\
                               self.time_delay + self.num_of_bins]

        #plt.figure()
        #plt.title("expected value")
        #plt.plot(self.bins[0:200], expect)
        #plt.figure()
        #plt.plot(np.array(range(len(paddel_expect))), paddel_expect)

        if len(self.data.shape) == 3:
            for index in ndindex(self.data.shape):
                i, j, k = index
                idx = int(np.floor((self.data[i, j, k] - self.bins[0]) / \
                                    self.bin_lengh))
                if idx > 200 or idx == 200:
                    idx = 199
                if self.data[i, j, k] > self.threshold:
                    self.sharpened_data[i, j, k] = expect[idx]
                if self.data[i, j, k] < self.threshold or \
                   self.data[i, j, k] == self.threshold:
                    self.sharpened_data[i, j, k] = self.data[i, j, k]

        return expect

    def field_smoothing(self):
        expect = self.update_stratergy()
        field = self.data - self.sharpened_data
        max_f = field.max()
        self.field = field
        """Here perform the subsampling stratergy and field smoothing
        """
        if len(self.data.shape) == 3:
            self.field[np.exp(self.field) < otsu(np.exp(self.field))] = 0
            #for i in range(4):
                #elf.field = ndimage.gaussian_filter(self.field,
                #                                     sigma=(2, 2, 2),
                #                                     order=0)
                #self.field = self.max_pooling(self.field)

            for i in range(4):
                shape_x = self.field.shape[0]
                shape_y = self.field.shape[1]
                shape_z = self.field.shape[2]
                self.field = ndimage.gaussian_filter(self.field,
                                                     sigma=(2, 2, 2),
                                                     order=0)
                self.field = self.field[0:shape_x:2, 0:shape_y:2, 0:shape_z:2]

            self.field = self.field * (max_f/self.field.max()) * 1/3
            self.field = self.Smoothfield_3D(self.field,
                                             self.data_shape[0]/self.field.shape[0],
                                             self.data_shape[1]/self.field.shape[1],
                                             self.data_shape[2]/self.field.shape[2])
        hist, h = np.histogram(self.field,
                               bins=self.num_of_bins)

        self.estimate_img = self.data - self.field
        threshold_u = otsu(self.estimate_img)
        hist_u, h_u = np.histogram(self.estimate_img,
                                   bins=self.num_of_bins,
                                   range=(threshold_u,
                                          self.estimate_img.max()))
        new_f = self.kernel_density_estimate(hist)
        self.estimate_u = self.kernel_density_estimate(hist_u)
        self.bins_u[:] = h_u[:]
        #plt.plot(np.array(range(200)), self.sample_f)
        #plt.figure()
        #plt.plot(np.array(h[1:201]), new_f)
        self.error = np.sum(np.abs(new_f-self.sample_f))
        print(self.error)
        self.sample_f[:] = new_f[:]
        new_bin = np.where(new_f == new_f.max())[0][0]
        #print(new_bin)
        #print("self_delay")
        #print(self.time_delay)
        #histu_median = self.finding_mid_point(self.estimate_u)
        #print(self.hist_median-histu_median)
        #self.difference = difference
        self.time_delay = new_bin
        if self.time_delay < 0:
            self.time_delay = 0
        if self.time_delay > 200:
            self.time_delay = 200

    def optimization_converge(self):
        print("error is")
        self.field_smoothing()
        while(self.error > 5):
            self.field_smoothing()

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

    def finding_mid_point(self, hist):
        total = np.sum(hist)
        num = 0
        for i in range(len(hist)):
            if(num > total/2):
                return i
            else:
                num = num + hist[i]

    def Smoothfield_2D(self, inputimg, xscale, yscale):
        output = zoom(inputimg, (xscale, yscale))
        return output

    def Smoothfield_3D(self, inputimg, xscale, yscale, zscale):
        output = zoom(inputimg, (xscale, yscale, zscale))
        return output

    def max_pooling(self, inputimg):
        shape_x = inputimg.shape[0]
        shape_y = inputimg.shape[1]
        shape_z = inputimg.shape[2]
        xscale = int(np.floor(shape_x/2))
        yscale = int(np.floor(shape_y/2))
        zscale = int(np.floor(shape_z/2))
        output = np.zeros((xscale, yscale, zscale))
        threshold = otsu(inputimg)
        median = np.median(inputimg[inputimg > threshold])
        print(median)
        for i in range(xscale):
            for j in range(yscale):
                for k in range(zscale):
                    local_max = inputimg[2*i:2*i+1,
                                         2*j:2*j+1,
                                         2*k:2*k+1].max()
                    output[i, j, k] = local_max

        return output

"""
Example for using class, the estimage data is Correct.estimate_img,
The calculated field is Correct.field
"""
dname = "/Users/tiwanyan/ANTs/Images/Raw/"
ft1 = dname + "Q_0011_T1.nii.gz"
t1 = nib.load(ft1).get_data()
Correct = NP_correction(t1)
Correct.optimization_converge()
#Correct.field_smoothing()
#show_volume(np.exp(Correct.estimate_img), t1, 0.5)
