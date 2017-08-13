from __future__ import division, print_function

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
from Goal0_1 import show_volume
from sklearn import mixture
from dipy.segment.mask import median_otsu
from dipy.segment.threshold import otsu
from scipy.misc import imresize
import scipy.ndimage as ndimage


class n3_correction(object):
    """Define a class for N3 bias correciton
    """
    def __init__(self, data, kernel="triangle", init_u="init_v"):

        #size = np.round((np.array(data.shape)/ratio))
        #size = size.astype(int)
        #self.resize_data = Setfieldpoints(data, size[0], size[1], size[2], "median")
        #self.resize_data = imresize(data,size)
        self.data = data
        self.sharpened_data = np.zeros((self.data.shape))
        self.field = np.zeros((self.data.shape))
        self.estimate_img = np.zeros((self.data.shape))
        self.estimate_img[:] = self.data
        self.kernel = kernel
        self.data_shape = data.shape
        self.num_of_bins = 200
        self.h_scale = 2
        self.exp_stat = np.zeros((self.num_of_bins))
        self.time_delay = 100
        self.difference = 1
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

    def wiener_decon_filter(self):
        """This function perform a wiener deconvolution filter
        """
        paddel_sample_f = np.zeros((len(self.sample_f) + \
                                    len(self.kernel_dhist) - 1),)
        paddel_sample_v = np.zeros((len(self.sample_f) + \
                                    len(self.kernel_dhist) - 1),)
        paddel_sample_f[:self.num_of_bins] = self.sample_f
        paddel_sample_v[:self.num_of_bins] = self.kernel_dhist
        ff = fft(self.sample_f)
        #vf = fft(paddel_sample_v)
        vf = fft(self.kernel_dhist)
        gf = np.zeros(len(self.sample_f), dtype=complex)
        for n in np.array(range(self.num_of_bins)):
            c = np.conjugate(ff[n])
            gf[n] = c / ((ff[n] * ff[n]).real + 0.1)

        #uf = np.zeros(self.num_of_bins, dtype=complex)

        uf = vf * gf

        uf = np.conjugate(uf)
        u = ifft(uf)
        u_real = u.real
        for n in np.array(range(len(u_real))):
            u_real[n] = np.max((u_real[n], 0.0))

        u_real = self.kernel_density_estimate(u_real)

        plt.figure()
        plt.title("estimate_u")
        plt.plot(np.array(range(len(self.sample_f))), u_real)

        #estimate_u = u[self.time_delay:\
        #               self.time_delay + self.num_of_bins]
        return u_real

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
            numerator[n] = self.bins_u[n+1] * paddel_estimate_u[n]

        f_numerator = ff * fft(numerator)
        inumerator = ifft(f_numerator)

        denominator = fft(paddel_estimate_u)
        f_denominator = ff * denominator
        idenominator = ifft(f_denominator)

        paddel_expect = inumerator.real / idenominator.real
        '''
        for i in np.array(range(200)):
            if i == 199:
                i = 198
            k = np.where((self.data>self.bins[i])&(self.data<self.bins[i+1]))
            n = len(k[0])
            self.exp_stat[i] = np.median(self.data[k[0],k[1],k[2]]-self.field[k[0],k[1],k[2]])
        '''
        plt.figure()
        plt.title("exp_stat")
        plt.plot(self.bins[0:200],self.exp_stat)

#        expect_smooth = paddel_expect[int(round((self.num_of_bins/2))):\
#                               int(round((self.num_of_bins/2))) + \
#                                          self.num_of_bins]

        expect = paddel_expect[self.time_delay:\
                               self.time_delay + self.num_of_bins]

        expect_p = expect[0:self.num_of_bins:4]
        #expect_obj = cubic_bspline(expect_p, self.num_of_bins + 15,
        #                           spacing="uniform")
        expect_obj = cubic_bspline(expect_p, self.num_of_bins + 40,
                                   spacing="uniform")
        expect_smooth = expect_obj.cubicbspline_2d()[19:self.num_of_bins+19]
        plt.figure()
        plt.title("expected value")
        plt.plot(self.bins[0:200], expect)
        plt.figure()
        plt.plot(np.array(range(len(paddel_expect))), paddel_expect)

#        kernel_estimate_expect = self.kernel_density_estimate(expect)

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

        if len(self.data.shape) == 2:
            for index in ndindex(self.data.shape):
                i, j = index
                idx = int(np.floor((self.data[i, j] - self.bins[0]) / \
                                    self.bin_lengh))
                if idx > 200 or idx == 200:
                    idx = 199
                self.sharpened_data[i, j] = expect_smooth[idx]

        return expect_smooth

    def optimization_converge(self):
        expect_smooth = self.update_stratergy()
        field = self.data - self.sharpened_data
        max_f = field.max()
        self.field = field
        if len(self.data.shape) == 2:
            field_extract = Setfieldpoints_2d(field, 4, 6, "median")
            #np.exp(field[0:160:10, 0:239:10, 0:200:10])
            self.field = Smoothfield_2D(field_extract,
                                        self.data_shape[0]/4,
                                        self.data_shape[1]/6)
            #field_extract_obj = cubic_bspline(field_extract,
            #                                  self.data_shape)
            #self.field = field_extract_obj.cubicbspline_3d()
        if len(self.data.shape) == 3:

            self.field[np.exp(self.field)<otsu(np.exp(self.field))]=0
            for i in range(4):
                shape_x = self.field.shape[0]
                shape_y = self.field.shape[1]
                shape_z = self.field.shape[2]
                self.field = ndimage.gaussian_filter(self.field, sigma=(2, 2, 2), order=0)
                self.field = self.field[0:shape_x:2,0:shape_y:2,0:shape_z:2]
            self.field = self.field * (max_f/self.field.max()) * 1/3
            self.field = Smoothfield_3D(self.field,
                                        self.data_shape[0]/self.field.shape[0],
                                        self.data_shape[1]/self.field.shape[1],
                                        self.data_shape[2]/self.field.shape[2])
            '''
            field_extract = Setfieldpoints(field, 20, 30, 25, "subsample")
            #np.exp(field[0:160:10, 0:239:10, 0:200:10])
            self.field = Smoothfield_3D(field_extract,
                                        self.data_shape[0]/20,
                                        self.data_shape[1]/30,
                                        self.data_shape[2]/25)
            field_2 = Setfieldpoints(self.field, 4, 5, 6, "median")
            self.field = Smoothfield_3D(field_2,
                                        self.data_shape[0]/4,
                                        self.data_shape[1]/5,
                                        self.data_shape[2]/6)
            '''
        hist, h = np.histogram(self.field,
                               bins=self.num_of_bins)

        f_median = np.median(self.field)
        self.estimate_img = self.data - self.field
        '''
        if len(self.data.shape) == 3:
            for index in ndindex(self.data.shape):
                i, j, k = index
                if self.data[i, j, k] > self.threshold:
                    self.estimate_img[i, j, k] = self.data[i, j, k]-self.field[i,j,k]
                if self.data[i, j, k] < self.threshold or \
                   self.data[i, j, k] == self.threshold:
                    self.estimate_img[i, j, k] = self.data[i, j, k]
        '''
        #threshold = otsu(self.estimate_img)
        threshold_u = otsu(self.estimate_img)
        hist_u, h_u = np.histogram(self.estimate_img,
                                   bins=self.num_of_bins,
                                   range=(threshold_u,
                                          self.estimate_img.max()))
        new_f = self.kernel_density_estimate(hist)
        self.estimate_u = self.kernel_density_estimate(hist_u)
        self.bins_u[:] = h_u[:]
        plt.plot(np.array(range(200)), self.sample_f)
        plt.figure()
        plt.plot(np.array(h[1:201]), new_f)
#        plt.figure()
#        plt.plot(np.array(h_u[1:201]), hist_u)
        print(np.sum(np.abs(new_f-self.sample_f)))
        self.sample_f[:] = new_f[:]
        #self.estimate_u = self.wiener_decon_filter()
        #plt.figure()
        #plt.plot(np.array(h_u[1:201]), self.estimate_u)
        new_bin = np.where(new_f == new_f.max())[0][0]
        #new_bin = self.finding_mid_point(new_f)
        print(new_bin)
        print("self_delay")
        print(self.time_delay)
        histu_median = self.finding_mid_point(self.estimate_u)
        print(self.hist_median-histu_median)
        difference = self.hist_median - histu_median
        #ratio = difference / self.difference
        self.difference = difference
        #print(ratio)
        #self.time_delay = self.time_delay + int(round(0.5*(new_bin - self.time_delay)))+ int(round(0.5*(self.hist_median - histu_median)))
        self.time_delay = new_bin
        if self.time_delay < 0:
            self.time_delay = 0
        if self.time_delay > 200:
            self.time_delay = 200
#        self.time_delay = new_bin + (1/2) * (self.time_delay - new_bin)
#        self.time_delay = int(round(self.time_delay + (1/10) * (new_bin - self.time_delay)))
#        self.time_delay = np.where(hist == hist.max())[0][0]
#        self.time_delay = np.where(abs(h-np.median(self.field)<0.01))[0][0]
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

    def finding_mid_point(self, hist):
        total = np.sum(hist)
        num = 0
        for i in range(len(hist)):
            if(num > total/2):
                return i
            else:
                num = num + hist[i]


def Smoothfield_2D(inputimg, xscale, yscale):
    output = zoom(inputimg, (xscale, yscale))
    return output


def Smoothfield_3D(inputimg, xscale, yscale, zscale):
    output = zoom(inputimg, (xscale, yscale, zscale))
    return output


def Setfieldpoints_2d(inputimg, xscale, yscale, method="mean"):
    xsize = inputimg.shape[0]
    ysize = inputimg.shape[1]
    output = np.zeros(shape=(xscale, yscale))
    xsubsize = np.floor(xsize / xscale)
    ysubsize = np.floor(ysize / yscale)
    xsubsize = xsubsize.astype(int)
    ysubsize = ysubsize.astype(int)
    for j in np.array(range(xscale)):
        for k in np.array(range(yscale)):
            if method == "mean":
                output[j, k] = \
                    np.mean(inputimg[j*xsubsize:(j+1)*xsubsize,
                                     k*ysubsize:(k+1)*ysubsize])
            if method == "median":
                output[j, k] = \
                    np.median(inputimg[j*xsubsize:(j+1)*xsubsize,
                                       k*ysubsize:(k+1)*ysubsize])
    return output
'''
def Setfieldpoints(inputimg, xscale, yscale, zscale, method="median"):
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
                output[j, k, i] = np.median(inputimg[j*xsubsize:(j+1)*xsubsize,
                                                   k*ysubsize:(k+1)*ysubsize,
                                                   i*zsubsize:(i+1)*zsubsize])
    return output
'''
def Setfieldpoints(inputimg, xscale, yscale, zscale, method="mean"):
    xsize = inputimg.shape[0]
    ysize = inputimg.shape[1]
    zsize = inputimg.shape[2]
    output = np.zeros(shape=(xscale, yscale, zscale))
    xsubsize = np.floor(xsize / xscale)
    xresi = np.floor((xsize % xscale) / 2)
    ysubsize = np.floor(ysize / yscale)
    yresi = np.floor((ysize % yscale) / 2)
    zsubsize = np.floor(zsize / zscale)
    zresi = np.floor((zsize % zscale) / 2)
    xsubsize = xsubsize.astype(int)
    ysubsize = ysubsize.astype(int)
    zsubsize = zsubsize.astype(int)
    for i in np.array(range(zscale)):
        for j in np.array(range(xscale)):
            for k in np.array(range(yscale)):
                x_left = xresi+j*xsubsize-xsubsize
                x_right = xresi+j*xsubsize+xsubsize
                y_left = yresi+k*ysubsize-ysubsize
                y_right = yresi+k*ysubsize+ysubsize
                z_left = zresi+i*zsubsize-zsubsize
                z_right = zresi+i*zsubsize+zsubsize
                if x_left < 0:
                    x_left = 0
                if x_right > xsize:
                    x_right = xsize
                if y_left < 0:
                    y_left = 0
                if y_right > ysize:
                    y_right = ysize
                if z_left < 0:
                    z_left = 0
                if z_right > zsize:
                    z_right = zsize
                if method == "mean":
                    output[j, k, i] = \
                        np.mean(inputimg[x_left:x_right,
                                         y_left:y_right,
                                         z_left:z_right])
                if method == "subsample":
                    output[j, k, i] = \
                                inputimg[xresi+j*xsubsize,
                                         yresi+k*ysubsize,
                                         zresi+i*zsubsize]
                if method == "median":
                    output[j, k, i] = \
                        np.median(inputimg[x_left:x_right,
                                         y_left:y_right,
                                         z_left:z_right])
    return output

dname = "/Users/tiwanyan/ANTs/Images/Raw/"
#dname = "/home/elef/Dropbox/Tingyi/Images/Raw/"
#t1_input = "/Raw/Q_0001_T1.nii.gz"
#ft1 = dname + "Q_0001_T1_N3.nii.gz"
#ft1 = dname + "Q_0001_T1.nii.gz"
ft1 = dname + "Q_0005_T1.nii.gz"
#ft1 = dname + "sub-A00039461_t1_bias.nii.gz"
#ft1_unb = dname + "sub-A00039461_t1_fast_nobias.nii.gz"
ft2 = dname + "output.nii.gz"
f_bias = dname + "outbias.nii.gz"

fmask = dname + "Q_0001_FS2Native_T1_Mask_Filled.nii.gz"

#ft1 = dname + "sub-A00039461_t1_bias.nii.gz"
t1 = nib.load(ft1).get_data()
t3 = nib.load(ft2).get_data()
bias = nib.load(f_bias).get_data()
#mask = nib.load(fmask).get_data()
#loc = np.array(np.where(mask == 1))
loc = np.array(np.where(t1 > 100))
t2 = t1[loc[0].min():loc[0].max(),
        loc[1].min():loc.max(),
        loc[2].min():loc[2].max()]

t2 = t2 + 10
t1 = t1 + 10
N3_correct = n3_correction(np.log(t1))
N3_correct.optimization_converge()


figure()
imshow(np.exp(N3_correct.data[:, :, 10]), cmap="gray")
figure()
imshow(np.exp(N3_correct.estimate_img[:, :, 10]), cmap="gray")
figure()
imshow(np.exp(N3_correct.field[:, :, 10]), cmap="gray")
