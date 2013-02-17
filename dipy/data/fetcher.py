import os
import urllib2
from os.path import join as pjoin
import numpy as np
import nibabel as nib
from dipy.core.gradients import gradient_table
from dipy.io.gradients import read_bvals_bvecs


def fetch_stanford_hardi():
    """ Download a HARDI dataset with 160 gradient directions
    """
    dipy_home = pjoin(os.path.expanduser('~'), '.dipy')
    url = 'https://stacks.stanford.edu/file/druid:yx282xq2090/'
    uraw = url + 'dwi.nii.gz'
    ubval = url + 'dwi.bvals'
    ubvec = url + 'dwi.bvecs'
    folder = pjoin(dipy_home, 'stanford_hardi')

    if not os.path.exists(folder):
        print('Creating new directory %s' % folder)
        os.makedirs(folder)
        print('Downloading raw HARDI data (87MB)...')
        opener = urllib2.urlopen(uraw)
        open(pjoin(folder, 'HARDI150.nii.gz'), 'wb').write(opener.read())

        opener = urllib2.urlopen(ubval)
        open(pjoin(folder, 'HARDI150.bval'), 'w').write(opener.read())

        opener = urllib2.urlopen(ubvec)
        open(pjoin(folder, 'HARDI150.bvec'), 'w').write(opener.read())

        print('Done.')
        print('Files copied in folder %s' % folder)
    else:
        print('Dataset already in place. If you want to fetch again please first remove folder %s ' % folder)


def read_stanford_hardi():
    """ Load Stanford HARDI dataset

    Returns
    -------
    img : obj,
        Nifti1Image
    gtab : obj,
        GradientTable
    """
    dipy_home = pjoin(os.path.expanduser('~'), '.dipy')
    folder = pjoin(dipy_home, 'stanford_hardi')
    fraw = pjoin(folder, 'HARDI150.nii.gz')
    fbval = pjoin(folder, 'HARDI150.bval')
    fbvec = pjoin(folder, 'HARDI150.bvec')

    bvals, bvecs = read_bvals_bvecs(fbval, fbvec)

    gtab = gradient_table(bvals, bvecs)
    img = nib.load(fraw)
    return img, gtab


def fetch_taiwan_ntu_dsi():
    """ Download a DSI dataset with 203 gradient directions
    """
    dipy_home = pjoin(os.path.expanduser('~'), '.dipy')
    uraw = 'http://dl.dropbox.com/u/2481924/taiwan_ntu_dsi.nii.gz'
    ubval = 'http://dl.dropbox.com/u/2481924/tawian_ntu_dsi.bval'
    ubvec = 'http://dl.dropbox.com/u/2481924/taiwan_ntu_dsi.bvec'
    ureadme = 'http://dl.dropbox.com/u/2481924/license_taiwan_ntu_dsi.txt'
    folder = pjoin(dipy_home, 'taiwan_ntu_dsi')
    if not os.path.exists(folder):
        print('Creating new directory %s' % folder)
        os.makedirs(folder)

        print('Downloading raw DSI data (91MB)...')

        opener = urllib2.urlopen(uraw)
        open(pjoin(folder, 'DSI203.nii.gz'), 'wb').write(opener.read())

        opener = urllib2.urlopen(ubval)
        open(pjoin(folder, 'DSI203.bval'), 'w').write(opener.read())

        opener = urllib2.urlopen(ubvec)
        open(pjoin(folder, 'DSI203.bvec'), 'w').write(opener.read())

        opener = urllib2.urlopen(ureadme)
        open(pjoin(folder, 'DSI203_license.txt'), 'w').write(opener.read())

        print('Done.')
        print('Files copied in folder %s' % folder)
        print('See DSI203_license.txt for LICENSE.')
        print('For the complete datasets please visit :')
        print('http://dsi-studio.labsolver.org')

    else:
        print('Dataset already in place. If you want to fetch again please first remove folder %s ' % folder)


def read_taiwan_ntu_dsi():
    """ Load Tawian NTU dataset

    Returns
    -------
    img : obj,
        Nifti1Image
    gtab : obj,
        GradientTable
    """
    dipy_home = pjoin(os.path.expanduser('~'), '.dipy')
    folder = pjoin(dipy_home, 'taiwan_ntu_dsi')
    fraw = pjoin(folder, 'DSI203.nii.gz')
    fbval = pjoin(folder, 'DSI203.bval')
    fbvec = pjoin(folder, 'DSI203.bvec')

    bvals, bvecs = read_bvals_bvecs(fbval, fbvec)
    bvecs[1:] = bvecs[1:] / np.sqrt(np.sum(bvecs[1:] * bvecs[1:], axis=1))[:, None]

    gtab = gradient_table(bvals, bvecs)
    img = nib.load(fraw)
    return img, gtab
