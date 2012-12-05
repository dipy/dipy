import os
import urllib2
from os.path import join as pjoin
import nibabel as nib
from dipy.core.gradients import gradient_table
from dipy.io.gradients import read_bvals_bvecs


def fetch_beijing_dti():
    """ Download a DTI dataset with 64 gradient directions
    """
    dipy_home = pjoin(os.path.expanduser('~'), '.dipy')
    url = 'https://www.dropbox.com/sh/ybl8vquht8pdc3s/'
    uraw = url + 'NfhK5WsYRO/DTI64.nii.gz?dl=1'
    ubval = url + 'HyX6GEQSSV/DTI64.bval?dl=1'
    ubvec = url + 'Xtrvr5jSOs/DTI64.bvec?dl=1'
    ureadme = url + 'HoGsocT6hZ/beijingEnhanced.txt?dl=1'
    folder = pjoin(dipy_home, 'beijing_dti')

    if not os.path.exists(folder):
        print('Creating new directory %s' % folder)
        os.makedirs(folder)
        print('Downloading raw DTI data (51MB)...')
        opener = urllib2.urlopen(uraw)
        open(pjoin(folder, 'DTI64.nii.gz'), 'wb').write(opener.read())

        opener = urllib2.urlopen(ubval)
        open(pjoin(folder, 'DTI64.bval'), 'w').write(opener.read())

        opener = urllib2.urlopen(ubvec)
        open(pjoin(folder, 'DTI64.bvec'), 'w').write(opener.read())

        opener = urllib2.urlopen(ureadme)
        open(pjoin(folder, 'beijingEnhanced.txt'), 'w').write(opener.read())

        print('Done.')
        print('Files copied in folder %s' % folder)
        print('See BeijingEnhanced.txt for DATA AGREEMENT LICENSE.')
        print('This is only a single brain DTI dataset which we use for some of the dipy tutorials.')
        print('For the complete datasets please visit :')
        print('http://fcon_1000.projects.nitrc.org/indi/retro/BeijingEnhanced.html')

    else:
        raise IOError('Dataset already in place. If you want to fetch again please first remove folder %s ' % folder)


def read_beijing_dti():
    """ Load Beijing dataset

    Returns
    -------
    img : obj,
        Nifti1Image
    gtab : obj,
        GradientTable
    """
    dipy_home = pjoin(os.path.expanduser('~'), '.dipy')
    folder = pjoin(dipy_home, 'beijing_dti')
    fraw = pjoin(folder, 'DTI64.nii.gz')
    fbval = pjoin(folder, 'DTI64.bval')
    fbvec = pjoin(folder, 'DTI64.bvec')
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

        opener = urllib2.urlopen(ubvec)
        open(pjoin(folder, 'DSI203_license.txt'), 'w').write(opener.read())

        print('Done.')
        print('Files copied in folder %s' % folder)
        print('See DSI203_license.txt for LICENSE.')

    else:
        raise IOError('Dataset already in place. If you want to fetch again please first remove folder %s ' % folder)


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
    gtab = gradient_table(bvals, bvecs)
    img = nib.load(fraw)
    return img, gtab


