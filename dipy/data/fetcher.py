from __future__ import division, print_function, absolute_import

import os
import sys
import textwrap
import contextlib

if sys.version_info[0] < 3:
    from urllib2 import urlopen
else:
    from urllib.request import urlopen

from os.path import join as pjoin
from hashlib import md5
from shutil import copyfileobj

import numpy as np
import nibabel as nib

import zipfile
from dipy.core.gradients import gradient_table
from dipy.io.gradients import read_bvals_bvecs


_bad_md5_message="""The downloaded file, {}, does not have the expected md5
checksum of "{}". This could mean that that something is wrong with the file or
that the upstream file has been updated. You can try downloading the file again
or updating to the newest version of dipy."""


class FetcherError(Exception):
    pass


def _log(msg):
    print(msg)


def fetch_data(files, folder):
    """Downloads files to folder and checks their md5 checksums

    Parameters
    ----------
    files : dictionary
        For each file in `files` the value should be (url, md5). The file will
        be downloaded from url if the file does not already exist or if the
        file exists but the md5 checksum does not match.
    folder : str
        The directory where to save the file, the directory will be created if
        it does not already exist.

    Raises
    ------
    FetcherError
        Raises if the md5 checksum of the file does not match the expected
        value. The downloaded file is not deleted when this error is raised.

    """
    if not os.path.exists(folder):
        _log("Creating new folder {}".format(folder))
        os.makedirs(folder)

    all_skip = True
    for f in files:
        url, md5 = files[f]
        fullpath = pjoin(folder, f)
        if os.path.exists(fullpath) and (_get_file_md5(fullpath) == md5):
            continue
        all_skip = False
        _log('Downloading "{}" to {}'.format(f, folder))
        _get_file_data(fullpath, url)
        if _get_file_md5(fullpath) != md5:
            msg = _bad_md5_message.format(fullpath, md5)
            msg = textwrap.fill(msg)
            raise FetcherError(msg)

    if all_skip:
        _log("All files already in {}.".format(folder))
    else:
        _log("Files successfully downloaded to {}".format(folder))


def fetch_scil_b0():
    """ Download b=0 datasets from multiple MR systems (GE, Philips, Siemens) and
        different magnetic fields (1.5T and 3T)
    """
    zipname = 'datasets_multi-site_all_companies'
    url = 'http://scil.dinf.usherbrooke.ca/wp-content/data/'
    uraw = url + zipname + '.zip'
    dipy_home = pjoin(os.path.expanduser('~'), '.dipy')
    folder = pjoin(dipy_home, zipname)

    if not os.path.exists(folder):
        print('Creating new directory %s' % folder)
        os.makedirs(folder)
        print('Downloading SCIL b=0 datasets from multiple sites and multiple companies (9.2MB)...')
        opener = urlopen(uraw)
        open(folder+'.zip', 'wb').write(opener.read())

        print('Unziping '+folder+'.zip ...')
        zip = zipfile.ZipFile(folder+'.zip', 'r')
        zip.extractall(dipy_home)

        print('Done.')
        print('Files copied in folder %s' % dipy_home)
    else:
        print('Dataset already in place. If you want to fetch again please first remove folder %s ' % dipy_home)


def read_scil_b0():
    """ Load GE 3T b0 image form the scil b0 dataset.

    Returns
    -------
    img : obj,
        Nifti1Image
    """
    dipy_home = os.path.join(os.path.expanduser('~'), '.dipy')
    file = dipy_home+'/datasets_multi-site_all_companies/3T/GE/b0.nii.gz'
    return nib.load(file)


def _get_file_md5(filename):
    """Compute the md5 checksum of a file"""
    md5_data = md5()
    with open(filename, 'rb') as f:
        for chunk in iter(lambda: f.read(128*md5_data.block_size), b''):
            md5_data.update(chunk)
    return md5_data.hexdigest()


def check_md5(filename, stored_md5):
    """
    Computes the md5 of filename and check if it matches with the supplied string md5

    Input
    -----
    filename : string
        Path to a file.
    md5 : string
        Known md5 of filename to check against.

    """
    computed_md5 = _get_file_md5(filename)
    if stored_md5 != computed_md5:
        print ("MD5 checksum of filename", filename, "failed. Expected MD5 was", stored_md5,
               "but computed MD5 was", computed_md5, '\n',
               "Please check if the data has been downloaded correctly or if the upstream data has changed.")


def _get_file_data(fname, url):
    with contextlib.closing(urlopen(url)) as opener:
        with open(fname, 'wb') as data:
            copyfileobj(opener, data)


def fetch_isbi2013_2shell():
    """ Download a 2-shell software phantom dataset
    """
    dipy_home = pjoin(os.path.expanduser('~'), '.dipy')
    url = 'https://dl.dropboxusercontent.com/u/2481924/isbi2013_merlet/'
    uraw = url + '2shells-1500-2500-N64-SNR-30.nii.gz'
    ubval = url + '2shells-1500-2500-N64.bval'
    ubvec = url + '2shells-1500-2500-N64.bvec'
    folder = pjoin(dipy_home, 'isbi2013')

    md5_list = ['42911a70f232321cf246315192d69c42', # data
                '90e8cf66e0f4d9737a3b3c0da24df5ea', # bval
                '4b7aa2757a1ccab140667b76e8075cb1'] # bvec

    url_list = [uraw, ubval, ubvec]
    fname_list = ['phantom64.nii.gz', 'phantom64.bval', 'phantom64.bvec']

    if not os.path.exists(folder):
        print('Creating new directory %s' % folder)
        os.makedirs(folder)
        print('Downloading raw 2-shell synthetic data (20MB)...')

        for i in range(len(md5_list)):
            _get_file_data(pjoin(folder, fname_list[i]), url_list[i])
            check_md5(pjoin(folder, fname_list[i]), md5_list[i])

        print('Done.')
        print('Files copied in folder %s' % folder)
    else:
        print('Dataset is already in place. If you want to fetch it again, please first remove the folder %s ' % folder)


def read_isbi2013_2shell():
    """ Load ISBI 2013 2-shell synthetic dataset

    Returns
    -------
    img : obj,
        Nifti1Image
    gtab : obj,
        GradientTable
    """
    dipy_home = pjoin(os.path.expanduser('~'), '.dipy')
    folder = pjoin(dipy_home, 'isbi2013')
    fraw = pjoin(folder, 'phantom64.nii.gz')
    fbval = pjoin(folder, 'phantom64.bval')
    fbvec = pjoin(folder, 'phantom64.bvec')

    md5_dict = {'data': '42911a70f232321cf246315192d69c42',
                'bval': '90e8cf66e0f4d9737a3b3c0da24df5ea',
                'bvec': '4b7aa2757a1ccab140667b76e8075cb1'}

    check_md5(fraw, md5_dict['data'])
    check_md5(fbval, md5_dict['bval'])
    check_md5(fbvec, md5_dict['bvec'])

    bvals, bvecs = read_bvals_bvecs(fbval, fbvec)

    gtab = gradient_table(bvals, bvecs)
    img = nib.load(fraw)
    return img, gtab


def fetch_sherbrooke_3shell():
    """ Download a 3shell HARDI dataset with 192 gradient directions
    """
    dipy_home = pjoin(os.path.expanduser('~'), '.dipy')
    url = 'https://dl.dropboxusercontent.com/u/2481924/sherbrooke_data/'
    uraw = url + '3shells-1000-2000-3500-N193.nii.gz'
    ubval = url + '3shells-1000-2000-3500-N193.bval'
    ubvec = url + '3shells-1000-2000-3500-N193.bvec'
    folder = pjoin(dipy_home, 'sherbrooke_3shell')

    md5_list = ['0b735e8f16695a37bfbd66aab136eb66', # data
                'e9b9bb56252503ea49d31fb30a0ac637', # bval
                '0c83f7e8b917cd677ad58a078658ebb7'] # bvec

    url_list = [uraw, ubval, ubvec]
    fname_list = ['HARDI193.nii.gz', 'HARDI193.bval', 'HARDI193.bvec']

    if not os.path.exists(folder):
        print('Creating new directory %s' % folder)
        os.makedirs(folder)
        print('Downloading raw 3-shell data (184MB)...')

        for i in range(len(md5_list)):
            _get_file_data(pjoin(folder, fname_list[i]), url_list[i])
            check_md5(pjoin(folder, fname_list[i]), md5_list[i])

        print('Done.')
        print('Files copied in folder %s' % folder)
    else:
        print('Dataset is already in place. If you want to fetch it again, please first remove the folder %s ' % folder)


def read_sherbrooke_3shell():
    """ Load Sherbrooke 3-shell HARDI dataset

    Returns
    -------
    img : obj,
        Nifti1Image
    gtab : obj,
        GradientTable
    """
    dipy_home = pjoin(os.path.expanduser('~'), '.dipy')
    folder = pjoin(dipy_home, 'sherbrooke_3shell')
    fraw = pjoin(folder, 'HARDI193.nii.gz')
    fbval = pjoin(folder, 'HARDI193.bval')
    fbvec = pjoin(folder, 'HARDI193.bvec')
    md5_dict = {'data': '0b735e8f16695a37bfbd66aab136eb66',
                'bval': 'e9b9bb56252503ea49d31fb30a0ac637',
                'bvec': '0c83f7e8b917cd677ad58a078658ebb7'}

    check_md5(fraw, md5_dict['data'])
    check_md5(fbval, md5_dict['bval'])
    check_md5(fbvec, md5_dict['bvec'])

    bvals, bvecs = read_bvals_bvecs(fbval, fbvec)

    gtab = gradient_table(bvals, bvecs)
    img = nib.load(fraw)
    return img, gtab


def fetch_stanford_labels():
    """Download reduced freesurfer aparc image from stanford web site."""
    dipy_home = pjoin(os.path.expanduser('~'), '.dipy')
    folder = pjoin(dipy_home, 'stanford_hardi')
    baseurl = 'https://stacks.stanford.edu/file/druid:yx282xq2090/'

    files = {}
    files["aparc-reduced.nii.gz"] = (baseurl + "aparc-reduced.nii.gz",
                                     '742de90090d06e687ce486f680f6d71a')
    files["label-info.txt"] = (baseurl + "label_info.txt",
                               '39db9f0f5e173d7a2c2e51b07d5d711b')
    fetch_data(files, folder)
    return files, folder


def read_stanford_labels():
    """Read stanford hardi data and label map"""
    # First get the hardi data
    fetch_stanford_hardi()
    hard_img, gtab = read_stanford_hardi()

    # Fetch and load
    files, folder = fetch_stanford_labels()
    labels_file = pjoin(folder, "aparc-reduced.nii.gz")
    labels_img = nib.load(labels_file)
    return hard_img, gtab, labels_img


def fetch_stanford_hardi():
    """ Download a HARDI dataset with 160 gradient directions
    """
    dipy_home = pjoin(os.path.expanduser('~'), '.dipy')
    url = 'https://stacks.stanford.edu/file/druid:yx282xq2090/'
    uraw = url + 'dwi.nii.gz'
    ubval = url + 'dwi.bvals'
    ubvec = url + 'dwi.bvecs'
    folder = pjoin(dipy_home, 'stanford_hardi')

    md5_list = ['0b18513b46132b4d1051ed3364f2acbc', # data
                '4e08ee9e2b1d2ec3fddb68c70ae23c36', # bval
                '4c63a586f29afc6a48a5809524a76cb4'] # bvec

    url_list = [uraw, ubval, ubvec]
    fname_list = ['HARDI150.nii.gz', 'HARDI150.bval', 'HARDI150.bvec']

    if not os.path.exists(folder):
        print('Creating new directory %s' % folder)
        os.makedirs(folder)
        print('Downloading raw HARDI data (87MB)...')

        for i in range(len(md5_list)):
            _get_file_data(pjoin(folder, fname_list[i]), url_list[i])
            check_md5(pjoin(folder, fname_list[i]), md5_list[i])

        print('Done.')
        print('Files copied in folder %s' % folder)
    else:
        print('Dataset is already in place. If you want to fetch it again, please first remove the folder %s ' % folder)


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
    md5_dict = {'data': '0b18513b46132b4d1051ed3364f2acbc',
                'bval': '4e08ee9e2b1d2ec3fddb68c70ae23c36',
                'bvec': '4c63a586f29afc6a48a5809524a76cb4'}

    check_md5(fraw, md5_dict['data'])
    check_md5(fbval, md5_dict['bval'])
    check_md5(fbvec, md5_dict['bvec'])

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

    md5_list = ['950408c0980a7154cb188666a885a91f', # data
                '602e5cb5fad2e7163e8025011d8a6755', # bval
                'a95eb1be44748c20214dc7aa654f9e6b', # bvec
                '7fa1d5e272533e832cc7453eeba23f44'] # license

    url_list = [uraw, ubval, ubvec, ureadme]
    fname_list = ['DSI203.nii.gz', 'DSI203.bval', 'DSI203.bvec', 'DSI203_license.txt']

    if not os.path.exists(folder):
        print('Creating new directory %s' % folder)
        os.makedirs(folder)
        print('Downloading raw DSI data (91MB)...')

        for i in range(len(md5_list)):
            _get_file_data(pjoin(folder, fname_list[i]), url_list[i])
            check_md5(pjoin(folder, fname_list[i]), md5_list[i])

        print('Done.')
        print('Files copied in folder %s' % folder)
        print('See DSI203_license.txt for LICENSE.')
        print('For the complete datasets please visit :')
        print('http://dsi-studio.labsolver.org')

    else:
        print('Dataset is already in place. If you want to fetch it again, please first remove the folder %s ' % folder)


def read_taiwan_ntu_dsi():
    """ Load Taiwan NTU dataset

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
    md5_dict = {'data': '950408c0980a7154cb188666a885a91f',
                'bval': '602e5cb5fad2e7163e8025011d8a6755',
                'bvec': 'a95eb1be44748c20214dc7aa654f9e6b',
                'license': '7fa1d5e272533e832cc7453eeba23f44'}

    check_md5(fraw, md5_dict['data'])
    check_md5(fbval, md5_dict['bval'])
    check_md5(fbvec, md5_dict['bvec'])
    check_md5(pjoin(folder, 'DSI203_license.txt'), md5_dict['license'])

    bvals, bvecs = read_bvals_bvecs(fbval, fbvec)
    bvecs[1:] = bvecs[1:] / np.sqrt(np.sum(bvecs[1:] * bvecs[1:], axis=1))[:, None]

    gtab = gradient_table(bvals, bvecs)
    img = nib.load(fraw)
    return img, gtab
