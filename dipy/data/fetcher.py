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
import tarfile

import numpy as np
import nibabel as nib

import zipfile
from dipy.core.gradients import gradient_table
from dipy.io.gradients import read_bvals_bvecs


class FetcherError(Exception):
    pass


def _log(msg):
    print(msg)

dipy_home = pjoin(os.path.expanduser('~'), '.dipy')


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
        _log("Creating new folder %s" % (folder))
        os.makedirs(folder)

    all_skip = True
    for f in files:
        url, md5 = files[f]
        fullpath = pjoin(folder, f)
        if os.path.exists(fullpath) and (_get_file_md5(fullpath) == md5):
            continue
        all_skip = False
        _log('Downloading "%s" to %s' % (f, folder))
        _get_file_data(fullpath, url)
        if _get_file_md5(fullpath) != md5:
            msg = """The downloaded file, %s, does not have the expected md5
checksum of "%s". This could mean that that something is wrong with the file or
that the upstream file has been updated. You can try downloading the file again
or updating to the newest version of dipy.""" % (fullpath, md5)
            msg = textwrap.fill(msg)
            raise FetcherError(msg)

    if all_skip:
        _log("All files already in %s." % (folder))
    else:
        _log("Files successfully downloaded to %s" % (folder))


def _already_there_msg(folder):
    """
    Prints a message indicating that a certain data-set is already in place
    """
    msg = 'Dataset is already in place. If you want to fetch it again '
    msg += 'please first remove the folder %s ' % folder
    print(msg)



def fetch_scil_b0():
    """ Download b=0 datasets from multiple MR systems (GE, Philips, Siemens) and
        different magnetic fields (1.5T and 3T)
    """
    zipname = 'datasets_multi-site_all_companies'
    url = 'http://scil.dinf.usherbrooke.ca/wp-content/data/'
    uraw = url + zipname + '.zip'
    folder = pjoin(dipy_home, zipname)

    if not os.path.exists(folder):
        print('Creating new directory %s' % folder)
        os.makedirs(folder)
        msg = 'Downloading SCIL b=0 datasets from multiple sites and'
        msg += 'multiple companies (9.2MB)...'
        print()
        opener = urlopen(uraw)
        open(folder+'.zip', 'wb').write(opener.read())

        print('Unziping '+folder+'.zip ...')
        zip = zipfile.ZipFile(folder+'.zip', 'r')
        zip.extractall(dipy_home)

        print('Done.')
        print('Files copied in folder %s' % dipy_home)
    else:
        _already_there_msg(folder)


def read_scil_b0():
    """ Load GE 3T b0 image form the scil b0 dataset.

    Returns
    -------
    img : obj,
        Nifti1Image
    """
    file = pjoin(dipy_home,
                 'datasets_multi-site_all_companies',
                 '3T',
                 'GE',
                 'b0.nii.gz')

    return nib.load(file)


def read_siemens_scil_b0():
    """ Load Siemens 1.5T b0 image form the scil b0 dataset.

    Returns
    -------
    img : obj,
        Nifti1Image
    """
    file = pjoin(dipy_home,
                 'datasets_multi-site_all_companies',
                 '1.5T',
                 'Siemens',
                 'b0.nii.gz')

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
        msg = "MD5 checksum of filename " + filename + " failed.\n"
        msg += "Expected MD5 was " + stored_md5 + "\n"
        msg += "Current MD5 is " + computed_md5 + "\n"
        msg += "Please check if the data has been downloaded "
        msg += "correctly or if the upstream data has changed."
        print (msg)


def _get_file_data(fname, url):
    with contextlib.closing(urlopen(url)) as opener:
        with open(fname, 'wb') as data:
            copyfileobj(opener, data)


def fetch_isbi2013_2shell():
    """ Download a 2-shell software phantom dataset
    """
    url = 'https://dl.dropboxusercontent.com/u/2481924/isbi2013_merlet/'
    uraw = url + '2shells-1500-2500-N64-SNR-30.nii.gz'
    ubval = url + '2shells-1500-2500-N64.bval'
    ubvec = url + '2shells-1500-2500-N64.bvec'
    folder = pjoin(dipy_home, 'isbi2013')

    md5_list = ['42911a70f232321cf246315192d69c42',  # data
                '90e8cf66e0f4d9737a3b3c0da24df5ea',  # bval
                '4b7aa2757a1ccab140667b76e8075cb1']  # bvec

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
        _already_there_msg(folder)


def read_isbi2013_2shell():
    """ Load ISBI 2013 2-shell synthetic dataset

    Returns
    -------
    img : obj,
        Nifti1Image
    gtab : obj,
        GradientTable
    """
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
    url = 'https://dl.dropboxusercontent.com/u/2481924/sherbrooke_data/'
    uraw = url + '3shells-1000-2000-3500-N193.nii.gz'
    ubval = url + '3shells-1000-2000-3500-N193.bval'
    ubvec = url + '3shells-1000-2000-3500-N193.bvec'
    folder = pjoin(dipy_home, 'sherbrooke_3shell')

    md5_list = ['0b735e8f16695a37bfbd66aab136eb66',  # data
                'e9b9bb56252503ea49d31fb30a0ac637',  # bval
                '0c83f7e8b917cd677ad58a078658ebb7']  # bvec

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
        _already_there_msg(folder)


def read_sherbrooke_3shell():
    """ Load Sherbrooke 3-shell HARDI dataset

    Returns
    -------
    img : obj,
        Nifti1Image
    gtab : obj,
        GradientTable
    """
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
    url = 'https://stacks.stanford.edu/file/druid:yx282xq2090/'
    uraw = url + 'dwi.nii.gz'
    ubval = url + 'dwi.bvals'
    ubvec = url + 'dwi.bvecs'
    folder = pjoin(dipy_home, 'stanford_hardi')

    md5_list = ['0b18513b46132b4d1051ed3364f2acbc',  # data
                '4e08ee9e2b1d2ec3fddb68c70ae23c36',  # bval
                '4c63a586f29afc6a48a5809524a76cb4']  # bvec

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
        _already_there_msg(folder)


def read_stanford_hardi():
    """ Load Stanford HARDI dataset

    Returns
    -------
    img : obj,
        Nifti1Image
    gtab : obj,
        GradientTable
    """
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


def fetch_stanford_t1():
    url = 'https://stacks.stanford.edu/file/druid:yx282xq2090/'
    url_t1 = url + 't1.nii.gz'
    folder = pjoin(dipy_home, 'stanford_hardi')
    file_md5 = 'a6a140da6a947d4131b2368752951b0a'
    files = {"t1.nii.gz": (url_t1, file_md5)}
    fetch_data(files, folder)
    return files, folder


def read_stanford_t1():
    files, folder = fetch_stanford_t1()
    f_t1 = pjoin(folder, 't1.nii.gz')
    img = nib.load(f_t1)
    return img


def fetch_stanford_pve_maps():
    url = 'https://stacks.stanford.edu/file/druid:yx282xq2090/'
    url_pve_csf = url + 'pve_csf.nii.gz'
    url_pve_gm = url + 'pve_gm.nii.gz'
    url_pve_wm = url + 'pve_wm.nii.gz'
    folder = pjoin(dipy_home, 'stanford_hardi')
    file_csf_md5 = '2c498e4fed32bca7f726e28aa86e9c18'
    file_gm_md5 = '1654b20aeb35fc2734a0d7928b713874'
    file_wm_md5 = '2e244983cf92aaf9f9d37bc7716b37d5'
    files = {"pve_csf.nii.gz": (url_pve_csf, file_csf_md5),
             "pve_gm.nii.gz": (url_pve_gm, file_gm_md5),
             "pve_wm.nii.gz": (url_pve_wm, file_wm_md5)}
    fetch_data(files, folder)
    return files, folder


def read_stanford_pve_maps():
    files, folder = fetch_stanford_pve_maps()
    f_pve_csf = pjoin(folder, 'pve_csf.nii.gz')
    f_pve_gm = pjoin(folder, 'pve_gm.nii.gz')
    f_pve_wm = pjoin(folder, 'pve_wm.nii.gz')
    img_pve_csf = nib.load(f_pve_csf)
    img_pve_gm = nib.load(f_pve_gm)
    img_pve_wm = nib.load(f_pve_wm)
    return (img_pve_csf, img_pve_gm, img_pve_wm)


def fetch_taiwan_ntu_dsi():
    """ Download a DSI dataset with 203 gradient directions
    """
    uraw = 'http://dl.dropbox.com/u/2481924/taiwan_ntu_dsi.nii.gz'
    ubval = 'http://dl.dropbox.com/u/2481924/tawian_ntu_dsi.bval'
    ubvec = 'http://dl.dropbox.com/u/2481924/taiwan_ntu_dsi.bvec'
    ureadme = 'http://dl.dropbox.com/u/2481924/license_taiwan_ntu_dsi.txt'
    folder = pjoin(dipy_home, 'taiwan_ntu_dsi')

    md5_list = ['950408c0980a7154cb188666a885a91f',  # data
                '602e5cb5fad2e7163e8025011d8a6755',  # bval
                'a95eb1be44748c20214dc7aa654f9e6b',  # bvec
                '7fa1d5e272533e832cc7453eeba23f44']  # license

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
        _already_there_msg(folder)


def read_taiwan_ntu_dsi():
    """ Load Taiwan NTU dataset

    Returns
    -------
    img : obj,
        Nifti1Image
    gtab : obj,
        GradientTable
    """
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


def fetch_syn_data():
    """ Download t1 and b0 volumes from the same session
    """
    url = 'https://dl.dropboxusercontent.com/u/5918983/'
    t1 = url + 't1.nii.gz'
    b0 = url + 'b0.nii.gz'

    folder = pjoin(dipy_home, 'syn_test')

    md5_list = ['701bda02bb769655c7d4a9b1df2b73a6',  # t1
                'e4b741f0c77b6039e67abb2885c97a78']  # b0

    url_list = [t1, b0]
    fname_list = ['t1.nii.gz', 'b0.nii.gz']

    if not os.path.exists(folder):
        print('Creating new directory %s' % folder)
        os.makedirs(folder)
        print('Downloading t1 and b0 volumes from the same session (12MB)...')

        for i in range(len(md5_list)):
            _get_file_data(pjoin(folder, fname_list[i]), url_list[i])
            check_md5(pjoin(folder, fname_list[i]), md5_list[i])

        print('Done.')
        print('Files copied in folder %s' % folder)
    else:
        _already_there_msg(folder)


def read_syn_data():
    """ Load t1 and b0 volumes from the same session

    Returns
    -------
    t1 : obj,
        Nifti1Image
    b0 : obj,
        Nifti1Image
    """
    folder = pjoin(dipy_home, 'syn_test')
    t1_name = pjoin(folder, 't1.nii.gz')
    b0_name = pjoin(folder, 'b0.nii.gz')

    md5_dict = {'t1': '701bda02bb769655c7d4a9b1df2b73a6',
                'b0': 'e4b741f0c77b6039e67abb2885c97a78'}

    check_md5(t1_name, md5_dict['t1'])
    check_md5(b0_name, md5_dict['b0'])

    t1 = nib.load(t1_name)
    b0 = nib.load(b0_name)
    return t1, b0

mni_notes = \
"""
    Notes
    -----
    The templates were downloaded from the MNI (McGill University) `website <http://www.bic.mni.mcgill.ca/ServicesAtlases/ICBM152NLin2009>`_
    in July 2015.

    The following publications should be referenced when using these templates:

    .. [1] VS Fonov, AC Evans, K Botteron, CR Almli, RC McKinstry, DL Collins
           and BDCG, Unbiased average age-appropriate atlases for pediatric
           studies, NeuroImage, 54:1053-8119, DOI: 10.1016/j.neuroimage.2010.07.033

    .. [2] VS Fonov, AC Evans, RC McKinstry, CR Almli and DL Collins,
            Unbiased nonlinear average age-appropriate brain templates from
            birth to adulthood, NeuroImage, 47:S102
            Organization for Human Brain Mapping 2009 Annual Meeting,
            DOI: http://dx.doi.org/10.1016/S1053-8119(09)70884-5

    License for the MNI templates:
    -----------------------------
    Copyright (C) 1993-2004, Louis Collins McConnell Brain Imaging Centre,
    Montreal Neurological Institute, McGill University. Permission to use,
    copy, modify, and distribute this software and its documentation for any
    purpose and without fee is hereby granted, provided that the above
    copyright notice appear in all copies. The authors and McGill University
    make no representations about the suitability of this software for any
    purpose. It is provided "as is" without express or implied warranty. The
    authors are not responsible for any data loss, equipment damage, property
    loss, or injury to subjects or patients resulting from the use or misuse
    of this software package.
"""


def fetch_viz_icons():
    """ Download icons for visualization
    """
    url = 'https://dl.dropboxusercontent.com/u/2481924/'
    fname = 'icomoon.tar.gz'
    icomoon = url + fname
    folder = pjoin(dipy_home, 'icons')

    url_list = [icomoon]
    md5_list = ['94a07cba06b4136b6687396426f1e380']
    fname_list = [fname]

    if not os.path.exists(folder):
        print('Creating new directory %s' % folder)
        os.makedirs(folder)
        print('Downloading icons ...')
        for i in range(len(md5_list)):
            _get_file_data(pjoin(folder, fname_list[i]), url_list[i])
            new_path = pjoin(folder, fname_list[i])
            check_md5(new_path, md5_list[i])
            ar = tarfile.open(new_path)
            ar.extractall(path=folder)
            ar.close()

        print('Done.')
        print('Files copied in folder %s' % folder)
    else:
        msg = 'Dataset is already in place. If you want to fetch it again, '
        msg += 'please first remove the folder %s '
        print(msg % folder)


def read_viz_icons(style='icomoon', fname='infinity.png'):
    """ Read specific icon from specific style

    Parameters
    ----------
    style: str
        Current icon style. Default is icomoon.
    fname: str
        Filename of icon. This should be found in folder HOME/.dipy/style/.
        Default is infinity.png.

    Returns
    --------
    path: str
        Complete path of icon.

    """

    folder = pjoin(dipy_home, 'icons', style)
    return pjoin(folder, fname)


def fetch_bundles_2_subjects():
    """ Download 2 subjects with their bundles
    """
    url = 'https://dl.dropboxusercontent.com/u/2481924/'
    fname = 'bundles_2_subjects.tar.gz'
    url = url + fname
    folder = pjoin(dipy_home, 'exp_bundles_and_maps')

    url_list = [url]
    md5_list = ['97756fbef11ce2df31f1bedf1fc7aac7']
    fname_list = [fname]

    if not os.path.exists(folder):
        print('Creating new directory %s' % folder)
        os.makedirs(folder)
        print('Downloading dataset ...')
        for i in range(len(md5_list)):
            _get_file_data(pjoin(folder, fname_list[i]), url_list[i])
            new_path = pjoin(folder, fname_list[i])
            check_md5(new_path, md5_list[i])
            ar = tarfile.open(new_path)
            ar.extractall(path=folder)
            ar.close()

        print('Done.')
        print('Files copied in folder %s' % folder)
    else:
        msg = 'Dataset is already in place. If you want to fetch it again, '
        msg += 'please first remove the folder %s '
        print(msg % folder)


def read_bundles_2_subjects(subj_id='subj_1', metrics=['fa'],
                            bundles=['af.left', 'cst.right', 'cc_1']):

    dname = pjoin(dipy_home, 'exp_bundles_and_maps', 'bundles_2_subjects')

    from nibabel import trackvis as tv

    res = {}

    if 't1' in metrics:
        img = nib.load(pjoin(dname, subj_id, 't1_warped.nii.gz'))
        data = img.get_data()
        affine = img.get_affine()
        res['t1'] = data

    if 'fa' in metrics:
        img_fa = nib.load(pjoin(dname, subj_id, 'fa_1x1x1.nii.gz'))
        fa = img_fa.get_data()
        affine = img_fa.get_affine()
        res['fa'] = fa

    res['affine'] = affine

    for bun in bundles:

        streams, hdr = tv.read(pjoin(dname, subj_id,
                                     'bundles', 'bundles_' + bun + '.trk'),
                               points_space="rasmm")
        streamlines = [s[0] for s in streams]
        res[bun] = streamlines

    return res


def fetch_mni_template():
    """
    Fetch the MNI T2 and T1 template files (~35 MB)
    """
    folder = pjoin(dipy_home, 'mni_template')
    baseurl = \
    'https://digital.lib.washington.edu/researchworks/bitstream/handle/1773/33312/'

    fname_list = ['COPYING',
                  'mni_icbm152_t2_tal_nlin_asym_09a.nii',
                  'mni_icbm152_t1_tal_nlin_asym_09a.nii']
    md5_list = ['6e2168072e80aa4c0c20f1e6e52ec0c8',
                'f41f2e1516d880547fbf7d6a83884f0d',
                '1ea8f4f1e41bc17a94602e48141fdbc8']
    files = {}
    for f, m in zip(fname_list, md5_list):
        files[f] = (baseurl + f, m)
    fetch_data(files, folder)
    return files, folder


def read_mni_template(contrast="T2"):
    """
    Read the MNI template from disk

    Parameters
    ----------
    contrast : list or string, optional
        Which of the contrast templates to read. Two contrasts are available:
        "T1" and "T2", so you can either enter one of these strings as input,
        or a list containing both of them.

    Returns
    -------
    list : contains the nibabel.Nifti1Image objects requested, according to the
        order they were requested in the input.

    Examples
    --------
    Get only the T2 file:
    >>> T2_nifti = read_mni_template("T2") # doctest: +SKIP
    Get both files in this order:
    >>> T1_nifti, T2_nifti = read_mni_template(["T1", "T2"]) # doctest: +SKIP
    """

    files, folder = fetch_mni_template()
    file_dict = {"T1": pjoin(folder, 'mni_icbm152_t1_tal_nlin_asym_09a.nii'),
                 "T2": pjoin(folder, 'mni_icbm152_t2_tal_nlin_asym_09a.nii')}
    if isinstance(contrast, str):
        return nib.load(file_dict[contrast])
    else:
        out_list = []
        for k in contrast:
            out_list.append(nib.load(file_dict[k]))
    return out_list


# Add the references to both MNI-related functions:
read_mni_template.__doc__ += mni_notes
fetch_mni_template.__doc__ += mni_notes


def fetch_cenir_multib(with_raw=False):
    """
    Fetch 'HCP-like' data, collected at multiple b-values

    Parameters
    ----------
    with_raw : bool
        Whether to fetch the raw data. Per default, this is False, which means
        that only eddy-current/motion corrected data is fetched
    """
    folder = pjoin(dipy_home, 'cenir_multib')

    fname_list = ['4D_dwi_eddycor_B200.nii.gz',
                  'dwi_bvals_B200', 'dwi_bvecs_B200',
                  '4D_dwieddycor_B400.nii.gz',
                  'bvals_B400', 'bvecs_B400',
                  '4D_dwieddycor_B1000.nii.gz',
                  'bvals_B1000', 'bvecs_B1000',
                   '4D_dwieddycor_B2000.nii.gz',
                  'bvals_B2000', 'bvecs_B2000',
                  '4D_dwieddycor_B3000.nii.gz',
                  'bvals_B3000', 'bvecs_B3000']

    md5_list = ['fd704aa3deb83c1c7229202cb3db8c48',
                '80ae5df76a575fe5bf9f1164bb0d4cfb',
                '18e90f8a3e6a4db2457e5b1ba1cc98a9',
                '3d0f2b8ef7b6a4a3aa5c4f7a90c9cfec',
                'c38056c40c9cc42372232d6e75c47f54',
                '810d79b4c30cb7dff3b2000017d5f72a',
                'dde8037601a14436b2173f4345b5fd17',
                '97de6a492ae304f39e0b418b6ebac64c',
                'f28a0faa701bdfc66e31bde471a5b992',
                'c5e4b96e3afdee99c0e994eff3b2331a',
                '9c83b8d5caf9c3def240f320f2d2f56c',
                '05446bd261d57193d8dbc097e06db5ff',
                'f0d70456ce424fda2cecd48e64f3a151',
                '336accdb56acbbeff8dac1748d15ceb8',
                '27089f3baaf881d96f6a9da202e3d69b']
    if with_raw:
        fname_list.extend(['4D_dwi_B200.nii.gz', '4D_dwi_B400.nii.gz',
                            '4D_dwi_B1000.nii.gz', '4D_dwi_B2000.nii.gz',
                            '4D_dwi_B3000.nii.gz'])
        md5_list.extend(['a8c36e76101f2da2ca8119474ded21d5',
                        'a0e7939f6d977458afbb2f4659062a79',
                        '87fc307bdc2e56e105dffc81b711a808',
                        '7c23e8a5198624aa29455f0578025d4f',
                        '4e4324c676f5a97b3ded8bbb100bf6e5'])

    files = {}
    baseurl = \
    'https://digital.lib.washington.edu/researchworks/bitstream/handle/1773/33311/'

    for f, m in zip(fname_list, md5_list):
        files[f] = (baseurl + f, m)

    fetch_data(files, folder)
    return files, folder


def read_cenir_multib(bvals=None):
    """
    Read CENIR multi b-value data

    Parameters
    ----------
    bvals : list
        The b-values to read from file (200, 400, 1000, 2000, 3000).

    Returns
    -------
    gtab : a GradientTable class instance
    img : nibabel.Nifti1Image

    Notes
    -----
    Details of acquisition and processing are availble
    """
    files, folder = fetch_cenir_multib(with_raw=False)
    if bvals is None:
        bvals = [200, 400, 1000, 2000, 3000]
    file_dict = {
    200:{'DWI': pjoin(folder, '4D_dwi_eddycor_B200.nii.gz'),
           'bvals': pjoin(folder, 'dwi_bvals_B200'),
           'bvecs': pjoin(folder, 'dwi_bvecs_B200')},
    400:{'DWI': pjoin(folder, '4D_dwieddycor_B400.nii.gz'),
           'bvals': pjoin(folder, 'bvals_B400'),
           'bvecs': pjoin(folder, 'bvecs_B400')},
    1000:{'DWI': pjoin(folder, '4D_dwieddycor_B1000.nii.gz'),
           'bvals': pjoin(folder, 'bvals_B1000'),
           'bvecs': pjoin(folder, 'bvecs_B1000')},
    2000:{'DWI': pjoin(folder, '4D_dwieddycor_B2000.nii.gz'),
           'bvals': pjoin(folder, 'bvals_B2000'),
           'bvecs': pjoin(folder, 'bvecs_B2000')},
    3000:{'DWI': pjoin(folder, '4D_dwieddycor_B3000.nii.gz'),
           'bvals': pjoin(folder, 'bvals_B3000'),
           'bvecs': pjoin(folder, 'bvecs_B3000')}}
    data = []
    bval_list = []
    bvec_list = []
    for bval in bvals:
        data.append(nib.load(file_dict[bval]['DWI']).get_data())
        bval_list.extend(np.loadtxt(file_dict[bval]['bvals']))
        bvec_list.append(np.loadtxt(file_dict[bval]['bvecs']))

    # All affines are the same, so grab the last one:
    aff = nib.load(file_dict[bval]['DWI']).get_affine()
    return (gradient_table(bval_list, np.concatenate(bvec_list, -1)),
            nib.Nifti1Image(np.concatenate(data, -1), aff))


CENIR_notes = \
"""
Notes
-----
Details of the acquisition and processing, and additional meta-data are avalible
through `UW researchworks <https://digital.lib.washington.edu/researchworks/handle/1773/33311>`_
"""

fetch_cenir_multib.__doc__ += CENIR_notes
read_cenir_multib.__doc__ += CENIR_notes
