from __future__ import division, print_function, absolute_import

import os
import sys
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

import tarfile
import zipfile
from dipy.core.gradients import gradient_table
from dipy.io.gradients import read_bvals_bvecs

# Set a user-writeable file-system location to put files:
dipy_home = pjoin(os.path.expanduser('~'), '.dipy')


class FetcherError(Exception):
    pass


def _log(msg):
    """Helper function to keep track of things.
    For now, just prints the message
    """
    print(msg)


def _already_there_msg(folder):
    """
    Prints a message indicating that a certain data-set is already in place
    """
    msg = 'Dataset is already in place. If you want to fetch it again '
    msg += 'please first remove the folder %s ' % folder
    _log(msg)


def _get_file_md5(filename):
    """Compute the md5 checksum of a file"""
    md5_data = md5()
    with open(filename, 'rb') as f:
        for chunk in iter(lambda: f.read(128*md5_data.block_size), b''):
            md5_data.update(chunk)
    return md5_data.hexdigest()


def check_md5(filename, stored_md5=None):
    """
    Computes the md5 of filename and check if it matches with the supplied
    string md5

    Input
    -----
    filename : string
        Path to a file.
    md5 : string
        Known md5 of filename to check against. If None (default), checking is
        skipped
    """
    if stored_md5 is not None:
        computed_md5 = _get_file_md5(filename)
        if stored_md5 != computed_md5:
            msg = """The downloaded file, %s, does not have the expected md5
   checksum of "%s". Instead, the md5 checksum was: "%s". This could mean that
   something is wrong with the file or that the upstream file has been updated.
   You can try downloading the file again or updating to the newest version of
   dipy.""" % (filename, stored_md5,
                computed_md5)
            raise FetcherError(msg)


def _get_file_data(fname, url):
    with contextlib.closing(urlopen(url)) as opener:
        with open(fname, 'wb') as data:
            copyfileobj(opener, data)


def fetch_data(files, folder, data_size=None):
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
    data_size : str, optional
        A string describing the size of the data (e.g. "91 MB") to be logged to
        the screen. Default does not produce any information about data size.
    Raises
    ------
    FetcherError
        Raises if the md5 checksum of the file does not match the expected
        value. The downloaded file is not deleted when this error is raised.

    """
    if not os.path.exists(folder):
        _log("Creating new folder %s" % (folder))
        os.makedirs(folder)

    if data_size is not None:
        _log('Data size is approximately %s' % data_size)

    all_skip = True
    for f in files:
        url, md5 = files[f]
        fullpath = pjoin(folder, f)
        if os.path.exists(fullpath) and (_get_file_md5(fullpath) == md5):
            continue
        all_skip = False
        _log('Downloading "%s" to %s' % (f, folder))
        _get_file_data(fullpath, url)
        check_md5(fullpath, md5)
    if all_skip:
        _already_there_msg(folder)
    else:
        _log("Files successfully downloaded to %s" % (folder))


def _make_fetcher(name, folder, baseurl, remote_fnames, local_fnames,
                  md5_list=None, doc="", data_size=None, msg=None,
                  unzip=False):
    """ Create a new fetcher

    Parameters
    ----------
    name : str
        The name of the fetcher function.
    folder : str
        The full path to the folder in which the files would be placed locally.
        Typically, this is something like 'pjoin(dipy_home, 'foo')'
    baseurl : str
        The URL from which this fetcher reads files
    remote_fnames : list of strings
        The names of the files in the baseurl location
    local_fnames : list of strings
        The names of the files to be saved on the local filesystem
    md5_list : list of strings, optional
        The md5 checksums of the files. Used to verify the content of the
        files. Default: None, skipping checking md5.
    doc : str, optional.
        Documentation of the fetcher.
    data_size : str, optional.
        If provided, is sent as a message to the user before downloading
        starts.
    msg : str, optional.
        A message to print to screen when fetching takes place. Default (None)
        is to print nothing
    unzip : bool, optional
        Whether to unzip the file(s) after downloading them. Supports zip, gz,
        and tar.gz files.
    returns
    -------
    fetcher : function
        A function that, when called, fetches data according to the designated
        inputs
    """
    def fetcher():
        files = {}
        for i, (f, n), in enumerate(zip(remote_fnames, local_fnames)):
            files[n] = (baseurl + f, md5_list[i] if
                        md5_list is not None else None)
        fetch_data(files, folder, data_size)

        if msg is not None:
            print(msg)
        if unzip:
            for f in local_fnames:
                split_ext = os.path.splitext(f)
                if split_ext[-1] == '.gz' or split_ext[-1] == '.bz2':
                    if os.path.splitext(split_ext[0])[-1] == '.tar':
                        ar = tarfile.open(pjoin(folder, f))
                        ar.extractall(path=folder)
                        ar.close()
                    else:
                        raise ValueError('File extension is not recognized')
                elif split_ext[-1] == '.zip':
                    z = zipfile.ZipFile(pjoin(folder, f), 'r')
                    z.extractall(folder)
                    z.close()
                else:
                    raise ValueError('File extension is not recognized')

        return files, folder

    fetcher.__name__ = name
    fetcher.__doc__ = doc
    return fetcher


fetch_isbi2013_2shell = _make_fetcher(
    "fetch_isbi2013_2shell",
    pjoin(dipy_home, 'isbi2013'),
    'https://dl.dropboxusercontent.com/u/2481924/isbi2013_merlet/',
    ['2shells-1500-2500-N64-SNR-30.nii.gz',
     '2shells-1500-2500-N64.bval',
     '2shells-1500-2500-N64.bvec'],
    ['phantom64.nii.gz', 'phantom64.bval', 'phantom64.bvec'],
    ['42911a70f232321cf246315192d69c42',
     '90e8cf66e0f4d9737a3b3c0da24df5ea',
     '4b7aa2757a1ccab140667b76e8075cb1'],
    doc="Download a 2-shell software phantom dataset",
    data_size="")

fetch_stanford_labels = _make_fetcher(
    "fetch_stanford_labels",
    pjoin(dipy_home, 'stanford_hardi'),
    'https://stacks.stanford.edu/file/druid:yx282xq2090/',
    ["aparc-reduced.nii.gz", "label_info.txt"],
    ["aparc-reduced.nii.gz", "label_info.txt"],
    ['742de90090d06e687ce486f680f6d71a',
     '39db9f0f5e173d7a2c2e51b07d5d711b'],
    doc="Download reduced freesurfer aparc image from stanford web site")

fetch_sherbrooke_3shell = _make_fetcher(
    "fetch_sherbrooke_3shell",
    pjoin(dipy_home, 'sherbrooke_3shell'),
    'https://dl.dropboxusercontent.com/u/2481924/sherbrooke_data/',
    ['3shells-1000-2000-3500-N193.nii.gz',
     '3shells-1000-2000-3500-N193.bval',
     '3shells-1000-2000-3500-N193.bvec'],
    ['HARDI193.nii.gz', 'HARDI193.bval', 'HARDI193.bvec'],
    ['0b735e8f16695a37bfbd66aab136eb66',
     'e9b9bb56252503ea49d31fb30a0ac637',
     '0c83f7e8b917cd677ad58a078658ebb7'],
    doc="Download a 3shell HARDI dataset with 192 gradient direction")


fetch_stanford_hardi = _make_fetcher(
    "fetch_stanford_hardi",
    pjoin(dipy_home, 'stanford_hardi'),
    'https://stacks.stanford.edu/file/druid:yx282xq2090/',
    ['dwi.nii.gz', 'dwi.bvals', 'dwi.bvecs'],
    ['HARDI150.nii.gz', 'HARDI150.bval', 'HARDI150.bvec'],
    ['0b18513b46132b4d1051ed3364f2acbc',
     '4e08ee9e2b1d2ec3fddb68c70ae23c36',
     '4c63a586f29afc6a48a5809524a76cb4'],
    doc="Download a HARDI dataset with 160 gradient directions")

fetch_stanford_t1 = _make_fetcher(
    "fetch_stanford_t1",
    pjoin(dipy_home, 'stanford_hardi'),
    'https://stacks.stanford.edu/file/druid:yx282xq2090/',
    ['t1.nii.gz'],
    ['t1.nii.gz'],
    ['a6a140da6a947d4131b2368752951b0a'])

fetch_stanford_pve_maps = _make_fetcher(
    "fetch_stanford_pve_maps",
    pjoin(dipy_home, 'stanford_hardi'),
    'https://stacks.stanford.edu/file/druid:yx282xq2090/',
    ['pve_csf.nii.gz', 'pve_gm.nii.gz', 'pve_wm.nii.gz'],
    ['pve_csf.nii.gz', 'pve_gm.nii.gz', 'pve_wm.nii.gz'],
    ['2c498e4fed32bca7f726e28aa86e9c18',
     '1654b20aeb35fc2734a0d7928b713874',
     '2e244983cf92aaf9f9d37bc7716b37d5'])

fetch_taiwan_ntu_dsi = _make_fetcher(
    "fetch_taiwan_ntu_dsi",
    pjoin(dipy_home, 'taiwan_ntu_dsi'),
    "http://dl.dropbox.com/u/2481924/",
    ['taiwan_ntu_dsi.nii.gz', 'tawian_ntu_dsi.bval',
     'taiwan_ntu_dsi.bvec', 'license_taiwan_ntu_dsi.txt'],
    ['DSI203.nii.gz', 'DSI203.bval', 'DSI203.bvec', 'DSI203_license.txt'],
    ['950408c0980a7154cb188666a885a91f',
     '602e5cb5fad2e7163e8025011d8a6755',
     'a95eb1be44748c20214dc7aa654f9e6b',
     '7fa1d5e272533e832cc7453eeba23f44'],
    doc="Download a DSI dataset with 203 gradient directions",
    msg="See DSI203_license.txt for LICENSE. For the complete datasets please visit : http://dsi-studio.labsolver.org",
    data_size="91MB")

fetch_syn_data = _make_fetcher(
    "fetch_syn_data",
    pjoin(dipy_home, 'syn_test'),
    'https://dl.dropboxusercontent.com/u/5918983/',
    ['t1.nii.gz', 'b0.nii.gz'],
    ['t1.nii.gz', 'b0.nii.gz'],
    ['701bda02bb769655c7d4a9b1df2b73a6',
     'e4b741f0c77b6039e67abb2885c97a78'],
    data_size="12MB",
    doc="Download t1 and b0 volumes from the same session")

fetch_mni_template = _make_fetcher(
    "fetch_mni_template",
    pjoin(dipy_home, 'mni_template'),
    'https://digital.lib.washington.edu/researchworks/bitstream/handle/1773/33312/',
    ['COPYING',
     'mni_icbm152_t2_tal_nlin_asym_09a.nii',
     'mni_icbm152_t1_tal_nlin_asym_09a.nii'],
    ['COPYING',
     'mni_icbm152_t2_tal_nlin_asym_09a.nii',
     'mni_icbm152_t1_tal_nlin_asym_09a.nii'],
    ['6e2168072e80aa4c0c20f1e6e52ec0c8',
     'f41f2e1516d880547fbf7d6a83884f0d',
     '1ea8f4f1e41bc17a94602e48141fdbc8'],
    doc = "Fetch the MNI T2 and T1 template files",
    data_size="35MB")

fetch_scil_b0 = _make_fetcher(
    "fetch_scil_b0",
    dipy_home,
    'http://scil.dinf.usherbrooke.ca/wp-content/data/',
    ['datasets_multi-site_all_companies.zip'],
    ['datasets_multi-site_all_companies.zip'],
    None,
    data_size="9.2MB",
    doc="Download b=0 datasets from multiple MR systems (GE, Philips, Siemens) and different magnetic fields (1.5T and 3T)",
    unzip=True)

fetch_viz_icons = _make_fetcher("fetch_viz_icons",
                                pjoin(dipy_home, "icons"),
                                'https://dl.dropboxusercontent.com/u/2481924/',
                                ['icomoon.tar.gz'],
                                ['icomoon.tar.gz'],
                                ['94a07cba06b4136b6687396426f1e380'],
                                data_size="12KB",
                                doc="Download icons for dipy.viz",
                                unzip=True)

fetch_bundles_2_subjects = _make_fetcher(
    "fetch_bundles_2_subjects",
    pjoin(dipy_home, 'exp_bundles_and_maps'),
    'https://dl.dropboxusercontent.com/u/2481924/',
    ['bundles_2_subjects.tar.gz'],
    ['bundles_2_subjects.tar.gz'],
    ['97756fbef11ce2df31f1bedf1fc7aac7'],
    data_size="234MB",
    doc="Download 2 subjects from the SNAIL dataset with their bundles",
    unzip=True)


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


def read_isbi2013_2shell():
    """ Load ISBI 2013 2-shell synthetic dataset

    Returns
    -------
    img : obj,
        Nifti1Image
    gtab : obj,
        GradientTable
    """
    files, folder = fetch_isbi2013_2shell()
    fraw = pjoin(folder, 'phantom64.nii.gz')
    fbval = pjoin(folder, 'phantom64.bval')
    fbvec = pjoin(folder, 'phantom64.bvec')
    bvals, bvecs = read_bvals_bvecs(fbval, fbvec)
    gtab = gradient_table(bvals, bvecs)
    img = nib.load(fraw)
    return img, gtab


def read_sherbrooke_3shell():
    """ Load Sherbrooke 3-shell HARDI dataset

    Returns
    -------
    img : obj,
        Nifti1Image
    gtab : obj,
        GradientTable
    """
    files, folder = fetch_sherbrooke_3shell()
    fraw = pjoin(folder, 'HARDI193.nii.gz')
    fbval = pjoin(folder, 'HARDI193.bval')
    fbvec = pjoin(folder, 'HARDI193.bvec')
    bvals, bvecs = read_bvals_bvecs(fbval, fbvec)
    gtab = gradient_table(bvals, bvecs)
    img = nib.load(fraw)
    return img, gtab


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


def read_stanford_hardi():
    """ Load Stanford HARDI dataset

    Returns
    -------
    img : obj,
        Nifti1Image
    gtab : obj,
        GradientTable
    """
    files, folder = fetch_stanford_hardi()
    fraw = pjoin(folder, 'HARDI150.nii.gz')
    fbval = pjoin(folder, 'HARDI150.bval')
    fbvec = pjoin(folder, 'HARDI150.bvec')
    bvals, bvecs = read_bvals_bvecs(fbval, fbvec)
    gtab = gradient_table(bvals, bvecs)
    img = nib.load(fraw)
    return img, gtab


def read_stanford_t1():
    files, folder = fetch_stanford_t1()
    f_t1 = pjoin(folder, 't1.nii.gz')
    img = nib.load(f_t1)
    return img


def read_stanford_pve_maps():
    files, folder = fetch_stanford_pve_maps()
    f_pve_csf = pjoin(folder, 'pve_csf.nii.gz')
    f_pve_gm = pjoin(folder, 'pve_gm.nii.gz')
    f_pve_wm = pjoin(folder, 'pve_wm.nii.gz')
    img_pve_csf = nib.load(f_pve_csf)
    img_pve_gm = nib.load(f_pve_gm)
    img_pve_wm = nib.load(f_pve_wm)
    return (img_pve_csf, img_pve_gm, img_pve_wm)


def read_taiwan_ntu_dsi():
    """ Load Taiwan NTU dataset

    Returns
    -------
    img : obj,
        Nifti1Image
    gtab : obj,
        GradientTable
    """
    files, folder = fetch_taiwan_ntu_dsi()
    fraw = pjoin(folder, 'DSI203.nii.gz')
    fbval = pjoin(folder, 'DSI203.bval')
    fbvec = pjoin(folder, 'DSI203.bvec')
    bvals, bvecs = read_bvals_bvecs(fbval, fbvec)
    bvecs[1:] = (bvecs[1:] /
                 np.sqrt(np.sum(bvecs[1:] * bvecs[1:], axis=1))[:, None])
    gtab = gradient_table(bvals, bvecs)
    img = nib.load(fraw)
    return img, gtab


def read_syn_data():
    """ Load t1 and b0 volumes from the same session

    Returns
    -------
    t1 : obj,
        Nifti1Image
    b0 : obj,
        Nifti1Image
    """
    files, folder = fetch_syn_data()
    t1_name = pjoin(folder, 't1.nii.gz')
    b0_name = pjoin(folder, 'b0.nii.gz')
    t1 = nib.load(t1_name)
    b0 = nib.load(b0_name)
    return t1, b0

mni_notes = \
    """
    Notes
    -----
    The templates were downloaded from the MNI (McGill University)
    `website <http://www.bic.mni.mcgill.ca/ServicesAtlases/ICBM152NLin2009>`_
    in July 2015.

    The following publications should be referenced when using these templates:

    .. [1] VS Fonov, AC Evans, K Botteron, CR Almli, RC McKinstry, DL Collins
           and BDCG, Unbiased average age-appropriate atlases for pediatric
           studies, NeuroImage, 54:1053-8119,
           DOI: 10.1016/j.neuroimage.2010.07.033

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
    bvals : list or int
        The b-values to read from file (200, 400, 1000, 2000, 3000).

    Returns
    -------
    gtab : a GradientTable class instance
    img : nibabel.Nifti1Image

    """
    files, folder = fetch_cenir_multib(with_raw=False)
    if bvals is None:
        bvals = [200, 400, 1000, 2000, 3000]
    if isinstance(bvals, int):
        bvals = [bvals]
    file_dict = {200: {'DWI': pjoin(folder, '4D_dwi_eddycor_B200.nii.gz'),
                       'bvals': pjoin(folder, 'dwi_bvals_B200'),
                       'bvecs': pjoin(folder, 'dwi_bvecs_B200')},
                 400: {'DWI': pjoin(folder, '4D_dwieddycor_B400.nii.gz'),
                       'bvals': pjoin(folder, 'bvals_B400'),
                       'bvecs': pjoin(folder, 'bvecs_B400')},
                 1000: {'DWI': pjoin(folder, '4D_dwieddycor_B1000.nii.gz'),
                        'bvals': pjoin(folder, 'bvals_B1000'),
                        'bvecs': pjoin(folder, 'bvecs_B1000')},
                 2000: {'DWI': pjoin(folder, '4D_dwieddycor_B2000.nii.gz'),
                        'bvals': pjoin(folder, 'bvals_B2000'),
                        'bvecs': pjoin(folder, 'bvecs_B2000')},
                 3000: {'DWI': pjoin(folder, '4D_dwieddycor_B3000.nii.gz'),
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
    return (nib.Nifti1Image(np.concatenate(data, -1), aff),
            gradient_table(bval_list, np.concatenate(bvec_list, -1)))


CENIR_notes = \
    """
    Notes
    -----
    Details of the acquisition and processing, and additional meta-data are
    available through `UW researchworks <https://digital.lib.washington.edu/researchworks/handle/1773/33311>`_
    """

fetch_cenir_multib.__doc__ += CENIR_notes
read_cenir_multib.__doc__ += CENIR_notes


def read_viz_icons(style='icomoon', fname='infinity.png'):
    """ Read specific icon from specific style

    Parameters
    ----------
    style : str
        Current icon style. Default is icomoon.
    fname : str
        Filename of icon. This should be found in folder HOME/.dipy/style/.
        Default is infinity.png.

    Returns
    --------
    path : str
        Complete path of icon.

    """

    folder = pjoin(dipy_home, 'icons', style)
    return pjoin(folder, fname)


def read_bundles_2_subjects(subj_id='subj_1', metrics=['fa'],
                            bundles=['af.left', 'cst.right', 'cc_1']):
    r""" Read images and streamlines from 2 subjects of the SNAIL dataset

    Parameters
    ----------
    subj_id : string
        Either ``subj_1`` or ``subj_2``.
    metrics : list
        Either ['fa'] or ['t1'] or ['fa', 't1']
    bundles : list
        Example ['af.left', 'cst.right', 'cc_1']. See all the available bundles
        in the ``exp_bundles_maps/bundles_2_subjects`` directory of your
        ``$HOME/.dipy`` folder.

    Returns
    -------
    dix : dict
        Dictionary with data of the metrics and the bundles as keys.

    Notes
    -----
    If you are using these datasets please cite the following publications.

    References
    ----------

    .. [1] Renaud, E., M. Descoteaux, M. Bernier, E. Garyfallidis,
    K. Whittingstall, "Morphology of thalamus, LGN and optic radiation do not
    influence EEG alpha waves", Plos One (under submission), 2015.

    .. [2] Garyfallidis, E., O. Ocegueda, D. Wassermann,
    M. Descoteaux. Robust and efficient linear registration of fascicles in the
    space of streamlines , Neuroimage, 117:124-140, 2015.

    """

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
