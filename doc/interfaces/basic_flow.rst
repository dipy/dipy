.. _basic_flow:

====================================================
Get Started with a basic DIPY Workflow command lines
====================================================

In a terminal, let's get some data. This is multi-shell dataset which was
kindly provided by Hansen and Jespersen (more details about the data are
provided in their paper [Hansen2016]_). First let's create a folder::

    $ mkdir data_folder

Download the data in the data_folder::

    $ dipy_fetch cfin_multib --out_dir data_folder

Move to the folder with the data::

    $ cd data_folder/cfin_multib
    $ ls or dir

    dwi.bval
    dwi.bvec
    dwi.nii
    T1.nii

Investigate the bvals::

    $ dipy_info dwi.nii



Visualize the diffusion data::

    dipy_horizon __DTI_AX_ep2d_2_5_iso_33d_20141015095334_4.nii

.. figure:: https://github.com/dipy/dipy_data/blob/master/cfin_basic1.png?raw=true
    :width: 70 %
    :alt: alternate text
    :align: center

    Visualization of a slice from the first volume of the diffusion data

Create brain mask::

    dipy_median_otsu dwi.nii.gz --out_dir out_work/

Visualize the mask::

    dipy_horizon out_work/brain_mask.nii.gz

.. figure:: https://github.com/dipy/dipy_data/blob/master/cfin_basic2.png?raw=true
    :width: 70 %
    :alt: alternate text
    :align: center

    Visualization of a slice from the first volume of brain mask


Perform DTI::

    dipy_fit_dti dwi.nii.gz dwi.bval dwi.bvec out_work/brain_mask.nii.gz --out_dir out_work/

The default options of the script generate the following files ad.nii.gz, evecs.nii.gz, md.nii.gz,
rgb.nii.gz, fa.nii.gz, mode.nii.gz, tensors.nii.gz, evals.nii.gz, ga.nii.gz and rd.nii.gz.

Visualize DEC map::

    dipy_horizon out_work/rgb.nii.gz

.. figure:: https://github.com/dipy/dipy_data/blob/master/cfin_basic3.png?raw=true
    :width: 70 %
    :alt: alternate text
    :align: center

    Visualization of a slice from the first volume of DEC image

Create peaks::

    dipy_fit_csa dwi.nii.gz dwi.bval dwi.bvec out_work/brain_mask.nii.gz --out_dir out_work/

Create seeding mask (this creates seed_mask in new folder)::

    dipy_mask out_work/fa.nii.gz 0.4 --out_dir out_work/ --out_mask seed_mask.nii.gz

To avoid that we ommit the --out_dir option::

    dipy_mask out_work/fa.nii.gz 0.4 --out_mask seed_mask.nii.gz

Create tracks using peaks::

    dipy_track_det out_work/peaks.pam5 out_work/fa.nii.gz out_work/seed_mask.nii.gz --out_tractogram 'out_work/tracks_from_peaks.trk'

Create tracks using sh cone::

    dipy_track_det peaks.pam5 fa.nii.gz seed_mask.nii.gz --out_tractogram 'tracks_from_sh.trk' --use_sh


For more information about each command line, you can got to :ref:`workflows_reference`.