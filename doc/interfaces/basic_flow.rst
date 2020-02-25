.. _basic_flow:

====================================================
A basic DIPY command line series
====================================================

Using a terminal, let's download a dataset. This is multi-shell dataset, which was
kindly provided by Hansen and Jespersen (more details about the data are
provided in their paper [Hansen2016]_). First let's create a folder::

    $ mkdir data_folder

Download the data in the data_folder::

    $ dipy_fetch cfin_multib --out_dir data_folder

Move to the folder with the data::

    $ cd data_folder/cfin_multib
    $ ls

    __DTI_AX_ep2d_2_5_iso_33d_20141015095334_4.bval
    __DTI_AX_ep2d_2_5_iso_33d_20141015095334_4.bvec
    __DTI_AX_ep2d_2_5_iso_33d_20141015095334_4.nii
    T1.nii

Let's rename the long filenames to something that is easier to read::

    $ mv __DTI_AX_ep2d_2_5_iso_33d_20141015095334_4.bval dwi.bval
    $ mv __DTI_AX_ep2d_2_5_iso_33d_20141015095334_4.bvec dwi.bvec
    $ mv __DTI_AX_ep2d_2_5_iso_33d_20141015095334_4.nii dwi.nii

We can use ``dipy_info`` to check the bval and nii files ::

    $ dipy_info *.bval *.nii

    INFO:-----------------
    INFO:Looking at T1.nii
    INFO:-----------------
    INFO:Data size (256, 256, 176)
    INFO:Data type uint16
    INFO:Data min 0 max 888 avg 62.940581408413976
    INFO:2nd percentile 0.0 98th percentile 377.0
    INFO:Native coordinate system PSR
    INFO:Affine Native to RAS matrix
    [[   0.       0.01     1.     -89.569]
    [  -1.       0.       0.     138.451]
    [   0.       1.      -0.01  -131.289]
    [   0.       0.       0.       1.   ]]
    INFO:Voxel size [1. 1. 1.]
    INFO:---------------------------------------------------------
    INFO:Looking at __DTI_AX_ep2d_2_5_iso_33d_20141015095334_4.nii
    INFO:---------------------------------------------------------
    INFO:Data size (96, 96, 19, 496)
    INFO:Data type int16
    INFO:Data min 0 max 1257 avg 58.62918037280702 of vol 0
    INFO:2nd percentile 0.0 98th percentile 234.0 of vol 0
    INFO:Native coordinate system LAS
    INFO:Affine Native to RAS matrix
    [[  -2.498    0.084    0.067  113.641]
    [   0.069    2.451   -0.488 -104.142]
    [   0.082    0.486    2.451  -31.504]
    [   0.       0.       0.       1.   ]]
    INFO:Voxel size [2.5 2.5 2.5]

    (base) C:\Users\elef\Data\examples\data_folder\cfin_multib>dipy_info *.bval *.nii
    INFO:----------------------------------------------------------
    INFO:Looking at __DTI_AX_ep2d_2_5_iso_33d_20141015095334_4.bval
    INFO:----------------------------------------------------------
    INFO:b-values
    [   0.  200.  200.  200.  200.  200.  200.  200.  200.  200.  200.  200.
    200.  200.  200.  200.  200.  200.  200.  200.  200.  200.  200.  200.
    200.  200.  200.  200.  200.  200.  200.  200.  200.  200.  400.  400.
    400.  400.  400.  400.  400.  400.  400.  400.  400.  400.  400.  400.
    400.  400.  400.  400.  400.  400.  400.  400.  400.  400.  400.  400.
    400.  400.  400.  400.  400.  400.  400.  600.  600.  600.  600.  600.
    600.  600.  600.  600.  600.  600.  600.  600.  600.  600.  600.  600.
    600.  600.  600.  600.  600.  600.  600.  600.  600.  600.  600.  600.
    600.  600.  600.  600.  800.  800.  800.  800.  800.  800.  800.  800.
    800.  800.  800.  800.  800.  800.  800.  800.  800.  800.  800.  800.
    800.  800.  800.  800.  800.  800.  800.  800.  800.  800.  800.  800.
    800. 1000. 1000. 1000. 1000. 1000. 1000. 1000. 1000. 1000. 1000. 1000.
    1000. 1000. 1000. 1000. 1000. 1000. 1000. 1000. 1000. 1000. 1000. 1000.
    1000. 1000. 1000. 1000. 1000. 1000. 1000. 1000. 1000. 1000. 1200. 1200.
    1200. 1200. 1200. 1200. 1200. 1200. 1200. 1200. 1200. 1200. 1200. 1200.
    1200. 1200. 1200. 1200. 1200. 1200. 1200. 1200. 1200. 1200. 1200. 1200.
    1200. 1200. 1200. 1200. 1200. 1200. 1200. 1400. 1400. 1400. 1400. 1400.
    1400. 1400. 1400. 1400. 1400. 1400. 1400. 1400. 1400. 1400. 1400. 1400.
    1400. 1400. 1400. 1400. 1400. 1400. 1400. 1400. 1400. 1400. 1400. 1400.
    1400. 1400. 1400. 1400. 1600. 1600. 1600. 1600. 1600. 1600. 1600. 1600.
    1600. 1600. 1600. 1600. 1600. 1600. 1600. 1600. 1600. 1600. 1600. 1600.
    1600. 1600. 1600. 1600. 1600. 1600. 1600. 1600. 1600. 1600. 1600. 1600.
    1600. 1800. 1800. 1800. 1800. 1800. 1800. 1800. 1800. 1800. 1800. 1800.
    1800. 1800. 1800. 1800. 1800. 1800. 1800. 1800. 1800. 1800. 1800. 1800.
    1800. 1800. 1800. 1800. 1800. 1800. 1800. 1800. 1800. 1800. 2000. 2000.
    2000. 2000. 2000. 2000. 2000. 2000. 2000. 2000. 2000. 2000. 2000. 2000.
    2000. 2000. 2000. 2000. 2000. 2000. 2000. 2000. 2000. 2000. 2000. 2000.
    2000. 2000. 2000. 2000. 2000. 2000. 2000. 2200. 2200. 2200. 2200. 2200.
    2200. 2200. 2200. 2200. 2200. 2200. 2200. 2200. 2200. 2200. 2200. 2200.
    2200. 2200. 2200. 2200. 2200. 2200. 2200. 2200. 2200. 2200. 2200. 2200.
    2200. 2200. 2200. 2200. 2400. 2400. 2400. 2400. 2400. 2400. 2400. 2400.
    2400. 2400. 2400. 2400. 2400. 2400. 2400. 2400. 2400. 2400. 2400. 2400.
    2400. 2400. 2400. 2400. 2400. 2400. 2400. 2400. 2400. 2400. 2400. 2400.
    2400. 2600. 2600. 2600. 2600. 2600. 2600. 2600. 2600. 2600. 2600. 2600.
    2600. 2600. 2600. 2600. 2600. 2600. 2600. 2600. 2600. 2600. 2600. 2600.
    2600. 2600. 2600. 2600. 2600. 2600. 2600. 2600. 2600. 2600. 2800. 2800.
    2800. 2800. 2800. 2800. 2800. 2800. 2800. 2800. 2800. 2800. 2800. 2800.
    2800. 2800. 2800. 2800. 2800. 2800. 2800. 2800. 2800. 2800. 2800. 2800.
    2800. 2800. 2800. 2800. 2800. 2800. 2800. 3000. 3000. 3000. 3000. 3000.
    3000. 3000. 3000. 3000. 3000. 3000. 3000. 3000. 3000. 3000. 3000. 3000.
    3000. 3000. 3000. 3000. 3000. 3000. 3000. 3000. 3000. 3000. 3000. 3000.
    3000. 3000. 3000. 3000.]
    INFO:Total number of b-values 496
    INFO:Number of gradient shells 15
    INFO:Number of b0s 1 (b0_thr 50)

    INFO:-----------------
    INFO:Looking at T1.nii
    INFO:-----------------
    INFO:Data size (256, 256, 176)
    INFO:Data type uint16
    INFO:Data min 0 max 888 avg 62.940581408413976
    INFO:2nd percentile 0.0 98th percentile 377.0
    INFO:Native coordinate system PSR
    INFO:Affine Native to RAS matrix
    [[   0.       0.01     1.     -89.569]
    [  -1.       0.       0.     138.451]
    [   0.       1.      -0.01  -131.289]
    [   0.       0.       0.       1.   ]]
    INFO:Voxel size [1. 1. 1.]
    INFO:---------------------------------------------------------
    INFO:Looking at __DTI_AX_ep2d_2_5_iso_33d_20141015095334_4.nii
    INFO:---------------------------------------------------------
    INFO:Data size (96, 96, 19, 496)
    INFO:Data type int16
    INFO:Data min 0 max 1257 avg 58.62918037280702 of vol 0
    INFO:2nd percentile 0.0 98th percentile 234.0 of vol 0
    INFO:Native coordinate system LAS
    INFO:Affine Native to RAS matrix
    [[  -2.498    0.084    0.067  113.641]
    [   0.069    2.451   -0.488 -104.142]
    [   0.082    0.486    2.451  -31.504]
    [   0.       0.       0.       1.   ]]
    INFO:Voxel size [2.5 2.5 2.5]


We can visualize the data using ``dipy_horizon`` ::

    $ dipy_horizon dwi.nii

.. figure:: https://github.com/dipy/dipy_data/blob/master/cfin_basic1.png?raw=true
    :width: 70 %
    :alt: alternate text
    :align: center

    Visualization of a slice from the first volume of the diffusion data

We can use ``dipy_median_otsu`` to build a brain mask for the diffusion data::

    dipy_median_otsu dwi.nii --out_dir --median_radius 2 --numpass 1 --vol_idx 0 1 out_work/

Visualize the mask using ``dipy_horizon``::

    $ dipy_horizon out_work/brain_mask.nii.gz

.. figure:: https://github.com/dipy/dipy_data/blob/master/cfin_basic2.png?raw=true
    :width: 70 %
    :alt: alternate text
    :align: center

    Visualization of a slice from the first volume of brain mask


Perform DTI using ``dipy_fit_dti`. The input of this function is the DWI data, b-values and b-vector files and the brain mask that we calculated in the previous step::

    $ dipy_fit_dti dwi.nii dwi.bval dwi.bvec out_work/brain_mask.nii.gz --out_dir out_work/

The default options of the script generate the following files ad.nii.gz, evecs.nii.gz, md.nii.gz,
rgb.nii.gz, fa.nii.gz, mode.nii.gz, tensors.nii.gz, evals.nii.gz, ga.nii.gz and rd.nii.gz.

Visualize DEC map::

    $ dipy_horizon out_work/rgb.nii.gz

.. figure:: https://github.com/dipy/dipy_data/blob/master/cfin_basic3.png?raw=true
    :width: 70 %
    :alt: alternate text
    :align: center

    Visualization of a slice from the first volume of DEC image

We can now move to more advanced reconstruction models. One of the fastest we can use is Constant Solid Angle (CSA) ::

    $ dipy_fit_csa dwi.nii dwi.bval dwi.bvec out_work/brain_mask.nii.gz --out_dir out_work/

Now, to move into doing some tracking we will need some seeds. We can generate seeds in the following way ::

    $ dipy_mask out_work/fa.nii.gz 0.4 --out_dir out_work/ --out_mask seed_mask.nii.gz

Create tracks using peaks::

    dipy_track_det out_work/peaks.pam5 out_work/fa.nii.gz out_work/seed_mask.nii.gz --out_tractogram tracks_from_peaks.trk

We can visualize the result using ``dipy_horizon``. The ``--cluster`` option allows to directly see the clusters of the tractogram::

    $ dipy_horizon out_work/tracts_from_peaks.trk --cluster


.. figure:: https://github.com/dipy/dipy_data/blob/master/some_tracks.png?raw=true
    :width: 70 %
    :alt: alternate text
    :align: center

    Showing tracks from the specific dataset. This dataset contains only a few slices.


Alternatively, we can create deterministic tracks using the maximum value of a spherical harmonics cone::

    $ dipy_track_det peaks.pam5 fa.nii.gz seed_mask.nii.gz --out_tractogram 'tracks_from_sh.trk' --use_sh


For more information about each command line, you see :ref:`workflows_reference`.

These commands are not by any fetch of imagination what we propose as a complete solution to tracking
but a mere introduction to DIPY's command interfaces, medical imaging requires a number of steps that
depend on the goal of the analysis strategy and the questions you are looking to answer.

If you are using any of these commands do cite the relevant papers.

References
----------

.. [Garyfallidis14] Garyfallidis, E., M. Brett, B. Amirbekian, A. Rokem,
    S. Van Der Walt, M. Descoteaux, and I. Nimmo-Smith.
    "DIPY, a library for the analysis of diffusion MRI data".
    Frontiers in Neuroinformatics, 1-18, 2014.

.. [Aganj2010] Aganj I, Lenglet C, Sapiro G, Yacoub E, Ugurbil K, Harel N.
   "Reconstruction of the orientation distribution function in single- and
   multiple-shell q-ball imaging within constant solid angle", Magnetic
   Resonance in Medicine. 2010 Aug;64(2):554-66. doi: 10.1002/mrm.22365

.. [Garyfallidis12] Garyfallidis E., "Towards an accurate brain tractography",
   PhD thesis, University of Cambridge, 2012.

.. [Hansen2016] Hansen, B, Jespersen, SN (2016). "Data for evaluation of fast
   kurtosis strategies, b-value optimization and exploration of diffusion MRI
   contrast". Scientific Data 3: 160072 doi:10.1038/sdata.2016.72
