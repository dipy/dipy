.. _basic_flow:

====================================================
Get Started with a basic DIPY Workflow command lines
====================================================

In a terminal, Let's get some data. This is multi-shell dataset which was
kindly provided by Hansen and Jespersen (more details about the data are
provided in their paper [Hansen2016]_)::

    dipy_fetch cfin_multib

Create brain mask::

    dipy_median_otsu dwi.nii.gz --out_dir out_work/

Create stopping criteria::

    dipy_fit_dti dwi.nii.gz dwi.bval dwi.bvec out_work/brain_mask.nii.gz --out_dir out_work/

Create peaks::

    dipy_fit_csd dwi.nii.gz dwi.bval dwi.bvec out_work/brain_mask.nii.gz --out_dir out_work/

Create seeding mask (this creates seed_mask in new folder)::

    dipy_mask out_work/fa.nii.gz 0.4 --out_dir out_work/ --out_mask seed_mask.nii.gz

To avoid that we ommit the --out_dir option::

    dipy_mask out_work/fa.nii.gz 0.4 --out_mask seed_mask.nii.gz

Create tracks using peaks::

    dipy_track_det out_work/peaks.pam5 out_work/fa.nii.gz out_work/seed_mask.nii.gz --out_tractogram 'out_work/tracks_from_peaks.trk'

Create tracks using sh cone::

    dipy_track_det peaks.pam5 fa.nii.gz seed_mask.nii.gz --out_tractogram 'tracks_from_sh.trk' --use_sh


For more information about each command line, you can got to :ref:`workflows_reference`.