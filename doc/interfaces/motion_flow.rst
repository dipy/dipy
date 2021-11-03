.. _motion_correction_flow:

=================================
Between-Volumes Motion Correction
=================================

This tutorial walks through the steps to perform Between-Volumes Motion
Correction using DIPY.

You can try this using your own data; we will be using the data in DIPY.
You can check how to :ref:`fetch the DIPY data<data_fetch>`.

-----------------
Motion Correction
-----------------

During a dMRI acquisition, the subject motion inevitable. This motion implies
a misalignment between N volumes on a dMRI dataset. A common way to solve this
issue is to perform a registration on each acquired volume to a
reference b = 0. [JenkinsonSmith01]_

This preprocessing is an highly recommended step that should be executed before
any dMRI dataset analysis.

We will use the ``sherbrooke_3shell`` dataset in DIPY to showcase the motion
correction process. As with any other workflow in DIPY, you can also use your
own data!

You may want to create an output directory to save the resulting corrected dataset::

    mkdir motion_output

To run the motion correction method, we need to specify the path of the input
data, b-vectors file and b-values file. This path may contain wildcards to process
multiple inputs at once. You can also specify the optional arguments. In this case,
we will be just specifying the output directory (``out_dir``). For more information,
use the help option (-h) to obtain all optional arguments.

To run the motion correction on the data it suffices to execute the
``dipy_correct_motion`` command, e.g.::

    dipy_correct_motion data/sherbrooke_3shell/HARDI193.nii.gz  data/sherbrooke_3shell/HARDI193.bval data/sherbrooke_3shell/HARDI193.bvec --out_dir "motion_output"

This command will apply the motion correction procedure to the input MR image
and write the artefact-free result to the ``motion_output`` directory.

In case no output directory is specified, the corrected output volume
is saved to the current directory by default.

References
----------
.. [JenkinsonSmith01] Jenkinson, M., Smith, S., 2001. A global optimisation
   method for robust affine registration of brain images. Med Image Anal 5
   (2), 143–56.