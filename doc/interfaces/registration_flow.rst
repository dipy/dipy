.. _registration_flow:

============
Registration
============

This tutorial walks through the steps to perform image-based and
streamline-based registration using DIPY.
Multiple registration methods are available in DIPY.

You can try these methods using your own data; we will be using the data in
DIPY. You can check how to :ref:`fetch the DIPY data<data_fetch>`.

-------------------
Affine Registration
-------------------

Affine registration is an optimization strategy used to maximize the mutual
information between two volumes or images. The affine transformation allows
linear coordinate changes that include shearing, scaling, rotation and
translation.

The workflow for the affine registration requires the paths to the static image
file, and to the moving image file. Static image is used as a reference and
moving image is transformed during optimization.

You may want to create an output directory to save the transformed image::

    mkdir affine_reg_output

Run the following command::

    dipy_align_affine <path_to_static_file> <path_to_moving_file> --out_dir "affine_reg_output" --out_affine "affine_reg.txt"

This command will apply affine transformation on the moving image file and save
the transformed image and the affine matrix to the directory
``affine_reg_output``.

In case you did not specify the output directory, the transformed image file
and affine matrix would be saved in the input directory by default. If you did
not specify the name of the output affine matrix, the affine matrix will be
saved to a file named ``affine.txt`` by default, located in the output
directory also by default.

------------------------------------
Symmetric Diffeomorphic Registration
------------------------------------

Symmetric Diffeomorphic Registration is performed using the ``Symmetric
Normalization (SyN)`` algorithm proposed by Avants et al. [Avants09]_ (also
implemented in the ANTs software [Avants11]_). It is an optimization technique
that brings the moving image closer to the static image.

Create a directory in which to save the transformed image (e.g.:
``syn_reg_output``)::
    
    mkdir syn_reg_output

To run the symmetric normalization registration method, we need to specify the
paths to the static image file, and to the moving image file, followed by
optional arguments. In this case, we will be specifying the metric (``metric``),
the output directory (``out_dir``) and the file name of the output warped image
(``out_warped``). You can use cc (Cross Correlation), ssd (Sum Squared
Difference) or em (Expectation-Maximization) as metrics.

We will run the command as::

    dipy_align_syn <path_to_static_file> <path_to_moving_file> --metric "cc" --out_dir "syn_reg_output" --out_warped "syn_reg_warped.nii.gz"

This command will perform symmetric diffeomorphic registration and save it to 
the specified output directory.

In case you did not specify the output directory, the transformed files would
be saved in the input directory by deafult. If you did not specify the file
name of the output warped image, the warped file will be saved as
``warped_moved.nii.gz`` by default.

----------------------
Apply a Transformation
----------------------

We can apply a transformation computed previously to an image. In order to do
so, we need to specify the path of the static image file, moving image file,
and transform map file followed by optional arguments. In this case, we will
be specifying the transform type (``transform_type``) and the output directory
(``out_dir``).

Create a directory in which to save the transfomed files (e.g.:
``transform_reg_output``)::

    mkdir transform_reg_output

For a ``diffeomorphic`` transformation, we would run the command as::

    dipy_apply_transform <path_to_static_file> <path_to_moving_file> <path_to_transform_map_file> --transform_type "diffeomorphic" --out_dir "transform_reg_output"

This command will transform the moving image and save the transformed files 
to the specified output directory.

-----------------------------
Streamline-based Registration
-----------------------------

Streamline-based registration (SLR) [Garyfallidis15]_ is performed to align
bundles of streamlines directly in the space of streamlines. The aim is to
align the moving streamlines with the static streamlines.

The workflow for streamline-based registration requires the paths to the 
static streamlines file, and to the moving streamlines file, followed by
optional arguments. In this case, we will be specifying the number of points
for discretizing each streamline (``nb_pts``) and the output directory
(``out_dir``).

Create a directory in which to save the transformed files (e.g.:
``sl_reg_output``)::

    mkdir sl_reg_output

Then, run the command as::

    dipy_slr <path_to_static_file> <path_to_moving_file> --nb_pts 25 --out_dir "sl_reg_output"

This command will perform streamline-based registration and save the 
transformed files to the specified output directory.

References
----------

.. [Avants09] Avants, B. B., Epstein, C. L., Grossman, M., & Gee, J. C. (2009).
   Symmetric Diffeomorphic Image Registration with Cross-Correlation:
   Evaluating Automated Labeling of Elderly and Neurodegenerative Brain, 12(1),
   26-41.

.. [Avants11] Avants, B. B., Tustison, N., & Song, G. (2011). Advanced
   Normalization Tools (ANTS), 1-35.

.. [Garyfallidis15] Garyfallidis et al., “Robust and efficient linear registration
   of white-matter fascicles in the space of streamlines”, Neuroimage,
   117:124-140, 2015.