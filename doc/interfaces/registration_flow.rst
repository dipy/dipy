.. _registration_flow:

============
Registration
============

This tutorial walks through the steps to perform image-based and
streamline-based registration using DIPY. Multiple registration methods are
available in DIPY.

You can try these methods using your own data; we will be using the data in
DIPY. You can check how to :ref:`fetch the DIPY data<data_fetch>`.

------------------
Image Registration
------------------

DIPY's image registration workflow can be used to register a moving image to a
static image by applying different transformations, such as center of mass,
translation, and rigid body or full affine (including translation, rotation,
scaling and shearing) transformations. During such a registration process, the
static image is considered to be the reference, and the moving image is
transformed to the space of the static image. Registration methods use some
sort of optimization process, and a given metric or criterion (like maximizing
the mutual information between the two input images) that is optimized during
the process, to achieve the goal.

The DIPY image registration workflow applies the specified type of
transformation to the input images, and hence, users are expected to choose the
type of transformation that best matches the requirements of their problem.
Alternatively, the workflow allows to perform registration in a progressive
manner. For example, using affine registration with ``progressive`` set to
``True`` will involve center of mass, translation, rigid body and full affine
registration; meanwhile, if ``progressive`` is set to ``False`` for an affine
registration, it will include only center of mass and affine registration. The
progressive registration will be slower but will improve the quality.

We will first create a directory in which to save the transformed image and the
affine matrix (e.g.: ``image_reg_output``)::

    mkdir image_reg_output

To run the image registration, we need to specify the paths to the static image
file, and to the moving image file, followed by the optional arguments. In this
case, we will be specifying the type of registration to be performed
(``transform``) and the output directory (``out_dir``).

To perform center of mass registration, we will call the ``dipy_align_affine``
command with the ``transform`` parameter set to ``com`` e.g.::

    dipy_align_affine <path_to_static_file> <path_to_moving_file> --transform "com" --out_dir "image_reg_output"

This command will save the transformed image and the affine matrix to the
specified output directory.

If we are to use an affine transformation type during the registration process,
we would call the ``dipy_align_affine`` command as, e.g.::

    dipy_align_affine <path_to_static_file> <path_to_moving_file> --transform "affine" --out_dir "affine_reg_output" --out_affine "affine_reg.txt"

This command will apply an affine transformation on the moving image file, and
save the transformed image and the affine matrix to the ``affine_reg_output``
directory.

In case you did not specify the output directory, the transformed image file
and affine matrix would be saved to the current by default. If you did not
specify the name of the output affine matrix, the affine matrix will be saved
to a file named ``affine.txt`` by default, located in the current directory
also by default.

------------------------------------
Symmetric Diffeomorphic Registration
------------------------------------

Symmetric Diffeomorphic Registration is performed using the Symmetric
Normalization (SyN) algorithm proposed by Avants et al. [Avants09]_ (also
implemented in the ANTs software [Avants11]_). It is an optimization technique
that brings the moving image closer to the static image.

Create a directory in which to save the transformed image (e.g.:
``syn_reg_output``)::
    
    mkdir syn_reg_output

To run the symmetric normalization registration method, we need to specify the
paths to the static image file, and to the moving image file, followed by
optional arguments. In this case, we will be specifying the metric (``metric``),
the output directory (``out_dir``) and the file name of the output warped image
(``out_warped``). You can use cc (cross correlation), ssd (sum squared
differences) or em (expectation-maximization) as metrics.

The symmetric diffeomorphic registration method in DIPY is run through the
``dipy_align_syn`` command, e.g.::

    dipy_align_syn <path_to_static_file> <path_to_moving_file> --metric "cc" --out_dir "syn_reg_output" --out_warped "syn_reg_warped.nii.gz"

In case you did not specify the output directory, the transformed files would
be saved in the current directory by default. If you did not specify the file
name of the output warped image, the warped file will be saved as
``warped_moved.nii.gz`` by default.

----------------------
Apply a Transformation
----------------------

We can apply a transformation computed previously to an image. In order to do
so, we need to specify the path of the static image file, moving image file,
and transform map file, which is a text(*.txt) file containing the affine matrix
for the affine case and a nifti file containing the mapping displacement field
in each voxel with this shape (x, y, z, 3, 2) for the diffeomorphic case,
followed by optional arguments. In this case, we will be specifying the
transform type (``transform_type``) and the output directory (``out_dir``).

Create a directory in which to save the transformed files (e.g.:
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