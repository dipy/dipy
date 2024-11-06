.. _recipes:

:octicon:`mortar-board` Recipes
===============================

**How do I do X in DIPY?** This page contains a collection of recipes that will
help you quickly resolve common basic/advanced operation. If you have a question
that is not answered here, please ask on the `dipy discussions`_ or even better,
answer it yourself and contribute to the documentation!


.. dropdown:: How do I convert my tractograms?
   :animate: fade-in-slide-down

   We recommend to look at the tutorials :ref:`Streamline File Formats<sphx_glr_examples_built_file_formats_streamline_formats.py>`

   .. code-block:: Python

        from dipy.io.streamline import load_tractogram, save_tractogram

        # convert from fib to trk
        my_fib_file_path = '/<my_folder>/<my_path>/my_tractogram.fib'
        my_trk_file_path = '/<my_folder>/<my_path>/my_tractogram.trk'
        my_trk = load_tractogram(my_fib_file_path, 'same')
        save_tractogram(my_trk.streamlines, my_trk_file_path, 'same')


.. dropdown:: How do I convert my Spherical Harmonics from MRTRIX3 to DIPY?
   :animate: fade-in-slide-down

   Available from DIPY 1.9.0+. See `this thread <https://github.com/dipy/dipy/discussions/2959#discussioncomment-7481675>`_ for more information.

   .. code-block:: Python

        from dipy.reconst.shm import convert_sh_descoteaux_tournier
        convert_sh_descoteaux_tournier(sh_coeffs)

.. dropdown:: How do I convert my tensors from FSL to DIPY  or MRTRIX3 to DIPY?
   :animate: fade-in-slide-down

   Available with DIPY 1.9.0+

   .. code-block:: Python

        from dipy.reconst.utils import convert_tensors
        from dipy.io.image import load_nifti, save_nifti

        data, affine, img = load_nifti('my_data.nii.gz', return_img=True)
        # convert from FSL to DIPY
        otensor = convert_tensors(data, 'fsl', 'dipy')
        save_nifti(otensor, data, affine, image.header)
        # convert from MRTRIX3 to DIPY
        otensor = convert_tensors(data, 'mrtrix', 'dipy')
        save_nifti(otensor, data, affine, image.header)


.. dropdown:: How do I save/read the output fields from the Symmetric Diffeomorphic Registration?
   :animate: fade-in-slide-down

   The mapping fields is saved in a Nifti file. The data in the file is organized with
   shape (X, Y, Z, 3, 2), such that the forward mapping in each voxel is in
   `data[i, j, k, :, 0]` and the backward mapping in each voxel is in
   `data[i, j, k, :, 1]`.

   The output fields from the Symmetric Diffeomorphic Registration can be saved and
   read using the functions `write_mapping` and `read_mapping` respectively.

   .. code-block:: Python

        from dipy.align import read_mapping, write_mapping
        from dipy.align.imwarp import DiffeomorphicMap, SymmetricDiffeomorphicRegistration

        # Load your static and moving images and setup the registration parameters.
        # Then apply the registration

        sdr = SymmetricDiffeomorphicRegistration(metric, level_iters)
        mapping = sdr.optimize(static, moving)

        # Save the mapping
        write_mapping(mapping,'outputField.nii.gz')

        # If you need to read the mapping back
        mapping = read_mapping('outputField.nii.gz', static_filename, moving_filename)




.. include:: links_names.inc