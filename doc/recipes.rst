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


.. dropdown:: How do I convert my Spherical Harmonics from DIPY to MRTRIX3?
   :animate: fade-in-slide-down

   *Available only on master branch. See `this thread <https://github.com/dipy/dipy/discussions/2959#discussioncomment-7481675>`_ for more information.

   .. code-block:: Python

        from dipy.reconst.shm import convert_sh
        convert_sh('dti.nii.gz', 'dti.bval', 'dti.bvec', 'fsl', 'dti.nii.gz')

.. dropdown:: How do I convert my tensors from DIPY to FSL?
   :animate: fade-in-slide-down

   *Available only on master branch

   .. code-block:: Python

       from dipy.io import convert_tensor
       convert_tensor('dti.nii.gz', 'dti.bval', 'dti.bvec', 'fsl', 'dti.nii.gz')


.. include:: links_names.inc