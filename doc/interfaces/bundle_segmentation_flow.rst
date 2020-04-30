.. _bundle_segmentation_flow:

===============================================
White Matter Bundle Extraction with RecoBundles
===============================================

This tutorial explains how we can use RecoBundles [1]_ to extract
bundles from input tractograms.


First we need to download streamline atlas [2]_ with 30 bundles in MNI space from:

    `<https://figshare.com/articles/Atlas_of_30_Human_Brain_Bundles_in_MNI_space/12089652>`_

Let's say we have an input target tractogram named ``target.trk`` and atlas we
download named ``whole_brain_MNI.trk``.

Visualizing target and atlas tractograms before registration::

    dipy_horizon "target.trk" "whole_brain_MNI.trk" --random_color

.. figure:: https://github.com/dipy/dipy_data/blob/master/tractograms_initial.png?raw=true
    :width: 70 %
    :alt: alternate text
    :align: center

    Atlas and target tractograms before registration.

------------------------------------
Streamline-Based Linear Registration
------------------------------------

For extracting bundles from tractogram we first need our target tractogram to
be in common space (atlas space). We will register target tractogram to
model atlas’ space using streamline-based linear registeration (SLR) [3]_.

Following workflows require two positional input arguments; ``Static`` and
``Moving`` .trk files. ``Static`` would be the ``atlas``  and ``Moving`` would be
our ``target``  tractogram.

Run the following workflow::

    dipy_slr "whole_brain_MNI.trk" "target.trk" --force

SLR workflow will save transformed tractogram as ``moved.trk``.

Visualizing target and atlas tractograms after registration::

    dipy_horizon "moved.trk" "whole_brain_MNI.trk" --random_color

.. figure:: https://github.com/dipy/dipy_data/blob/master/tractograms_after_registration.png?raw=true
    :width: 70 %
    :alt: alternate text
    :align: center

    Atlas and target tractograms after registration.

-----------
Recobundles
-----------

Create an ``out_dir`` folder (eg: rb_output)::

    mkdir rb_output

For Recobundles workflow, we will be using 30 model bundles downloaded earlier.
Run the following workflow::

    dipy_recobundles "moved.trk" "bundles/*.trk" --force --mix_names --out_dir "rb_output"

This workflow will extract 30 bundles from the tractogram.
Example of extracted Left Arcuate fasciculus (AF_L) bundle:

.. figure:: https://github.com/dipy/dipy_data/blob/master/AF_L_rb.png?raw=true
    :width: 70 %
    :alt: alternate text
    :align: center

    Extracted Left Arcuate fasciculus (AF_L) from input tractogram

Example of extracted Left Arcuate fasciculus (AF_L) bundle visualized along
model AF_L bundle used as reference in RecoBundles:

.. figure:: https://github.com/dipy/dipy_data/blob/master/AF_L_rb_with_model.png?raw=true
    :width: 70 %
    :alt: alternate text
    :align: center

    Extracted Left Arcuate fasciculus (AF_L) in Pink and model AF_L bundle in green color.

Output of recobundles will be in native space. To get bundles in subject's
original space, run following commands::

    mkdir org_output

    dipy_labelsbundles 'target.trk' 'rb_output/*.npy' --mix_names --out_dir "org_output"



For more information about each command line, you can go to
`<https://github.com/dipy/dipy/blob/master/dipy/workflows/segment.py>`_

If you are using any of these commands do cite the relevant papers and
DIPY [4]_.

.. [1] Garyfallidis et al. Recognition of white matter bundles using local and
    global streamline-based registration and clustering, Neuroimage, 2017

.. [2] Yeh F.C., Panesar S., Fernandes D., Meola A., Yoshino M.,
    Fernandez-Miranda J.C., Vettel J.M., Verstynen T.
    Population-averaged atlas of the macroscale human structural
    connectome and its network topology.
    Neuroimage, 2018.

.. [3] Garyfallidis et al., “Robust and efficient linear registration of
    white-matter fascicles in the space of streamlines”, Neuroimage,
    117:124-140, 2015.

.. [4] Garyfallidis, E., M. Brett, B. Amirbekian, A. Rokem,
    S. Van Der Walt, M. Descoteaux, and I. Nimmo-Smith.
    "DIPY, a library for the analysis of diffusion MRI data".
    Frontiers in Neuroinformatics, 1-18, 2014.
