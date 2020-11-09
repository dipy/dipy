.. _bundle_segmentation_flow:

=================================================
White Matter Bundle Segmentation with RecoBundles
=================================================

This tutorial explains how we can use RecoBundles [Garyfallidis17]_ to extract
bundles from input tractograms.


First, we need to download a reference streamline atlas. Here, we downloaded an atlas with
30 bundles in MNI space [Yeh18]_ from:

    `<https://figshare.com/articles/Atlas_of_30_Human_Brain_Bundles_in_MNI_space/12089652>`_

For this tutorial, you can use your own tractography data or you can download a single subject
tractogram from:

    `<https://figshare.com/articles/hcp_tractogram_zip/7013003>`_

Let's say we have an input target tractogram named ``streamlines.trk`` and the atlas we
downloaded, named ``whole_brain_MNI.trk``.

Visualizing the target and atlas tractograms before registration::

    dipy_horizon "streamlines.trk" "whole_brain_MNI.trk" --random_color

.. figure:: https://github.com/dipy/dipy_data/blob/master/tractograms_initial.png?raw=true
    :width: 70 %
    :alt: alternate text
    :align: center

    Atlas and target tractograms before registration.

------------------------------------
Streamline-Based Linear Registration
------------------------------------

To extract the bundles from the tractogram, we first need move our target tractogram to
be in the same space as the atlas (MNI, in this case). We can directly register the target tractogram to
the space of the atlas, using streamline-based linear registration (SLR) [Garyfallidis15]_.

The following workflows require two positional input arguments; ``static`` and
``moving`` .trk files. In our case, the ``static`` input is the atlas and the ``moving`` is
our ``target``  tractogram (``streamlines.trk``).

Run the following workflow::

    dipy_slr "whole_brain_MNI.trk" "streamlines.trk" --force

Per default, the SLR workflow will save a transformed tractogram as ``moved.trk``.

Visualizing the target and atlas tractograms after registration::

    dipy_horizon "moved.trk" "whole_brain_MNI.trk" --random_color

.. figure:: https://github.com/dipy/dipy_data/blob/master/tractograms_after_registration.png?raw=true
    :width: 70 %
    :alt: alternate text
    :align: center

    Atlas and target tractograms after registration.

-----------
RecoBundles
-----------

Create an ``out_dir`` folder (e.g., ``rb_output``), into which output will be placed::

    mkdir rb_output

For the RecoBundles workflow, we will use the 30 model bundles downloaded earlier.
Run the following workflow::

    dipy_recobundles "moved.trk" "bundles/*.trk" --force --mix_names --out_dir "rb_output"

This workflow will extract 30 bundles from the tractogram.
Example of extracted Left Arcuate fasciculus (AF_L) bundle (visualized with ``dipy_horizon``):

.. figure:: https://github.com/dipy/dipy_data/blob/master/AF_L_rb.png?raw=true
    :width: 70 %
    :alt: alternate text
    :align: center

    Extracted Left Arcuate fasciculus (AF_L) from input tractogram

Example of extracted Left Arcuate fasciculus (AF_L) bundle visualized along
with the model AF_L bundle used as reference in RecoBundles:

.. figure:: https://github.com/dipy/dipy_data/blob/master/AF_L_rb_with_model.png?raw=true
    :width: 70 %
    :alt: alternate text
    :align: center

    Extracted Left Arcuate fasciculus (AF_L) in pink and model AF_L bundle in green color.

Output of RecoBundles will be in native space. To get bundles in subject's
original space, run following commands::

    mkdir org_output

    dipy_labelsbundles 'streamlines.trk' 'rb_output/*.npy' --mix_names --out_dir "org_output"



For more information about each command line, please visit DIPY website `<https://dipy.org/>`_ .

If you are using any of these commands please be sure to cite the relevant papers and
DIPY [Garyfallidis14]_.

----------
References
----------

.. [Garyfallidis17] Garyfallidis et al. Recognition of white matter bundles using local and
    global streamline-based registration and clustering, Neuroimage, 2017

.. [Yeh18] Yeh F.C., Panesar S., Fernandes D., Meola A., Yoshino M.,
    Fernandez-Miranda J.C., Vettel J.M., Verstynen T.
    Population-averaged atlas of the macroscale human structural
    connectome and its network topology.
    Neuroimage, 2018.

.. [Garyfallidis15] Garyfallidis et al., “Robust and efficient linear registration of
    white-matter fascicles in the space of streamlines”, Neuroimage,
    117:124-140, 2015.

.. [Garyfallidis14] Garyfallidis, E., M. Brett, B. Amirbekian, A. Rokem,
    S. Van Der Walt, M. Descoteaux, and I. Nimmo-Smith.
    "DIPY, a library for the analysis of diffusion MRI data".
    Frontiers in Neuroinformatics, 1-18, 2014.
