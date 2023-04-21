.. _bundlewarp_registration_flow:

=========================================================
Nonrigid White Matter Bundle Registration with BundleWarp
=========================================================

This tutorial explains how we can use BundleWarp [Chandio2023]_ to nonlinearly
register two bundles.


First, we need to download static and moving bundles for this tutorial. Here,
we two uncinate fasciculus bundles in the left hemisphere of the brain from:

    `<hhttps://figshare.com/articles/dataset/Test_Bundles_for_DIPY/22557733>`_


Let's say we have moving bundle (bundle to be registered) named ``m_UF_L.trk``
and static/fixed bundle named ``s_UF_L.trk``.

Visualizing the moving and static bundles before registration::

    dipy_horizon "m_UF.trk" "s_UF_LI.trk" --random_color

.. figure:: https://github.com/dipy/dipy_data/blob/master/before_bw_registration.png?raw=true
    :width: 70 %
    :alt: alternate text
    :align: center

    Moving bundle in green and static bundle in pink before registration.


BundleWarp provides the capacity to either partially or fully deform the
moving bundle using a single regularization parameter, alpha (represented with
λ in BundleWarp paper). Where alpha controls the trade-off between regularizing
the deformation and having points match very closely. The lower the value of
alpha, the more closely the bundles would match. Here, we investigate how to
warp moving bundle with different levels of deformations using BundleWarp
registration method [Chandio23]_.

--------------------------------------------
Partially Deformable BundleWarp Registration
--------------------------------------------

Here, we partially deform/warp moving bundle to align it with static bundle.
partial deformations improve linear registration while preserving the anatomical
shape and structures of the moving bundle. Here, we use relatively higher value
of alpha=0.03. By default, BundleWarp partially deforms the bundle to preserve
the key characteristics of the original bundle.

The following BundleWarp workflows requirse two positional input arguments;
 ``static`` and ``moving`` .trk files. In our case, the ``static`` input bundle
is the ``s_UF_L.trk`` and the ``moving`` is ``m_UF_L.trk``.

Run the following workflow::

    dipy_bundlewarp "s_UF_L.trk" "m_UF_L.trk" --alpha 0.01 --force

Per default, the BundleWarp workflow will save a nonlinearly transformed bundle
as ``nonlinearly_moved.trk``.

Visualizing the moved and static bundles after registration::

    dipy_horizon "nonlinearly_moved.trk" "s_UF_L.trk" --random_color

.. figure:: https://github.com/dipy/dipy_data/blob/master/partially_deformable_bw_registration.png?raw=true
    :width: 70 %
    :alt: alternate text
    :align: center

    Partially moved bundle in green and static bundle in pink after registration.

----------------------------------------
Fully Deformable BundleWarp Registration
----------------------------------------

Here, we fully deform/warp moving bundle to make it completely aligned with
the static bundle. Here, we use lower value of alpha=0.01.
NOTE: Be caustious with setting lower value of alpha as it can completely
change the original anatomical shape of the moving bundle.

Run the following workflow::

    dipy_bundlewarp "s_UF_L.trk" "m_UF_L.trk" --alpha 0.01 --force

Per default, the BundleWarp workflow will save a nonlinearly transformed bundle
as ``nonlinearly_moved.trk``.

Visualizing the moved and static bundles after registration::

    dipy_horizon "nonlinearly_moved.trk" "s_UF_L.trk" --random_color

.. figure:: https://github.com/dipy/dipy_data/blob/master/fully_deformable_bw_registration.png?raw=true
    :width: 70 %
    :alt: alternate text
    :align: center

    Fully moved bundle in green and static bundle in pink after registration.



For more information about each command line, please visit DIPY website `<https://dipy.org/>`_ .

If you are using any of these commands please be sure to cite the relevant papers and
DIPY [Garyfallidis14]_.

----------
References
----------

.. [Chandio2023] Chandio et al. "BundleWarp, streamline-based nonlinear
            registration of white matter tracts." bioRxiv (2023): 2023-01

.. [Garyfallidis14] Garyfallidis, E., M. Brett, B. Amirbekian, A. Rokem,
    S. Van Der Walt, M. Descoteaux, and I. Nimmo-Smith.
    "DIPY, a library for the analysis of diffusion MRI data".
    Frontiers in Neuroinformatics, 1-18, 2014.
