.. _bundlewarp_registration_flow:

=========================================================
Nonrigid White Matter Bundle Registration with BundleWarp
=========================================================

This tutorial explains how we can use BundleWarp :footcite:p`Chandio2023` to
nonlinearly register two bundles.


First, we need to download static and moving bundles for this tutorial. Here,
we two uncinate fasciculus bundles in the left hemisphere of the brain from:

    `<https://figshare.com/articles/dataset/Test_Bundles_for_DIPY/22557733>`_


Let's say we have a moving bundle (bundle to be registered) named ``m_UF_L.trk``
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
registration method :footcite:p`Chandio2023`.

--------------------------------------------
Partially Deformable BundleWarp Registration
--------------------------------------------

Here, we partially deform/warp the moving bundle to align it with the static bundle.
partial deformations improve linear registration while preserving the anatomical
shape and structures of the moving bundle. Here, we use a relatively higher value
of alpha=0.5. By default, BundleWarp partially deforms the bundle to preserve
the key characteristics of the original bundle.


The following BundleWarp workflow requires two positional input arguments;
``static`` and ``moving`` .trk files. In our case, the ``static`` input bundle
is the ``s_UF_L.trk``, and the ``moving`` is ``m_UF_L.trk``.

Run the following workflow::

    dipy_bundlewarp "s_UF_L.trk" "m_UF_L.trk" --alpha 0.5 --force

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
NOTE: Be cautious with setting lower value of alpha as it can completely
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
DIPY :footcite:p:`Garyfallidis2014a`.

----------
References
----------

.. footbibliography::
