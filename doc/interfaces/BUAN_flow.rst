.. _BUAN_flow:

================================================================================
Tutorial for BUndle ANalytics (BUAN) framework using DIPY Workflow command lines
================================================================================

This tutorial walks through the steps for reproducing Bundle Analytics results
on Parkinson's Progression Markers Initiative (PPMI) data derivatives.
Bundle Analytics is a framework for comparing bundle profiles and shapes of
different groups. In this example, we will be comparing healthy controls and
patients with parkinson's disease. We will be using PPMI data derivatives generated
using DIPY.


First we need to download streamline atlas of 30 bundles in MNI space from::

    https://figshare.com/articles/Atlas_of_30_Human_Brain_Bundles_in_MNI_space/12089652

For this tutorial we will be using a test sample of DIPY Processed Parkinson's
Progression Markers Initiative (PPMI) Data Derivatives. It can be downloaded
from the link below::

     https://doi.org/10.35092/yhjc.12098397

There are two parts of Bundle Analytics group comparison framework,
bundle profile analysis and bundle shape similarity.

-------------------------------------------------------
We first start with group comparison of bundle profiles
-------------------------------------------------------

For generating bundle profile data (saved as .h5 files):
You must have downloaded bundles folder of 30 atlas bundles and subjects folder
with PPMI data derivatives.

NOTE: Make sure all output folders are empty and do not get overridden.
Following workflows require specific input directory structure.

Create an ``out_dir`` folder (eg: bundle_profiles)::

    mkdir bundle_profiles

Run thee following workflow::

    dipy_buan_profiles bundles/ subjects_small/ --out_dir "bundle_profiles"


For running Linear Mixed Models (LMM) on generated .h5 files from the previous
step:

Create an ``out_dir`` folder (eg: lmm_plots)::

    mkdir lmm_plots

and run the following workflow::

    dipy_buan_lmm "bundle_profiles/*" --out_dir "lmm_plots"

--------------------------------------------------------------------------
Calculating shape similarity of same type of bundles across the population
--------------------------------------------------------------------------

Create an ``out_dir`` folder (eg: sm_plots)::

    mkdir sm_plots

run following workflow::

    dipy_buan_shapes subjects/ --out_dir "sm_plots"


For more information about each command line, you can got to :ref:`workflows_reference`.

If you are using any of these commands do cite the relevant papers.

.. [Chandio19] Chandio, B.Q., S. Koudoro, D. Reagan, J. Harezlak,
    E. Garyfallidis, Bundle Analytics: a computational and statistical
    analyses framework for tractometric studies, Proceedings of:
    International Society of Magnetic Resonance in Medicine (ISMRM),
    Montreal, Canada, 2019.

.. [Garyfallidis14] Garyfallidis, E., M. Brett, B. Amirbekian, A. Rokem,
    S. Van Der Walt, M. Descoteaux, and I. Nimmo-Smith.
    "DIPY, a library for the analysis of diffusion MRI data".
    Frontiers in Neuroinformatics, 1-18, 2014.