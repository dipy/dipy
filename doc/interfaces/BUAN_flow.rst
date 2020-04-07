.. _BUAN_flow:

=========================================================================
Get Started with a BUndle ANalytics (BUAN) DIPY Workflow command lines
=========================================================================

This documents walks through the steps for reproducing Bundle Analytics results
on Parkinson's Progression Markers Initiative (PPMI) data derivatives.

First we need to download streamline atlas of 30 bundles in MNI space from
[https://figshare.com/articles/Atlas_of_30_Human_Brain_Bundles_in_MNI_space/12089652]

Next, we need to download DIPY Processed Parkinson's Progression Markers
Initiative (PPMI) Data Derivatives from here [insert link here]

There are two parts of Bundle Analytics group comparison framework,
bundle profile analysis and bundle shape similarity.

We first start with group comparison of bundle profiles.

For generating bundle profile data (saved as .h5 files):
You must have downloaded bundles folder of 30 atlas bundles and subjects folder
with PPMI data derivatives.

Create an out_dir folder (eg: bundle_profiles)::

    mkdir bundle_profiles

Run following workflow::

    dipy_buan_profiles bundles/ subjects/ --out_dir "bundle_profiles"


For running Linear mixed models (LMM) on generated .h5 files from the previous
step:

Create an out_dir folder (eg: lmm_plots) ::

    mkdir lmm_plots

and run following workflow::

    dipy_buan_lmm "bundle_profiles/*" --out_dir "lmm_plots"

For calculating shape similarity of same type of bundles across the population:

Create an out_dir folder (eg: sm_plots)::

    mkdir sm_plots

run following workflow::

    dipy_buan_shapes subjects/ --out_dir "sm_plots"


For more information about each command line, you can got to :ref:`workflows_reference`.