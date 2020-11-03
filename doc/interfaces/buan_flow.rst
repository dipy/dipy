.. _buan_flow:

=================================
BUndle ANalytics (BUAN) framework
=================================

This tutorial walks through the steps for reproducing Bundle Analytics [1]_
results on Parkinson's Progression Markers Initiative (PPMI) [2]_ data derivatives.
Bundle Analytics is a framework for comparing bundle profiles and shapes of
different groups. In this example, we will be comparing healthy controls and
patients with parkinson's disease. We will be using PPMI data derivatives generated
using DIPY [3]_.


First we need to download streamline atlas [4]_ with 30 white matter bundles
in MNI space from

    `<https://figshare.com/articles/Atlas_of_30_Human_Brain_Bundles_in_MNI_space/12089652>`_

For this tutorial we will be using a test sample of DIPY Processed Parkinson's
Progression Markers Initiative (PPMI) Data Derivatives. It can be downloaded
from the link below

     `<https://doi.org/10.35092/yhjc.12098397>`_

.. note::

    If you prefer to run experiments on the complete dataset to reproduce the paper [1]_
    results please see the "Reproducing results on larger dataset" section at end of
    the page for more information.

There are two parts of Bundle Analytics group comparison framework,
bundle profile analysis and bundle shape similarity analysis.

-----------------------------------
Group Comparison of Bundle Profiles
-----------------------------------

For generating bundle profile data (saved as .h5 files):
You must have downloaded bundles folder of 30 atlas bundles and subjects folder
with PPMI data derivatives.

Following workflows require specific input directory structure but don't worry
as data you downloaded is already in the required format. We will be using ``bundles``
folder you downloaded from streamline atlas link and ``subjects_small`` folder
downloaded from test data link.

.. note::

    Make sure all the output folders are empty and do not get overridden.

Create an ``out_dir`` folder (eg: bundle_profiles)::

    mkdir bundle_profiles

Run the following workflow::

    dipy_buan_profiles bundles/ subjects_small/ --out_dir "bundle_profiles"


For running Linear Mixed Models (LMM) on generated .h5 files from the previous
step:

Create an ``out_dir`` folder (eg: lmm_plots)::

    mkdir lmm_plots

And run the following workflow::

    dipy_buan_lmm "bundle_profiles/*" --out_dir "lmm_plots"

This workflow will generate 30 bundles group comparison plots per anatomical measures.
Plots will look like the following example:

.. figure:: https://github.com/dipy/dipy_data/blob/master/AF_L_fa.png?raw=true
    :width: 70 %
    :alt: alternate text
    :align: center

    Result plot for left arcuate fasciculus (AF_L) on FA measure

We can also visualize and highlight the specific location of group differences on the bundle by providing
output pvalues file from dipy_buan_lmm workflow. User can specify at what level of
significance they want to see group differences by providing threshold value of pvalue to ``buan_thr`` (default 0.05).
The color of the highlighted area can be specified by providing RGB color values to ``buan_highlight`` (Default Red)

Run the following commandline for visualizing group differences on the model bundle::

    dipy_horizon bundles/AF_L.trk lmm_plots/AF_L_fa_pvalues.npy --buan --buan_thr 0.05

Where, ``AF_L.trk `` is located in your model bundle folder ``bundles`` and
``AF_L_fa_pvalues.npy`` is saved in output folder ``lmm_plots`` of dipy_buan_lmm workflow

Output of this commandline is interactive visualization window. Example snapshot:

.. figure:: https://github.com/dipy/dipy_data/blob/master/AF_L_highlighted.png?raw=true
    :width: 70 %
    :alt: alternate text
    :align: center

    Result plot for left arcuate fasciculus (AF_L) with highlighted group differences
    area in red color on the bundle where pvalues < 0.05.

Let's use a different highlight color this time on ``CST_L`` bundle::

     dipy_horizon bundles/CST_L.trk lmm_plots/CST_L_fa_pvalues.npy --buan --buan_thr 0.05 --buan_highlight 1 1 0

.. figure:: https://github.com/dipy/dipy_data/blob/master/CST_L_highlighted.png?raw=true
    :width: 50 %
    :alt: alternate text
    :align: center

    Result plot for left corticospinal tract left (CST_L) with highlighted group differences
    area in yellow color on the bundle where pvalues < 0.05.
-----------------------------------------------------------
Shape similarity of specific bundles across the populations
-----------------------------------------------------------

Create an ``out_dir`` folder (eg: sm_plots)::

    mkdir sm_plots

Run the following workflow::

    dipy_buan_shapes subjects_small/ --out_dir "sm_plots"

This workflow will generate 30 bundles shape similarity plots. Shape similarity
score ranges between 0-1, where 1 being highest similarity and 0 being lowest.
Plots will look like the following example:

.. figure:: https://github.com/dipy/dipy_data/blob/master/SM_moved_UF_R__recognized.png?raw=true
    :width: 50 %
    :alt: alternate text
    :align: center

    Result plot for right uncinate fasciculus (UF_R) for 10 subjects.
    First 5 subjects belong to the healthy control group and last 5 subjects belong to patient group.
    In the diagonal, we have shape similarity score of 1 as it is calculated between a bundle and itself.

--------------------------------------
Reproducing results on larger dataset:
--------------------------------------

Complete dataset of DIPY Processed Parkinson's Progression Markers Initiative (PPMI)
Data Derivatives can be downloaded from the link below:

     `<https://doi.org/10.35092/yhjc.12033390>`_

Please note this is a large data file and might take some time to run. If you
only want to test the workflows use the test sample data.

All steps will be the same as mentioned above except this time the data donwloaded
will have different folder name ``subjects`` instead of ``subjects_small``.

For more information about each command line, you can go to
`<https://github.com/dipy/dipy/blob/master/dipy/workflows/stats.py>`_

If you are using any of these commands do cite the relevant papers.

.. [1] Chandio, B.Q., Risacher, S.L., Pestilli, F., Bullock, D.,
    Yeh, FC., Koudoro, S., Rokem, A., Harezlak, J., and Garyfallidis, E.
    Bundle analytics, a computational framework for investigating the
    shapes and profiles of brain pathways across populations.
    Sci Rep 10, 17149 (2020)

.. [2] Marek, Kenneth and Jennings, Danna and Lasch, Shirley and Siderowf,
    Andrew and Tanner, Caroline and Simuni, Tanya and Coffey, Chris and Kieburtz,
    Karl and Flagg, Emily and Chowdhury, Sohini and others.
    The parkinson progression marker initiative (PPMI).
    Progress in neurobiology, 2011.

.. [3] Garyfallidis, E., M. Brett, B. Amirbekian, A. Rokem,
    S. Van Der Walt, M. Descoteaux, and I. Nimmo-Smith.
    "DIPY, a library for the analysis of diffusion MRI data".
    Frontiers in Neuroinformatics, 1-18, 2014.

.. [4] Yeh F.C., Panesar S., Fernandes D., Meola A., Yoshino M.,
    Fernandez-Miranda J.C., Vettel J.M., Verstynen T.
    Population-averaged atlas of the macroscale human structural
    connectome and its network topology.
    Neuroimage, 2018.





