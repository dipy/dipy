.. _buan_flow:

=================================
BUndle ANalytics (BUAN) framework
=================================

This tutorial walks through the steps for reproducing Bundle Analytics :footcite:p:`Chandio2020a`
results on Parkinson's Progression Markers Initiative (PPMI) :footcite:p:`Marek2011`
data derivatives. Bundle Analytics is a framework for comparing bundle
profiles and shapes of different groups. In this example, we will be comparing
healthy controls and patients with parkinson's disease. We will be using PPMI
data derivatives generated using DIPY :footcite:p:`Garyfallidis2014a`.


First we need to download streamline atlas :footcite:p:`Yeh2018` with 30 white
matter bundles in MNI space from

    `<https://figshare.com/articles/Atlas_of_30_Human_Brain_Bundles_in_MNI_space/12089652>`_

For this tutorial we will be using a test sample of DIPY Processed Parkinson's
Progression Markers Initiative (PPMI) Data Derivatives. It can be downloaded
from the link below

     `<https://doi.org/10.35092/yhjc.12098397>`_

.. note::

    If you prefer to run experiments on the complete dataset to reproduce the paper
    :footcite:p:`Chandio2020a` results please see the "Reproducing results on larger
    dataset" section at end of the page for more information.

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

Where, ``bundles`` folder has following model bundles::

    bundles/
    ├── AF_L.trk
    ├── AF_R.trk
    ├── CCMid.trk
    ├── CC_ForcepsMajor.trk
    ├── CC_ForcepsMinor.trk
    ├── CST_L.trk
    ├── CST_R.trk
    ├── EMC_L.trk
    ├── EMC_R.trk
    ├── FPT_L.trk
    ├── FPT_R.trk
    ├── IFOF_L.trk
    ├── IFOF_R.trk
    ├── ILF_L.trk
    ├── ILF_R.trk
    ├── MLF_L.trk
    ├── MLF_R.trk
    ├── ML_L.trk
    ├── ML_R.trk
    ├── MdLF_L.trk
    ├── MdLF_R.trk
    ├── OPT_L.trk
    ├── OPT_R.trk
    ├── OR_L.trk
    ├── OR_R.trk
    ├── STT_L.trk
    ├── STT_R.trk
    ├── UF_L.trk
    ├── UF_R.trk
    └── V.trk

The ``subjects_small`` directory has following structure::

    subjects_small
    ├── control
    │   ├── 3805
    │   │   ├── anatomical_measures
    │   │   ├── org_bundles
    │   │   └── rec_bundles
    │   ├── 3806
    │   │   ├── anatomical_measures
    │   │   ├── org_bundles
    │   │   └── rec_bundles
    │   ├── 3809
    │   │   ├── anatomical_measures
    │   │   ├── org_bundles
    │   │   └── rec_bundles
    │   ├── 3850
    │   │   ├── anatomical_measures
    │   │   ├── org_bundles
    │   │   └── rec_bundles
    │   └── 3851
    │       ├── anatomical_measures
    │       ├── org_bundles
    │       └── rec_bundles
    └── patient
        ├── 3383
        │   ├── anatomical_measures
        │   ├── org_bundles
        │   └── rec_bundles
        ├── 3385
        │   ├── anatomical_measures
        │   ├── org_bundles
        │   └── rec_bundles
        ├── 3387
        │   ├── anatomical_measures
        │   ├── org_bundles
        │   └── rec_bundles
        ├── 3392
        │   ├── anatomical_measures
        │   ├── org_bundles
        │   └── rec_bundles
        └── 3552
            ├── anatomical_measures
            ├── org_bundles
            └── rec_bundles

And each subject folder has the following structure::

    ├── anatomical_measures
    │   ├── ad.nii.gz
    │   ├── csa_peaks.pam5
    │   ├── fa.nii.gz
    │   ├── md.nii.gz
    │   └── rd.nii.gz
    ├── org_bundles
    │   ├── streamlines_moved_AF_L__labels__recognized_orig.trk
    │   ├── streamlines_moved_AF_R__labels__recognized_orig.trk
    │   ├── streamlines_moved_CCMid__labels__recognized_orig.trk
    │   . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
    │   . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
    │   . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
    │   ├── streamlines_moved_UF_L__labels__recognized_orig.trk
    │   ├── streamlines_moved_UF_R__labels__recognized_orig.trk
    │   └── streamlines_moved_V__labels__recognized_orig.trk
    └── rec_bundles
        ├── moved_AF_L__recognized.trk
        ├── moved_AF_R__recognized.trk
        ├── moved_CCMid__recognized.trk
        . . . . . . . . . . . . . . . .
        . . . . . . . . . . . . . . . .
        . . . . . . . . . . . . . . . .
        ├── moved_UF_L__recognized.trk
        ├── moved_UF_R__recognized.trk
        └── moved_V__recognized.trk

If you want to run this tutorial on your data, make sure that the directory structure is
The same as shown above. Where, ``anatomical_measures`` folder has nifti files for DTI measures such as
FA, MD, and CSA/CSD pam5 files. The ``org_bundles`` folder has extracted bundles in native space.
The ``rec_bundles`` folder has extracted bundles in common space.

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
output p-values file from dipy_buan_lmm workflow. The user can specify at what level of
significance they want to see group differences by providing threshold value of p-value to ``buan_thr`` (default 0.05).
The color of the highlighted area can be specified by providing RGB color values to ``buan_highlight`` (Default Red)

Run the following commandline for visualizing group differences on the model bundle::

    dipy_horizon bundles/AF_L.trk lmm_plots/AF_L_fa_pvalues.npy --buan --buan_thr 0.05

Where, ``AF_L.trk `` is located in your model bundle folder ``bundles`` and
``AF_L_fa_pvalues.npy`` is saved in output folder ``lmm_plots`` of dipy_buan_lmm workflow

The output of this commandline is an interactive visualization window. Example snapshot:

.. figure:: https://github.com/dipy/dipy_data/blob/master/AF_L_highlighted.png?raw=true
    :width: 70 %
    :alt: alternate text
    :align: center

    Result plot for left arcuate fasciculus (AF_L) with a highlighted area on the bundle in red color.
    The highlighted area represents the segments on bundles with significant group differences
    that have pvalues < 0.05.

Let's use a different highlight color this time on ``CST_L`` bundle::

     dipy_horizon bundles/CST_L.trk lmm_plots/CST_L_fa_pvalues.npy --buan --buan_thr 0.05 --buan_highlight 1 1 0

.. figure:: https://github.com/dipy/dipy_data/blob/master/CST_L_highlighted.png?raw=true
    :width: 50 %
    :alt: alternate text
    :align: center

    Result plot for left corticospinal tract left (CST_L) with a highlighted area on the bundle
    in yellow color. The highlighted area represents the segments on bundles with significant
    group differences that have pvalues < 0.05.


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

All steps will be the same as mentioned above except this time the data downloaded
will have different folder name ``subjects`` instead of ``subjects_small``.

For more information about each command line, you can go to
`<https://github.com/dipy/dipy/blob/master/dipy/workflows/stats.py>`_

If you are using any of these commands do cite the relevant papers.

----------
References
----------

.. footbibliography::
