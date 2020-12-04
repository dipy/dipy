.. _tracking_flow:

========
Tracking
========

This tutorial walks through the steps to perform fiber tracking using DIPY.
Multiple tracking methods are available in DIPY.

You can try these methods using your own data; we will be using the data in
DIPY. You can check how to :ref:`fetch the DIPY data<data_fetch>`.

--------------------
Local Fiber Tracking
--------------------

Local fiber tracking is an approach used to model white matter fibers by
creating streamlines from local directional information. The idea is as follows:
if the local directionality of a tract/pathway segment is known, one can
integrate along those directions to build a complete representation of that
structure. Local fiber tracking is widely used in the field of diffusion MRI
because it is simple and robust.

In order to perform local fiber tracking, three things are needed:

#. A method for getting directions from a diffusion data set.
#. A method for identifying when the tracking must stop.
#. A set of seeds from which to begin tracking.

We will be using the ``stanford_hardi`` dataset for the local tracking command
line interface demonstration purposes.

We will start by creating the directory to which to save the stopping file
(e.g.: ``stop_file``)::

    mkdir stop_file

The workflow for each local tracking method requires the paths to the peaks and
metrics files, stopping files, and seeding files. The default value for the
local tracking method to be used (specified through the ``tracking_method``
parameter) is ``eudx`` (EuDX tracking [Garyfallidis12]_). A number of optional
arguments, such as the number of seeds per dimension within a voxel
(``seed_density``), or the output directory (``out_dir``), can also be provided.

To get the stopping file, we will create a binary file of the GFA of the CSA
reconstruction method and set a stopping criterion by calling the ``dipy_mask``
command::

    dipy_mask recons-csa/gfa.nii.gz 0.25 --out_dir "stop_file"

We will use the white matter (WM) partial volume effects (PVE) map corresponding
to the diffusion data being tracked, and available through the
``stanford_pve_maps`` dataset, as the seeding file (e.g.: ``pve_wm.nii.gz``).

There are four tracking methods as described below.

EuDX Tracking
*************

We will first create the directory in which to save the output tractogram file
(e.g.: ``eudx_tracking_output``)::

    mkdir eudx_tracking_output

Then, to perform the EuDX tracking we will run the ``dipy_track`` command as::

    dipy_track recons-csa/peaks.pam5 stop_file/mask.nii.gz data/stanford_hardi/pve_wm.nii.gz --out_dir "eudx_tracking_output"

Deterministic Tracking
**********************

Deterministic maximum direction getter is the deterministic version of the 
probabilistic direction getter. It can be used with the same local models and
has the same parameters. Deterministic maximum fiber tracking follows the
trajectory of the most probable pathway within the tracking constraint (e.g.
max angle). In other words, it follows the direction with the highest
probability from a distribution, as opposed to the probabilistic direction
getter which draws the direction from the distribution. Therefore, the maximum
deterministic direction getter is equivalent to the probabilistic direction
getter returning always the maximum value of the distribution.

Deterministic maximum fiber tracking is an alternative to EuDX deterministic
tractography, and unlike EuDX, it does not follow the peaks of the local models,
but uses the entire orientation distributions.

We will first create the directory in which to save the output tractogram file
(e.g.: ``det_tracking_output``)::

    mkdir det_tracking_output

Then, to perform the deterministic tracking we will run the ``dipy_track``
command as::

    dipy_track recons-csa/peaks.pam5 stop_file/mask.nii.gz data/stanford_hardi/pve_wm.nii.gz --seed_density 2 --tracking_method "det" --out_dir "det_tracking_output"

Probabilistic Tracking
**********************

Probabilistic fiber tracking is a way of reconstructing white matter connections
using diffusion MR imaging. Like deterministic fiber tracking, the probabilistic
approach follows the trajectory of a possible pathway step by step starting at a
seed; however, unlike deterministic tracking, the tracking direction at each
point along the path is chosen at random from a distribution. The distribution
at each point is different and depends on the observed diffusion data at that
point. The distribution of tracking directions at each point can be represented
as a probability mass function (PMF) if the possible tracking directions are
restricted to discrete numbers of well distributed points on a sphere.

We will first create the directory in which to save the output tractogram file
(e.g.: ``prob_tracking_output``)::

    mkdir prob_tracking_output

Then, to perform the probabilistic tracking we will run the ``dipy_track``
command as::

    dipy_track recons-csa/peaks.pam5 stop_file/mask.nii.gz data/stanford_hardi/pve_wm.nii.gz --seed_density 2 --tracking_method "prob" --out_dir "prob_tracking_output"

Closest Peaks Tracking
**********************

Closest peaks fiber tracking is a type of deterministic tracking. Deterministic
streamlines follow a predictable path through the data by selecting the same
diffusion orientation when evaluating the same point through the propagation
process. There may be several estimated directions at each point (such as a
voxel center): closest peaks tracking selects one of these estimated directions
on the basis of closeness of match to the previous direction of the streamline.

We will first create the directory in which to save the output tractogram
file (e.g.: ``closest_peaks_output``)::

    mkdir closest_peaks_output

Then, to perform the closest peaks tracking we will run the ``dipy_track``
command as::

    dipy_track recons-csa/peaks.pam5 stop_file/mask.nii.gz data/stanford_hardi/pve_wm.nii.gz --seed_density 2 --tracking_method "cp" --out_dir "closest_peaks_output"

---------------------------------
Particle Filtering Tracking (PFT)
---------------------------------

Particle Filtering Tracking (PFT) [Girard2014]_ uses tissue partial volume
estimation (PVE) to reconstruct trajectories connecting the gray matter, and
not prematurely stopping in the white matter or in the corticospinal fluid. It
relies on a stopping criterion that identifies the tissue where the streamline
stopped. If the streamline reaches the gray matter, the trajectory is kept. If
the streamline incorrectly stopped in the white matter or in the cerebrospinal
fluid, PFT uses anatomical information to find an alternative streamline
segment to extend the trajectory. When this segment is found, the tractography
continues until the streamline reaches the gray matter.

We will use the ``stanford_hardi`` dataset in DIPY to showcase this tracking
method. As with any other workflow in DIPY, you can also use your own data!

We will first create a directory in which to save the output tractogram file
(e.g.: ``pft_output``)::
    
    mkdir pft_output

To run the Particle Filtering Tracking method, we need to specify the paths
to the diffusion input file, white matter partial volume estimate, grey matter
partial volume estimate, and cerebrospinal fluid partial volume estimate for
tracking, and seeding file followed by optional arguments. In this case, we
will be specifying the threshold for the Probability Mass Function that controls
the randomness or probabilistic nature of the tracking (``pmf_threshold``),
and the output directory (``out_dir``).

White matter, grey matter, and cerebrospinal fluid volume files are available
through the ``stanford_pve_maps`` dataset.

The Particle Filtering Tracking is performed by calling the ``dipy_track_pft``
command, e.g.::

    dipy_track_pft recons-csa/peaks.pam5 data/stanford_hardi/pve_wm.nii.gz data/stanford_hardi/pve_gm.nii.gz data/stanford_hardi/pve_csf.nii.gz data/stanford_hardi/pve_wm.nii.gz --pmf_threshold 0.5 --out_dir "pft_output"

This command will save the tractogram file to the specified output directory.

----------------------------------
Overview of Fiber Tracking Methods
----------------------------------

.. |image1| image:: https://github.com/dipy/dipy_data/blob/master/eudx_tracking.png?raw=true
   :align: middle
.. |image2| image:: https://github.com/dipy/dipy_data/blob/master/deterministic_tracking.png?raw=true
   :align: middle
.. |image3| image:: https://github.com/dipy/dipy_data/blob/master/closest_peaks_tracking.png?raw=true
   :align: middle

+-----------------------------+-----------------------------+
|    Fiber Tracking Method    |           Output            |
+=============================+=============================+
|        EuDX Tracking        |          |image1|           |
+-----------------------------+-----------------------------+
|    Deterministic Tracking   |          |image2|           |
+-----------------------------+-----------------------------+
|    Closest Peaks Tracking   |          |image3|           |
+-----------------------------+-----------------------------+

----------
References
----------

.. [Garyfallidis12] Garyfallidis E., "Towards an accurate brain tractography",
    PhD thesis, University of Cambridge, 2012.

.. [Girard2014] Girard, G., Whittingstall, K., Deriche, R., & Descoteaux, M.
    Towards quantitative connectivity analysis: reducing tractography biases.
    NeuroImage, 98, 266-278, 2014.