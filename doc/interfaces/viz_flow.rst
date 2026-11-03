.. _viz_flow:

===============================
dMRI Visualization with Skyline
===============================

Skyline is DIPY's interactive viewer for diffusion MRI data. It shows images,
tractograms, peaks, ODFs, ROIs and surfaces together in one 3D scene and is
started from the command line with ``dipy_skyline``.

Skyline replaces Horizon. ``dipy_horizon`` is deprecated: it still opens your
files, but it forwards them to Skyline and ignores the options that only
Horizon had (see `Moving from dipy_horizon`_).

To follow this tutorial, install DIPY with its visualization dependencies
(FURY 2.1 or later and ``imgui_bundle``). ::

  pip install "dipy[viz]" --upgrade

Let's explore the options that Skyline provides. ::

  dipy_skyline --help


------------------
The Skyline window
------------------

Every command in this tutorial opens the same window: the 3D scene on the
right and a control panel on the left. The top of the panel has three buttons:

* **Add Visualization** opens a menu to load *3D/4D Images*, *Peaks*, *ODFs*,
  *Surfaces*, *Tractograms* or *ROIs* through a file dialog. For *ODFs*, pick
  the *Input SH basis* before *Select files...*.
* **Change Background Color** sets the scene background.
* **Take Snapshot** saves the current view as a PNG file.

Below them, the **Global Slices** block moves the X, Y and Z slices of every
dataset whose *Synchronize Slices* toggle is on, and the toggle next to each
axis shows or hides that axis for all datasets at once. The dataset that
defines the slice coordinates is shown as *Reference*.

Loaded datasets are grouped as *Images*, *Tractograms*, *Peaks*, *ODFs*,
*ROIs* and *Surfaces*. Each group can be collapsed or hidden, and each dataset
opens its own controls and can be removed from the scene.

You can also start with an empty window and load everything from the menu,
which opens automatically. ::

  dipy_skyline run


-------------------------
Visualize 3D Brain Images
-------------------------

Using a terminal, let's download a dataset called ``mni_template``. You can
skip this step if you already have the dataset downloaded. ::

  dipy_fetch mni_template

To see more details about ``dipy_fetch`` you can refer to :ref:`data_fetch`.

This command will download the data in your ``.dipy`` folder placed in your
home directory.

Let's load the T1 and T2 templates together.

**For macOS and Linux** ::

  dipy_skyline ~/.dipy/mni_template/mni_icbm152_t1_tal_nlin_asym_09a.nii ~/.dipy/mni_template/mni_icbm152_t2_tal_nlin_asym_09a.nii

**For Windows (cmd)** ::

  dipy_skyline %USERPROFILE%/.dipy/mni_template/mni_icbm152_t1_tal_nlin_asym_09a.nii %USERPROFILE%/.dipy/mni_template/mni_icbm152_t2_tal_nlin_asym_09a.nii

**For Windows (PowerShell)** ::

  dipy_skyline "\$home/.dipy/mni_template/mni_icbm152_t1_tal_nlin_asym_09a.nii" "\$home/.dipy/mni_template/mni_icbm152_t2_tal_nlin_asym_09a.nii"

Skyline reads ``.nii`` and ``.nii.gz`` images. Each image has these controls:

* *Synchronize Slices* links the image to the Global Slices.
* *Opacity* and the *Slice* sliders for the X, Y and Z planes, each with a
  visibility toggle.
* *Intensities* sets the displayed intensity range by percentiles.
* *Colormap* picks one of Gray, Inferno, Magma, Plasma, Viridis, Cool, Hot,
  Bone, Copper, Pink, Spring, Summer, Autumn, Winter, Jet, Cividis, Distinct
  or Divergent.
* *Interpolation* switches between Linear and Nearest.

Instead of listing files one by one, you can pass a quoted glob pattern or a
folder. Skyline then loads every matching file and skips the formats it does
not support. ::

  dipy_skyline "\$HOME/.dipy/mni_template/*.nii"
  dipy_skyline ~/.dipy/mni_template

On Windows, use ``%USERPROFILE%`` (cmd) or ``\$home`` (PowerShell) instead of
``\$HOME`` and ``~``, as above. The remaining examples show the macOS and Linux
form only.


------------------------
Visualize 4D Brain Image
------------------------

Let's download a diffusion dataset called ``stanford_hardi``. ::

  dipy_fetch stanford_hardi

This dataset has a NIfTI file with 160 volumes.

**For macOS and Linux** ::

  dipy_skyline ~/.dipy/stanford_hardi/HARDI150.nii.gz

**For Windows (cmd)** ::

  dipy_skyline %USERPROFILE%/.dipy/stanford_hardi/HARDI150.nii.gz

**For Windows (PowerShell)** ::

  dipy_skyline "\$home/.dipy/stanford_hardi/HARDI150.nii.gz"

For 4D images, the *Directions* slider selects which volume is shown.

Color images, such as the DEC map that ``dipy_fit_dti`` saves as
``rgb.nii.gz``, are shown in RGB automatically when the file uses the NIfTI
RGB data type. For other 4D images whose last dimension is 3 (RGB) or 4
(RGBA), an *RGB* toggle appears in the image controls. You can also choose
from the command line: ``--rgb`` forces RGB mode and ``--no-rgb`` turns it
off. ::

  dipy_skyline rgb.nii.gz --rgb


---------------------
Visualize Brain ROIs
---------------------

Binary masks given with ``--rois`` are drawn as 3D contours over the anatomy.
The ``mni_template`` dataset includes a brain mask for the 2009c template. ::

  dipy_skyline ~/.dipy/mni_template/mni_icbm152_t1_tal_nlin_asym_09c.nii --rois ~/.dipy/mni_template/mni_icbm152_t1_tal_nlin_asym_09c_mask.nii

Each ROI has an *Opacity* slider and a color picker. ROIs must be ``.nii`` or
``.nii.gz`` files.


--------------------------
Visualize Brain Tractogram
--------------------------

Let's download a dataset called ``bundle_atlas_hcp842``. ::

  dipy_fetch bundle_atlas_hcp842

Skyline supports the following tractogram formats.

* .trk
* .trx
* .dpy
* .tck
* .vtk
* .vtp
* .fib

``.trk`` and ``.trx`` files carry their own spatial reference. The other
formats are placed using the header of the MNI ICBM 2009c template.

Let's load the whole-brain atlas and group its streamlines into clusters.

**For macOS and Linux** ::

  dipy_skyline ~/.dipy/bundle_atlas_hcp842/Atlas_80_Bundles/whole_brain/whole_brain_MNI.trk --cluster

**For Windows (cmd)** ::

  dipy_skyline %USERPROFILE%/.dipy/bundle_atlas_hcp842/Atlas_80_Bundles/whole_brain/whole_brain_MNI.trk --cluster

**For Windows (PowerShell)** ::

  dipy_skyline "\$home/.dipy/bundle_atlas_hcp842/Atlas_80_Bundles/whole_brain/whole_brain_MNI.trk" --cluster

With ``--cluster``, the streamlines are grouped with QuickBundlesX and each
cluster is drawn as its centroid. These options control the clustering:

* ``--cluster_thr`` is the distance threshold in mm (default 15.0). For small
  animal brains you may need something smaller, such as 2.0.
* ``--cluster_size_thr`` hides clusters with fewer streamlines than this
  value. By default it is the 50th percentile of the cluster sizes.
* ``--cluster_length_thr`` hides clusters whose average length in mm is below
  this value. By default it is the 25th percentile of the cluster lengths.

The same values can be changed in the tractogram controls (*Threshold*,
*Size* and *Length*). Click a centroid to select or deselect its cluster,
then use these keys:

* ``e`` expands the selected clusters into their streamlines.
* ``c`` collapses them back into centroids.
* ``a`` selects all clusters and ``d`` deselects all.
* ``h`` hides the deselected clusters and ``s`` shows all clusters again.

The download button in the cluster controls saves the selected clusters as a
new ``.trx`` tractogram.

Without ``--cluster``, every streamline is drawn. Opening a large tractogram
this way can be computationally expensive. The *Cluster* toggle in the
tractogram controls switches between both modes at any time.

**Colors.** By default, streamlines are colored by their local direction. Use
``--tract_colors random`` to give each tractogram its own distinguishable
color, or three numbers between 0 and 1 for a single RGB color. ::

  dipy_skyline ~/.dipy/bundle_atlas_hcp842/Atlas_80_Bundles/bundles/AF_L.trk ~/.dipy/bundle_atlas_hcp842/Atlas_80_Bundles/bundles/CST_L.trk --tract_colors random
  dipy_skyline ~/.dipy/bundle_atlas_hcp842/Atlas_80_Bundles/bundles/AF_L.trk --tract_colors "1 0 0"

The color picker in the tractogram controls changes the color afterwards.

**Lines or tubes.** Streamlines are drawn as tubes by default. For large
tractograms, ``--performance_version`` draws them as lines, which is much
faster. The *Line Type* switch in the tractogram controls changes this at
any time, and *Opacity* and *Thickness* adjust the lines.

**Tractograms with anatomy.** The atlas is in MNI space, so it lines up with
the MNI template. ::

  dipy_skyline ~/.dipy/mni_template/mni_icbm152_t1_tal_nlin_asym_09a.nii ~/.dipy/bundle_atlas_hcp842/Atlas_80_Bundles/whole_brain/whole_brain_MNI.trk --cluster

**BUAN p-values.** Pass the p-values file saved by ``dipy_buan_lmm`` with
``--buan_pvals`` to color a bundle by the significance of group differences
along its length. ::

  dipy_skyline AF_L.trk --buan_pvals AF_L_fa_pvalues.npy

The *Hue*, *Saturation* and *Value* sliders tune the colors. *Upload BUAN P
values* in the tractogram controls loads a p-values file into a running
window. See :ref:`buan_flow` for how these files are produced.


-------------------------
Visualize Peaks and ODFs
-------------------------

Peaks and spherical harmonic ODFs come from reconstruction workflows. Here we
use the ``recons_csd_output/peaks.pam5`` file saved by ``dipy_fit_csd`` in
:ref:`reconstruction_flow`.

A ``.pam5`` file given as an input file is shown as peaks. ::

  dipy_skyline recons_csd_output/peaks.pam5

To show its ODF glyphs, pass the same file to ``--odfs``. Use ``run`` as the
input file when you only load files through options. ::

  dipy_skyline run --odfs recons_csd_output/peaks.pam5

Peaks and ODFs can also come from NIfTI files. ``--peaks`` accepts volumes of
shape (X, Y, Z, 3*N) or (X, Y, Z, N, 3), and ``--odfs`` accepts 4D volumes of
SH coefficients, such as the ``peaks_dirs.nii.gz`` and ``shm.nii.gz`` files
that ``dipy_fit_csd --extract_pam_values`` writes. ::

  dipy_skyline run --peaks recons_csd_output/peaks_dirs.nii.gz --odfs recons_csd_output/shm.nii.gz

``--sh_basis`` declares the SH convention of the ODF files:
``descoteaux07`` (the default, current DIPY), ``descoteaux07_legacy`` (older
DIPY), ``tournier07`` (MRtrix 0.2) or ``tournier19`` (MRtrix3). ::

  dipy_skyline run --odfs wm_fod.nii.gz --sh_basis tournier19

Peaks and ODFs have *Synchronize Slices*, *Scale*, *Opacity* and *Slice*
controls, so you can step through them together with an image such as FA.


-----------------------
Visualize Brain Surface
-----------------------

Skyline reads FreeSurfer surfaces (``.pial``) and GIFTI surfaces (``.gii``,
``.gii.gz``).

Using terminal, let's download a brain surface.

**For macOS and Linux** ::

  wget https://github.com/maharshi-gor/dipy_data/raw/surface_data/surfaces/lh.pial

**For Windows (PowerShell)** ::

  wget https://github.com/maharshi-gor/dipy_data/raw/surface_data/surfaces/lh.pial -O lh.pial

**For macOS users**, if you do not have ``wget`` on your terminal you can set
it up with the following command. ::

  brew install wget

If you are still getting an error you can download the surface by clicking
`here <https://github.com/maharshi-gor/dipy_data/raw/surface_data/surfaces/lh.pial>`_.

The previous step downloads the file into your current directory. To load the
surface, ::

  dipy_skyline lh.pial

Each surface has an *Opacity* slider. ``--glass_brain`` draws surfaces as dark
translucent shells on a white background, which is useful for showing
tractograms inside the brain. ::

  dipy_skyline lh.pial --glass_brain


-----------------------------
Save Figures Without a Window
-----------------------------

Use ``--stealth`` to render the scene offscreen and save it as a PNG instead
of opening a window. ``--out_dir`` and ``--out_stealth_png`` choose where the
image goes. ::

  dipy_skyline ~/.dipy/mni_template/mni_icbm152_t1_tal_nlin_asym_09a.nii --stealth --out_dir figures --out_stealth_png t1.png

Setting the environment variable ``DIPY_OFFSCREEN`` to ``1`` or ``true`` has
the same effect without changing the command, which is convenient on servers
and in scripts.

``--bg_color`` sets the background as three numbers between 0 and 1, for
example ``--bg_color 1 1 1`` for white. In an open window, the *Take Snapshot*
button saves the current view.


------------------------
Moving from dipy_horizon
------------------------

Most ``dipy_horizon`` options have a direct ``dipy_skyline`` equivalent.

.. list-table::
   :header-rows: 1

   * - ``dipy_horizon``
     - ``dipy_skyline``
   * - ``--cluster``, ``--cluster_thr``
     - unchanged
   * - ``--clusters_gt N``
     - ``--cluster_size_thr N``
   * - ``--length_gt L``
     - ``--cluster_length_thr L``
   * - ``--random_colors``
     - ``--tract_colors random``
   * - ``--buan`` with a ``.npy`` input file
     - ``--buan_pvals file.npy``
   * - ``--roi_images`` with binary images
     - ``--rois file.nii.gz``
   * - ``--rgb``, ``--bg_color``, ``--stealth``, ``--out_dir``,
       ``--out_stealth_png``
     - unchanged
   * - ``--buan_thr``, ``--buan_highlight``, ``--roi_colors``
     - *Hue*, *Saturation*, *Value* sliders and the color pickers in the
       window
   * - ``--length_lt``, ``--clusters_lt``, ``--native_coords``,
       ``--emergency_header``, ``--disable_order_transparency``
     - not available
