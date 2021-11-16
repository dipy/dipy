.. _faq:

==========================
Frequently Asked Questions
==========================

-----------
Theoretical
-----------

1. **What is a b-value?**

  The b-value $b$ or *diffusion weighting* is a function of the
  strength, duration and temporal spacing and timing parameters of the
  specific paradigm. This function is derived from the Bloch-Torrey
  equations. In the case of the classical Stejskal-Tanner
  pulsed gradient spin-echo (PGSE) sequence, at the time of readout
  $b=\gamma^{2}G^{2}\delta^{2}\left(\Delta-\frac{\delta}{3}\right)$
  where $\gamma$ is the gyromagnetic ratio, $\delta$ denotes the pulse
  width, $G$ is the gradient amplitude and $\Delta$ the center-to-center
  spacing. $\gamma$ is a constant, but we can change the other
  three parameters and in that way control the b-value.

2. **What is q-space?**

  Q-space is the space of one or more 3D spin displacement wave vectors
  $\mathbf{q}$. The vector $\mathbf{q}$
  parametrizes the space of diffusion gradients. It is related to the
  applied magnetic gradient $\mathbf{g}$ by the formula
  $\mathbf{q}=(2\pi)^{-1}\gamma\delta\mathbf{g}$.
  Every single vector $\mathbf{q}$ has the same orientation as the
  direction of diffusion gradient $\mathbf{g}$ and length proportional
  to the strength $g$ of the gradient field. Every single point in
  q-space corresponds to a possible 3D volume of the MR signal for a specific
  gradient direction and strength. Therefore if, for example, we have
  programmed the scanner to apply 60 gradient directions, then our data
  should have 60 diffusion volumes, with each volume obtained for a specific
  gradient. A Diffusion Weighted Image (DWI) is the volume acquired
  from only one direction gradient.

3. **What does DWI stand for?**

  Diffusion Weighted Imaging (DWI) is MRI imaging designed to be sensitive
  to diffusion. A diffusion weighted image is a volume of voxel data gathered
  by applying only one gradient direction
  using a diffusion sequence. We expect that the signal in any voxel
  should be low if there is greater mobility of water molecules along
  the specified gradient direction and it should be high if there is
  less movement in that direction. Yes, it is counterintuitive but correct!
  However, greater mobility gives greater opportunity for the proton spins to
  be dephased, producing a smaller RF signal.

4. **Why dMRI and not DTI?**

  Diffusion MRI (dMRI or dwMRI) are the preferred terms if you want to speak
  about diffusion weighted MRI in general. DTI (diffusion tensor imaging) is
  just one of the many ways you can reconstruct the voxel from your measured
  signal. There are plenty of others, for example DSI, GQI, QBI, etc.

5. **What is the difference between Image coordinates and World coordinates?**

  Image coordinates have positive integer values and represent the centres
  $(i, j, k)$ of the voxels. There is an affine transform (stored in the
  nifti file) that takes the image coordinates and transforms them into
  millimeter (mm) in real world space. World coordinates have floating point
  precision and your dataset has 3 real dimensions e.g. $(x, y, z)$.

6. **We generated dMRI datasets with nonisotropic voxel sizes. What do we do?**

  You need to resample your raw data to an isotropic size. Have a look at
  the module ``dipy.align.aniso2iso``. (We think it is a mistake to
  acquire nonisotropic data because the directional resolution of the data
  will depend on the orientation of the gradient with respect to the
  voxels, being lower when aligned with a longer voxel dimension.)

7. **Why are non-isotropic voxel sizes a bad idea in diffusion?**

  If, for example, you have $2 \times 2 \times 4\ \textrm{mm}^3$ voxels, the
  last dimension will be averaged over the double distance and less detail
  will be captured compared to the other two dimensions. Furthermore, with
  very anisotropic voxels the uncertainty on orientation estimates will
  depend on the position of the subject in the scanner.

---------
Practical
---------

1. **Why Python and not MATLAB or some other language?**

  Python is free, batteries included, very well-designed, painless to read
  and easy to use.
  There is nothing else like it. Give it a go.
  Once with Python, always with Python.

2. **Isn't Python slow?**

  True, sometimes Python can be slow if you are using multiple nested
  ``for`` loops, for example.
  In that case, we use Cython_, which takes execution up to C speed.

3. **What numerical libraries do you use in Python?**

  The best ever designed numerical library - NumPy_.

2. **Which Python console do you recommend?**

  IPython_

3. **What do you use for visualization?**

  For 3D visualization, we use ``dipy.viz`` which depends in turn on ``FURY``::

    from dipy.viz import window, actor

  For 2D visualization we use matplotlib_.

4. **Which file formats do you support?**

  Nifti (.nii), Dicom (Siemens(read-only)), Trackvis (.trk), DIPY (.dpy),
  Numpy (.npy, ,npz), text and any other formats supported by nibabel and
  pydicom.

  You can also read/save in Matlab version v4 (Level 1.0), v6 and v7 to 7.2,
  using `scipy.io.loadmat`. For higher versions >= 7.3, you can use pytables_
  or any other python-to-hdf5 library e.g. h5py.

  For object serialization, you can use ``dipy.io.pickles`` functions
  ``load_pickle``, ``save_pickle``.

5. **What is dpy**?

  ``dpy`` is an ``hdf5`` file format that we use in DIPY to store
  tractography and other information. This allows us to store huge
  tractography and load different parts of the datasets
  directly from the disk as if it were in memory.

6. **Which python editor should I use?**

  Any text editor would do the job but we prefer the following: PyCharm, Sublime, Aptana, Emacs, Vim and Eclipse (with PyDev).

7. **I have problems reading my dicom files using nibabel, what should I do?**

  Use Chris Rorden's dcm2nii to transform them into nifti files.
  http://www.cabiatl.com/mricro/mricron/dcm2nii.html
  Or you can make your own reader using pydicom.
  http://code.google.com/p/pydicom/
  and then use nibabel to store the data as nifti.

8. **Where can I find diffusion data?**

  There are many sources for openly available diffusion MRI. the :mod:`dipy.data` module can be used to download some sample datasets that we use in our examples. In addition there are a lot of large research-grade datasets available through the following sources:

    - http://fcon_1000.projects.nitrc.org/
    - https://www.humanconnectome.org/
    - https://openneuro.org/

.. include:: links_names.inc
