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
  where $\gamma$ is the gyromagnetic radio, $\delta$ denotes the pulse
  width, $G$ is the gradient amplitude and $\Delta$ the centre to
  centre spacing. $\gamma$ is a constant, but we can change the other
  three parameters and in that way control the b-value.

2. **What is q-space?**
  
  Q-space is the space of one or more 3D spin displacement wave vectors
  $\mathbf{q}$ as shown in equation \ref{eq:fourier}. The vector $\mathbf{q}$
  parametrises the space of diffusion gradients. It is related to the
  applied magnetic gradient $\mathbf{g}$ by the formula $\mathbf{q}=(2\pi)^{-1}\gamma\delta\mathbf{g}$.
  Every single vector $\mathbf{q}$ has the same orientation as the
  direction of diffusion gradient $\mathbf{g}$ and length proportional
  to the strength $g$ of the gradient field. Every single point in
  q-space corresponds to a possible 3D volume of the MR signal for a specific
  gradient direction and strength. Therefore if, for example, we have
  programmed the scanner to apply 60 gradient directions then our data
  should have 60 diffusion volumes with each volume obtained for a specific
  gradient. A Diffusion Weighted Image (DWI) is the volume acquired
  from only one direction gradient.
  
3. **What DWI stands for?**
   
  Diffusion Weighted Imaging (DWI) is MRI imaging designed to be sensitive
  to diffusion. A diffusion weighted image is a volume of voxel data gathered 
  by applying only one gradient direction
  using a diffusion sequence. We expect that the signal in any voxel
  should be low if there is greater mobility of water molecules along
  the specified gradient direction and it should be high if there is
  less movement in that direction. Yes, it is counterintuitive but correct!
  However greater mobility gives greater opportunity for the proton spins to be dephased
  producing a smaller RF signal.

4. **Why dMRI and not DTI?**

  Diffusion MRI (dMRI or dwMRI) are prefered terms if you want to speak about diffusion weighted MRI in general. 
  DTI (diffusion tensor imaging) is just one of the many ways you can reconstruct the voxel from your measured signal. 
  There are plenty of others for example DSI, GQI, QBI etc.     

5. **What is the recommended practice for registration of diffusion datasets?**

  Registration can be tricky. But this is what usually works for us for normal healthy adult subjects. 
  We register the FA (fractional anisotropy) images to the FMRIB_FA_1mm template which is in MNI space 
  using ``flirt`` and ``fnirth`` from FSL. Then we can apply the warping displacements in any other scalar volumes 
  that we have to register that scalar volume into the MNI space. We need the corresponding inverse displacements 
  to map a tractography into MNI space. 

6. **What is the difference between Image coordinates and World coordinates?**

  Image coordinates have positive integer values and represent the centres $(i, j, k)$ of the voxels. There is an affine transform 
  (stored in the nifti file) that takes the image coordinates and transforms them to millimeter (mm) in real world space. 
  World coordinates have floating point precision and your dataset have 3 real dimensions e.g. $(x, y, z)$.
  
7. **Why 'tracks' and not 'tracts'?**

  Tractography is only an approximation or simulation - if you prefer - of the real tracts (brain neural fiber pathways 
  or brain nerves). Therefore we prefer to call these simulated tracts as tracks (trajectories or curves represented as sequences of
  points joined by line segments) so that others will be clear 
  that they are not the real tracts (fibers) but only an estimate or suggestion. 
  We hope that in the future tractography could reach a point that what you see on
  your screen is a very faithful representation of what is actually in the white matter of the brain. 
  However the field is not yet at this level of detail.    

8. **Why use 'deterministic' and not 'probabilistic' tractography?**

  We wanted to create at the outset a tractographic method which will help us and you to get closer to datasets 
  in a very efficient way. Therefore, we created first the ``EuDX`` (Euler Delta Crossings) algorithm which is a tracking method 
  which can work both with model or model-free input and resolve also
  crossing fibers with a high order of crossings. Also it is very fast to calculate (~2 minutes for 1 million tracks ). 
  We hope that at a later stage we will be able to incorporate and test more methods e.g. probabilistic, global and graph-theoretic.
  
9. **We made the mistake in our lab of generating datasets with nonisotropic voxel sizes wusehat do we do?**
  
  You need to resample your raw data to an isotropic size. Have a look at the module ``dipy.align.noniso2iso``. 
  (We think it is a mistake to acquire nonisotropic data because the directional resolution of the data will depend on
  the orientation of the gradient with respect to the voxels, being lower when aligned with a longer voxel dimension.)
  
10. **Why nonisotropic voxel sizes are a bad idea in diffusion?**
  
  If for example you have $2 \times 2 \times 4 \textrm{mm}^3$ voxels, the last dimension will
  be averaged over the double distance and less detail will be captured compared
  to the other two dimensions. Furthermore, with very nonisotropic voxels 
  the uncertainty on orientation estimates will depend on the position of 
  the subject in the scanner.

---------
Practical
---------

1. **Why python and not matlab or some other language?**

  python is free, batteries included, very well designed,  painless to read and easy to use. 
  There is nothing else like it. Give it a go. 
  Once with python always with python. 
  
2. **Isn't python slow?**

  True, some times python can be slow if you are using for example multiple nested for loops. 
  In that case we use cython which takes execution up to C speed.
  
3. **What numerical libraries do you use in python?**

  The best ever designed numerical library numpy.   
  
2. **Which python console do your recommend?**

  ``ipython``

3. **What do you use for visualization?**

  We use ``fosvtk(fvtk)`` which depends in turn on ``python-vtk``:: 
  
  from dipy.viz import fvtk

4. **What about interactive visualization?**

  There is already interaction in the ``fvtk`` module but we have started a new project 
  only for visualization which we plan to integrate in ``dipy`` in the near future for more information 
  have a look at http://fos.me

5. **Which file formats do you support?**
  
  Nifti (.nii), Dicom (Siemens(read-only)), Trackvis (.trk), Dipy (.dpy), Numpy (.npy, ,npz), text 
  and any other formats supported by nibabel and pydicom. 
  
  You can also read/save in Matlab version v4 (Level 1.0), v6 and v7 to 7.2 using scipy.io.loadmat. For higher versions >= 7.3
  you can use pytables or any other python to hdf5 library e.g. h5py .
  
  For object serialization you can used dipy.io.pickles function load_pickle, save_pickle.  
  
6. **What is dpy**?

  ``dpy`` is an ``hdf5`` file format which we use in dipy to store tractography and other information. 
  This allows us to store huge tractographies and load different parts of the datasets 
  directly from the disk as if it were in memory.

7. **Which python editor should I use?**

  Any text editor would do the job but we prefer the following Aptana, Emacs, Vim and Eclipse (with PyDev).
  
8. **I have problems reading my dicom files using nibabel, what should I do?**

  Use Chris Roden's dcm2nii to transform them to nifti files.  
  http://www.cabiatl.com/mricro/mricron/dcm2nii.html
  Or you can make your own reader using pydicom   
  http://code.google.com/p/pydicom/
  and then use nibabel to store the data as niftis.
  
  
\