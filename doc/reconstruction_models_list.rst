.. list-table:: Reconstruction methods available in DIPY
   :widths: 10 8 8 8 56 10
   :header-rows: 1

   * - Method
     - Single Shell
     - Multi Shell
     - Cartesian
     - Paper Data Descriptions
     - References
   * - :ref:`DTI (SLS, WLS, NNLS) <sphx_glr_examples_built_reconstruction_reconst_dti.py>`
     - :bdg-success:`Yes`
     - :bdg-success:`Yes`
     - :bdg-success:`Yes`
     - Typical b-value = 1000 s/mm^2, maximum b-value 1200 s/mm^2 (some success up to 1500 s/mm^2)
     - :cite:t:`Basser1994a`
   * - :ref:`DTI (RESTORE) <sphx_glr_examples_built_reconstruction_restore_dti.py>`
     - :bdg-success:`Yes`
     - :bdg-success:`Yes`
     - :bdg-success:`Yes`
     - Typical b-value = 1000 s/mm^2, maximum b-value 1200 s/mm^2 (some success up to 1500 s/mm^2)
     - :cite:t:`Chang2005`, :cite:t:`Chung2006`, :cite:t:`Yendiki2014`
   * - :ref:`FwDTI <sphx_glr_examples_built_reconstruction_reconst_fwdti.py>`
     - :bdg-danger:`No`
     - :bdg-success:`Yes`
     - :bdg-danger:`No`
     - DTI-style acquisition, multiple b=0, all shells should be within maximum b-value of 1000 s/mm^2 (or 32 directions evenly distributed 500 s/mm^2 and 1500 s/mm^2 per :cite:t:`NetoHenriques2017`)
     - :cite:t:`Pasternak2009`, :cite:t:`NetoHenriques2017`
   * - :ref:`DKI - Standard <sphx_glr_examples_built_reconstruction_reconst_dki.py>`
     - :bdg-danger:`No`
     - :bdg-success:`Yes`
     - :bdg-danger:`No`
     - Dual spin echo diffusion-weighted 2D EPI images were acquired with b values of 0, 500, 1000, 1500, 2000, and 2500 s/mm^2 (max b value of 2000 suggested as sufficient in brain tissue); at least 15 directions
     - :cite:t:`Jensen2005`
   * - :ref:`DKI+ Constraints <sphx_glr_examples_built_reconstruction_reconst_dki.py>`
     - :bdg-danger:`No`
     - :bdg-success:`Yes`
     - :bdg-danger:`No`
     - None
     - :cite:t:`DelaHaije2020`
   * - :ref:`DKI - Micro (WMTI) <sphx_glr_examples_built_reconstruction_reconst_dki_micro.py>`
     - :bdg-danger:`No`
     - :bdg-success:`Yes`
     - :bdg-danger:`No`
     - DKI-style acquisition: at least two non-zero b shells (max b value 2000), minimum of 15 directions; typically b-values in increments of 500 from 0 to 2000, 30 directions
     - :cite:t:`Fieremans2011`, :cite:t:`Tabesh2011`
   * - :ref:`Mean Signal DKI <sphx_glr_examples_built_reconstruction_reconst_msdki.py>`
     - :bdg-danger:`No`
     - :bdg-success:`Yes`
     - :bdg-danger:`No`
     - b-values in increments of 500 from 0 to 2000 s/mm^2, 30 directions
     - :cite:t:`NetoHenriques2018`
   * - :ref:`CSA <sphx_glr_examples_built_reconstruction_reconst_csa.py>`
     - :bdg-success:`Yes`
     - :bdg-danger:`No`
     - :bdg-danger:`No`
     - HARDI data (preferably 7T) with at least 200 diffusion images at b=3000 s/mm^2, or multi-shell data with high angular resolution
     - :cite:t:`Aganj2010`
   * - Westins CSA
     - :bdg-success:`Yes`
     - :bdg-danger:`No`
     - :bdg-danger:`No`
     -
     -
   * - :ref:`IVIM <sphx_glr_examples_built_reconstruction_reconst_ivim.py>`
     - :bdg-danger:`No`
     - :bdg-success:`Yes`
     - :bdg-danger:`No`
     - low b-values are needed
     - :cite:t:`LeBihan1988`
   * - :ref:`IVIM Variable Projection <sphx_glr_examples_built_reconstruction_reconst_ivim.py>`
     - :bdg-danger:`No`
     - :bdg-success:`Yes`
     - :bdg-danger:`No`
     -
     - :cite:t:`Fadnavis2019`
   * - SDT
     - :bdg-success:`Yes`
     - :bdg-danger:`No`
     - :bdg-danger:`No`
     - QBI-style acquisition (60-64 directions, b-value 1000 s/mm^2)
     - :cite:t:`Descoteaux2009`
   * - :ref:`DSI <sphx_glr_examples_built_reconstruction_reconst_dsi.py>`
     - :bdg-danger:`No`
     - :bdg-danger:`No`
     - :bdg-success:`Yes`
     - 515 diffusion encodings, b-values from 12,000 to 18,000 s/mm^2. Acceleration in subsequent studies with ~100 diffusion encoding directions in half sphere of the q-space with b-values = 1000, 2000, 3000 s/mm^2)
     - :cite:t:`Wedeen2008`, :cite:t:`Sotiropoulos2013`
   * - :ref:`DSID <sphx_glr_examples_built_reconstruction_reconst_dsid.py>`
     - :bdg-danger:`No`
     - :bdg-danger:`No`
     - :bdg-success:`Yes`
     - 203 diffusion encodings (isotropic 3D grid points in the q-space contained within a sphere with radius 3.6), maximum b-value = 4000 s/mm^2
     - :cite:t:`CanalesRodriguez2010`
   * - :ref:`GQI - GQI2 <sphx_glr_examples_built_reconstruction_reconst_gqi.py>`
     - :bdg-danger:`No`
     - :bdg-success:`Yes`
     - :bdg-success:`Yes`
     - Fits any sampling scheme with at least one non-zero b-shell, benefits from more directions. Recommended 23 b-shells ranging from 0 to 4000 in a 258 direction grid-sampling scheme
     - :cite:t:`Yeh2010`
   * - :ref:`SFM <sphx_glr_examples_built_reconstruction_reconst_sfm.py>`
     - :bdg-success:`Yes`
     - :bdg-success:`Yes`
     - :bdg-danger:`No`
     - At least 40 directions, b-value above 1000 s/mm^2
     - :cite:t:`Rokem2015`
   * - :ref:`Q-Ball (OPDT) <sphx_glr_examples_built_reconstruction_reconst_csa.py>`
     - :bdg-success:`Yes`
     - :bdg-danger:`No`
     - :bdg-danger:`No`
     - At least 64 directions, maximum b-values 3000-4000 s/mm^2, multi-shell, isotropic voxel size
     - :cite:t:`Tuch2004`, :cite:t:`Descoteaux2007`, :cite:t:`TristanVega2009b`
   * - :ref:`SHORE <sphx_glr_examples_built_reconstruction_reconst_shore.py>`
     - :bdg-danger:`No`
     - :bdg-success:`Yes`
     - :bdg-danger:`No`
     - Multi-shell HARDI data (500, 1000, and 2000 s/mm^2; minimum 2 non-zero b-shells) or DSI (514 images in a cube of five lattice-units, one b=0)
     - :cite:t:`Merlet2013`, :cite:t:`Ozarslan2008`, :cite:t:`Ozarslan2009`
   * - :ref:`MAP-MRI <sphx_glr_examples_built_reconstruction_reconst_mapmri.py>`
     - :bdg-danger:`No`
     - :bdg-success:`Yes`
     - :bdg-danger:`No`
     - Six unit sphere shells with b = 1000, 2000, 3000, 4000, 5000, 6000 s/mm^2 along 19, 32, 56, 87, 125, and 170 directions (see :cite:t:`Olson2019` for candidate sub-sampling schemes)
     - :cite:t:`Ozarslan2013`, :cite:t:`Olson2019`
   * - :ref:`MAP+ Constraints <sphx_glr_examples_built_reconstruction_reconst_mapmri.py>`
     - :bdg-danger:`No`
     - :bdg-success:`Yes`
     - :bdg-danger:`No`
     -
     - :cite:t:`DelaHaije2020`
   * - MAPL
     - :bdg-danger:`No`
     - :bdg-success:`Yes`
     - :bdg-danger:`No`
     - Multi-shell similar to WU-Minn HCP, with minimum of 60 samples from 2 shells b-value 1000 and 3000 s/mm^2
     - :cite:t:`Fick2016b`
   * - :ref:`CSD <sphx_glr_examples_built_reconstruction_reconst_csd.py>`
     - :bdg-success:`Yes`
     - :bdg-danger:`No`
     - :bdg-danger:`No`
     - Minimum: 20 gradient directions and a b-value of 1000 s/mm^2; benefits additionally from 60 direction HARDI data with b-value = 3000 s/mm^2 or multi-shell
     - :cite:t:`Tournier2004`, :cite:t:`Tournier2007`, :cite:t:`Descoteaux2007`
   * - :ref:`MS/MT CSD <sphx_glr_examples_built_reconstruction_reconst_mcsd.py>`
     - :bdg-danger:`No`
     - :bdg-success:`Yes`
     - :bdg-danger:`No`
     - 5 b=0, 50 directions at 3 non-zero b-shells: b=1000, 2000, 3000 s/mm^2
     - :cite:t:`Jeurissen2014`
   * - :ref:`FORECAST <sphx_glr_examples_built_reconstruction_reconst_forecast.py>`
     - :bdg-danger:`No`
     - :bdg-success:`Yes`
     - :bdg-danger:`No`
     - Multi-shell 64 direction b-values of 1000, 2000 s/mm^2 as in :cite:t:`Alexander2017`. Original model used 1480 s/mm^2 with 92 directions and 36 b=0
     - :cite:t:`Anderson2005`, :cite:t:`Alexander2017`
   * - :ref:`RUMBA-SD <sphx_glr_examples_built_reconstruction_reconst_rumba.py>`
     - :bdg-success:`Yes`
     - :bdg-success:`Yes`
     - :bdg-success:`Yes`
     - HARDI data with 64 directions at b = 2500 s/mm^2, 3 b=0 images (full original acquisition: 256 directions on a sphere at b = 2500 s/mm^2, 36 b=0 volumes)
     - :cite:t:`CanalesRodriguez2015`
   * - :ref:`QTI <sphx_glr_examples_built_reconstruction_reconst_qti.py>`
     - :bdg-danger:`No`
     - :bdg-success:`Yes`
     - :bdg-danger:`No`
     - Evenly distributed geometric sampling scheme of 216 measurements, 5 b-values (50, 250, 50, 1000, 200 s/mm^2), measurement tensors of four shapes: stick, prolate, sphere, and plane
     - :cite:t:`Westin2016`
   * - :ref:`QTI+ <sphx_glr_examples_built_reconstruction_reconst_qtiplus.py>`
     - :bdg-danger:`No`
     - :bdg-success:`Yes`
     - :bdg-danger:`No`
     - At least one b=0, minimum of 39 acquisitions with spherical and linear encoding; optimal 120 (see :cite:t:`Morez2023`), ideal 217 see Table 1 in :cite:t:`Herberthson2021`
     - :cite:t:`Herberthson2021`, :cite:t:`Morez2023`
   * - Ball & Stick
     - :bdg-success:`Yes`
     - :bdg-success:`Yes`
     - :bdg-danger:`No`
     - Three b=0, 60 evenly distributed directions per :cite:t:`Jones1999` at b-value 1000 s/mm^2
     - :cite:t:`Behrens2003`
   * - :ref:`QTau-MRI <sphx_glr_examples_built_reconstruction_reconst_qtdmri.py>`
     - :bdg-danger:`No`
     - :bdg-success:`Yes`
     - :bdg-danger:`No`
     - Minimum 200 volumes of multi-spherical dMRI (multi-shell, multi-diffusion time; varying gradient directions, gradient strengths, and diffusion times)
     - :cite:t:`Fick2018`
   * - Power Map
     - :bdg-success:`Yes`
     - :bdg-success:`Yes`
     - :bdg-danger:`No`
     - HARDI data with 60 directions at b-value = 3000 s/mm^2, 7 b=0 (Minimum: HARDI data with at least 30 directions)
     - :cite:t:`DellAcqua2014`
   * - :ref:`SMT / SMT2 <sphx_glr_examples_built_reconstruction_reconst_msdki.py>`
     - :bdg-danger:`No`
     - :bdg-success:`Yes`
     - :bdg-danger:`No`
     - 72 directions at each of 5 evenly spaced b-values from 0.5 to 2.5 ms/μm^2, 5 b-values from 3 to 5 ms/μm^2, 5 b-values from 5.5 to 7.5 ms/μm^2, and 3 b-values from 8 to 9 ms/μm^2 / b=0 ms/μm^2, and along 33 directions at b-values from 0.2–3 ms/μm^2 in steps of 0.2 ms/μm^2 (24 point spherical design and 9 directions identified for rapid kurtosis estimation)
     - :cite:t:`Kaden2016b`, :cite:t:`NetoHenriques2019`
   * - :ref:`CTI <sphx_glr_examples_built_reconstruction_reconst_cti.py>`
     - :bdg-danger:`No`
     - :bdg-success:`Yes`
     - :bdg-danger:`No`
     -
     - :cite:t:`NetoHenriques2020`, :cite:t:`NetoHenriques2021b`, :cite:t:`Novello2022`
