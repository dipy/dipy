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
     - Typical b-value = 1000s/mm^2, maximum b-value 1200s/mm^2 (some success up to 1500s/mm^2)
     - `Basser 1994 <https://www.ncbi.nlm.nih.gov/pubmed/8130344>`__
   * - :ref:`DTI (RESTORE) <sphx_glr_examples_built_reconstruction_restore_dti.py>`
     - :bdg-success:`Yes`
     - :bdg-success:`Yes`
     - :bdg-success:`Yes`
     - Typical b-value = 1000s/mm^2, maximum b-value 1200s/mm^2 (some success up to 1500s/mm^2)
     - Yendiki2013, Chang2005, Chung2006
   * - :ref:`FwDTI <sphx_glr_examples_built_reconstruction_reconst_fwdti.py>`
     - :bdg-danger:`No`
     - :bdg-success:`Yes`
     - :bdg-danger:`No`
     - DTI-style acquistion, multiple b=0, all shells should be within maximum b-value of 1000 (or 32 directions evenly distributed 500mm/s^2 and 1500mm/s^2 per Henriques 2017)
     - `Pasternak 2009 <https://www.ncbi.nlm.nih.gov/pubmed/19623619>`__, `Henriques et al., 2017 <https://github.com/ReScience-Archives/Henriques-Rokem-Garyfallidis-St-Jean-Peterson-Correia-2017/raw/master/article/Henriques-Rokem-Garyfallidis-St-Jean-Peterson-Correia-2017.pdf>`__
   * - :ref:`DKI - Standard <sphx_glr_examples_built_reconstruction_reconst_dki.py>`
     - :bdg-danger:`No`
     - :bdg-success:`Yes`
     - :bdg-danger:`No`
     - Dual spin echo diffusion-weighted 2D EPI images were acquired with b values of 0, 500, 1000, 1500, 2000, and 2500 s/mm^2 (max b value of 2000 suggested as sufficient in brain tissue); at least 15 directions
     - `Jensen2005 <https://www.ncbi.nlm.nih.gov/pubmed/15906300>`__
   * - :ref:`DKI+ Constraints <sphx_glr_examples_built_reconstruction_reconst_dki.py>`
     - :bdg-danger:`No`
     - :bdg-success:`Yes`
     - :bdg-danger:`No`
     - None
     - `Tom Dela Haije 2020 <https://doi.org/10.1016/j.neuroimage.2019.116405>`__
   * - :ref:`DKI - Micro (WMTI) <sphx_glr_examples_built_reconstruction_reconst_dki_micro.py>`
     - :bdg-danger:`No`
     - :bdg-success:`Yes`
     - :bdg-danger:`No`
     - DKI-style acquisition: at least two non-zero b shells (max b value 2000), minimum of 15 directions; typically b-values in increments of 500 from 0 to 2000, 30 directions
     - `Fieremans 2011 <https://www.sciencedirect.com/science/article/pii/S1053811911006148>`__, `Tabesh 2010 <https://doi.org/10.1002/mrm.22655>`__
   * - :ref:`Mean Signal DKI <sphx_glr_examples_built_reconstruction_reconst_msdki.py>`
     - :bdg-danger:`No`
     - :bdg-success:`Yes`
     - :bdg-danger:`No`
     - b-values in increments of 500 from 0 to 2000, 30 directions
     - `Henriques, 2018 <https://www.repository.cam.ac.uk/handle/1810/281993>`__
   * - :ref:`CSA <sphx_glr_examples_built_reconstruction_reconst_csa.py>`
     - :bdg-success:`Yes`
     - :bdg-danger:`No`
     - :bdg-danger:`No`
     - HARDI data (preferably 7T) with at least 200 diffusion images at b=3000 s/mm^2, or multi-shell data with high angular resolution
     - `Aganj 2010 <https://www.ncbi.nlm.nih.gov/pubmed/20535807>`__
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
     - LeBihan 1984
   * - :ref:`IVIM Variable Projection <sphx_glr_examples_built_reconstruction_reconst_ivim.py>`
     - :bdg-danger:`No`
     - :bdg-success:`Yes`
     - :bdg-danger:`No`
     -
     - Fadnavis 2019
   * - SDT
     - :bdg-success:`Yes`
     - :bdg-danger:`No`
     - :bdg-danger:`No`
     - QBI-style acquisition (60-64 directions, b-value 1000mm/s^2)
     - Descoteaux 2009
   * - :ref:`DSI <sphx_glr_examples_built_reconstruction_reconst_dsi.py>`
     - :bdg-danger:`No`
     - :bdg-danger:`No`
     - :bdg-success:`Yes`
     - 515 diffusion encodings, b-values from 12,000 to 18,000 s/mm^2. Acceleration in subsequent studies with ~100 diffusion encoding directions in half sphere of the q-space with b-values = 1000, 2000, 3000s/mm2)
     - `Wedeen 2008 <https://doi.org/10.1016/j.neuroimage.2008.03.036>`__, `Sotiropoulos 2013 <https://doi.org/10.1016/j.neuroimage.2013.05.057>`__
   * - :ref:`DSID <sphx_glr_examples_built_reconstruction_reconst_dsid.py>`
     - :bdg-danger:`No`
     - :bdg-danger:`No`
     - :bdg-success:`Yes`
     - 203 diffusion encodings (isotropic 3D grid points in the q-space contained within a sphere with radius 3.6), maximum b-value=4000mm/s^2
     - `Canales-Rodriguez 2010 <https://doi.org/10.1016/j.neuroimage.2009.11.066>`__
   * - :ref:`GQI - GQI2 <sphx_glr_examples_built_reconstruction_reconst_gqi.py>`
     - :bdg-danger:`No`
     - :bdg-success:`Yes`
     - :bdg-success:`Yes`
     - Fits any sampling scheme with at least one non-zero b-shell, benefits from more directions. Recommended 23 b-shells ranging from 0 to 4000 in a 258 direction grid-sampling scheme
     - Yeh 2010
   * - :ref:`SFM <sphx_glr_examples_built_reconstruction_reconst_sfm.py>`
     - :bdg-success:`Yes`
     - :bdg-success:`Yes`
     - :bdg-danger:`No`
     - At least 40 directions, b-value above 1000mm/s^2
     - `Rokem 2015 <https://doi.org/10.1371/journal.pone.0123272>`__
   * - :ref:`Q-Ball (OPDT) <sphx_glr_examples_built_reconstruction_reconst_csa.py>`
     - :bdg-success:`Yes`
     - :bdg-danger:`No`
     - :bdg-danger:`No`
     - At least 64 directions, maximum b-values 3000-4000mm/s^2, multi-shell, isotropic voxel size
     - `Tuch 2004 <https://doi.org/10.1002/mrm.20279>`__, `Descoteaux 2007 <https://www.ncbi.nlm.nih.gov/pubmed/17763358>`__, `Tristan-Vega 2010 <https://doi.org/10.1007/978-3-642-04271-3_51>`__
   * - :ref:`SHORE <sphx_glr_examples_built_reconstruction_reconst_shore.py>`
     - :bdg-danger:`No`
     - :bdg-success:`Yes`
     - :bdg-danger:`No`
     - Multi-shell HARDI data (500, 1000, and 2000 s/mm^2; minimum 2 non-zero b-shells) or DSI (514 images in a cube of five lattice-units, one b=0)
     - Merlet 2013, Özarslan 2009, Özarslan 2008
   * - :ref:`MAP-MRI <sphx_glr_examples_built_reconstruction_reconst_mapmri.py>`
     - :bdg-danger:`No`
     - :bdg-success:`Yes`
     - :bdg-danger:`No`
     - Six unit sphere shells with b = 1000, 2000, 3000, 4000, 5000, 6000 s/mm^2 along 19, 32, 56, 87, 125, and 170 directions (see `Olson 2019 <https://doi.org/10.1016/j.neuroimage.2019.05.078>`__ for candidate sub-sampling schemes)
     - `Ozarslan 2013 <https://doi.org/10.1016%2Fj.neuroimage.2013.04.016>`__, `Olson 2019 <https://doi.org/10.1016/j.neuroimage.2019.05.078>`__
   * - :ref:`MAP+ Constraints <sphx_glr_examples_built_reconstruction_reconst_mapmri.py>`
     - :bdg-danger:`No`
     - :bdg-success:`Yes`
     - :bdg-danger:`No`
     -
     - `Tom Dela Haije < https://doi.org/10.1016/j.neuroimage.2019.116405>`__
   * - MAPL
     - :bdg-danger:`No`
     - :bdg-success:`Yes`
     - :bdg-danger:`No`
     - Multi-shell similar to WU-Minn HCP, with minimum of 60 samples from 2 shells b-value 1000 and 3000s/mm^2
     - `Fick 2016 <https://doi.org/10.1016/j.neuroimage.2016.03.046>`__
   * - :ref:`CSD <sphx_glr_examples_built_reconstruction_reconst_csd.py>`
     - :bdg-success:`Yes`
     - :bdg-danger:`No`
     - :bdg-danger:`No`
     - Minimum: 20 gradient directions and a b-value of 1000 s/mm^2; benefits additionally from 60 direction HARDI data with b-value = 3000s/mm^2 or multi-shell
     - Tournier 2017, Descoteaux 2008, Tournier 2007
   * - :ref:`SMS/MT CSD <sphx_glr_examples_built_reconstruction_reconst_mcsd.py>`
     - :bdg-danger:`No`
     - :bdg-success:`Yes`
     - :bdg-danger:`No`
     - 5 b=0, 50 directions at 3 non-zero b-shells: b=1000, b=2000, b=3000
     - `Jeurissen 2014 <https://www.ncbi.nlm.nih.gov/pubmed/25109526>`__
   * - :ref:`ForeCast <sphx_glr_examples_built_reconstruction_reconst_forecast.py>`
     - :bdg-danger:`No`
     - :bdg-success:`Yes`
     - :bdg-danger:`No`
     - Multi-shell 64 direction b-values of 1000, 2000s/mm^2 as in `Alexander 2017 <https://doi.org/10.1038%2Fsdata.2017.181>`__. Original model used 1480 s/mm^2  with 92 directions and 36 b=0
     - Anderson 2005, `Alexander 2017 <https://doi.org/10.1038%2Fsdata.2017.181>`__
   * - :ref:`RUMBA-SD <sphx_glr_examples_built_reconstruction_reconst_rumba.py>`
     - :bdg-success:`Yes`
     - :bdg-success:`Yes`
     - :bdg-success:`Yes`
     - HARDI data with 64 directions at b = 2500s/mm^2, 3 b=0 images (full original acquisition: 256 directions on a sphere at b = 2500s/mm^2, 36 b=0 volumes)
     - `Canales-Rodríguez 2015 <https://doi.org/10.1371/journal.pone.0138910>`__
   * - :ref:`QTI <sphx_glr_examples_built_reconstruction_reconst_qti.py>`
     - :bdg-danger:`No`
     - :bdg-success:`Yes`
     - :bdg-danger:`No`
     - Evenly distributed geometric sampling scheme of 216 measurements, 5 b-values (50, 250, 50, 1000, 200mm/s^2), measurement tensors of four shapes: stick, prolate, sphere, and plane
     - `Westin 2016 <https://doi.org/10.1016/j.neuroimage.2016.02.039>`__
   * - :ref:`QTI+ <sphx_glr_examples_built_reconstruction_reconst_qtiplus.py>`
     - :bdg-danger:`No`
     - :bdg-success:`Yes`
     - :bdg-danger:`No`
     - At least one b=0, minimum of 39 acquisitions with spherical and linear encoding; optimal 120 (see `Morez 2023 <https://doi.org/10.1002/hbm.26175>`__), ideal 217 see `Herberthson 2021 Table 1 <https://www.sciencedirect.com/science/article/pii/S1053811921004754?via%3Dihub#tbl0001>`__
     - `Herberthson 2021 <https://doi.org/10.1016/j.neuroimage.2021.118198>`__, `Morez 2023 <https://doi.org/10.1002/hbm.26175>`__
   * - Ball & Stick
     - :bdg-success:`Yes`
     - :bdg-success:`Yes`
     - :bdg-danger:`No`
     - Three b=0, 60 evenly distributed directions per `Jones 1999 <https://doi.org/10.1002/(SICI)1522-2594(199909)42:3%3C515::AID-MRM14%3E3.0.CO;2-Q>`__ at b-value 1000mm/s^2
     - `Behrens 2003 <https://onlinelibrary.wiley.com/doi/full/10.1002/mrm.10609>`__
   * - :ref:`QTau-MRI <sphx_glr_examples_built_reconstruction_reconst_qtdmri.py>`
     - :bdg-danger:`No`
     - :bdg-success:`Yes`
     - :bdg-danger:`No`
     - Minimum 200 volumes of multi-spherical dMRI (multi-shell, multi-diffusion time; varying gradient directions, gradient strengths, and diffusion times)
     - Fick 2017
   * - Power Map
     - :bdg-success:`Yes`
     - :bdg-success:`Yes`
     - :bdg-danger:`No`
     - HARDI data with 60 directions at b-value = 3000 s/mm^2, 7 b=0 (Minimum: HARDI data with at least 30 directions)
     - `DellAcqua2014 <http://archive.ismrm.org/2014/0730.html>`__
   * - :ref:`SMT / SMT2 <sphx_glr_examples_built_reconstruction_reconst_msdki.py>`
     - :bdg-danger:`No`
     - :bdg-success:`Yes`
     - :bdg-danger:`No`
     - 72 directions at each of 5 evenly spaced b-values from 0.5 to 2.5 ms/μm2, 5 b-values from 3 to 5 ms/μm2, 5 b-values from 5.5 to 7.5 ms/μm2, and 3 b-values from 8 to 9 ms/μm2 /  b=0 ms/μm^-2, and along 33 directions at b-values from 0.2–3 ms/μm^-2 in steps of 0.2 ms/μm^−2 (24 point spherical design and 9 directions identified for rapid kurtosis estimation)
     - `NetoHe2019 <https://doi.org/10.1002/mrm.27606>`__, `Kaden2016b <https://www.nature.com/articles/sdata201672>`__
   * - :ref:`CTI <sphx_glr_examples_built_reconstruction_reconst_cti.py>`
     - :bdg-danger:`No`
     - :bdg-success:`Yes`
     - :bdg-danger:`No`
     -
     - `NetoHe2020 <https://www.sciencedirect.com/science/article/pii/S1053811920300926>`__, `NovelloL2022 <https://pubmed.ncbi.nlm.nih.gov/35339682/>`__, `NetHe2021 <https://onlinelibrary.wiley.com/doi/10.1002/mrm.28938>`__

