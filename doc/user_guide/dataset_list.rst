.. list-table:: Datasets available in DIPY
   :widths: 10 8 8 8 56 10
   :header-rows: 1

   * - Name
     - Synthetic/Phantom/Human/Animal
     - Data features (structural; diffusion; label information)
     - Scanner
     - DIPY name
     - Citations
   * - Tractogram file formats examples
     - Synthetic
     - Tractogram file formats (`.dpy`, `.fib`, `.tck`, `.trk`)
     -
     - bundle_file_formats_example
     - Rheault, F. (2019). Bundles for tractography file format testing and example (Version 1.0) [Data set]. Zenodo. https://doi.org/10.5281/zenodo.3352379
   * - CENIR HCP-like dataset
     -
     - Multi-shell data: b-vals: [200, 400, 1000, 2000, 3000] (s/mm^2); [20, 20, 202, 204, 206] gradient directions; Corrected for Eddy currents
     -
     - cenir_multib
     -
   * - CFIN dataset
     -
     - T1; Multi-shell data: b-vals: [200, 400, 600, 800, 1000, 1200, 1400, 1600, 1800, 2000, 2200, 2400, 2600, 2800, 3000] (s/mm^2); 496 gradient directions
     -
     - cfin_multib
     - Hansen, B., Jespersen, S.. Data for evaluation of fast kurtosis strategies, b-value optimization and exploration of diffusion MRI contrast. Sci Data 3, 160072 (2016). doi:10.1038/sdata.2016.72
   * - Gold standard streamlines IO testing
     - Synthetic
     - Tractogram file formats (`.dpy`, `.fib`, `.tck`, `.trk`)
     -
     - gold_standard_io
     - Rheault, F. (2019). Gold standard for tractogram io testing (Version 1.0) [Data set]. Zenodo. https://doi.org/10.5281/zenodo.2651349
   * - HCP842 bundle atlas
     - Human
     - Whole brain/bundle-wise tractograms in MNI space; 80 bundles
     - Human Connectome Project (HCP) scanner
     - bundle_atlas_hcp842
     - Garyfallidis, E., et al. Recognition of white matter bundles using local and global streamline-based registration and clustering. NeuroImage 170 (2017): 283-297; Yeh, F.-C., et al. Population-averaged atlas of the macroscale human structural connectome and its network topology. NeuroImage 178 (2018): 57-68. <a href='https://figshare.com/articles/Advanced_Atlas_of_80_Bundles_in_MNI_space/7375883'>figshare.com/articles/Advanced_Atlas_of_80_Bundles_in_MNI_space/7375883</a>
   * - HCP bundle FA
     - Human
     - Fractional Anisotropy (FA); 2 bundles
     -
     - bundle_fa_hcp
     -
   * - HCP tractogram
     - Human
     - Whole brain tractogram
     - Human Connectome Project (HCP) scanner
     - target_tractogram_hcp
     -
   * - ISBI 2013
     - Phantom
     - Multi shell data: b-vals: [0, 1500, 2500] (s/mm^2); 64 gradient directions
     -
     - isbi2013_2shell
     - Daducci, A., et al. Quantitative Comparison of Reconstruction Methods for Intra-Voxel Fiber Recovery From Diffusion MRI. IEEE Transactions on Medical Imaging, vol. 33, no. 2, pp. 384-399, Feb. 2014. <a href='http://hardi.epfl.ch/static/events/2013_ISBI/testing_data.html'>HARDI reconstruction challenge 2013</a>
   * - IVIM dataset
     - Human
     - Multi shell data: b-vals: [0, 10, 20, 30, 40, 60, 80, 100, 120, 140, 160, 180, 200, 300, 400, 500, 600, 700, 800, 900, 1000] (s/mm^2); 21 gradient directions
     -
     - fetch_ivim
     - Peterson, Eric (2016): IVIM dataset. figshare. Dataset. <a href='https://doi.org/10.6084/m9.figshare.3395704.v1'>figshare.com/articles/dataset/IVIM_dataset/3395704/1</a>
   * - MNI template
     - Human
     - MNI 2009a T1, T2; 2009c T1, T1 mask
     -
     - mni_template
     - Fonov, V.S., Evans, A.C., Botteron, K., Almli, C.R., McKinstry, R.C., Collins, D.L., BDCG. Unbiased average age-appropriate atlases for pediatric studies. NeuroImage, Volume 54, Issue 1, January 2011, ISSN 1053–8119, doi:10.1016/j.neuroimage.2010.07.033; Fonov, V.S., Evans, A.C., McKinstry, R.C., Almli, C.R., Collins, D.L. Unbiased nonlinear average age-appropriate brain templates from birth to adulthood, NeuroImage, Volume 47, Supplement 1, July 2009, Page S102 Organization for Human Brain Mapping 2009 Annual Meeting, doi:10.1016/S1053-8119(09)70884-5 <a href='https://www.bic.mni.mcgill.ca/ServicesAtlases/ICBM152NLin2009'>ICBM 152 Nonlinear atlases version 2009</a>
   * - qt-dMRI C57Bl6 mice dataset
     - Animal
     - 2 C57Bl6 mice test-retest qt-dMRI; Corpus callosum (CC) bundle masks
     -
     - qtdMRI_test_retest_2subjects
     - Wassermann, D., Santin, M., Philippe, A.-C., Fick, R., Deriche, R., Lehericy, S., Petiet, A. (2017). Test-Retest qt-dMRI datasets for "Non-Parametric GraphNet-Regularized Representation of dMRI in Space and Time" [Data set]. Zenodo. https://doi.org/10.5281/zenodo.996889
   * - SCIL b0
     -
     - b0
     - GE (1.5, 3 T), Philips (3 T); Siemens (1.5, 3 T)
     - scil_b0
     - <a href='http://scil.dinf.usherbrooke.ca'>Sherbrooke Connectivity Imaging Lab (SCIL)</a>
   * - Sherbrooke 3 shells
     - Human
     - Multi shell data: b-vals: [0, 1000, 2000; 3500] (s/mm^2); 193 gradient directions
     -
     - sherbrooke_3shell
     - <a href='http://scil.dinf.usherbrooke.ca'>Sherbrooke Connectivity Imaging Lab (SCIL)</a>
   * - SNAIL dataset
     -
     - 2 subjects: T1; Fractional Anisotropy (FA); 27 bundles
     -
     - bundles_2_subjects
     -
   * - Stanford HARDI
     - Human
     - HARDI-like multi-shell data: b-vals: [0, 2000] (s/mm^2); 160 gradient directions
     - GE Discovery MR750
     - stanford_hardi
     - <a href='https://purl.stanford.edu/ng782rw8378'>Human brain diffusion-weighted MRI, collected with high diffusion-weighting angular resolution and repeated measurements at multiple diffusion-weighting strengths</a>. Rokem, A., Yeatman, J.D., Pestilli, F., Kay, K.N., Mezer A., van der Walt, S., and Wandell, B.A. (2015) Evaluating the Accuracy of Diffusion MRI Models in White Matter. PLoS ONE 10(4): e0123272. doi:10.1371/journal.pone.0123272
   * - Stanford labels
     - Human
     - Gray matter region labels
     - GE Discovery MR750
     - stanford_labels
     - <a href='https://purl.stanford.edu/ng782rw8378'>Human brain diffusion-weighted MRI, collected with high diffusion-weighting angular resolution and repeated measurements at multiple diffusion-weighting strengths</a>. Rokem, A., Yeatman, J.D., Pestilli, F., Kay, K.N., Mezer A., van der Walt, S., and Wandell, B.A. (2015) Evaluating the Accuracy of Diffusion MRI Models in White Matter. PLoS ONE 10(4): e0123272. doi:10.1371/journal.pone.0123272
   * - Stanford PVE maps
     - Human
     - Partial Volume Effects (PVE) maps: Gray matter (GM), White matter (WM); Cerebrospinal Fluid (CSF)
     - GE Discovery MR750
     - fetch_stanford_pve_maps
     - <a href='https://purl.stanford.edu/ng782rw8378'>Human brain diffusion-weighted MRI, collected with high diffusion-weighting angular resolution and repeated measurements at multiple diffusion-weighting strengths</a>. Rokem, A., Yeatman, J.D., Pestilli, F., Kay, K.N., Mezer A., van der Walt, S., and Wandell, B.A. (2015) Evaluating the Accuracy of Diffusion MRI Models in White Matter. PLoS ONE 10(4): e0123272. doi:10.1371/journal.pone.0123272
   * - Stanford T1
     - Human
     - T1
     - GE Discovery MR750
     - stanford_t1
     - <a href='https://purl.stanford.edu/ng782rw8378'>Human brain diffusion-weighted MRI, collected with high diffusion-weighting angular resolution and repeated measurements at multiple diffusion-weighting strengths</a>. Rokem, A., Yeatman, J.D., Pestilli, F., Kay, K.N., Mezer A., van der Walt, S., and Wandell, B.A. (2015) Evaluating the Accuracy of Diffusion MRI Models in White Matter. PLoS ONE 10(4): e0123272. doi:10.1371/journal.pone.0123272
   * - SyN data
     - Human
     - T1; b0
     -
     - syn_data
     -
   * - Taiwan NTU DSI
     -
     - DSI-like data; Multi-shell data: b-vals: [0, 308 ,615, 923, 1231, 1538, 1538, 1846, 1846, 2462, 2769, 3077, 3385, 3692, 4000] (s/mm^2); 203 gradient directions
     - Siemens Trio
     - taiwan_ntu_dsi
     - National Taiwan University (NTU) Hospital Advanced Biomedical MRI Lab DSI MRI data
   * - Tissue data
     - Human
     - T1; denoised T1; Power map
     -
     - tissue_data
     -
