.. raw:: html

    <style>
        table {
        font-family: arial, sans-serif;
        border-collapse: collapse;
        width: 100%;
        }
        td, th {
        border: 1px solid #dddddd;
        text-align: left;
        padding: 8px;
        }
        td.red { color: red;}
        td.green {color: green;}
        tr:nth-child(even) {
        background-color: #f8f8f8;
        }
    </style>

    <table>
    <tr>
        <th>Name</th>
        <th>Synthetic/Phantom/Human/Animal</th>
        <th>Data features (structural; diffusion; label information)</th>
        <th>Scanner</th>
        <th>DIPY name</th>
        <th>Citations</th>
    </tr>
    <tr>
        <td>Tractogram file formats examples</td>
        <td>Synthetic</td>
        <td>Tractogram file formats (`.dpy`, `.fib`, `.tck`, `.trk`)</td>
        <td></td>
        <td>bundle_file_formats_example</td>
        <td>Rheault, F. (2019). Bundles for tractography file format testing and example (Version 1.0) [Data set]. Zenodo. http://doi.org/10.5281/zenodo.3352379</td>
    </tr>
    <tr>
        <td>CENIR HCP-like dataset</td>
        <td></td>
        <td>Multi-shell data: b-vals: [200, 400, 1000, 2000, 3000] (s/mm^2); [20, 20, 202, 204, 206] gradient directions; Corrected for Eddy currents</td>
        <td></td>
        <td>cenir_multib</td>
        <td></td>
    </tr>
    <tr>
        <td>CFIN dataset</td>
        <td></td>
        <td>T1; Multi-shell data: b-vals: [200, 400, 600, 800, 1000, 1200, 1400, 1600, 1800, 2000, 2200, 2400, 2600, 2800, 3000] (s/mm^2); 496 gradient directions</td>
        <td></td>
        <td>cfin_multib</td>
        <td>Hansen, B., Jespersen, S.. Data for evaluation of fast kurtosis strategies, b-value optimization and exploration of diffusion MRI contrast. Sci Data 3, 160072 (2016). doi:10.1038/sdata.2016.72</td>
    </tr>
    <tr>
        <td>Gold standard streamlines IO testing</td>
        <td>Synthetic</td>
        <td>Tractogram file formats (`.dpy`, `.fib`, `.tck`, `.trk`)</td>
        <td></td>
        <td>gold_standard_io</td>
        <td>Rheault, F. (2019). Gold standard for tractogram io testing (Version 1.0) [Data set]. Zenodo. http://doi.org/10.5281/zenodo.2651349</td>
    </tr>
    <tr>
        <td>HCP842 bundle atlas</td>
        <td>Human</td>
        <td>Whole brain/bundle-wise tractograms in MNI space; 80 bundles</td>
        <td>Human Connectome Project (HCP) scanner</td>
        <td>bundle_atlas_hcp842</td>
        <td>Garyfallidis, E., et al. Recognition of white matter bundles using local and global streamline-based registration and clustering. NeuroImage 170 (2017): 283-297; Yeh, F.-C., et al. Population-averaged atlas of the macroscale human structural connectome and its network topology. NeuroImage 178 (2018): 57-68. <a href='https://figshare.com/articles/Advanced_Atlas_of_80_Bundles_in_MNI_space/7375883'>figshare.com/articles/Advanced_Atlas_of_80_Bundles_in_MNI_space/7375883</a></td>
    </tr>
    <tr>
        <td>HCP bundle FA</td>
        <td>Human</td>
        <td>Fractional Anisotropy (FA); 2 bundles</td>
        <td></td>
        <td>bundle_fa_hcp</td>
        <td></td>
    </tr>
    <tr>
        <td>HCP tractogram</td>
        <td>Human</td>
        <td>Whole brain tractogram</td>
        <td>Human Connectome Project (HCP) scanner</td>
        <td>target_tractogram_hcp</td>
        <td></td>
    </tr>
    <tr>
        <td>ISBI 2013</td>
        <td>Phantom</td>
        <td>Multi shell data: b-vals: [0, 1500, 2500] (s/mm^2); 64 gradient directions</td>
        <td></td>
        <td>isbi2013_2shell</td>
        <td>Daducci, A., et al. Quantitative Comparison of Reconstruction Methods for Intra-Voxel Fiber Recovery From Diffusion MRI. IEEE Transactions on Medical Imaging, vol. 33, no. 2, pp. 384-399, Feb. 2014. <a href='http://hardi.epfl.ch/static/events/2013_ISBI/testing_data.html'>HARDI reconstruction challenge 2013</a></td>
    </tr>
    <tr>
        <td>IVIM dataset</td>
        <td>Human</td>
        <td>Multi shell data: b-vals: [0, 10, 20, 30, 40, 60, 80, 100, 120, 140, 160, 180, 200, 300, 400, 500, 600, 700, 800, 900, 1000] (s/mm^2); 21 gradient directions</td>
        <td></td>
        <td>fetch_ivim</td>
        <td>Peterson, Eric (2016): IVIM dataset. figshare. Dataset. <a href='https://doi.org/10.6084/m9.figshare.3395704.v1'>figshare.com/articles/dataset/IVIM_dataset/3395704/1</a></td>
    </tr>
    <tr>
        <td>MNI template</td>
        <td>Human</td>
        <td>MNI 2009a T1, T2; 2009c T1, T1 mask</td>
        <td></td>
        <td>mni_template</td>
        <td>Fonov, V.S., Evans, A.C., Botteron, K., Almli, C.R., McKinstry, R.C., Collins, D.L., BDCG. Unbiased average age-appropriate atlases for pediatric studies. NeuroImage, Volume 54, Issue 1, January 2011, ISSN 1053–8119, doi:10.1016/j.neuroimage.2010.07.033; Fonov, V.S., Evans, A.C., McKinstry, R.C., Almli, C.R., Collins, D.L. Unbiased nonlinear average age-appropriate brain templates from birth to adulthood, NeuroImage, Volume 47, Supplement 1, July 2009, Page S102 Organization for Human Brain Mapping 2009 Annual Meeting, doi:10.1016/S1053-8119(09)70884-5 <a href='http://www.bic.mni.mcgill.ca/ServicesAtlases/ICBM152NLin2009'>ICBM 152 Nonlinear atlases version 2009</a> </td>
    </tr>
    <tr>
        <td>qt-dMRI C57Bl6 mice dataset</td>
        <td>Animal</td>
        <td>2 C57Bl6 mice test-retest qt-dMRI; Corpus callosum (CC) bundle masks</td>
        <td></td>
        <td>qtdMRI_test_retest_2subjects</td>
        <td>Wassermann, D., Santin, M., Philippe, A.-C., Fick, R., Deriche, R., Lehericy, S., Petiet, A. (2017). Test-Retest qt-dMRI datasets for "Non-Parametric GraphNet-Regularized Representation of dMRI in Space and Time" [Data set]. Zenodo. http://doi.org/10.5281/zenodo.996889</td>
    </tr>
    <tr>
        <td>SCIL b0</td>
        <td></td>
        <td>b0</td>
        <td>GE (1.5, 3 T), Philips (3 T); Siemens (1.5, 3 T)</td>
        <td>scil_b0</td>
        <td><a href='http://scil.dinf.usherbrooke.ca'>Sherbrooke Connectivity Imaging Lab (SCIL)</a></td>
    </tr>
    <tr>
        <td>Sherbrooke 3 shells</td>
        <td>Human</td>
        <td>Multi shell data: b-vals: [0, 1000, 2000; 3500] (s/mm^2); 193 gradient directions</td>
        <td></td>
        <td>sherbrooke_3shell</td>
        <td><a href='http://scil.dinf.usherbrooke.ca'>Sherbrooke Connectivity Imaging Lab (SCIL)</a></td>
    </tr>
    <tr>
        <td>SNAIL dataset</td>
        <td></td>
        <td>2 subjects: T1; Fractional Anisotropy (FA); 27 bundles</td>
        <td></td>
        <td>bundles_2_subjects</td>
        <td></td>
    </tr>
    <tr>
        <td>Stanford HARDI</td>
        <td>Human</td>
        <td>HARDI-like multi-shell data: b-vals: [0, 2000] (s/mm^2); 160 gradient directions</td>
        <td>GE Discovery MR750</td>
        <td>stanford_hardi</td>
        <td><a href='https://purl.stanford.edu/ng782rw8378'>Human brain diffusion-weighted MRI, collected with high diffusion-weighting angular resolution and repeated measurements at multiple diffusion-weighting strengths</a>. Rokem, A., Yeatman, J.D., Pestilli, F., Kay, K.N., Mezer A., van der Walt, S., and Wandell, B.A. (2015) Evaluating the Accuracy of Diffusion MRI Models in White Matter. PLoS ONE 10(4): e0123272. doi:10.1371/journal.pone.0123272</td>
    </tr>
    <tr>
        <td>Stanford labels</td>
        <td>Human</td>
        <td>Gray matter region labels</td>
        <td>GE Discovery MR750</td>
        <td>stanford_labels</td>
        <td><a href='https://purl.stanford.edu/ng782rw8378'>Human brain diffusion-weighted MRI, collected with high diffusion-weighting angular resolution and repeated measurements at multiple diffusion-weighting strengths</a>. Rokem, A., Yeatman, J.D., Pestilli, F., Kay, K.N., Mezer A., van der Walt, S., and Wandell, B.A. (2015) Evaluating the Accuracy of Diffusion MRI Models in White Matter. PLoS ONE 10(4): e0123272. doi:10.1371/journal.pone.0123272</td>
    </tr>
    <tr>
        <td>Stanford PVE maps</td>
        <td>Human</td>
        <td>Partial Volume Effects (PVE) maps: Gray matter (GM), White matter (WM); Cerebrospinal Fluid (CSF)</td>
        <td>GE Discovery MR750</td>
        <td>fetch_stanford_pve_maps</td>
        <td><a href='https://purl.stanford.edu/ng782rw8378'>Human brain diffusion-weighted MRI, collected with high diffusion-weighting angular resolution and repeated measurements at multiple diffusion-weighting strengths</a>. Rokem, A., Yeatman, J.D., Pestilli, F., Kay, K.N., Mezer A., van der Walt, S., and Wandell, B.A. (2015) Evaluating the Accuracy of Diffusion MRI Models in White Matter. PLoS ONE 10(4): e0123272. doi:10.1371/journal.pone.0123272</td>
    </tr>
    <tr>
        <td>Stanford T1</td>
        <td>Human</td>
        <td>T1</td>
        <td>GE Discovery MR750</td>
        <td>stanford_t1</td>
        <td><a href='https://purl.stanford.edu/ng782rw8378'>Human brain diffusion-weighted MRI, collected with high diffusion-weighting angular resolution and repeated measurements at multiple diffusion-weighting strengths</a>. Rokem, A., Yeatman, J.D., Pestilli, F., Kay, K.N., Mezer A., van der Walt, S., and Wandell, B.A. (2015) Evaluating the Accuracy of Diffusion MRI Models in White Matter. PLoS ONE 10(4): e0123272. doi:10.1371/journal.pone.0123272</td>
    </tr>
    <tr>
        <td>SyN data</td>
        <td>Human</td>
        <td>T1; b0</td>
        <td></td>
        <td>syn_data</td>
        <td></td>
    </tr>
    <tr>
        <td>Taiwan NTU DSI</td>
        <td></td>
        <td>DSI-like data; Multi-shell data: b-vals: [0, 308 ,615, 923, 1231, 1538, 1538, 1846, 1846, 2462, 2769, 3077, 3385, 3692, 4000] (s/mm^2); 203 gradient directions</td>
        <td>Siemens Trio</td>
        <td>taiwan_ntu_dsi</td>
        <td>National Taiwan University (NTU) Hospital Advanced Biomedical MRI Lab DSI MRI data</td>
    </tr>
    <tr>
        <td>Tissue data</td>
        <td>Human</td>
        <td>T1; denoised T1; Power map</td>
        <td></td>
        <td>tissue_data</td>
        <td></td>
    </tr>
    </table>
