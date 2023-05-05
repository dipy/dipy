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
        <th>Method</th>
        <th>Single Shell</th>
        <th>Multi Shell</th>
        <th>Cartesian</th>
        <th>Paper Data Descriptions</th>
        <th>References</th>
    </tr>
    <tr>
        <td><a href='#diffusion-tensor-imaging'>DTI (SLS, WLS, NNLS)</a></td>
        <td class='green'>Yes</td>
        <td class='green'>Yes</td>
        <td class='green'>Yes</td>
        <td>- All shells should be < 1000</td>
        <td><a href='https://www.ncbi.nlm.nih.gov/pubmed/8130344'>Basser 1994</a></td>
    </tr>
    <tr>
        <td><a href='#diffusion-tensor-imaging'>FwDTI</a></td>
        <td class='red'>No</td>
        <td class='green'>Yes</td>
        <td class='red'>No</td>
        <td>DTI-style acquistion, multiple b=0, all shells should be < 1000 (or 32 directions evenly distributed 500mm/s^2 and 1500mm/s^2 per Henriques 2017)</td>
        <td>
            <a href='https://www.ncbi.nlm.nih.gov/pubmed/19623619'>Pasternak 2009</a>,
            <a href='https://github.com/ReScience-Archives/Henriques-Rokem-Garyfallidis-St-Jean-Peterson-Correia-2017/raw/master/article/Henriques-Rokem-Garyfallidis-St-Jean-Peterson-Correia-2017.pdf'>Henriques et al., 2017</a>
        </td>
    </tr>
    <tr>
        <td><a href='#diffusion-kurtosis-imaging'>DKI - Standard</a></td>
        <td class='red'>No</td>
        <td class='green'>Yes</td>
        <td class='red'>No</td>
        <td>Dual spin echo diffusion-weighted 2D EPI images were acquired with b values of 0, 500, 1000, 1500, 2000, and 2500 s/mm^2 (max b value of 2000 suggested as sufficient in brain tissue); at least 15 directions</td>
        <td><a href='https://www.ncbi.nlm.nih.gov/pubmed/15906300'>Jensen 2005</a></td>
    </tr>
    <tr>
        <td><a href='#diffusion-kurtosis-imaging'>DKI - Micro (WMTI)</a></td>
        <td class='red'>No</td>
        <td class='green'>Yes</td>
        <td class='red'>No</td>
        <td>DKI-style acquisition: at least two non-zero b shells (max b value 2000), minimum of 15 directions; typically b-values in increments of 500 from 0 to 2000, 30 directions</td>
        <td>
            <a href='https://www.sciencedirect.com/science/article/pii/S1053811911006148'>Fieremans 2011</a>,
            <a href='https://doi.org/10.1002/mrm.22655'>Tabesh 2010</a>
        </td>
    </tr>
    <tr>
        <td><a href='#diffusion-kurtosis-imaging'>Mean Signal DKI</a></td>
        <td class='red'>No</td>
        <td class='green'>Yes</td>
        <td class='red'>No</td>
        <td>b-values in increments of 500 from 0 to 2000, 30 directions</td>
        <td><a href='https://www.repository.cam.ac.uk/handle/1810/281993'>Henriques, 2018</a></td>
    </tr>
    <tr>
        <td><a href='#q-ball-constant-solid-angle'>CSA</a></td>
        <td class='green'>Yes</td>
        <td class='red'>No</td>
        <td class='red'>No</td>
        <td>HARDI data (preferably 7T) with at least 200 diffusion images at b=3000 s/mm^2, or multi-shell data with high angular resolution</td>
        <td><a href='https://www.ncbi.nlm.nih.gov/pubmed/20535807'>Aganj 2010</a></td>
    </tr>
    <tr>
        <td>Westin's CSA</td>
        <td class='green'>Yes</td>
        <td class='red'>No</td>
        <td class='red'>No</td>
        <td></td>
        <td></td>
    </tr>
    <tr>
        <td><a href='#intravoxel-incoherent-motion-ivim'>IVIM</a></td>
        <td class='red'>No</td>
        <td class='green'>Yes</td>
        <td class='red'>No</td>
        <td>- low b-values are needed</td>
        <td><a href=''>LeBihan 1984</a></td>
    </tr>
    <tr>
        <td><a href='#intravoxel-incoherent-motion-ivim'>IVIM Variable Projection</a></td>
        <td class='red'>No</td>
        <td class='green'>Yes</td>
        <td class='red'>No</td>
        <td></td>
        <td><a href=''>Fadnavis 2019</a></td>
    </tr>
    <tr>
        <td>SDT</td>
        <td class='green'>Yes</td>
        <td class='red'>No</td>
        <td class='red'>No</td>
        <td>QBI-style acquisition (60-64 directions, b-value 1000mm/s^2)</td>
        <td><a href=''>Descoteaux 2009</a></td>
    </tr>
    <tr>
        <td><a href='#diffusion-spectrum-imaging'>DSI</a></td>
        <td class='red'>No</td>
        <td class='red'>No</td>
        <td class='green'>Yes</td>
        <td>515 diffusion encodings, b-values from 12,000 to 18,000 s/mm^2. Acceleration in subsequent studies with ~100 diffusion encoding directions in half sphere of the q-space with b-values = 1000, 2000, 3000s/mm2)</td>
        <td>
            <a href='https://doi.org/10.1016/j.neuroimage.2008.03.036'>Wedeen 2008</a>,
            <a href='https://doi.org/10.1016/j.neuroimage.2013.05.057'>Sotiropoulos 2013</a>
        </td>
        </tr>
    <tr>
        <td><a href='#dsi-with-deconvolution'>DSID</a></td>
        <td class='red'>No</td>
        <td class='red'>No</td>
        <td class='green'>Yes</td>
        <td>203 diffusion encodings (isotropic 3D grid points in the q-space contained within a sphere with radius 3.6), maximum b-value=4000mm/s^2</td>
        <td><a href='https://doi.org/10.1016/j.neuroimage.2009.11.066'>Canales-Rodriguez 2010</a></td>
    </tr>
    <tr>
        <td><a href='#generalized-q-sampling-imaging'>GQI - GQI2</a></td>
        <td class='red'>No</td>
        <td class='green'>Yes</td>
        <td class='green'>Yes</td>
        <td> Fits any sampling scheme with at least one non-zero b-shell, benefits from more directions. Recommended 23 b-shells ranging from 0 to 4000 in a 258 direction grid-sampling scheme</td>
        <td><a href=''>Yeh 2010</a></td>
    </tr>
    <tr>
        <td><a href='#sparse-fascicle-model'>SFM</a></td>
        <td class='green'>Yes</td>
        <td class='green'>Yes</td>
        <td class='red'>No</td>
        <td>At least 40 directions, b-value above 1000mm/s^2</td>
        <td><a href='https://doi.org/10.1371/journal.pone.0123272'>Rokem 2015</a></td>
    </tr>
    <tr>
        <td><a href='#q-ball-constant-solid-angle'>Q-Ball (OPDT)</a></td>
        <td class='green'>Yes</td>
        <td class='red'>No</td>
        <td class='red'>No</td>
        <td>At least 64 directions, maximum b-values 3000-4000mm/s^2, multi-shell, isotropic voxel size</td>
        <td>
            <a href='https://doi.org/10.1002/mrm.20279'>Tuch 2004</a>,
            <a href='https://www.ncbi.nlm.nih.gov/pubmed/17763358'>Descoteaux 2007</a>,
            <a href='https://doi.org/10.1007/978-3-642-04271-3_51'>Tristan-Vega 2010</a>
        </td>
    </tr>
    <tr>
        <td><a href='#simple-harmonic-oscillator-based-reconstruction-and-estimation'>SHORE</a></td>
        <td class='red'>No</td>
        <td class='green'>Yes</td>
        <td class='red'>No</td>
        <td>Multi-shell HARDI data (500, 1000, and 2000 s/mm^2; minimum 2 non-zero b-shells) or DSI (514 images in a cube of five lattice-units, one b=0)</td>
        <td>
            <a href=''>Merlet 2013</a>,
            <a href=''>Özarslan 2009</a>,
            <a href=''>Özarslan 2008</a></td>
    </tr>
    <tr>
        <td><a href='#mean-apparent-propagator-map-mri'>MAP-MRI</a></td>
        <td class='red'>No</td>
        <td class='green'>Yes</td>
        <td class='red'>No</td>
        <td>Six unit sphere shells with b = 1000, 2000, 3000, 4000, 5000, 6000 s/mm^2 along 19, 32, 56, 87, 125, and 170 directions (see <a href='https://doi.org/10.1016/j.neuroimage.2019.05.078'>Olson 2019</a> for candidate sub-sampling schemes)</td>
        <td>
            <a href='https://doi.org/10.1016%2Fj.neuroimage.2013.04.016'>Ozarslan 2013</a>
            <a href='https://doi.org/10.1016/j.neuroimage.2019.05.078'>Olson 2019</a>
        </td>
    </tr>
    <tr>
        <td>MAPL</td>
        <td class='red'>No</td>
        <td class='green'>Yes</td>
        <td class='red'>No</td>
        <td></td>
        <td><a href=''>Fick 2016</a></td>
    </tr>
    <tr>
        <td><a href='#constrained-spherical-deconvolution'>CSD</a></td>
        <td class='green'>Yes</td>
        <td class='red'>No</td>
        <td class='red'>No</td>
        <td></td>
        <td>
            <a href=''>Tournier 2017</a>
            <a href=''>Descoteaux 2008</a>
            <a href=''>Tournier 2007</a></td>
    </tr>
    <tr>
        <td><a href='#reconst-mcsd'>SMS/MT CSD</a></td>
        <td class='red'>No</td>
        <td class='green'>Yes</td>
        <td class='red'>No</td>
        <td>5 b=0, 50 directions at 3 non-zero b-shells: b=1000, b=2000, b=3000</td>
        <td><a href='https://www.ncbi.nlm.nih.gov/pubmed/25109526'>Jeurissen 2014</a></td>
    </tr>
    <tr>
        <td><a href='#fiber-orientation-estimated-using-continuous-axially-symmetric-tensors'>ForeCast</a></td>
        <td class='red'>No</td>
        <td class='green'>Yes</td>
        <td class='red'>No</td>
        <td></td>
        <td><a href=''>Anderson 2005</a></td>
    </tr>
    <tr>
        <td><a href='#robust-and-unbiased-model-based-spherical-deconvolution-(rumba-sd)'>RUMBA-SD</a></td>
        <td class='green'>Yes</td>
        <td class='green'>Yes</td>
        <td class='green'>Yes</td>
        <td></td>
        <td><a href='https://doi.org/10.1371/journal.pone.0138910'>Canales-Rodríguez 2015</a></td>
    </tr>
    <tr>
        <td><a href='#q-space-trajectory-imaging'>QTI</a></td>
        <td class='red'>No</td>
        <td class='green'>Yes</td>
        <td class='red'>No</td>
        <td>Evenly distributed geometric sampling scheme of 216 measurements, 5 b-values (50, 250, 50, 1000, 200mm/s^2), measurement tensors of four shapes: stick, prolate, sphere, and plane</td>
        <td><a href='https://doi.org/10.1016/j.neuroimage.2016.02.039'>Westin 2016</a></td>
    </tr>
    <tr>
        <td><a href='#q-space-trajectory-imaging-with-positivity-constraints'>QTI+</a></td>
        <td class='red'>No</td>
        <td class='green'>Yes</td>
        <td class='red'>No</td>
        <td>At least one b=0, minimum of 39 acquisitions with spherical and linear encoding; optimal 120 (see <a href='https://doi.org/10.1002/hbm.26175'>Morez 2023</a>), ideal 217 see <a href='https://www.sciencedirect.com/science/article/pii/S1053811921004754?via%3Dihub#tbl0001'>Herberthson 2021 Table 1</a></td>
        <td>
            <a href='https://doi.org/10.1016/j.neuroimage.2021.118198'>Herberthson 2021</a>
            <a href='https://doi.org/10.1002/hbm.26175'>Morez 2023</a>
        </td>
    </tr>
    <tr>
        <td>Ball & Stick</td>
        <td class='green'>Yes</td>
        <td class='green'>Yes</td>
        <td class='red'>No</td>
        <td>Three b=0, 60 evenly distributed directions per <a href='https://doi.org/10.1002/(SICI)1522-2594(199909)42:3%3C515::AID-MRM14%3E3.0.CO;2-Q'>Jones 1999</a> at b-value 1000mm/s^2</td>
        <td><a href='https://onlinelibrary.wiley.com/doi/full/10.1002/mrm.10609'>Behrens 2003</a></td>
    </tr>
        <tr>
        <td><a href='#studying-diffusion-time-dependence-using-qt-dmri'>QTau-MRI</a></td>
        <td class='red'>No</td>
        <td class='green'>Yes</td>
        <td class='red'>No</td>
        <td>Minimum 200 volumes of multi-spherical dMRI (multi-shell, multi-diffusion time; varying gradient directions, gradient strengths, and diffusion times)</td>
        <td><a href=''>Fick 2017</a></td>
    </tr>
    </tr>
        <tr>
        <td>Power Map</td>
        <td class='green'>No</td>
        <td class='green'>Yes</td>
        <td class='red'>No</td>
        <td>HARDI data with 60 directions at b-value = 3000 s/mm^2, 7 b=0 (Minimum: HARDI data with at least 30 directions)</td>
        <td><a href='http://archive.ismrm.org/2014/0730.html'>DellAcqua2014</a></td>
    </tr>
    </tr>
        <tr>
        <td><a href='#reconst-msdki'>SMT / SMT2</a></td>
        <td class='red'>No</td>
        <td class='green'>Yes</td>
        <td class='red'>No</td>
        <td>72 directions at each of 5 evenly spaced b-values from 0.5 to 2.5 ms/μm2, 5 b-values from 3 to 5 ms/μm2, 5 b-values from 5.5 to 7.5 ms/μm2, and 3 b-values from 8 to 9 ms/μm2 /  b=0 ms/μm^-2, and along 33 directions at b-values from 0.2–3 ms/μm^-2 in steps of 0.2 ms/μm^−2 (24 point spherical design and 9 directions identified for rapid kurtosis estimation)</td>
        <td><a href='https://doi.org/10.1002/mrm.27606'>NetoHe2019</a>, <a href='https://www.nature.com/articles/sdata201672'>Kaden2016b</a></td>
    </tr>
    </table>
