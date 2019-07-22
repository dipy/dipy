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
        <th>Methods</th>
        <th>Single Shell</th>
        <th>Multi Shell</th>
        <th>Cartesian</th>
        <th>Ideal Data Requirements</th>
        <th>Citations</th>
    </tr>
    <tr>
        <td><a href='#diffusion-tensor-imaging'>DTI (SLS, WLS, NNLS)</a></td>
        <td class='green'>Yes</td>
        <td class='green'>Yes</td>
        <td class='green'>Yes</td>
        <td></td>
        <td><a href='https://www.ncbi.nlm.nih.gov/pubmed/8130344'>Basser 1994</a></td>
    </tr>
    <tr>
        <td><a href='#diffusion-tensor-imaging'>FwDTI</a></td>
        <td class='red'>No</td>
        <td class='green'>Yes</td>
        <td class='red'>No</td>
        <td></td>
        <td><a href='https://www.ncbi.nlm.nih.gov/pubmed/19623619'>Pasternak 2009</a></td>
    </tr>
    <tr>
        <td><a href='#diffusion-kurtosis-imaging'>DKI - Standard</a></td>
        <td class='red'>No</td>
        <td class='green'>Yes</td>
        <td class='red'>No</td>
        <td></td>
        <td><a href='https://www.ncbi.nlm.nih.gov/pubmed/15906300'>Jensen 2005</a></td>
    </tr>
    <tr>
        <td><a href='#diffusion-kurtosis-imaging'>DKI - Micro</a></td>
        <td class='red'>No</td>
        <td class='green'>Yes</td>
        <td class='red'>No</td>
        <td></td>
        <td><a href='https://www.sciencedirect.com/science/article/pii/S1053811911006148'>Fieremans 2011</a></td>
    </tr>
    <tr>
        <td><a href='#q-ball-constant-solid-angle'>CSA</a></td>
        <td class='green'>Yes</td>
        <td class='red'>No</td>
        <td class='red'>No</td>
        <td></td>
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
        <td></td>
        <td><a href=''>LeBihan 1984</a></td>
    </tr>
    <tr>
        <td>SDT</td>
        <td class='green'>Yes</td>
        <td class='red'>No</td>
        <td class='red'>No</td>
        <td></td>
        <td><a href=''>Descoteaux 2009</a></td>
    </tr>
    <tr>
        <td><a href='#diffusion-spectrum-imaging'>DSI</a></td>
        <td class='red'>No</td>
        <td class='red'>No</td>
        <td class='green'>Yes</td>
        <td></td>
        <td><a href=''>Wedeen 2008</a></td>
    </tr>
    <tr>
        <td><a href='#dsi-with-deconvolution'>DSID</a></td>
        <td class='red'>No</td>
        <td class='red'>No</td>
        <td class='green'>Yes</td>
        <td></td>
        <td><a href=''>Canales-Rodriguez 2010</a></td>
    </tr>
    <tr>
        <td><a href='#generalized-q-sampling-imaging'>GQI - GQI2</a></td>
        <td class='red'>No</td>
        <td class='green'>Yes</td>
        <td class='green'>Yes</td>
        <td></td>
        <td><a href=''>Yeh 2010</a></td>
    </tr>
    <tr>
        <td><a href='#sparse-fascicle-model'>SFM</a></td>
        <td class='green'>Yes</td>
        <td class='green'>Yes</td>
        <td class='red'>No</td>
        <td></td>
        <td><a href=''>Rokem 2015</a></td>
    </tr>
    <tr>
        <td><a href='#q-ball-constant-solid-angle'>Q-Ball</a></td>
        <td class='green'>Yes</td>
        <td class='red'>No</td>
        <td class='red'>No</td>
        <td></td>
        <td><a href=''>Tuch 2004</a></td>
    </tr>
    <tr>
        <td><a href='#simple-harmonic-oscillator-based-reconstruction-and-estimation'>SHORE</a></td>
        <td class='red'>No</td>
        <td class='green'>Yes</td>
        <td class='red'>No</td>
        <td></td>
        <td><a href=''>Merlet 2013</a></td>
    </tr>
    <tr>
        <td><a href='#mean-apparent-propagator-map-mri'>MAP-MRI</a></td>
        <td class='red'>No</td>
        <td class='green'>Yes</td>
        <td class='red'>No</td>
        <td></td>
        <td><a href=''>Ozarslan 2013</a></td>
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
        <td><a href=''>Tournier 2017</a></td>
    </tr>
    <tr>
        <td>M-CSD</td>
        <td class='red'>No</td>
        <td class='green'>Yes</td>
        <td class='red'>No</td>
        <td></td>
        <td><a href=''>Jeurissen 2009</a></td>
    </tr>
    <tr>
        <td>MS/MT CSD</td>
        <td class='red'>No</td>
        <td class='green'>Yes</td>
        <td class='red'>No</td>
        <td></td>
        <td></td>
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
        <td>OPDT</td>
        <td class='green'>Yes</td>
        <td class='red'>No</td>
        <td class='red'>No</td>
        <td></td>
        <td><a href=''>Tristan-Vega 2010</a></td>
    </tr>
    <tr>
        <td>ActiveAx</td>
        <td class='red'>No</td>
        <td class='green'>Yes</td>
        <td class='red'>No</td>
        <td></td>
        <td></td>
    </tr>
    <tr>
        <td>Noddi / NoddiX</td>
        <td class='red'>No</td>
        <td class='green'>Yes</td>
        <td class='red'>No</td>
        <td></td>
        <td></td>
    </tr>
    <tr>
        <td>Ball & Stick</td>
        <td class='green'>Yes</td>
        <td class='green'>Yes</td>
        <td class='red'>No</td>
        <td></td>
        <td></td>
    </tr>
        <tr>
        <td><a href='#studying-diffusion-time-dependence-using-qt-dmri'>QTau-MRI</a></td>
        <td class='red'>No</td>
        <td class='green'>Yes</td>
        <td class='red'>No</td>
        <td></td>
        <td><a href=''>Fick 2017</a></td>
    </tr>
    </table>
