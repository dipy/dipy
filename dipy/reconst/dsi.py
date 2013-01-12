import numpy as np
from scipy.ndimage import map_coordinates
from dipy.reconst.recspeed import pdf_to_odf
from scipy.fftpack import fftn, fftshift, ifftshift
from dipy.reconst.odf import OdfModel, OdfFit, gfa
from dipy.reconst.cache import Cache
from dipy.reconst.multi_voxel import multi_voxel_model
from dipy.reconst.recspeed import local_maxima, remove_similar_vertices


@multi_voxel_model
class DiffusionSpectrumModel(OdfModel, Cache):

    def __init__(self, 
                 gtab,
                 method='standard',
                 qgrid_size=16,
                 r_start=2.1,
                 r_end=6.,
                 r_step=0.2,
                 filter_width=32,
                 normalize_peaks=False):
        r""" Diffusion Spectrum Imaging

        The main idea here is that you can create the diffusion propagator
        $P(\mathbf{r})$ (probability density function of the average spin displacements) by
        applying 3D FFT to the signal values $S(\mathbf{q})$

        ..math::
            :nowrap:
                \begin{eqnarray}
                    P(\mathbf{r}) & = & S_{0}^{-1}\int S(\mathbf{q})\exp(-i2\pi\mathbf{q}\cdot\mathbf{r})d\mathbf{r}
                \end{eqnarray}

        where $\mathbf{r}$ is the displacement vector and $\mathbf{q}$ is the
        wavector which corresponds to different gradient directions. 

        The standard method is based on [1]_ and the deconvolution method is based
        on [2]_.

        The main assumption for this model is fast gradient switching and that
        the acquisition gradients will sit on a keyhole Cartesian grid in
        q_space [3]_.

        Parameters
        ----------
        gtab : GradientTable,
            Gradient directions and bvalues container class
        method : ('standard' or 'deconv')
            Method used to calculate the ODFs. 'standard' is the method
            proposed by Wedeen et. al [1]_ and 'deconv' is the method proposed
            by Canales-Rodriguez et al. [2]_
        qgrid_size : int,
            sets the size of the q_space grid. For example if qgrid_size is 16
            then the shape of the grid will be ``(16, 16, 16)``.
        r_start : float,
            ODF is sampled radially in the PDF. This parameters shows where the
            sampling should start.
        r_end : float,
            Radial endpoint of ODF sampling
        r_step : float,
            Step size of the ODf sampling from r_start to r_end
        filter_width : float,
            Strength of the hanning filter

        References
        ----------
        .. [1]  Wedeen V.J et. al, "Mapping Complex Tissue Architecture With
        Diffusion Spectrum Magnetic Resonance Imaging", MRM 2005.

        .. [2] Canales-Rodriguez E.J et. al, "Deconvolution in Diffusion Spectrum
        Imaging", Neuroimage, 2010.

        .. [3] Garyfallidis E, "Towards an accurate brain tractography", PhD
        thesis, University of Cambridge, 2012.

        Examples
        --------
        Here we create an example where we provide the data, a gradient table 
        and a reconstruction sphere and calculate generalized FA for the first 
        voxel in the data.

        >>> from dipy.data import dsi_voxels, get_sphere
        >>> data, gtab = dsi_voxels()
        >>> sphere = get_sphere('symmetric724')
        >>> from dipy.reconst.dsi import DiffusionSpectrumModel
        >>> ds = DiffusionSpectrumModel(gtab)
        >>> ds.direction_finder.config(sphere=sphere, 
                                       min_separation_angle=25,
                                       relative_peak_threshold=.35)
        >>> dsfit = ds.fit(data)
        >>> np.round(dsfit.gfa[0, 0, 0], 2)
        0.12

        Notes
        ------
        A. Have in mind that DSI expects gradients on both hemispheres. If your
        gradients span only one hemisphere you need to duplicate the data and
        project them to the other hemisphere before calling this class. The
        function dipy.reconst.dsi.half_to_full_qspace can be used for this
        purpose.

        B. If you increase the size of the grid (parameter qgrid_size) you will
        most likely also need to update the r_* parameters. This is because
        the added zero padding from the increase of gqrid_size also introduces
        a scaling of the PDF.

        C. We assume that data have only one b0 volume.

        See Also
        --------
        dipy.reconst.gqi.GeneralizedQSampling

        """

        self.bvals = gtab.bvals
        self.bvecs = gtab.bvecs
        self.normalize_peaks = normalize_peaks

        if method == 'standard':
            #3d volume for Sq
            self.qgrid_size = qgrid_size
            #necessary shifting for centering
            self.origin = self.qgrid_size // 2
            #hanning filter width
            self.filter_width = filter_width
            #odf collecting radius
            self.qradius = np.arange(r_start, r_end, r_step)
            self.create_qspace()
            self.hanning_filter()
        if method == 'deconv':
            raise NotImplementedError()
        b0 = np.min(self.bvals)
        self.dn = (self.bvals > b0).sum()

    def create_qspace(self):
        """ create the 3D grid which will hold the signal values
        """
        #create the q-table from bvecs and bvals
        bv = self.bvals
        bmin = np.sort(bv)[1]
        bv = np.sqrt(bv/bmin)
        qtable = np.vstack((bv,bv,bv)).T*self.bvecs
        qtable = np.floor(qtable+.5)
        self.qtable = qtable
        self.qradiusn = len(self.qradius)
        #center and index in qspace volume
        self.qgrid = qtable + self.origin
        self.qgrid = self.qgrid.astype('i8')

    def hanning_filter(self):
        """ create a hanning window

        The signal is premultiplied by a Hanning window before 
        Fourier transform in order to ensure a smooth attenuation 
        of the signal at high q values.

        """
        #calculate r - hanning filter free parameter
        r = np.sqrt(self.qtable[:, 0] ** 2 + 
                    self.qtable[:, 1] ** 2 + self.qtable[:, 2] ** 2)
        #setting hanning filter width and hanning
        self.filter = .5*np.cos(2*np.pi*r/self.filter_width)

    def fit(self, data):
        return DiffusionSpectrumFit(self, data)


class DiffusionSpectrumFit(OdfFit):

    def __init__(self, model, data):
        """ Calculates PDF and ODF and other properties for a single voxel

        Parameters:
        -----------
        model: object,
            DiffusionSpectrumModel
        data: 1d ndarray,
            signal values
        """
        self.model = model
        self.data = data
        self.qgrid_sz = self.model.qgrid_size
        self.dn = self.model.dn
        self._gfa = None
        self.npeaks = 5
        self._peak_values = None
        self._peak_indices = None

    def pdf(self):
        """ Applies the 3D FFT in the q-space grid to generate 
        the diffusion propagator
        """
        values = self.data * self.model.filter
        #create the signal volume
        Sq = np.zeros((self.qgrid_sz, self.qgrid_sz, self.qgrid_sz))
        #fill q-space
        for i in range(self.dn):
            qx, qy, qz = self.model.qgrid[i]
            Sq[qx, qy, qz] += values[i]
        #apply fourier transform
        Pr=fftshift(np.abs(np.real(fftn(ifftshift(Sq), 3 * (self.qgrid_sz, )))))
        return Pr

    def odf(self, sphere):
        r""" Calculates the real discrete odf for a given discrete sphere

        ..math::
            :nowrap:
                \begin{equation}
                    \psi_{DSI}(\hat{\mathbf{u}})=\int_{0}^{\infty}P(r\hat{\mathbf{u}})r^{2}dr
                \end{equation}

        where $\hat{\mathbf{u}}$ is the unit vector which corresponds to a
        sphere point.
        """
        interp_coords = self.model.cache_get('interp_coords',
                                             key=sphere)
        if interp_coords is None:
            interp_coords = pdf_interp_coords(sphere,
                                              self.model.qradius,
                                              self.model.origin)
            self.model.cache_set('interp_coords', sphere, interp_coords)

        Pr = self.pdf()

        #calculate the orientation distribution function
        odf = pdf_odf(Pr, sphere, self.model.qradius, interp_coords)

        # We compute the gfa here, since we have the ODF available and
        # the storage requirements are minimal.  Otherwise, we'd need to
        # recompute the ODF at significant computational expense.
        self._gfa = gfa(odf)
        pk, ind = local_maxima(odf, sphere.edges)

        relative_peak_threshold = self.model.direction_finder._config["relative_peak_threshold"]
        min_separation_angle = self.model.direction_finder._config["min_separation_angle"]

        # Remove small peaks.
        gt_threshold = pk >= (relative_peak_threshold * pk[0])
        pk = pk[gt_threshold]
        ind = ind[gt_threshold]

        # Keep peaks which are unique, which means remove peaks that are too
        # close to a larger peak.
        _, where_uniq = remove_similar_vertices(sphere.vertices[ind],
                                                min_separation_angle,
                                                return_index=True)
        pk = pk[where_uniq]
        ind = ind[where_uniq]

        # Calculate peak metrics
        #global_max = max(global_max, pk[0])
        n = min(self.npeaks, len(pk))
        #qa_array[i, :n] = pk[:n] - odf.min()
        self._peak_values = np.zeros(self.npeaks)
        self._peak_indices = np.zeros(self.npeaks)
        if self.model.normalize_peaks:
            self._peak_values[:n] = pk[:n] / pk[0]
        else:
            self._peak_values[:n] = pk[:n]
        self._peak_indices[:n] = ind[:n]

        return odf

    @property
    def gfa(self):
        if self._gfa is None:
            # Borrow default sphere from direction finder
            self.odf(self.model.direction_finder._config["sphere"])
        return self._gfa

    @property
    def peak_values(self):
        if self._peak_values is None:
            self.odf(self.model.direction_finder._config["sphere"])
        return self._peak_values

    @property
    def peak_indices(self):
        if self._peak_indices is None:
            self.odf(self.model.direction_finder._config["sphere"])
        return self._peak_indices


def pdf_interp_coords(sphere, rradius, origin):
    """ Precompute coordinates for ODF calculation from the PDF

    Parameters:
    -----------
    sphere : object,
            Sphere
    rradius : array, shape (M,)
            line interpolation points
    origin : array, shape (3,)
            center of the grid
    """
    interp_coords = rradius * sphere.vertices[np.newaxis].T
    interp_coords = interp_coords.reshape((3, -1))
    interp_coords = origin + interp_coords
    return interp_coords


def pdf_odf(Pr, sphere, rradius, interp_coords):
    r""" Calculates the real ODF from the diffusion propagator(PDF) Pr

    Parameters
    ----------
    Pr : array, shape (X, X, X)
        probability density function
    sphere : Sphere
        reconstruction sphere
    rradius : array, shape (N,)
        interpolation range on the radius
    interp_coords : array, shape (N, 3)
        coordinates in the pdf for interpolating the odf
    """
    verts_no = sphere.vertices.shape[0]
    odf = np.zeros(verts_no)
    rradius_no = len(rradius)
    #interp_coords = pdf_interp_coords(sphere, rradius, origin)
    PrIs = map_coordinates(Pr, interp_coords, order=1)
    pdf_to_odf(odf, PrIs, rradius, verts_no, rradius_no)
    return odf


def half_to_full_qspace(data, gtab):
    """ Half to full Cartesian grid mapping

    Useful when dMRI data are provided in one qspace hemisphere as DiffusionSpectrum
    expects data to be in full qspace.

    Parameters
    ----------
    data : array, shape (X, Y, Z, W)
        where (X, Y, Z) volume size and W number of gradient directions
    gtab : GradientTable
        container for b-values and b-vectors (gradient directions)

    Returns
    -------
    new_data : array, shape (X, Y, Z, 2 * W -1)
    new_gtab : GradientTable

    Notes
    -----
    We assume here that only on b0 is provided with the initial data. If that
    is not the case then you will need to write your own preparation function
    before providing the gradients and the data to the DiffusionSpectrumModel class.
    """
    bvals = gtab.bvals
    bvecs = gtab.bvecs
    bvals = np.append(bvals, bvals[1:])
    bvecs = np.append(bvecs, - bvecs[1:], axis=0)
    data = np.append(data, data[..., 1:], axis=-1)
    gtab.bvals = bvals.copy()
    gtab.bvecs = bvecs.copy()
    return data, gtab


def project_hemisph_bvecs(gtab):
    """ Project any near identical bvecs to the other hemisphere

    Parameters:
    -----------
    gtab : object,
            GradientTable

    Notes
    -------
    Useful only when working with some types of dsi data.
    """
    bvals = gtab.bvals
    bvecs = gtab.bvecs
    bvs = bvals[1:]
    bvcs = bvecs[1:]
    b = bvs[:,None] * bvcs
    bb = np.zeros((len(bvs), len(bvs)))
    pairs = []
    for (i, vec) in enumerate(b):
        for (j, vec2) in enumerate(b):
            bb[i, j] = np.sqrt(np.sum((vec - vec2) ** 2))
        I = np.argsort(bb[i])
        for j in I:
            if j != i:
                break
        if (j, i) in pairs:
            pass
        else:
            pairs.append((i, j))
    bvecs2=bvecs.copy()
    for (i, j) in pairs:
        bvecs2[1 + j] =- bvecs2[1 + j]
    return bvecs2, pairs


if __name__ == '__main__':
    pass
