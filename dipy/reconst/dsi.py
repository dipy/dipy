import numpy as np
from scipy.ndimage import map_coordinates
from scipy.fftpack import fftn, fftshift, ifftshift
from dipy.reconst.odf import OdfModel, OdfFit
from dipy.reconst.cache import Cache
from dipy.reconst.multi_voxel import multi_voxel_fit


class DiffusionSpectrumModel(OdfModel, Cache):

    def __init__(self,
                 gtab,
                 qgrid_size=17,
                 r_start=2.1,
                 r_end=6.,
                 r_step=0.2,
                 filter_width=32,
                 normalize_peaks=False):
        r""" Diffusion Spectrum Imaging

        The theoretical idea underlying this method is that the diffusion
        propagator $P(\mathbf{r})$ (probability density function of the average
        spin displacements) can be estimated by applying 3D FFT to the signal
        values $S(\mathbf{q})$

        ..math::
            :nowrap:
                \begin{eqnarray}
                    P(\mathbf{r}) & = & S_{0}^{-1}\int S(\mathbf{q})\exp(-i2\pi\mathbf{q}\cdot\mathbf{r})d\mathbf{r}
                \end{eqnarray}

        where $\mathbf{r}$ is the displacement vector and $\mathbf{q}$ is the
        wave vector which corresponds to different gradient directions. Method
        used to calculate the ODFs. Here we implement the method proposed by
        Wedeen et al. [1]_.

        The main assumption for this model is fast gradient switching and that
        the acquisition gradients will sit on a keyhole Cartesian grid in
        q_space [3]_.

        Parameters
        ----------
        gtab : GradientTable,
            Gradient directions and bvalues container class
        qgrid_size : int,
            has to be an odd number. Sets the size of the q_space grid.
            For example if qgrid_size is 17 then the shape of the grid will be
            ``(17, 17, 17)``.
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
        .. [1]  Wedeen V.J et al., "Mapping Complex Tissue Architecture With
        Diffusion Spectrum Magnetic Resonance Imaging", MRM 2005.

        .. [2] Canales-Rodriguez E.J et al., "Deconvolution in Diffusion
        Spectrum Imaging", Neuroimage, 2010.

        .. [3] Garyfallidis E, "Towards an accurate brain tractography", PhD
        thesis, University of Cambridge, 2012.

        Examples
        --------
        In this example where we provide the data, a gradient table
        and a reconstruction sphere, we calculate generalized FA for the first
        voxel in the data with the reconstruction performed using DSI.

        >>> import warnings
        >>> from dipy.data import dsi_voxels, default_sphere
        >>> data, gtab = dsi_voxels()
        >>> from dipy.reconst.dsi import DiffusionSpectrumModel
        >>> ds = DiffusionSpectrumModel(gtab)
        >>> dsfit = ds.fit(data)
        >>> from dipy.reconst.odf import gfa
        >>> np.round(gfa(dsfit.odf(default_sphere))[0, 0, 0], 2)
        0.11

        Notes
        -----
        A. Have in mind that DSI expects gradients on both hemispheres. If your
        gradients span only one hemisphere you need to duplicate the data and
        project them to the other hemisphere before calling this class. The
        function dipy.reconst.dsi.half_to_full_qspace can be used for this
        purpose.

        B. If you increase the size of the grid (parameter qgrid_size) you will
        most likely also need to update the r_* parameters. This is because
        the added zero padding from the increase of gqrid_size also introduces
        a scaling of the PDF.

        C. We assume that data only one b0 volume is provided.

        See Also
        --------
        dipy.reconst.gqi.GeneralizedQSampling

        """

        self.bvals = gtab.bvals
        self.bvecs = gtab.bvecs
        self.normalize_peaks = normalize_peaks
        # 3d volume for Sq
        if qgrid_size % 2 == 0:
            raise ValueError('qgrid_size needs to be an odd integer')
        self.qgrid_size = qgrid_size
        # necessary shifting for centering
        self.origin = self.qgrid_size // 2

        # hanning filter width
        self.filter = hanning_filter(gtab, filter_width, self.origin)
        # odf sampling radius
        self.qradius = np.arange(r_start, r_end, r_step)
        self.qradiusn = len(self.qradius)
        # create qspace grid
        self.qgrid = create_qspace(gtab, self.origin)
        b0 = np.min(self.bvals)
        self.dn = (self.bvals > b0).sum()
        self.gtab = gtab

    @multi_voxel_fit
    def fit(self, data):
        return DiffusionSpectrumFit(self, data)


class DiffusionSpectrumFit(OdfFit):

    def __init__(self, model, data):
        """ Calculates PDF and ODF and other properties for a single voxel

        Parameters
        ----------
        model : object,
            DiffusionSpectrumModel
        data : 1d ndarray,
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

    def pdf(self, normalized=True):
        """ Applies the 3D FFT in the q-space grid to generate
        the diffusion propagator
        """
        values = self.data * self.model.filter
        # create the signal volume
        Sq = np.zeros((self.qgrid_sz, self.qgrid_sz, self.qgrid_sz))
        # fill q-space

        for i in range(len(values)):
            qx, qy, qz = self.model.qgrid[i]
            Sq[qx, qy, qz] += values[i]
        # apply fourier transform
        Pr = fftshift(np.real(fftn(ifftshift(Sq), 3 * (self.qgrid_sz, ))))
        # clipping negative values to 0 (ringing artefact)
        Pr = np.clip(Pr, 0, Pr.max())

        # normalize the propagator to obtain a pdf
        if normalized:
            Pr /= Pr.sum()

        return Pr

    def rtop_signal(self, filtering=True):
        """ Calculates the return to origin probability (rtop) from the signal
        rtop equals to the sum of all signal values

        Parameters
        ----------
        filtering : boolean, optional
            Whether to perform Hanning filtering. Default: True

        Returns
        -------
        rtop : float
            the return to origin probability
        """

        if filtering:
            values = self.data * self.model.filter
        else:
            values = self.data

        rtop = values.sum()

        return rtop

    def rtop_pdf(self, normalized=True):
        r""" Calculates the return to origin probability from the propagator, which is
        the propagator evaluated at zero (see Descoteaux et Al. [1]_,
        Tuch [2]_, Wu et al. [3]_)
        rtop = P(0)

        Parameters
        ----------
        normalized : boolean, optional
            Whether to normalize the propagator by its sum in order to obtain a
            pdf. Default: True.

        Returns
        -------
        rtop : float
            the return to origin probability

        References
        ----------
        .. [1] Descoteaux M. et al., "Multiple q-shell diffusion propagator
        imaging", Medical Image Analysis, vol 15, No. 4, p. 603-621, 2011.

        .. [2] Tuch D.S., "Diffusion MRI of Complex Tissue Structure",
         PhD Thesis, 2002.

        .. [3] Wu Y. et al., "Computation of Diffusion Function Measures
        in q -Space Using Magnetic Resonance Hybrid Diffusion Imaging",
        IEEE TRANSACTIONS ON MEDICAL IMAGING, vol. 27, No. 6, p. 858-865, 2008

        """

        Pr = self.pdf(normalized=normalized)

        center = self.qgrid_sz // 2

        rtop = Pr[center, center, center]
        return rtop

    def msd_discrete(self, normalized=True):
        r""" Calculates the mean squared displacement on the discrete propagator

        ..math::
            :nowrap:
                \begin{equation}
                    MSD:{DSI}=\int_{-\infty}^{\infty}\int_{-\infty}^{\infty}\int_{-\infty}^{\infty} P(\hat{\mathbf{r}}) \cdot \hat{\mathbf{r}}^{2} \ dr_x \ dr_y \ dr_z
                \end{equation}

        where $\hat{\mathbf{r}}$ is a point in the 3D Propagator space
        (see Wu et al. [1]_).

        Parameters
        ----------
        normalized : boolean, optional
            Whether to normalize the propagator by its sum in order to obtain a
            pdf. Default: True

        Returns
        -------
        msd : float
            the mean square displacement

        References
        ----------
        .. [1] Wu Y. et al., "Hybrid diffusion imaging", NeuroImage, vol 36,
        p. 617-629, 2007.

        """

        Pr = self.pdf(normalized=normalized)

        # create the r squared 3D matrix
        gridsize = self.qgrid_sz
        center = gridsize // 2
        a = np.arange(gridsize) - center
        x = np.tile(a, (gridsize, gridsize, 1))
        y = np.tile(a.reshape(gridsize, 1), (gridsize, 1, gridsize))
        z = np.tile(a.reshape(gridsize, 1, 1), (1, gridsize, gridsize))
        r2 = x ** 2 + y ** 2 + z ** 2

        msd = np.sum(Pr * r2) / float((gridsize ** 3))
        return msd

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

        # calculate the orientation distribution function
        return pdf_odf(Pr, self.model.qradius, interp_coords)


def create_qspace(gtab, origin):
    """ create the 3D grid which holds the signal values (q-space)

    Parameters
    ----------
    gtab : GradientTable
    origin : (3,) ndarray
        center of qspace

    Returns
    -------
    qgrid : ndarray
        qspace coordinates
    """
    # create the q-table from bvecs and bvals
    qtable = create_qtable(gtab, origin)

    # center and index in qspace volume
    qgrid = qtable + origin
    return qgrid.astype('i8')


def create_qtable(gtab, origin):
    """ create a normalized version of gradients

    Parameters
    ----------
    gtab : GradientTable
    origin : (3,) ndarray
        center of qspace

    Returns
    -------
    qtable : ndarray
    """

    bv = gtab.bvals
    bsorted = np.sort(bv[np.bitwise_not(gtab.b0s_mask)])
    for i in range(len(bsorted)):
        bmin = bsorted[i]
        try:
            if np.sqrt(bv.max() / bmin) > origin + 1:
                continue
            else:
                break
        except ZeroDivisionError:
            continue

    bv = np.sqrt(bv / bmin)
    qtable = np.vstack((bv, bv, bv)).T * gtab.bvecs
    return np.floor(qtable + .5)


def hanning_filter(gtab, filter_width, origin):
    """ create a hanning window

    The signal is premultiplied by a Hanning window before
    Fourier transform in order to ensure a smooth attenuation
    of the signal at high q values.

    Parameters
    ----------
    gtab : GradientTable
    filter_width : int
    origin : (3,) ndarray
        center of qspace

    Returns
    -------
    filter : (N,) ndarray
        where N is the number of non-b0 gradient directions

    """
    qtable = create_qtable(gtab, origin)
    # calculate r - hanning filter free parameter
    r = np.sqrt(qtable[:, 0] ** 2 + qtable[:, 1] ** 2 + qtable[:, 2] ** 2)
    # setting hanning filter width and hanning
    return .5 * np.cos(2 * np.pi * r / filter_width)


def pdf_interp_coords(sphere, rradius, origin):
    """ Precompute coordinates for ODF calculation from the PDF

    Parameters
    ----------
    sphere : object,
            Sphere
    rradius : array, shape (N,)
            line interpolation points
    origin : array, shape (3,)
            center of the grid

    """
    interp_coords = rradius * sphere.vertices[np.newaxis].T
    origin = np.reshape(origin, [-1, 1, 1])
    interp_coords = origin + interp_coords
    return interp_coords


def pdf_odf(Pr, rradius, interp_coords):
    r""" Calculates the real ODF from the diffusion propagator(PDF) Pr

    Parameters
    ----------
    Pr : array, shape (X, X, X)
        probability density function
    rradius : array, shape (N,)
        interpolation range on the radius
    interp_coords : array, shape (3, M, N)
        coordinates in the pdf for interpolating the odf
    """
    PrIs = map_coordinates(Pr, interp_coords, order=1)
    odf = (PrIs * rradius ** 2).sum(-1)
    return odf


def half_to_full_qspace(data, gtab):
    """ Half to full Cartesian grid mapping

    Useful when dMRI data are provided in one qspace hemisphere as
    DiffusionSpectrum expects data to be in full qspace.

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
    before providing the gradients and the data to the DiffusionSpectrumModel
    class.
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

    Parameters
    ----------
    gtab : object,
            GradientTable

    Notes
    -----
    Useful only when working with some types of dsi data.
    """
    bvals = gtab.bvals
    bvecs = gtab.bvecs
    bvs = bvals[1:]
    bvcs = bvecs[1:]
    b = bvs[:, None] * bvcs
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
    bvecs2 = bvecs.copy()
    for (i, j) in pairs:
        bvecs2[1 + j] = - bvecs2[1 + j]
    return bvecs2, pairs


class DiffusionSpectrumDeconvModel(DiffusionSpectrumModel):

    def __init__(self, gtab, qgrid_size=35, r_start=4.1, r_end=13.,
                 r_step=0.4, filter_width=np.inf, normalize_peaks=False):
        r""" Diffusion Spectrum Deconvolution

        The idea is to remove the convolution on the DSI propagator that is
        caused by the truncation of the q-space in the DSI sampling.

        ..math::
            :nowrap:
                \begin{eqnarray*}
                    P_{dsi}(\mathbf{r}) & = & S_{0}^{-1}\iiint\limits_{\| \mathbf{q} \| \le \mathbf{q_{max}}} S(\mathbf{q})\exp(-i2\pi\mathbf{q}\cdot\mathbf{r})d\mathbf{q} \\
                    & = & S_{0}^{-1}\iiint\limits_{\mathbf{q}} \left( S(\mathbf{q}) \cdot M(\mathbf{q}) \right) \exp(-i2\pi\mathbf{q}\cdot\mathbf{r})d\mathbf{q} \\
                    & = & P(\mathbf{r}) \otimes \left( S_{0}^{-1}\iiint\limits_{\mathbf{q}}  M(\mathbf{q}) \exp(-i2\pi\mathbf{q}\cdot\mathbf{r})d\mathbf{q} \right) \\
                \end{eqnarray*}

        where $\mathbf{r}$ is the displacement vector and $\mathbf{q}$ is the
        wave vector which corresponds to different gradient directions,
        $M(\mathbf{q})$ is a mask corresponding to your q-space sampling and
        $\otimes$ is the convolution operator [1]_.


        Parameters
        ----------
        gtab : GradientTable,
            Gradient directions and bvalues container class
        qgrid_size : int,
            has to be an odd number. Sets the size of the q_space grid.
            For example if qgrid_size is 35 then the shape of the grid will be
            ``(35, 35, 35)``.
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
        .. [1] Canales-Rodriguez E.J et al., "Deconvolution in Diffusion
        Spectrum Imaging", Neuroimage, 2010.

        .. [2] Biggs David S.C. et al., "Acceleration of Iterative Image
        Restoration Algorithms", Applied Optics, vol. 36, No. 8, p. 1766-1775,
        1997.

        """
        DiffusionSpectrumModel.__init__(self, gtab, qgrid_size,
                                        r_start, r_end, r_step,
                                        filter_width,
                                        normalize_peaks)

    @multi_voxel_fit
    def fit(self, data):
        return DiffusionSpectrumDeconvFit(self, data)


class DiffusionSpectrumDeconvFit(DiffusionSpectrumFit):

    def pdf(self):
        """ Applies the 3D FFT in the q-space grid to generate
        the DSI diffusion propagator, remove the background noise with a
        hard threshold and then deconvolve the propagator with the
        Lucy-Richardson deconvolution algorithm
        """
        values = self.data
        # create the signal volume
        Sq = np.zeros((self.qgrid_sz, self.qgrid_sz, self.qgrid_sz))
        # fill q-space
        for i in range(len(values)):
            qx, qy, qz = self.model.qgrid[i]
            Sq[qx, qy, qz] += values[i]
        # get deconvolution PSF
        DSID_PSF = self.model.cache_get('deconv_psf', key=self.model.gtab)
        if DSID_PSF is None:
            DSID_PSF = gen_PSF(self.model.qgrid, self.qgrid_sz,
                               self.qgrid_sz, self.qgrid_sz)
        self.model.cache_set('deconv_psf', self.model.gtab, DSID_PSF)
        # apply fourier transform
        Pr = fftshift(np.abs(np.real(fftn(ifftshift(Sq),
                                          3 * (self.qgrid_sz, )))))
        # threshold propagator
        Pr = threshold_propagator(Pr)
        # apply LR deconvolution
        Pr = LR_deconv(Pr, DSID_PSF, 5, 2)
        return Pr


def threshold_propagator(P, estimated_snr=15.):
    """
    Applies hard threshold on the propagator to remove background noise for the
    deconvolution.
    """
    P_thresholded = P.copy()
    threshold = P_thresholded.max() / float(estimated_snr)
    P_thresholded[P_thresholded < threshold] = 0
    return P_thresholded / P_thresholded.sum()


def gen_PSF(qgrid_sampling, siz_x, siz_y, siz_z):
    """
    Generate a PSF for DSI Deconvolution by taking the ifft of the binary
    q-space sampling mask and truncating it to keep only the center.
    """
    Sq = np.zeros((siz_x, siz_y, siz_z))
    # fill q-space
    for i in range(qgrid_sampling.shape[0]):
        qx, qy, qz = qgrid_sampling[i]
        Sq[qx, qy, qz] = 1
    return Sq * np.real(np.fft.fftshift(np.fft.ifftn(np.fft.ifftshift(Sq))))


def LR_deconv(prop, psf, numit=5, acc_factor=1):
    r"""
    Perform Lucy-Richardson deconvolution algorithm on a 3D array.

    Parameters
    ----------
    prop : 3-D ndarray of dtype float
        The 3D volume to be deconvolve
    psf : 3-D ndarray of dtype float
        The filter that will be used for the deconvolution.
    numit : int
        Number of Lucy-Richardson iteration to perform.
    acc_factor : float
        Exponential acceleration factor as in [1]_.

    References
    ----------
    .. [1] Biggs David S.C. et al., "Acceleration of Iterative Image
       Restoration Algorithms", Applied Optics, vol. 36, No. 8, p. 1766-1775,
       1997.

    """

    eps = 1e-16
    # Create the otf of the same size as prop
    otf = np.zeros_like(prop)
    # prop.ndim==3
    otf[otf.shape[0] // 2 - psf.shape[0] // 2:otf.shape[0] // 2 +
        psf.shape[0] // 2 + 1, otf.shape[1] // 2 - psf.shape[1] // 2:
        otf.shape[1] // 2 + psf.shape[1] // 2 + 1, otf.shape[2] // 2 -
        psf.shape[2] // 2:otf.shape[2] // 2 + psf.shape[2] // 2 + 1] = psf
    otf = np.real(np.fft.fftn(np.fft.ifftshift(otf)))
    # Enforce Positivity
    prop = np.clip(prop, 0, np.inf)
    prop_deconv = prop.copy()
    for it in range(numit):
        # Blur the estimate
        reBlurred = np.real(np.fft.ifftn(otf * np.fft.fftn(prop_deconv)))
        reBlurred[reBlurred < eps] = eps
        # Update the estimate
        prop_deconv = prop_deconv * (
            np.real(np.fft.ifftn(
                otf * np.fft.fftn((prop / reBlurred) + eps)))) ** acc_factor
        # Enforce positivity
        prop_deconv = np.clip(prop_deconv, 0, np.inf)
    return prop_deconv / prop_deconv.sum()


if __name__ == '__main__':
    pass
