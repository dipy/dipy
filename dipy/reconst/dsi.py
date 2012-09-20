import numpy as np
from scipy.ndimage import map_coordinates
from dipy.reconst.recspeed import pdf_to_odf
from scipy.fftpack import fftn, fftshift
from .odf import OdfModel, OdfFit
from .cache import Cache
from dipy.core.onetime import auto_attr


class DiffusionSpectrumModel(OdfModel, Cache):

    def __init__(self, gtab, method='standard'):
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

        The main assumptions for this model is fast gradient switching and that
        the acquisition gradients will sit on a keyhole Cartesian grid in
        q_space [3]_.
        
        Parameters
        ----------
        gtab: object, 
            GradientTable
        method: str, 
            'standard' or 'deconv'

        References
        ----------
        .. [1]  Wedeen V.J et. al, "Mapping Complex Tissue Architecture With
        Diffusion Spectrum Magnetic Resonance Imaging", MRM 2005.

        .. [2] Canales-Rodriguez E.J et a., "Deconvolution in Diffusion Spectrum
        Imaging", Neuroimage, 2010.

        .. [3] Garyfallidis E, "Towards an accurate brain tractography", PhD
        thesis, University of Cambridge, 2012.

        Examples
        --------
        Here we create an example where we provide the data, a gradient table 
        and a reconstruction sphere and calculate generalized FA for the first 
        voxel in the data.

        >>> from dipy.data import dsi_voxels
        >>> data, gtab = dsi_voxels()
        >>> from dipy.core.subdivide_octahedron import create_unit_sphere 
        >>> sphere = create_unit_sphere(5)
        >>> from dipy.reconst.dsi import DiffusionSpectrumModel
        >>> from dipy.reconst.odf import gfa
        >>> ds = DiffusionSpectrumModel(gtab)
        >>> np.round(gfa(ds.fit(data[0, 0, 0]).odf(sphere)), 2)
        0.12

        Notes
        ------
        Have in mind that DSI expects gradients on both hemispheres.
        If your gradients span only one hemisphere you need to duplicate        them and project them to the other hemisphere too.

        See Also
        --------
        dipy.reconst.gqi.GeneralizedQSampling

        """

        self.bvals = gtab.bvals
        self.bvecs = gtab.bvecs
        if method == 'standard':
            #3d volume for Sq
            self.qgrid_size = 16
            #necessary shifting for centering
            self.origin = 8
            #hanning filter width
            self.filter_width = 32.
            #odf collecting radius
            self.qradius = np.arange(2.1, 6, .2)
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
        """ Calculates PDF and ODF for a single voxel

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
    
    @auto_attr
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
        Pr=fftshift(np.abs(np.real(fftn(fftshift(Sq), 3 * (self.qgrid_sz, )))))
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
        Pr = self.pdf
        #calculate the orientation distribution function
        return pdf_odf(Pr, sphere, self.model.qradius, interp_coords)


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
    Pr: array, shape (X, X, X)
            probability density function
    sphere: object,
            Sphere
    rradius: array, shape (N,)
            interpolation range on the radius
    interp_coords: array, shape (N, 3)
            coordinates in the pdf for interpolating the odf
    """

    verts_no = sphere.vertices.shape[0]
    odf = np.zeros(verts_no)
    rradius_no = len(rradius)
    #interp_coords = pdf_interp_coords(sphere, rradius, origin)
    PrIs = map_coordinates(Pr, interp_coords, order=1)
    pdf_to_odf(odf, PrIs, rradius, verts_no, rradius_no)
    return odf


def project_hemisph_bvecs(bvals, bvecs):
    """ Project any near identical bvecs to the other hemisphere

    Parameters:
    -----------
    bvals: array, shape (N,)
            b-values
    bvecs: array, shape (N, 3)
            b-vectors

    Notes
    -------
    Useful when working with dsi data because the full q-space needs to be
    mapped in both hemispheres and perhaps only b-values and b-vectors are
    provided for the one hemisphere.
    """
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
