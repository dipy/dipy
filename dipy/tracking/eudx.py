import numpy as np

from dipy.tracking.propspeed import eudx_both_directions
from dipy.data import get_sphere


class EuDX(object):

    '''Euler Delta Crossings

    Generates tracks with termination criteria defined by a delta function [1]_
    and it has similarities with FACT algorithm [2]_ and Basser's method
    but uses trilinear interpolation.

    Can be used with any reconstruction method as DTI, DSI, QBI, GQI which can
    calculate an orientation distribution function and find the local peaks of
    that function. For example a single tensor model can give you only
    one peak a dual tensor model 2 peaks and quantitative anisotropy
    method as used in GQI can give you 3,4,5 or even more peaks.

    The parameters of the delta function are checking thresholds for the
    direction propagation magnitude and the angle of propagation.

    A specific number of seeds is defined randomly and then the tracks
    are generated for that seed if the delta function returns true.

    Trilinear interpolation is being used for defining the weights of
    the propagation.

    References
    ------------
    .. [1] Garyfallidis, Towards an accurate brain tractography, PhD thesis,
           University of Cambridge, 2012.
    .. [2] Mori et al. Three-dimensional tracking of axonal projections
           in the brain by magnetic resonance imaging. Ann. Neurol. 1999.

    Notes
    -----
    The coordinate system of the tractography is that of native space of image
    coordinates not native space world coordinates therefore voxel size is
    always considered as having size (1,1,1).  Therefore, the origin is at the
    center of the center of the first voxel of the volume and all i,j,k
    coordinates start from the center of the voxel they represent.

    '''

    def __init__(self, a, ind,
                 seeds=10000,
                 odf_vertices=None,
                 a_low=0.0239,
                 step_sz=0.5,
                 ang_thr=60.,
                 length_thr=0.,
                 total_weight=.5,
                 max_points=1000,
                 voxel_origin='center'):
        '''
        Euler integration with multiple stopping criteria and supporting
        multiple multiple fibres in crossings [1]_.

        Parameters
        ------------
        a : array, 
            Shape (I, J, K, Np), magnitude of the peak of a scalar anisotropic 
            function e.g. QA (quantitative anisotropy) where Np is the number of
            peaks or a different function of shape (I, J, K) e.g FA or GFA.
        ind : array, shape(x,y,z,Np)
            indices of orientations of the scalar anisotropic peaks found on the
            resampling sphere
        seeds : int or ndarray
            If an int is specified then a number of random seeds is generated
            everywhere in the volume. If an (N, 3) array of (I, J, K) coordinates
            is given then these will be used for seeds. The latter is useful when
            you need to track from specific regions e.g. the white/gray matter 
            interface or a specific ROI e.g. in the corpus callosum.
        odf_vertices : None or ndarray, shape (N, 3), optional
            sphere points which define a discrete representation of orientations
            for the peaks, the same for all voxels. Usually the same sphere is
            used as an input for a reconstruction algorithm e.g. DSI.
            None results in loading the vertices from a default sphere with
            362 points.
        a_low : float, optional
            low threshold for QA(typical 0.023)  or FA(typical 0.2) or any other
            anisotropic function
        step_sz : float, optional
            euler propagation step size
        ang_thr : float, optional
            if turning angle is bigger than this threshold then tracking stops.
        total_weight : float, optional
            total weighting threshold
        max_points : int, optional
            maximum number of points in a track. Used to stop tracks from looping for ever
        voxel_origin : str
            If `corner` then the streamlines will be shifted (0.5, 0.5, 0.5) else
            if `center` they will stay as they are. 

        Returns
        -------
        generator : obj
            By iterating this generator you can obtain all the streamlines.


        Examples
        --------
        >>> import nibabel as nib
        >>> from dipy.reconst.dti import TensorModel, quantize_evecs
        >>> from dipy.data import get_data
        >>> from dipy.core.gradients import gradient_table
        >>> fimg,fbvals,fbvecs = get_data('small_101D')
        >>> img = nib.load(fimg)
        >>> affine = img.get_affine()
        >>> data = img.get_data()
        >>> gtab = gradient_table(fbvals, fbvecs)
        >>> model = TensorModel(gtab)
        >>> ten = model.fit(data)
        >>> ind = quantize_evecs(ten.evecs)
        >>> eu = EuDX(a=ten.fa, ind=ind, seeds=100,a_low=.2)
        >>> tracks = [e for e in eu]

        Notes
        -------
        This works as an iterator class because otherwise it could fill your
        entire memory if you generate many tracks.  Something very common as
        you can easily generate millions of tracks if you have many seeds.

        References
        ----------
        .. [1] E. Garyfallidis (2012), "Towards an accurate brain
               tractography", PhD thesis, University of Cambridge, UK.

        '''
        self.a = np.ascontiguousarray(a.copy(), dtype='f8')
        self.ind = np.ascontiguousarray(ind.copy(), dtype='f8')
        self.a_low = a_low
        self.ang_thr = ang_thr
        self.step_sz = step_sz
        self.length_thr = length_thr
        self.total_weight = total_weight
        self.max_points = max_points
        self.voxel_shift = 0
        if voxel_origin == 'corner': 
            self.voxel_shift = 0.5
        if len(self.a.shape) == 3:
            self.a.shape = self.a.shape + (1,)
            self.ind.shape = self.ind.shape + (1,)
        # store number of maximum peacks
        x, y, z, g = self.a.shape
        self.Np = g
        if odf_vertices == None:
            sphere = get_sphere('symmetric724')
            vertices, faces = sphere.vertices, sphere.faces
            self.odf_vertices = vertices
        else:
            self.odf_vertices = np.ascontiguousarray(odf_vertices, dtype='f8')
        try:
            if len(seeds) > 0:
                self.seed_list = seeds
                self.seed_no = len(seeds)
        except TypeError:
            self.seed_no = seeds
            self.seed_list = None
        self.ind = self.ind.astype(np.double)

    def __iter__(self):
        ''' This is were all the fun starts '''
        x, y, z, g = self.a.shape
        # for all seeds
        for i in range(self.seed_no):
            if self.seed_list == None:
                rx = (x - 1) * np.random.rand()
                ry = (y - 1) * np.random.rand()
                rz = (z - 1) * np.random.rand()
                seed = np.ascontiguousarray(
                    np.array([rx, ry, rz]), dtype=np.float64)
            else:
                seed = np.ascontiguousarray(
                    self.seed_list[i], dtype=np.float64)
            # for all peaks
            for ref in range(g):
                track = eudx_both_directions(seed.copy(),
                                             ref,
                                             self.a,
                                             self.ind,
                                             self.odf_vertices,
                                             self.a_low,
                                             self.ang_thr,
                                             self.step_sz,
                                             self.total_weight,
                                             self.max_points)

                if track == None:
                    pass
                else:
                    if track.shape[0] > 1:
                        yield track + self.voxel_shift
