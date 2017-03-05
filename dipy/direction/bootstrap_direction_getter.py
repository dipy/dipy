from warnings import warn
import numpy as np

from dipy.reconst import shm
from dipy.core.geometry import cart2sphere
from dipy.tracking.local.interpolation import trilinear_interpolate4d

from dipy.data import default_sphere
from dipy.direction.closest_peak import BaseDirectionGetter


default_SH = 4
class BootOdfGen(object):

    def __init__(self, data, model, sphere, sh_order=None, tol=1e-2):
        if sh_order is None:
            if hasattr(model, "sh_order"):
                sh_order = model.sh_order
            else:
                sh_order = default_SH
            
        self.where_dwi = shm.lazy_index(~model.gtab.b0s_mask)
        if not isinstance(self.where_dwi, slice):
            msg = ("For optimal bootstrap tracking consider reordering the "
                   "diffusion volumes so that all the b0 volumes are at the "
                   "beginning")
            warn(msg)
        x, y, z = model.gtab.gradients[self.where_dwi].T
        r, theta, phi = cart2sphere(x, y, z)
        b_range = (r.max() - r.min()) / r.min()
        if b_range > tol:
            raise ValueError("BootOdfGen only supports single shell data")

        B, m, n = shm.real_sym_sh_basis(sh_order, theta, phi)
        H = shm.hat(B)
        R = shm.lcr_matrix(H)

        self.data = np.asarray(data, "float64")
        self.model = model
        self.sphere = sphere
        self.H = H
        self.R = R

    def get_pmf(self, point):
        """Produces an ODF from a SH bootstrap sample"""
        single_vox_data = trilinear_interpolate4d(self.data, point)

        non_b0_data = self.where_dwi
        dwidata = single_vox_data[non_b0_data]
        bootdata = shm.bootstrap_data_voxel(dwidata, self.H, self.R)
        single_vox_data[non_b0_data] = bootdata
        fit = self.model.fit(single_vox_data)
        pmf = fit.odf(self.sphere)
        return pmf

    def pmf_no_boot(self, point):
        data = trilinear_interpolate4d(self.data, point)
        fit = self.model.fit(data)
        return fit.odf(self.sphere)


class BootDirectionGetter(BaseDirectionGetter):

    def __init__(self, pmfgen, maxangle, sphere=default_sphere,
                 max_attempts=5, **kwargs):
        self.max_attempts = max_attempts
        super(BootDirectionGetter, self).__init__(pmfgen, maxangle, sphere,
                                                  **kwargs)

    @classmethod
    def from_data(cls, data, model, max_angle, sphere=default_sphere,
                  sh_order=None, max_attempts=5, **kwargs):
        """Create a BootDirectionGetter using HARDI data and an ODF type model
        
        Parameters
        ----------
        data : ndarray, float, (..., N)
            Diffusion MRI data with N volumes.
        model : dipy diffusion model
            Must provide fit with odf method.
        max_angle : float (0, 90)
            Maximum angle between tract segments. This angle can be more
            generous (larger) than values typically used with probabilistic
            direction getters.
        sphere : Sphere
            The sphere used to sample the diffusion ODF.
        sh_order : even int
            The order of the SH "model" used to estimate bootstrap residuals.
        max_attempts : int
            Max number of bootstrap samples used to find tracking direction
            before giving up.
        pmf_threshold : float
            Threshold for ODF functions.
        relative_peak_threshold : float in [0., 1.]
            Relative threshold for excluding ODF peaks.
        min_separation_angle : float in [0, 90]
            Angular threshold for excluding ODF peaks.
        
        """
        boot_gen = BootOdfGen(data, model, sphere, sh_order=sh_order)
        return cls(boot_gen, max_angle, sphere, **kwargs)

    def initial_direction(self, point):
        """Returns best directions at seed location to start tracking.

        Parameters
        ----------
        point : ndarray, shape (3,)
            The point in an image at which to lookup tracking directions.

        Returns
        -------
        directions : ndarray, shape (N, 3)
            Possible tracking directions from point. ``N`` may be 0, all
            directions should be unique.

        """
        odf = self.pmf_gen.pmf_no_boot(point)
        return self._peak_directions(odf)

    def get_direction(self, point, direction):
        """Attempt direction getting on a few bootstrap samples.
        """
        count = 0
        no_valid_direction = True
        super_get_direction = super(BootDirectionGetter, self).get_direction
        while no_valid_direction and count < self.max_attempts:
            count += 1
            no_valid_direction = super_get_direction(point, direction)
        return no_valid_direction