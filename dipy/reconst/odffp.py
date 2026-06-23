"""Orientation Distribution Function Fingerprinting (ODF-FP).

ODF-FP :footcite:p:`Baete2019,Filipiak2022` reconstructs the diffusion ODF by
matching the ODF reconstructed from the measured signal against a dictionary of
ODF "fingerprints" simulated from a multi-compartment biophysical model. Each
voxel is aligned so that its main peak points to the pole, normalized, and
matched to the most similar dictionary fingerprint by penalized cosine
similarity. The microstructure parameters of the matched fingerprint are then
assigned to the voxel.

By default both the dictionary ODFs and the measured ODFs are reconstructed
with Generalized Q-Sampling Imaging (GQI), which is fast and keeps the two ODF
estimates consistent.

References
----------
.. footbibliography::
"""

import numpy as np

from dipy.core.geometry import sphere2cart
from dipy.core.sphere import Sphere
from dipy.data import get_sphere
from dipy.direction import peak_directions
from dipy.direction.peaks import PeaksAndMetrics
from dipy.reconst.base import ReconstFit, ReconstModel
from dipy.reconst.gqi import GeneralizedQSamplingModel
from dipy.reconst.multi_voxel import multi_voxel_fit
from dipy.reconst.odffp_matching import select_best_match
from dipy.reconst.shm import real_sh_descoteaux, sf_to_sh, sh_to_sf, smooth_pinv

DEFAULT_RECON_EDGE = 1.2
DEFAULT_DICT_EDGE = 1.2

DEFAULT_FIT_PENALTY = 1e-5
MAX_FIT_PENALTY = 0.1

# SH order used to resample ODFs between tessellations.
SH_ORDER_MAX = 4

# SH order used to store the ODFs in the peaks (PAM5) output.
DEFAULT_PEAKS_SH_ORDER = 8


def _default_sphere():
    return get_sphere(name="repulsion724")


def _rotation_to_pole(direction, pole=(0.0, 0.0, 1.0)):
    """Rotation matrix taking the unit vector ``direction`` onto ``pole``."""
    pole = np.asarray(pole)
    rotation = np.eye(3)
    cross = np.cross(direction, pole)
    sin_sqr = np.sum(cross**2)
    if sin_sqr != 0:
        skew = np.array(
            [
                [0, -cross[2], cross[1]],
                [cross[2], 0, -cross[0]],
                [-cross[1], cross[0], 0],
            ]
        )
        rotation += skew + np.dot(skew, skew) * (1 - np.dot(direction, pole)) / sin_sqr
    return rotation


def _sh_operators(sphere):
    """Real descoteaux07 SH synthesis ``B`` and analysis ``inv_B`` matrices.

    These reproduce :func:`~dipy.reconst.shm.sf_to_sh`/``sh_to_sf`` with no
    regularization, so that ``inv_B`` and ``B`` match the SH fit and evaluation
    used by :meth:`OdffpModel.resample_odf`.
    """
    basis, _, order = real_sh_descoteaux(
        SH_ORDER_MAX, sphere.theta, sphere.phi, legacy=False
    )
    inv_basis = smooth_pinv(basis, np.sqrt(0.0) * (-order * (order + 1)))
    return basis, inv_basis


class OdffpDictionary:
    """Dictionary of ODF fingerprints and their microstructure parameters.

    The fingerprints are simulated from a multi-compartment model (intra- and
    extra-axonal and free water) and reconstructed with ``odf_recon_model``
    (GQI by default) on a symmetric ``sphere``.
    """

    IDX_VOID = 0
    IDX_ISO = 1
    PREDEFINED_IDX_NUM = 2

    MICRO_DA = 0
    MICRO_DE = 1
    MICRO_DR = 2
    MICRO_FIN = 3
    MICRO_PARAMS_NUM = 4

    def __init__(self, gtab, sphere=None, dict_file=None):
        self.gtab = gtab
        self.sphere = sphere if sphere is not None else _default_sphere()
        self.max_peaks_num = 0
        self.odf = None
        self.peak_dirs = None
        self.micro = None
        self.ratio = None
        self.peaks_per_voxel = None
        if dict_file is not None:
            self.load(dict_file)

    # -- signal simulation ----------------------------------------------------

    @staticmethod
    def _interval(values):
        return np.array([np.min(values), np.max(values)])

    def _random_fraction_volumes(self, p_iso, p_fib, peaks_per_voxel):
        fraction_volumes = np.zeros(peaks_per_voxel + 1)

        # Lower bounds are hard limits; variability stays in [0, p_random_max].
        p_random_max = 1 - (p_iso[0] + peaks_per_voxel * p_fib[0])
        p_random = np.hstack(
            (
                np.random.uniform(0, p_iso[1] - p_iso[0]),
                np.random.uniform(0, p_fib[1] - p_fib[0], size=peaks_per_voxel),
            )
        )
        p_random /= np.maximum(1e-8, np.sum(p_random))

        fraction_volumes[1:] = p_fib[0] + p_random_max * p_random[1:]
        fraction_volumes[0] = 1 - np.sum(fraction_volumes[1:])
        return fraction_volumes

    def _random_micro_parameters(
        self,
        f_in,
        D_iso,
        D_a,
        D_e,
        D_r,
        peaks_per_voxel,
        equal_fibers,
        assert_faster_D_a,
        tortuosity_approximation,
    ):
        micro = np.zeros((self.MICRO_PARAMS_NUM, peaks_per_voxel + 1))

        # Free-water compartment: D_a = 0, f_in = 0, and D_e = D_iso.
        micro[self.MICRO_DE, 0] = np.random.uniform(D_iso[0], D_iso[1])

        while True:
            if equal_fibers:
                micro[:, 1:] = np.tile(
                    [
                        [np.random.uniform(D_a[0], D_a[1])],
                        [np.random.uniform(D_e[0], D_e[1])],
                        [np.random.uniform(D_r[0], D_r[1])],
                        [np.random.uniform(f_in[0], f_in[1])],
                    ],
                    peaks_per_voxel,
                )
            else:
                micro[:, 1:] = np.array(
                    [
                        np.random.uniform(D_a[0], D_a[1], size=peaks_per_voxel),
                        np.random.uniform(D_e[0], D_e[1], size=peaks_per_voxel),
                        np.random.uniform(D_r[0], D_r[1], size=peaks_per_voxel),
                        np.random.uniform(f_in[0], f_in[1], size=peaks_per_voxel),
                    ]
                )

            if assert_faster_D_a and np.any(
                micro[self.MICRO_DA, 1:] < micro[self.MICRO_DE, 1:]
            ):
                continue

            if tortuosity_approximation:
                micro[self.MICRO_DR, 1:] = (1 - micro[self.MICRO_FIN, 1:]) * micro[
                    self.MICRO_DA, 1:
                ]
                if np.any(micro[self.MICRO_DR, 1:] < D_r[0]) or np.any(
                    micro[self.MICRO_DR, 1:] > D_r[1]
                ):
                    continue
            break
        return micro

    def _compute_dwi(self, ratio, micro, peak_dirs_idx):
        ratio = np.nan_to_num(ratio)
        micro = np.nan_to_num(micro)

        # Convert the b-values from s/mm^2 to ms/um^2.
        bvals = np.vstack(1e-3 * self.gtab.bvals)

        # Diffusion signal of free water.
        dwi = ratio[0] * np.exp(-bvals * micro[self.MICRO_DE, 0])

        # Add the diffusion signal of each fiber.
        for j in range(len(peak_dirs_idx)):
            dir_prod_sqr = (
                np.dot(self.gtab.bvecs, self.sphere.vertices[peak_dirs_idx[j]].T) ** 2
            )
            dwi_intra = np.exp(-bvals * micro[self.MICRO_DA, j + 1] * dir_prod_sqr)
            dwi_extra = np.exp(
                -bvals
                * (
                    micro[self.MICRO_DE, j + 1] * dir_prod_sqr
                    + micro[self.MICRO_DR, j + 1] * (1 - dir_prod_sqr)
                )
            )
            dwi += ratio[j + 1] * (
                micro[self.MICRO_FIN, j + 1] * dwi_intra
                + (1 - micro[self.MICRO_FIN, j + 1]) * dwi_extra
            )

        return 1e3 * dwi.T

    def _compute_odf_trace(self, odf_recon_model, ratio, micro, peak_dirs_idx):
        dwi = self._compute_dwi(ratio, micro, peak_dirs_idx)
        odf = odf_recon_model.fit(dwi).odf(self.sphere).T
        return odf[: len(self.sphere.vertices) // 2]

    def _peaks_per_voxel_cdf(self, total_dirs_num):
        """CDF of the random variable ``peaks_per_voxel``."""
        # The numbers of directions are in the proportion
        # 1 : 1*(k-1) : 1*(k-1)*(k-2) : ...
        cumulative_dirs_num = np.ones(self.max_peaks_num)
        dirs_per_peak = 1
        for i in range(1, self.max_peaks_num):
            dirs_per_peak *= total_dirs_num - i
            cumulative_dirs_num[i] = cumulative_dirs_num[i - 1] + dirs_per_peak
        return cumulative_dirs_num[:-1] / cumulative_dirs_num[-1]

    # -- generation and persistence -------------------------------------------

    def generate(
        self,
        dict_size=1000000,
        max_peaks_num=3,
        equal_fibers=False,
        p_iso=(0.0, 1.0),
        p_fib=(0.0, 1.0),
        f_in=(0.0, 1.0),
        D_iso=(2.0, 3.0),
        D_a=(1.5, 2.5),
        D_e=(1.5, 2.5),
        D_r=(0.5, 1.5),
        max_chunk_size=10000,
        odf_recon_model=None,
        assert_faster_D_a=False,
        tortuosity_approximation=False,
    ):
        """Randomly generate a dictionary of ODF fingerprints."""
        if odf_recon_model is None:
            odf_recon_model = GeneralizedQSamplingModel(
                self.gtab, sampling_length=DEFAULT_DICT_EDGE
            )

        dict_size = np.maximum(1, dict_size)
        self.max_peaks_num = np.maximum(1, max_peaks_num)
        self.peaks_per_voxel = np.zeros(dict_size, dtype=int)
        p_iso, p_fib = self._interval(p_iso), self._interval(p_fib)
        f_in, D_iso = self._interval(f_in), self._interval(D_iso)
        D_a = self._interval(D_a)
        D_e, D_r = self._interval(D_e), self._interval(D_r)

        # Total number of directions allowed by the tessellation (k).
        total_dirs_num = len(self.sphere.vertices) // 2

        # Unused elements are kept as NaNs for backward compatibility.
        self.peak_dirs = np.nan * np.zeros((2, self.max_peaks_num, dict_size))
        self.ratio = np.nan * np.zeros((self.max_peaks_num + 1, dict_size))
        self.micro = np.nan * np.zeros((4, self.max_peaks_num + 1, dict_size))
        self.odf = np.zeros((total_dirs_num, dict_size))

        # VOID element: empty voxels outside the mask (skipped in matching).
        self.ratio[0, self.IDX_VOID] = 0
        self.micro[self.MICRO_DE, 0, self.IDX_VOID] = 0
        self.peaks_per_voxel[self.IDX_VOID] = -1

        # ISO element: voxels with isotropic (free) water only.
        self.ratio[0, self.IDX_ISO] = 1
        self.micro[self.MICRO_DE, 0, self.IDX_ISO] = 3
        self.peaks_per_voxel[self.IDX_ISO] = 0
        self.odf[:, self.IDX_ISO] = np.squeeze(
            self._compute_odf_trace(
                odf_recon_model,
                self.ratio[:, self.IDX_ISO],
                self.micro[:, :, self.IDX_ISO],
                [],
            )
        )

        chunk_bounds = range(max_chunk_size, dict_size, max_chunk_size)
        for chunk_idx in np.split(
            range(self.PREDEFINED_IDX_NUM, dict_size), chunk_bounds
        ):
            chunk_size = len(chunk_idx)
            peak_dirs_idx = np.zeros((self.max_peaks_num, chunk_size), dtype=int)

            # Draw the numbers of peaks per voxel. Direction [0, 0, 1] is
            # obligatory, hence the leading 1.
            self.peaks_per_voxel[chunk_idx] = 1 + np.sum(
                np.random.uniform(size=(chunk_size, 1))
                > self._peaks_per_voxel_cdf(total_dirs_num),
                axis=1,
            )

            for i, j in zip(range(chunk_size), chunk_idx):
                # Direction [0, 0, 1] has index 0 in the tessellation.
                peak_dirs_idx[1 : self.peaks_per_voxel[j], i] = np.random.choice(
                    range(1, total_dirs_num), self.peaks_per_voxel[j] - 1, replace=False
                )

                self.peak_dirs[:, : self.peaks_per_voxel[j], j] = np.array(
                    [
                        self.sphere.phi[peak_dirs_idx[: self.peaks_per_voxel[j], i]],
                        self.sphere.theta[peak_dirs_idx[: self.peaks_per_voxel[j], i]]
                        - np.pi / 2,
                    ]
                )
                self.ratio[: self.peaks_per_voxel[j] + 1, j] = (
                    self._random_fraction_volumes(p_iso, p_fib, self.peaks_per_voxel[j])
                )
                self.micro[:, : self.peaks_per_voxel[j] + 1, j] = (
                    self._random_micro_parameters(
                        f_in,
                        D_iso,
                        D_a,
                        D_e,
                        D_r,
                        self.peaks_per_voxel[j],
                        equal_fibers,
                        assert_faster_D_a,
                        tortuosity_approximation,
                    )
                )

            self.odf[:, chunk_idx] = self._compute_odf_trace(
                odf_recon_model,
                self.ratio[:, chunk_idx],
                self.micro[:, :, chunk_idx],
                peak_dirs_idx,
            )

            # Sort the peaks of each voxel in descending order and recompute the
            # ODF when the main peak was not the obligatory [0, 0, 1].
            recompute_filter = np.zeros(chunk_size, dtype=bool)
            for i, j in zip(range(chunk_size), chunk_idx):
                if self.peaks_per_voxel[j] < 2:
                    continue
                sorted_idx = np.argsort(
                    -self.odf[peak_dirs_idx[: self.peaks_per_voxel[j], i], j]
                )
                seq_idx = np.arange(self.peaks_per_voxel[j])
                if np.any(sorted_idx != seq_idx):
                    self.micro[:, seq_idx + 1, j] = self.micro[:, sorted_idx + 1, j]
                    self.ratio[seq_idx + 1, j] = self.ratio[sorted_idx + 1, j]
                if sorted_idx[0] != 0:
                    recompute_filter[i] = True

            self.odf[:, chunk_idx[recompute_filter]] = self._compute_odf_trace(
                odf_recon_model,
                self.ratio[:, chunk_idx[recompute_filter]],
                self.micro[:, :, chunk_idx[recompute_filter]],
                peak_dirs_idx[:, recompute_filter],
            )

    def save(self, dict_file="odf_dict.npz"):
        """Save the dictionary to a ``.npz`` file."""
        np.savez(
            dict_file,
            odf=self.odf,
            peak_dirs=self.peak_dirs,
            micro=self.micro,
            ratio=self.ratio,
            peaks_per_voxel=self.peaks_per_voxel,
            max_peaks_num=self.max_peaks_num,
        )

    def load(self, dict_file):
        """Load a dictionary previously saved with :meth:`save`."""
        data = np.load(dict_file)
        self.odf = data["odf"]
        self.peak_dirs = data["peak_dirs"]
        self.micro = data["micro"]
        self.ratio = data["ratio"]
        self.peaks_per_voxel = data["peaks_per_voxel"]
        self.max_peaks_num = int(data["max_peaks_num"])


class OdffpModel(ReconstModel):
    """ODF-Fingerprinting reconstruction model."""

    def __init__(
        self,
        gtab,
        dictionary,
        *,
        penalty=DEFAULT_FIT_PENALTY,
        drop_negative_odf=True,
        zero_baseline_odf=False,
        output_dict_odf=True,
        num_threads=None,
        odf_recon_model=None,
    ):
        if not hasattr(dictionary, "odf") or dictionary.odf is None:
            raise ValueError("The specified ODF-dictionary is empty.")

        ReconstModel.__init__(self, gtab)
        self.dictionary = dictionary
        self.sphere = dictionary.sphere
        self.penalty = float(np.clip(penalty, 0.0, MAX_FIT_PENALTY))
        self.num_threads = num_threads
        self._drop_negative_odf = drop_negative_odf
        self._zero_baseline_odf = zero_baseline_odf
        self._output_dict_odf = output_dict_odf
        if odf_recon_model is None:
            odf_recon_model = GeneralizedQSamplingModel(
                gtab, sampling_length=DEFAULT_RECON_EDGE
            )
        self._odf_recon_model = odf_recon_model

        self._half_size = len(self.sphere.vertices) // 2
        self._basis, self._inv_basis = _sh_operators(self.sphere)
        self._dict_trace, _ = self._normalize_odf(dictionary.odf)

        # Penalty group of each fingerprint (negative -> ignored in matching).
        n_fibers = dictionary.peaks_per_voxel
        group = np.where(n_fibers < 0, -1, np.maximum(0, n_fibers - 1))
        self._group = np.ascontiguousarray(group, dtype=np.intp)
        self._n_groups = int(self._group.max()) + 1

        # Resampling operators, cached by main-peak vertex across voxels/fits.
        self._operators = {}

    @staticmethod
    def resample_odf(odf, in_sphere, out_sphere):
        """Resample full-sphere ODF(s) from ``in_sphere`` to ``out_sphere``.

        ``odf`` is a single ODF vector or a matrix of ODF row-vectors. Returns a
        half-sphere ODF trace.
        """
        sphere_half_size = len(in_sphere.vertices) // 2
        odf = np.atleast_2d(odf)
        if odf.shape[1] == sphere_half_size:
            odf = np.hstack((odf, odf))
        resampled = sh_to_sf(
            sf_to_sh(odf, in_sphere, legacy=False), out_sphere, legacy=False
        )
        return np.squeeze(resampled[:, :sphere_half_size])

    def _normalize_odf(self, odf):
        """Drop negatives/baseline and L2-normalize ODF column-vectors."""
        if self._drop_negative_odf:
            odf = np.maximum(0, odf)
        if self._zero_baseline_odf:
            odf = odf - np.min(odf, axis=0)
        odf_norm = np.maximum(1e-8, np.sqrt(np.sum(odf**2, axis=0)))
        return odf / odf_norm, odf_norm

    def _main_peak_vertices(self, odfs):
        """Vertex of each ODF's main peak (-1 when the ODF has no peak)."""
        vertices = np.full(len(odfs), -1)
        for i, odf in enumerate(odfs):
            _, _, indices = peak_directions(odf, self.sphere)
            if len(indices):
                vertices[i] = indices[0]
        return vertices

    def _resampling_operators(self, peak_vertices):
        """Rotation and SH operators aligning each main peak with the pole.

        The main peak is always a tessellation vertex, so the rotations come
        from a finite set; the spherical harmonics of any not-yet-seen rotation
        are evaluated in a single batched call and cached for reuse.
        """
        new = sorted(set(peak_vertices.tolist()) - self._operators.keys())
        if new:
            rotations = np.stack(
                [
                    np.eye(3) if v < 0 else _rotation_to_pole(self.sphere.vertices[v])
                    for v in new
                ]
            )
            rotated = np.einsum("pj,rjk->rpk", self.sphere.vertices, rotations)
            sphere = Sphere(xyz=rotated.reshape(-1, 3))
            basis, _, _ = real_sh_descoteaux(
                SH_ORDER_MAX, sphere.theta, sphere.phi, legacy=False
            )
            n_points, n_sh = len(self.sphere.vertices), basis.shape[1]
            basis = basis.reshape(len(new), n_points, n_sh)
            pad = np.zeros((len(new), n_sh, n_sh))
            inv_basis = np.linalg.pinv(np.concatenate((basis, pad), axis=1))[
                :, :, :n_points
            ]
            for i, v in enumerate(new):
                self._operators[v] = (rotations[i], basis[i], inv_basis[i])

        rotations = np.stack([self._operators[v][0] for v in peak_vertices])
        basis = np.stack([self._operators[v][1] for v in peak_vertices])
        inv_basis = np.stack([self._operators[v][2] for v in peak_vertices])
        return rotations, basis, inv_basis

    def _rotate_peak_dirs(self, peak_dirs, rotation):
        directions = np.array(
            sphere2cart(1, np.pi / 2 + peak_dirs[1, :], peak_dirs[0, :])
        )
        return np.dot(directions.T, rotation)

    @multi_voxel_fit(
        batched=True,
        shared_obj=("_dict_trace", "dictionary"),
        # Matching builds a (chunk x dictionary) similarity matrix, so the chunk
        # is kept small to bound its memory for large (~1M) dictionaries. Pass
        # ``vox_per_chunk`` to fit() to override.
        chunk_size={"serial": 1000, "ray": "auto"},
    )
    def fit(self, data, *, mask=None, **kwargs):
        """Match each voxel to its best ODF fingerprint.

        Decorated with ``@multi_voxel_fit(batched=True)``: the decorator chunks
        the volume and hands each batch (2-D) to this method, which aligns every
        ODF to the pole and matches the whole batch against the dictionary in a
        single parallel call. Returns an :class:`OdffpFit` for a single voxel
        (1-D input) or a :class:`~dipy.reconst.multi_voxel.MultiVoxelFit`. Pass
        the fit to :func:`odffp_peaks` to build a
        :class:`~dipy.direction.peaks.PeaksAndMetrics`.
        """
        single = data.ndim == 1
        batch = data.reshape(1, -1) if single else data
        n_vox = batch.shape[0]
        half = self._half_size

        input_odf = self._odf_recon_model.fit(batch).odf(self.sphere)
        peak_vertices = self._main_peak_vertices(input_odf)
        rotations, basis, inv_basis = self._resampling_operators(peak_vertices)

        # Align every ODF to the pole and match the batch to the dictionary.
        coeffs = input_odf @ self._inv_basis.T
        aligned = np.einsum("vk,vpk->vp", coeffs, basis)[:, :half]
        trace, norm = self._normalize_odf(aligned.T)
        similarity = np.ascontiguousarray(trace.T @ self._dict_trace)
        matched = select_best_match(
            similarity,
            self._group,
            self.penalty,
            self._n_groups,
            num_threads=self.num_threads,
        )

        if self._output_dict_odf:
            # Rotate the matched dictionary ODFs back to the voxel frame.
            scaled = norm[:, np.newaxis] * self.dictionary.odf[:, matched].T
            full = np.concatenate((scaled, scaled), axis=1)
            out_coeffs = np.einsum("vp,vkp->vk", full, inv_basis)
            output_odf = (out_coeffs @ self._basis.T)[:, :half]
        else:
            output_odf = input_odf[:, :half]

        peak_dirs = np.stack(
            [
                self._rotate_peak_dirs(
                    self.dictionary.peak_dirs[:, :, matched[i]], rotations[i]
                )
                for i in range(n_vox)
            ]
        )

        params = {
            "odf": output_odf,
            "peak_dirs": peak_dirs,
            "dict_idx": matched,
            "microstructure": np.moveaxis(self.dictionary.micro[..., matched], -1, 0),
            "compartment_volume": self.dictionary.ratio[:, matched].T,
        }
        if kwargs.pop("_raw", False):
            return params

        fits = np.empty(n_vox, dtype=object)
        for i in range(n_vox):
            fits[i] = OdffpFit(self, {k: v[i] for k, v in params.items()})
        return fits[0] if single else fits


class OdffpFit(ReconstFit):
    """Result of an :class:`OdffpModel` fit for a single voxel."""

    def __init__(self, model, params):
        self.model = model
        self._params = params

    def odf(self, sphere=None):
        """Matched fingerprint ODF, normalized to a unit maximum."""
        odf = self._params["odf"]
        if (
            sphere is not None
            and self.model is not None
            and sphere is not self.model.sphere
        ):
            odf = OdffpModel.resample_odf(odf, self.model.sphere, sphere)
        return odf / np.maximum(1e-8, np.max(odf))

    @property
    def peak_dirs(self):
        """Fiber directions of the matched fingerprint in the voxel frame."""
        return self._params["peak_dirs"]

    @property
    def dict_idx(self):
        """Index of the matched fingerprint in the dictionary."""
        return self._params["dict_idx"]

    @property
    def microstructure(self):
        """Microstructure parameters of the matched fingerprint."""
        return self._params["microstructure"]

    @property
    def compartment_volume(self):
        """Compartment volume fractions of the matched fingerprint."""
        return self._params["compartment_volume"]


OdffpModel._fit_class = OdffpFit


def odffp_peaks(fit, *, sh_order_max=DEFAULT_PEAKS_SH_ORDER):
    """Create a :class:`~dipy.direction.peaks.PeaksAndMetrics` from an ODF-FP fit.

    Mirrors :func:`dipy.reconst.force.force_peaks`: it takes the fit object and
    returns a ``PeaksAndMetrics`` holding the peak directions, indices and
    amplitudes, with the matched ODFs stored as SH coefficients on
    ``shm_coeff`` (reconstruct them with
    :func:`~dipy.reconst.shm.sh_to_sf`). The result can be written to disk
    with :func:`~dipy.io.peaks.save_pam`.

    Works for a single :class:`OdffpFit` and for the
    :class:`~dipy.reconst.multi_voxel.MultiVoxelFit` returned for a volume.

    Parameters
    ----------
    fit : OdffpFit or MultiVoxelFit
        The result of :meth:`OdffpModel.fit`.
    sh_order_max : int, optional
        Maximum SH order used to represent the stored ODFs.

    Returns
    -------
    peaks : PeaksAndMetrics
    """
    sphere = fit.model.sphere
    half = len(sphere.vertices) // 2
    half_sphere = Sphere(xyz=sphere.vertices[:half])

    odf = np.asarray(fit.odf())  # (..., half) on the reconstruction hemisphere
    peak_dirs = np.nan_to_num(np.asarray(fit.peak_dirs), nan=0.0)
    n_peaks = peak_dirs.shape[-2]
    lead = peak_dirs.shape[:-2]  # () for a single voxel, (X, Y, Z) for a volume
    n_vox = int(np.prod(lead)) if lead else 1

    # Matched ODFs stored as SH coefficients, like FORCE.
    shm_coeff = sf_to_sh(
        odf, half_sphere, sh_order_max=sh_order_max, legacy=False
    ).astype(np.float32)

    # Main-peak vertex on the hemisphere and its ODF amplitude.
    dirs = peak_dirs.reshape(n_vox, n_peaks, 3)
    odf_flat = odf.reshape(n_vox, half)
    valid = np.any(dirs != 0, axis=-1)  # (n_vox, n_peaks)
    flat_valid = valid.reshape(-1)
    flat_idx = np.zeros(n_vox * n_peaks, dtype=np.intp)
    flat_dirs = dirs.reshape(-1, 3)
    flat_idx[flat_valid] = (
        np.argmax(flat_dirs[flat_valid] @ sphere.vertices.T, axis=1) % half
    )
    idx = flat_idx.reshape(n_vox, n_peaks)
    values = np.take_along_axis(odf_flat, idx, axis=1)
    values[~valid] = 0.0
    indices = idx.astype(np.int32)
    indices[~valid] = -1

    peaks = PeaksAndMetrics()
    peaks.peak_dirs = peak_dirs.astype(np.float32)
    peaks.peak_values = values.reshape(lead + (n_peaks,)).astype(np.float32)
    peaks.peak_indices = indices.reshape(lead + (n_peaks,))
    peaks.shm_coeff = shm_coeff
    peaks.sphere = half_sphere
    return peaks
