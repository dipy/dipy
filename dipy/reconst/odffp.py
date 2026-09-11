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

from dipy.core.geometry import sphere2cart, vec2vec_rotmat
from dipy.core.sphere import Sphere
from dipy.data import get_sphere
from dipy.direction import peak_directions
from dipy.direction.peaks import PeaksAndMetrics
from dipy.reconst.base import ReconstFit, ReconstModel
from dipy.reconst.gqi import GeneralizedQSamplingModel
from dipy.reconst.multi_voxel import multi_voxel_fit
from dipy.reconst.odffp_matching import accumulate_block, finalize_match
from dipy.reconst.shm import (
    real_sh_descoteaux,
    sf_to_sh,
    sh_to_sf,
    sh_to_sf_matrix,
)

# Number of dictionary fingerprints matched per block. The matching is streamed
# over blocks so the full (n_voxels x n_dict) similarity is never materialized.
# A large block keeps the BLAS matmul efficient (one big GEMM beats many small
# ones) while still bounding the (n_voxels x block) tile for very large
# dictionaries; ~5e5 runs at materialize speed at roughly half the peak memory.
MATCH_BLOCK_SIZE = 524288


def _as_interval(values):
    """Return the interval spanned by a sequence.

    Parameters
    ----------
    values : array-like
        Values whose minimum and maximum define the interval.

    Returns
    -------
    interval : ndarray, shape (2,)
        Minimum and maximum values, in that order.
    """
    return np.array([np.min(values), np.max(values)])


def resample_odf(odf, in_sphere, out_sphere, *, sh_order_max=8):
    """Resample full-sphere ODF(s) from ``in_sphere`` to ``out_sphere``.

    Parameters
    ----------
    odf : ndarray
        A single ODF vector or an array of ODF row vectors. A half-sphere input
        is expanded using antipodal symmetry before fitting.
    in_sphere : Sphere
        Sphere on which ``odf`` is sampled.
    out_sphere : Sphere
        Sphere on which to evaluate the resampled ODF.
    sh_order_max : int, optional
        Maximum spherical-harmonic order used for resampling.

    Returns
    -------
    resampled : ndarray
        Resampled half-sphere ODF trace or traces.
    """
    sphere_half_size = len(in_sphere.vertices) // 2
    odf = np.atleast_2d(odf)
    if odf.shape[1] == sphere_half_size:
        odf = np.hstack((odf, odf))
    resampled = sh_to_sf(
        sf_to_sh(odf, in_sphere, sh_order_max=sh_order_max, legacy=False),
        out_sphere,
        sh_order_max=sh_order_max,
        legacy=False,
    )
    return np.squeeze(resampled[:, :sphere_half_size])


class OdffpDictionary:
    """Dictionary of ODF fingerprints and their microstructure parameters.

    The fingerprints are simulated from a multi-compartment model (intra- and
    extra-axonal and free water) and reconstructed with ``odf_recon_model``
    (GQI by default) on a symmetric ``sphere``.

    Parameters
    ----------
    gtab : GradientTable
        Acquisition gradient table used to simulate diffusion signals.
    sphere : Sphere, optional
        Sphere used to represent dictionary ODFs. Uses the full
        ``repulsion724`` sphere when omitted.
    dict_file : path-like, optional
        Dictionary archive previously written by :meth:`save`. If provided,
        the dictionary is loaded during initialization.
    """

    IDX_VOID = 0
    IDX_ISO = 1
    PREDEFINED_IDX_NUM = 2

    MICRO_DA = 0
    MICRO_DE = 1
    MICRO_DR = 2
    MICRO_FIN = 3
    MICRO_PARAMS_NUM = 4

    def __init__(self, gtab, *, sphere=None, dict_file=None):
        """Initialize an empty or previously saved ODF-FP dictionary."""
        self.gtab = gtab
        self.sphere = sphere if sphere is not None else get_sphere(name="repulsion724")
        self.max_peaks_num = 0
        self.odf = None
        self.peak_dirs = None
        self.micro = None
        self.ratio = None
        self.peaks_per_voxel = None
        if dict_file is not None:
            self.load(dict_file)

    def _random_fraction_volumes(self, p_iso, p_fib, peaks_per_voxel, rng):
        """Draw free-water and fiber-compartment volume fractions.

        Parameters
        ----------
        p_iso : ndarray, shape (2,)
            Minimum and maximum free-water volume fractions.
        p_fib : ndarray, shape (2,)
            Minimum and maximum volume fractions for each fiber compartment.
        peaks_per_voxel : int
            Number of fiber compartments.
        rng : numpy.random.Generator
            Random number generator.

        Returns
        -------
        fraction_volumes : ndarray, shape (peaks_per_voxel + 1,)
            Free-water fraction followed by one fraction per fiber.
        """
        fraction_volumes = np.zeros(peaks_per_voxel + 1)

        # Lower bounds are hard limits; variability stays in [0, p_random_max].
        p_random_max = 1 - (p_iso[0] + peaks_per_voxel * p_fib[0])
        p_random = np.hstack(
            (
                rng.uniform(0, p_iso[1] - p_iso[0]),
                rng.uniform(0, p_fib[1] - p_fib[0], size=peaks_per_voxel),
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
        rng,
    ):
        """Draw microstructure parameters for one dictionary fingerprint.

        Parameters
        ----------
        f_in : ndarray, shape (2,)
            Interval for the intra-axonal signal fraction.
        D_iso : ndarray, shape (2,)
            Interval for free-water isotropic diffusivity.
        D_a : ndarray, shape (2,)
            Interval for intra-axonal diffusivity.
        D_e : ndarray, shape (2,)
            Interval for extra-axonal axial diffusivity.
        D_r : ndarray, shape (2,)
            Interval for extra-axonal radial diffusivity.
        peaks_per_voxel : int
            Number of fiber compartments.
        equal_fibers : bool
            If True, assign identical microstructure parameters to all fibers.
        assert_faster_D_a : bool
            If True, require intra-axonal diffusivity to be no smaller than
            extra-axonal axial diffusivity.
        tortuosity_approximation : bool
            If True, derive radial diffusivity using the tortuosity
            approximation and reject draws outside ``D_r``.
        rng : numpy.random.Generator
            Random number generator.

        Returns
        -------
        micro : ndarray, shape (4, peaks_per_voxel + 1)
            Microstructure parameters for free water and each fiber.
        """
        micro = np.zeros((self.MICRO_PARAMS_NUM, peaks_per_voxel + 1))

        # Free-water compartment: D_a = 0, f_in = 0, and D_e = D_iso.
        micro[self.MICRO_DE, 0] = rng.uniform(D_iso[0], D_iso[1])

        while True:
            if equal_fibers:
                micro[:, 1:] = np.tile(
                    [
                        [rng.uniform(D_a[0], D_a[1])],
                        [rng.uniform(D_e[0], D_e[1])],
                        [rng.uniform(D_r[0], D_r[1])],
                        [rng.uniform(f_in[0], f_in[1])],
                    ],
                    peaks_per_voxel,
                )
            else:
                micro[:, 1:] = np.array(
                    [
                        rng.uniform(D_a[0], D_a[1], size=peaks_per_voxel),
                        rng.uniform(D_e[0], D_e[1], size=peaks_per_voxel),
                        rng.uniform(D_r[0], D_r[1], size=peaks_per_voxel),
                        rng.uniform(f_in[0], f_in[1], size=peaks_per_voxel),
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
        """Simulate diffusion-weighted signals for dictionary fingerprints.

        Parameters
        ----------
        ratio : ndarray
            Compartment volume fractions. The first row contains free-water
            fractions and subsequent rows contain fiber fractions.
        micro : ndarray
            Microstructure parameters, indexed by parameter, compartment, and
            optionally fingerprint.
        peak_dirs_idx : array-like
            Sphere-vertex indices of the fiber directions.

        Returns
        -------
        dwi : ndarray
            Simulated signal or signals, with gradients on the last axis.
        """
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
        """Simulate signals and reconstruct their half-sphere ODF traces.

        Parameters
        ----------
        odf_recon_model : ReconstModel
            Model used to reconstruct each simulated signal's ODF.
        ratio : ndarray
            Compartment volume fractions.
        micro : ndarray
            Microstructure parameters.
        peak_dirs_idx : array-like
            Sphere-vertex indices of the fiber directions.

        Returns
        -------
        odf : ndarray
            Reconstructed half-sphere ODF trace or traces.
        """
        dwi = self._compute_dwi(ratio, micro, peak_dirs_idx)
        odf = odf_recon_model.fit(dwi).odf(self.sphere).T
        return odf[: len(self.sphere.vertices) // 2]

    def _peaks_per_voxel_cdf(self, total_dirs_num):
        """Compute the CDF used to sample the number of fiber peaks.

        Parameters
        ----------
        total_dirs_num : int
            Number of candidate directions on the hemisphere.

        Returns
        -------
        cdf : ndarray, shape (max_peaks_num - 1,)
            Cumulative probabilities separating the possible peak counts.
        """
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
        *,
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
        rng=None,
    ):
        """Randomly generate a dictionary of ODF fingerprints.

        Parameters
        ----------
        dict_size : int, optional
            Number of fingerprints, including the predefined void and
            isotropic entries.
        max_peaks_num : int, optional
            Maximum number of fiber compartments in a fingerprint.
        equal_fibers : bool, optional
            If True, use identical microstructure parameters for all fibers in
            a fingerprint.
        p_iso : array-like, shape (2,), optional
            Interval for the free-water volume fraction.
        p_fib : array-like, shape (2,), optional
            Interval for each fiber-compartment volume fraction.
        f_in : array-like, shape (2,), optional
            Interval for the intra-axonal signal fraction.
        D_iso : array-like, shape (2,), optional
            Interval for free-water isotropic diffusivity, in um^2/ms.
        D_a : array-like, shape (2,), optional
            Interval for intra-axonal diffusivity, in um^2/ms.
        D_e : array-like, shape (2,), optional
            Interval for extra-axonal axial diffusivity, in um^2/ms.
        D_r : array-like, shape (2,), optional
            Interval for extra-axonal radial diffusivity, in um^2/ms.
        max_chunk_size : int, optional
            Maximum number of fingerprints simulated at once.
        odf_recon_model : ReconstModel, optional
            Model used to reconstruct simulated ODFs. By default, GQI is used
            with a sampling length of 1.2.
        assert_faster_D_a : bool, optional
            If True, reject samples whose intra-axonal diffusivity is smaller
            than their extra-axonal axial diffusivity.
        tortuosity_approximation : bool, optional
            If True, derive radial diffusivity from the intra-axonal fraction
            and diffusivity.
        rng : numpy.random.Generator, optional
            Random number generator. A new default generator is created when
            omitted.

        Notes
        -----
        The generated arrays replace any dictionary data currently stored on
        this instance.
        """
        if rng is None:
            rng = np.random.default_rng()
        if odf_recon_model is None:
            odf_recon_model = GeneralizedQSamplingModel(self.gtab, sampling_length=1.2)

        dict_size = np.maximum(1, dict_size)
        self.max_peaks_num = np.maximum(1, max_peaks_num)
        self.peaks_per_voxel = np.zeros(dict_size, dtype=int)
        p_iso, p_fib = _as_interval(p_iso), _as_interval(p_fib)
        f_in, D_iso = _as_interval(f_in), _as_interval(D_iso)
        D_a = _as_interval(D_a)
        D_e, D_r = _as_interval(D_e), _as_interval(D_r)

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
                rng.uniform(size=(chunk_size, 1))
                > self._peaks_per_voxel_cdf(total_dirs_num),
                axis=1,
            )

            for i, j in zip(range(chunk_size), chunk_idx):
                # Direction [0, 0, 1] has index 0 in the tessellation.
                peak_dirs_idx[1 : self.peaks_per_voxel[j], i] = rng.choice(
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
                    self._random_fraction_volumes(
                        p_iso, p_fib, self.peaks_per_voxel[j], rng
                    )
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
                        rng,
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
        """Save the dictionary to a NumPy archive.

        Parameters
        ----------
        dict_file : path-like, optional
            Destination ``.npz`` file.
        """
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
        """Load a dictionary previously saved with :meth:`save`.

        Parameters
        ----------
        dict_file : path-like
            Source ``.npz`` file.
        """
        data = np.load(dict_file)
        self.odf = data["odf"]
        self.peak_dirs = data["peak_dirs"]
        self.micro = data["micro"]
        self.ratio = data["ratio"]
        self.peaks_per_voxel = data["peaks_per_voxel"]
        self.max_peaks_num = int(data["max_peaks_num"])


class OdffpModel(ReconstModel):
    """ODF-Fingerprinting reconstruction model.

    Parameters
    ----------
    gtab : GradientTable
        Acquisition gradient table for the measured signal.
    dictionary : OdffpDictionary
        Generated or loaded dictionary used for fingerprint matching.
    penalty : float, optional
        Model-complexity penalty, clipped to the interval [0, 0.1].
    sh_order_max : int, optional
        Maximum SH order used for alignment, resampling, and matching.
    drop_negative_odf : bool, optional
        If True, set negative ODF samples to zero before normalization.
    zero_baseline_odf : bool, optional
        If True, subtract the minimum of each ODF before normalization.
    output_dict_odf : bool, optional
        If True, return the matched dictionary ODF rotated into the voxel
        frame. If False, return the measured ODF reconstruction.
    matching_precision : {"float32", "float64"}, optional
        Floating-point precision used for fingerprint matching.
    num_threads : int, optional
        Number of OpenMP threads used by the matching kernels. ``None`` uses
        the default number of threads.
    odf_recon_model : ReconstModel, optional
        Model used to reconstruct measured ODFs. By default, GQI is used with
        a sampling length of 1.2.
    """

    def __init__(
        self,
        gtab,
        dictionary,
        *,
        penalty=1e-5,
        sh_order_max=8,
        drop_negative_odf=True,
        zero_baseline_odf=False,
        output_dict_odf=True,
        matching_precision="float32",
        num_threads=None,
        odf_recon_model=None,
    ):
        """Initialize an ODF-FP reconstruction model."""
        if not hasattr(dictionary, "odf") or dictionary.odf is None:
            raise ValueError("The specified ODF-dictionary is empty.")
        if matching_precision not in ("float32", "float64"):
            raise ValueError("matching_precision must be 'float32' or 'float64'.")

        ReconstModel.__init__(self, gtab)
        self.dictionary = dictionary
        self.sphere = dictionary.sphere
        self.penalty = float(np.clip(penalty, 0.0, 0.1))
        self.sh_order_max = int(sh_order_max)
        self.num_threads = num_threads
        self._drop_negative_odf = drop_negative_odf
        self._zero_baseline_odf = zero_baseline_odf
        self._output_dict_odf = output_dict_odf
        self._match_dtype = np.dtype(matching_precision)
        if odf_recon_model is None:
            odf_recon_model = GeneralizedQSamplingModel(gtab, sampling_length=1.2)
        self._odf_recon_model = odf_recon_model

        self._half_size = len(self.sphere.vertices) // 2
        # Align each main peak to vertex 0, the dictionary's obligatory fiber.
        self._pole = self.sphere.vertices[0]
        self._sh_to_sf, self._sf_to_sh = sh_to_sf_matrix(
            self.sphere, sh_order_max=self.sh_order_max, legacy=False
        )

        # The pole-aligned query ODFs are order-``sh_order_max`` band-limited, so
        # their 362-sample cosine with a (full-resolution) dictionary trace
        # equals a low-dimensional dot product in that SH space. Project the
        # normalized dictionary traces onto the half-sphere SH basis once
        # ``(n_dict, n_sh)`` and project each query the same way: the
        # high-frequency dictionary content is orthogonal to the band-limited
        # query, so this is exact yet uses ~24x fewer matmul flops and ~24x
        # less dictionary memory. The match runs in the chosen precision
        # (float32 by default: ~1.8x faster again, flipping only sub-1e-7
        # near-tie matches).
        sh_basis = real_sh_descoteaux(
            self.sh_order_max,
            self.sphere.theta[: self._half_size],
            self.sphere.phi[: self._half_size],
            legacy=False,
        )[0]
        self._query_proj = np.linalg.pinv(sh_basis).T  # (half, n_sh)
        dict_trace, _ = self._normalize_odf(dictionary.odf)  # (half, n_dict)
        self._dict_trace = np.ascontiguousarray(
            (sh_basis.T @ dict_trace).T, dtype=self._match_dtype
        )  # (n_dict, n_sh)

        # Penalty group of each fingerprint (negative -> ignored in matching).
        n_fibers = dictionary.peaks_per_voxel
        group = np.where(n_fibers < 0, -1, np.maximum(0, n_fibers - 1))
        self._group = np.ascontiguousarray(group, dtype=np.intp)
        self._n_groups = int(self._group.max()) + 1

        # Resampling operators, cached by main-peak vertex across voxels/fits.
        self._operators = {}

    def _normalize_odf(self, odf):
        """Preprocess and L2-normalize ODF column vectors.

        Parameters
        ----------
        odf : ndarray, shape (n_samples, n_odfs)
            ODF column vectors.

        Returns
        -------
        normalized : ndarray
            Preprocessed ODFs with unit L2 norm where possible.
        norm : ndarray, shape (n_odfs,)
            Original norms after optional negative-value and baseline removal.
        """
        if self._drop_negative_odf:
            odf = np.maximum(0, odf)
        if self._zero_baseline_odf:
            odf = odf - np.min(odf, axis=0)
        odf_norm = np.maximum(1e-8, np.sqrt(np.sum(odf**2, axis=0)))
        return odf / odf_norm, odf_norm

    def _main_peak_vertices(self, odfs):
        """Find the sphere vertex of each ODF's main peak.

        Parameters
        ----------
        odfs : ndarray, shape (n_odfs, n_vertices)
            ODF row vectors sampled on :attr:`sphere`.

        Returns
        -------
        vertices : ndarray, shape (n_odfs,)
            Main-peak vertex indices, or -1 for ODFs without a peak.
        """
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
        are evaluated in a single batched call. The cache is replaced rather
        than mutated so concurrent fits always use a complete local snapshot.

        Parameters
        ----------
        peak_vertices : ndarray, shape (n_voxels,)
            Main-peak vertex index for each voxel. A value of -1 requests the
            identity rotation.

        Returns
        -------
        rotations : ndarray, shape (n_voxels, 3, 3)
            Matrices relating pole-aligned and voxel-frame directions.
        basis : ndarray
            SH synthesis matrix for each rotated sphere.
        inv_basis : ndarray
            SH analysis matrix for each rotated sphere.
        """
        operators = self._operators
        new = sorted(set(peak_vertices.tolist()) - operators.keys())
        if new:
            rotations = np.stack(
                [
                    np.eye(3)
                    if v < 0
                    else vec2vec_rotmat(self.sphere.vertices[v], self._pole)
                    for v in new
                ]
            )
            rotated = np.einsum("pj,rjk->rpk", self.sphere.vertices, rotations)
            sphere = Sphere(xyz=rotated.reshape(-1, 3))
            basis, _, _ = real_sh_descoteaux(
                self.sh_order_max, sphere.theta, sphere.phi, legacy=False
            )
            n_points, n_sh = len(self.sphere.vertices), basis.shape[1]
            basis = basis.reshape(len(new), n_points, n_sh)
            pad = np.zeros((len(new), n_sh, n_sh))
            inv_basis = np.linalg.pinv(np.concatenate((basis, pad), axis=1))[
                :, :, :n_points
            ]
            additions = {
                v: (rotations[i], basis[i], inv_basis[i]) for i, v in enumerate(new)
            }
            operators = {**operators, **additions}
            self._operators = operators

        rotations = np.stack([operators[v][0] for v in peak_vertices])
        basis = np.stack([operators[v][1] for v in peak_vertices])
        inv_basis = np.stack([operators[v][2] for v in peak_vertices])
        return rotations, basis, inv_basis

    def _rotate_peak_dirs(self, peak_dirs, rotation):
        """Rotate dictionary peak directions into a voxel frame.

        Parameters
        ----------
        peak_dirs : ndarray, shape (2, n_peaks)
            Dictionary peak azimuth and elevation angles.
        rotation : ndarray, shape (3, 3)
            Rotation from pole-aligned coordinates to the voxel frame.

        Returns
        -------
        directions : ndarray, shape (n_peaks, 3)
            Rotated Cartesian peak directions.
        """
        directions = np.array(
            sphere2cart(1, np.pi / 2 + peak_dirs[1, :], peak_dirs[0, :])
        )
        return np.dot(directions.T, rotation)

    def _match(self, query):
        """Match a batch of aligned ODF traces to the dictionary.

        The similarity matmul is fused with the penalized arg-max and streamed
        over blocks of the dictionary, so the full ``(n_voxels x n_dict)``
        similarity matrix is never materialized.

        Parameters
        ----------
        query : ndarray (n_voxels, n_sh), C-contiguous, matching precision
            SH coefficients of the L2-normalized, pole-aligned ODF traces, one
            row per voxel.

        Returns
        -------
        matched : ndarray (n_voxels,), intp
            Index of the best-matching fingerprint for each voxel.
        """
        n_vox = query.shape[0]
        n_dict = self._dict_trace.shape[0]
        group_best = np.full((n_vox, self._n_groups), -np.inf)
        group_idx = np.full((n_vox, self._n_groups), -1, dtype=np.intp)
        for start in range(0, n_dict, MATCH_BLOCK_SIZE):
            stop = min(start + MATCH_BLOCK_SIZE, n_dict)
            similarity = np.ascontiguousarray(query @ self._dict_trace[start:stop].T)
            accumulate_block(
                similarity,
                self._group[start:stop],
                group_best,
                group_idx,
                start,
                num_threads=self.num_threads,
            )
        return finalize_match(
            group_best, group_idx, self.penalty, num_threads=self.num_threads
        )

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

        Parameters
        ----------
        data : ndarray
            Diffusion signal for one voxel or a volume, with gradients on the
            last axis.
        mask : ndarray, optional
            Boolean mask selecting voxels to fit.
        **kwargs : dict
            Options consumed by :func:`multi_voxel_fit`, including ``engine``,
            ``n_jobs``, and ``vox_per_chunk``.

        Returns
        -------
        fit : OdffpFit or MultiVoxelFit
            Fitted fingerprint parameters for one voxel or a volume.
        """
        single = data.ndim == 1
        batch = data.reshape(1, -1) if single else data
        n_vox = batch.shape[0]
        half = self._half_size

        input_odf = self._odf_recon_model.fit(batch).odf(self.sphere)
        peak_vertices = self._main_peak_vertices(input_odf)
        rotations, basis, inv_basis = self._resampling_operators(peak_vertices)

        # Align every ODF to the pole and match the batch to the dictionary.
        coeffs = input_odf @ self._sf_to_sh
        aligned = np.einsum("vk,vpk->vp", coeffs, basis)[:, :half]
        trace, norm = self._normalize_odf(aligned.T)
        # Project the aligned traces into the SH subspace and match there.
        query = np.ascontiguousarray(
            trace.T @ self._query_proj, dtype=self._match_dtype
        )
        matched = self._match(query)

        if self._output_dict_odf:
            # Rotate the matched dictionary ODFs back to the voxel frame.
            scaled = norm[:, np.newaxis] * self.dictionary.odf[:, matched].T
            full = np.concatenate((scaled, scaled), axis=1)
            out_coeffs = np.einsum("vp,vkp->vk", full, inv_basis)
            output_odf = (out_coeffs @ self._sh_to_sf)[:, :half]
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
    """Result of an :class:`OdffpModel` fit for a single voxel.

    Parameters
    ----------
    model : OdffpModel or None
        Model that produced the fit. Worker-produced fits may use ``None``.
    params : dict
        Matched ODF, peak directions, dictionary index, microstructure
        parameters, and compartment volumes.
    """

    def __init__(self, model, params):
        """Initialize an ODF-FP fit from matched parameter arrays."""
        self.model = model
        self._params = params

    def odf(self, sphere=None):
        """Return the matched fingerprint ODF normalized to a unit maximum.

        Parameters
        ----------
        sphere : Sphere, optional
            Sphere on which to evaluate the ODF. By default, return the ODF on
            the model's reconstruction hemisphere.

        Returns
        -------
        odf : ndarray
            Normalized ODF samples.
        """
        odf = self._params["odf"]
        if (
            sphere is not None
            and self.model is not None
            and sphere is not self.model.sphere
        ):
            odf = resample_odf(
                odf, self.model.sphere, sphere, sh_order_max=self.model.sh_order_max
            )
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


def odffp_peaks(fit, *, sh_order_max=8):
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
