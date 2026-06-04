import math
from math import radians

from nibabel.streamlines.array_sequence import MEGABYTE, ArraySequence
import numpy as np

from dipy.core.sphere import HemiSphere, Sphere
from dipy.data import default_sphere
from dipy.tracking.jit.generic_tracker import GenericJITTracker
from dipy.tracking.jit.numba_njit.generate_streamlines_numba import (
    genStreamlinesMergeProb_generator,
)
from dipy.tracking.jit.numba_njit.num_streamlines_numba import (
    getNumStreamlinesProb_generator,
)


class NumbaTracker(GenericJITTracker):
    """
    Probabilistic tracker using Numba.

    Parameters
    ----------
    pmf : np.ndarray, shape (dimx, dimy, dimz, dimt)
        PMF volume.
    stop_map : np.ndarray, shape (dimx, dimy, dimz)
        Stopping metric (e.g. GFA or FA).
    stop_threshold : float
        Voxels with stop_map <= stop_threshold are endpoints.
    sphere: Sphere
        Sphere defining the directions in the PMF.
        If None, uses default_sphere.
        Default: None.
    max_angle : float
        Maximum turning angle in radians. Default: radians(60).
    step_size : float
        Step size in voxels. Default: 0.5.
    min_steps : int
        Minimum streamline length (steps) to keep. Default: 0.
    max_steps : float
        Maximum streamline length (steps) to keep. Default: inf.
    relative_peak_thresh : float
        Relative peak threshold for direction selection. Default: 0.25.
    min_separation_angle : float
        Minimum separation angle (radians) between peaks. Default: radians(45).
    pmf_threshold : float
        Minimum PMF value (relative to max) to consider a valid direction.
        Default: 0.1.
    rng_seed : int, optional
        Seed for random number generator
        default: 0
    chunk_size : int
        Seeds per propagate() call in generate_sft(). Default: 100000.
    precision : str
        "float32" or "float64".
        Default: "float32".
    """

    def __init__(
        self,
        pmf: np.ndarray,
        stop_map: np.ndarray,
        stop_threshold: float,
        sphere: Sphere = None,
        max_angle: float = radians(60),
        step_size: float = 0.5,
        min_steps: int = 0,
        max_steps: int = 500,
        relative_peak_thresh: float = 0.25,
        min_separation_angle: float = radians(45),
        pmf_threshold: float = 0.1,
        rng_seed: int = 0,
        chunk_size: int = 25000,
        precision: str = "float64",
    ):
        if precision not in ("float32", "float64"):
            raise ValueError(f"Unsupported precision: {precision}")
        if precision == "float32":
            self.REAL_DTYPE = np.float32
        else:
            self.REAL_DTYPE = np.float64

        if sphere is None:
            sphere = default_sphere

        self.dataf = np.ascontiguousarray(pmf, dtype=self.REAL_DTYPE)
        self.metric_map = np.ascontiguousarray(stop_map, dtype=self.REAL_DTYPE)
        self.sphere_vertices = np.ascontiguousarray(
            sphere.vertices, dtype=self.REAL_DTYPE
        )
        self.sphere_edges = np.ascontiguousarray(sphere.edges, dtype=np.int32)

        if self.sphere_vertices.shape[0] != self.dataf.shape[3]:
            raise ValueError(
                f"Number of vertices in sphere ({self.sphere_vertices.shape[0]}) "
                f"must match 4th dimension of PMF ({self.dataf.shape[3]})"
            )

        # This assumes that if you pass a sphere which is not
        # a HemiSphere, then it should be treated as asymmetric.
        self.sphere_symm = isinstance(sphere, HemiSphere)

        self.dimx, self.dimy, self.dimz, self.dimt = pmf.shape

        self.max_angle = float(max_angle)
        self.tc_threshold = float(stop_threshold)
        self.step_size = float(step_size)
        self.relative_peak_thresh = float(relative_peak_thresh)
        self.min_separation_angle = float(min_separation_angle)
        self.chunk_size = int(chunk_size)
        self.nedges = int(self.sphere_edges.shape[0])
        self.PMF_THRESHOLD_P = float(pmf_threshold)

        if rng_seed != 0:
            np.random.seed(rng_seed)

        self.min_steps = min_steps
        self.max_steps = max_steps

        self.MAX_SLINE_LEN = int(self.max_steps)

        self.nSlines = 0
        self.slines = None
        self.sline_lens = None

        self.getNumStreamlinesProb = getNumStreamlinesProb_generator(
            self.dimx,
            self.dimy,
            self.dimz,
            self.dimt,
            self.relative_peak_thresh,
            self.min_separation_angle,
            self.nedges,
            self.sphere_symm,
            self.PMF_THRESHOLD_P,
        )

        self.genStreamlinesMergeProb = genStreamlinesMergeProb_generator(
            self.dimx,
            self.dimy,
            self.dimz,
            self.dimt,
            self.sphere_symm,
            self.step_size,
            self.max_angle,
            self.tc_threshold,
            self.MAX_SLINE_LEN,
            self.PMF_THRESHOLD_P,
        )

    def _get_num_streamlines(self, seeds):
        nseed = len(seeds)

        shDir0 = np.zeros((nseed, self.dimt, 3), dtype=self.REAL_DTYPE)
        slineOutOff = np.zeros(nseed + 1, dtype=np.int32)

        self.getNumStreamlinesProb(
            seeds,
            self.dataf,
            self.sphere_vertices,
            self.sphere_edges,
            shDir0,
            slineOutOff,
        )

        __pval = slineOutOff[0]
        slineOutOff[0] = 0
        for jj in range(1, nseed + 1):
            __cval = slineOutOff[jj]
            slineOutOff[jj] = slineOutOff[jj - 1] + __pval
            __pval = __cval

        return shDir0, slineOutOff

    def _generate_streamlines(self, seeds, shDir0, slineOutOff):
        nSlines = int(slineOutOff[-1])

        slineSeed = np.full(nSlines, -1, dtype=np.int32)
        slineLen = np.zeros(nSlines, dtype=np.int32)
        sline = np.zeros((nSlines * self.MAX_SLINE_LEN * 2, 3), dtype=self.REAL_DTYPE)

        self.genStreamlinesMergeProb(
            seeds,
            self.dataf,
            self.metric_map,
            self.sphere_vertices,
            slineOutOff,
            shDir0,
            slineSeed,
            slineLen,
            sline,
        )
        return nSlines, slineLen, sline

    def propagate(self, seeds: np.ndarray):
        """
        Run full two-phase tracking for `seeds`.
        Results stored in self.slines, self.sline_lens, self.nSlines.
        """
        seeds = np.ascontiguousarray(seeds, dtype=self.REAL_DTYPE)

        shDir0, slineOutOff = self._get_num_streamlines(seeds)
        nSlines, slineLen, sline = self._generate_streamlines(
            seeds, shDir0, slineOutOff
        )

        self.nSlines = nSlines
        self.sline_lens = slineLen
        self.slines = sline

    def get_buffer_size(self) -> int:
        """Return estimated buffer size in MB"""
        if self.sline_lens is None:
            return 0
        total_pts = sum(
            len_
            for len_ in self.sline_lens[: self.nSlines]
            if self.min_steps <= len_ <= self.max_steps
        )
        return math.ceil(total_pts * 3 * self.REAL_DTYPE(0).itemsize / MEGABYTE)

    def as_generator(self):
        def _yield_slines():
            slines = self.slines
            sline_lens = self.sline_lens
            step = self.MAX_SLINE_LEN * 2  # points allocated per streamline

            for i in range(self.nSlines):
                npts = int(sline_lens[i])
                if npts < self.min_steps or npts > self.max_steps:
                    print("here")
                    continue
                yield np.asarray(
                    slines[i * step : i * step + npts], dtype=self.REAL_DTYPE
                )

        return _yield_slines()

    def as_array_sequence(self) -> ArraySequence:
        return ArraySequence(self.as_generator(), self.get_buffer_size())
