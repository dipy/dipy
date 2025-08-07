"""Benchmarks for functions related to streamline in ``dipy.tracking``module."""

import numpy as np

from dipy.data import get_fnames
from dipy.io.streamline import load_tractogram
from dipy.tracking import Streamlines
from dipy.tracking.streamline import length, set_number_of_points
from dipy.tracking.streamlinespeed import compress_streamlines


class BenchStreamlines:
    def setup(self):
        rng = np.random.default_rng(42)
        nb_streamlines = 20000
        min_nb_points = 2
        max_nb_points = 100

        def generate_streamlines(nb_streamlines, min_nb_points, max_nb_points, rng):
            streamlines = [
                rng.random((rng.integers(min_nb_points, max_nb_points), 3))
                for _ in range(nb_streamlines)
            ]
            return streamlines

        self.data = {}
        self.data["rng"] = rng
        self.data["nb_streamlines"] = nb_streamlines
        self.data["streamlines"] = generate_streamlines(
            nb_streamlines, min_nb_points, max_nb_points, rng=rng
        )
        self.data["streamlines_arrseq"] = Streamlines(self.data["streamlines"])

        fname = get_fnames(name="fornix")
        fornix = load_tractogram(fname, "same", bbox_valid_check=False).streamlines

        self.fornix_streamlines = Streamlines(fornix)

    def time_set_number_of_points(self):
        streamlines = self.data["streamlines"]
        set_number_of_points(streamlines, nb_points=50)

    def time_set_number_of_points_arrseq(self):
        streamlines = self.data["streamlines_arrseq"]
        set_number_of_points(streamlines, nb_points=50)

    def time_length(self):
        streamlines = self.data["streamlines"]
        length(streamlines)

    def time_length_arrseq(self):
        streamlines = self.data["streamlines_arrseq"]
        length(streamlines)

    def time_compress_streamlines(self):
        compress_streamlines(self.fornix_streamlines)
