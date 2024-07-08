import lazy_loader as lazy

__getattr__, __dir__, __all__ = lazy.attach(
    __name__,
    submodules=[
        "bootstrap_direction_getter",
        "closest_peak_direction_getter",
        "peaks",
        "probabilistic_direction_getter",
        "ptt_direction_getter",
    ],
    submod_attrs={
        "bootstrap_direction_getter": ["BootDirectionGetter"],
        "closest_peak_direction_getter": ["ClosestPeakDirectionGetter"],
        "peaks": [
            "PeaksAndMetrics",
            "peak_directions",
            "peak_directions_nl",
            "peaks_from_model",
            "peaks_from_positions",
            "reshape_peaks_for_visualization",
        ],
        "probabilistic_direction_getter": [
            "DeterministicMaximumDirectionGetter",
            "ProbabilisticDirectionGetter",
        ],
        "ptt_direction_getter": ["PTTDirectionGetter"],
    },
)

__all__ = [
    "BootDirectionGetter",
    "ClosestPeakDirectionGetter",
    "DeterministicMaximumDirectionGetter",
    "ProbabilisticDirectionGetter",
    "PTTDirectionGetter",
    "PeaksAndMetrics",
    "peak_directions",
    "peak_directions_nl",
    "peaks_from_model",
    "peaks_from_positions",
    "reshape_peaks_for_visualization",
]
