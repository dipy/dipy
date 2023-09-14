# Shared objects across Horizon's systems

class GlobalHorizon:
    def __init__(self):
        # window level sharing
        self.window_timer_cnt = 0

        # slicer level sharing
        self.slicer_opacity = 1
        self.slicer_colormap = 'gray'
        self.slicer_colormaps = ['gray', 'magma', 'viridis', 'jet', 'Pastel1', 'disting']
        self.slicer_colormap_cnt = 0
        self.slicer_axes = ['x', 'y', 'z']

        self.slicer_curr_x = None
        self.slicer_curr_y = None
        self.slicer_curr_z = None

        self.slicer_curr_actor_x = None
        self.slicer_curr_actor_y = None
        self.slicer_curr_actor_z = None

        self.slicer_orig_shape = None
        self.slicer_resliced_shape = None

        self.slicer_vol_idx = None
        self.slicer_vol = None

        self.slicer_peaks_actor_z = None
        self.slicer_rgb = False

        self.slicer_grid = False

        # tractogram level sharing
        self.cluster_thr = 15
        # self.cluster_lengths = []  # not used
        # self.cluster_sizes = []  # not used
        # self.cluster_thr_min_max = []  # not used
        self.streamline_actors = []
        self.centroid_actors = []
        self.cluster_actors = []
