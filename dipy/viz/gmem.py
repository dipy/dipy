class GlobalHorizon(object):
    window_timer_cnt = 0
    
    slicer_opacity = 1
    slicer_colormap = 'gray'
    slicer_colormaps = ['gray', 'magma', 'viridis', 'Pastel1', 'disting']
    slicer_axes = ['x', 'y', 'z']
    
    slicer_curr_x = None
    slicer_curr_y = None
    slicer_curr_z = None
    
    slicer_curr_actor_x = None
    slicer_curr_actor_y = None
    slicer_curr_actor_z = None
    
    slicer_orig_shape = None
    slicer_resliced_shape = None
    
    slicer_vol_idx = None
    slicer_vol = None

    slicer_peaks_actor_z = None

    slicer_grid = False
    
    cluster_thr = 15
    

HORIZON = GlobalHorizon()