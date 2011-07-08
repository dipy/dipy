from numpy import abs, array, asarray, atleast_3d, broadcast_arrays, cos, \
                  dot, empty, floor, mgrid, pi, sqrt

pn_edge = array([[0], [1]])

def propagate(voxel_location, cur_step, vox_size, over_step=1e-1):
    """takes a step just past the edge of the next voxel along cur_step

    given a location and a cur_step, finds the smallest step needed to move
    into the next voxel
    """

    #change cur_step from mm to vox dim
    vox_step = cur_step/vox_size

    space = voxel_location % 1
    dist = pn_edge - space
    step_size = dist/vox_step
    step_size = step_size.ravel()
    step_size.sort()
    #because all values in space are > 0 and < 1, 3 step_sizes are < 0 and 3
    #are > 0, take the smallest step > 0
    if step_size[3] == 0:
        raise ValueError('step_size[3] should not be zero')
    smallest_step = step_size[3] + over_step
    new_loc = voxel_location + smallest_step*vox_step

    #we need to know cur_step to pass to the next propagation step
    return new_loc

def fact_tracking(voxel_model, seeds, start_steps, over_step=1e-1):
    all_tracks = []
    seeds, start_steps = broadcast_arrays(seeds, start_steps)
    next_step = voxel_model.next_step
    vox_size = voxel_model.vox_size
    for ii in xrange(len(seeds)):
        vox_loc = seeds[ii]
        step = start_steps[ii]
        track = [vox_loc]
        step = next_step(vox_loc, step)
        while step is not None:
            vox_loc = propagate(vox_loc, step, vox_size, over_step)
            track.append(vox_loc)
            step = next_step(vox_loc, step)
        all_tracks.append(array(track)*vox_size)
    return all_tracks

class FactTensorModel(object):
    def _get_angel_limit(self):
        return arccos(self.dot_limit)*180/pi
    def _set_angel_limit(self, angle):
        if angle >= 0 and angle <= 90:
            self.dot_limit = cos(angle*pi/180)
        else:
            raise ValueError("angle should be between 0 and 180")
    angel_limit = property(_get_angel_limit, _set_angel_limit)

    def from_tensor(tensor, fa_limit=None, angle_limit=None):
        self.fa_vol = tensor.fa()
        self.evec1_vol = tensor.evec[..., 0]
        if fa_limit is not None:
            self.fa_limit = fa_limit
        if angle_limit is not None:
            self.angel_limit = angle_limit

    def next_step(vox_loc, prev_step):
        vox_loc = vox_loc.astype('int')
        if self.fa_vol[vox_loc] < self.fa_limit:
            return False
        step = self.evec1_vol[vox_loc]
        angle_dot = dot(step, prev_step)
        if abs(angle_dot) < self.dot_limit:
            return False
        if angle_dot > 0:
            return step
        else:
            return -step

def track_tensor(evec1_vol, fa_vol, start_loc, start_step, vox_size, fa_limit,
                 angle_limit):
    """makes tracks using tensor starting from stat_loc"""

    #evec1_vol = tensor.evecs[...,0]
    #fa_vol = tensor.fa()
    dot_limit = cos(pi*angle_limit/180)

    all_tracks = []

    for vox_loc in start_loc:
        track = [vox_loc]
        cur_step, fa = get_tensor_info(vox_loc, evec1_vol, fa_vol)
        if dot(cur_step, start_step) > 0:
            prev_step = cur_step
        else:
            prev_step = -cur_step
        while keep_tracking(cur_step, prev_step, fa, fa_limit, dot_limit):
            if dot(cur_step, prev_step) < 0:
                cur_step = -cur_step
            vox_loc, prev_step = propagate(vox_loc, cur_step, prev_step,
                                           vox_size)
            track.append(vox_loc)
            cur_step, fa = get_tensor_info(vox_loc, evec1_vol, fa_vol)
        if len(track) > 1:
            all_tracks.append((array(track)*vox_size, None, None))

    return all_tracks

def get_tensor_info(vox_loc, evec1_vol, fa_vol):
    """returns ev1 and fa for a given location"""
    ind = vox_loc.asytpe('int')
    step = evec1_vol[ind[0], ind[1], ind[2]]
    fa = fa_vol[ind[0], ind[1], ind[2]]
    return step, fa

def keep_tracking(cur_step, prev_step, fa, fa_limit, dot_limit):
    """checks the fa_limit and dot_limit for the current location"""

    if fa < fa_limit:
        return False

    d = dot(cur_step, prev_step)
    if abs(d) < dot_limit:
        return False

    return True

def seeds_from_mask(mask, density):
    """Takes a binary mask and returns seeds in voxels != 0

    places evanly spaced points in nonzero voxels of mask, spaces the points
    based on density. For example if density is [1, 2, 3], there will be 6
    points in each voxel, at x=.5, y=[.25, .75] and z=[.166, .5, .833].
    density = a is the same as density = [a, a, a]

    """
    mask = atleast_3d(mask)
    if mask.ndim != 3:
        raise ValueError('mask cannot be more than 3d')
    density = array(density, 'int')
    sp = empty(3)
    sp[:] = 1./density

    voxels = mask.nonzero()
    mg = mgrid[0:1:sp[0], 0:1:sp[1], 0:1:sp[2]]

    seeds = []
    for ii, jj, kk in zip(voxels, mg, sp):
        s = ii[:,None] + jj.ravel() + kk/2
        seeds.append(s.ravel())

    seeds = array(seeds).T
    return seeds

def target(tracks, voxel_dim, mask):
    assert mask.ndim == 3
    result_tracks = []
    for ii in tracks:
        ind = (ii/voxel_dim).T.astype('int')
        try:
            state = mask[ind[0], ind[1], ind[2]]
        except IndexError:
            1./0
        else:
            if state.any():
                result_tracks.append(ii)
    return result_tracks
