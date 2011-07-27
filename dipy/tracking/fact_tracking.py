from numpy import abs, array, asarray, atleast_3d, broadcast_arrays, cos, \
                  dot, empty, floor, mgrid, pi, signbit, sqrt, vstack

class FactPropogator(object):
    """Used for fact fiber tracking"""
    def __init__(self, voxel_size=(1,1,1), overstep=1e-1):
        """Creates a propogator instance

        Parameters:
        -----------
        voxel_size : array-like
            Size of voxels in data volume
        overstep : float
            A small number used to prevent the track from getting stuck at the
            edge of a voxel.
        """
        self.overstep = overstep
        self.voxel_size = asarray(voxel_size)

    def propagate(self, location, step):
        """takes a step just past the edge of the next voxel along step

        given a location and a step, finds the smallest step needed to move
        into the next voxel

        Parameters:
        -----------
        location : array-like, (3,)
            location to propagate
        step : array-like, (3,)
            direction in 3 space to propagate along
        """
        space = location % self.voxel_size
        dist = self.voxel_size*(~signbit(step)) - space
        step_sizes = dist/step
        smallest_step = step_sizes.min()
        assert smallest_step >= 0
        smallest_step += self.overstep
        new_location = location + smallest_step*step
        return new_location

class FixedStepPropogator(object):
    def __init__(self, step_size=.5):
        self.step_size = step_size
    def propagate(self, location, step):
        """Takes a step of step_size from location"""
        new_location = self.step_size*step + location
        return new_loctation

def track_streamlines(stepper, propogator, seeds, start_steps):
    """Creates streamlines using a stepper and propogator

    Creates streamlines starting at each of the seeds, and integrates them
    using the propogator, getting the direction of each step from stepper.

    Parameters:
    -----------
    stepper : object with next_step method
        The next_step method should take a location, (x, y, z) cordinates in
        3-space, and the previous step, a unit vector in 3-space, and return a
        new step
    propogator : object with a propagate method
        The propagate method should take a location and a step, as defined
        above and return a new location.
    seeds : array-like (N, 3)
        Locations where to start tracking.
    start_steps : array-like
        start_steps should be broadcastable with seeds and is used in place of
        prev_step when determining the initial step for each seed.

    Returns:
    --------
    tracks : list
        Each element of tracks is a streamline, an (M, 3) array of coordinates
        in 3-space.
    """
    start_steps = asarray(start_steps)
    norm_start_steps = sqrt((start_steps*start_steps).sum(-1))
    norm_start_steps = norm_start_steps.reshape((-1, 1))
    start_steps = start_steps/norm_start_steps
    seeds, start_steps = broadcast_arrays(seeds, start_steps)

    all_tracks = []
    next_step = stepper.next_step
    propagate = propogator.propagate
    for ii in xrange(len(seeds)):
        location = seeds[ii]
        step = start_steps[ii]
        track = [location]
        step = next_step(location, step)
        while step is not None:
            location = propagate(location, step)
            track.append(location)
            step = next_step(location, step)
        all_tracks.append(array(track))
    return all_tracks

class TensorStepper(object):
    """Used for tracking diffusion tensors, has next_step method"""
    def _get_angel_limit(self):
        return arccos(self.dot_limit)*180/pi
    def _set_angel_limit(self, angle):
        if angle >= 0 and angle <= 90:
            self.dot_limit = cos(angle*pi/180)
        else:
            raise ValueError("angle should be between 0 and 180")
    angel_limit = property(_get_angel_limit, _set_angel_limit)

    def __init__(self, voxel_size, fa_limit=None, angle_limit=None,
                 fa_vol=None, evec1_vol=None):
        self.voxel_size = voxel_size
        self.fa_limit = fa_limit
        self.angle_limit = angle_limit
        if fa_vol.shape != evec1_vol.shape[:-1]:
            msg = "the fa and eigen vector volumes are not the same shape"
            raise ValueError(msg)
        if evec1_vol.shape[-1] != 3:
            msg = "eigen vector volume should have vecetors of length 3 " + \
                  "along the last dimmension"
            raise ValueError(msg)
        self.fa_vol = fa_vol
        self.evec1_vol = evec1_vol

    def from_tensor(tensor):
        self.fa_vol = tensor.fa()
        self.evec1_vol = tensor.evec[..., 0]

    def next_step(location, prev_step):
        """Returns the nearest neighbor tensor for location"""
        vox_loc = location/self.voxel_size
        vox_loc = tuple(int(ii) for ii in vox_loc)
        try:
            if self.fa_vol[vox_loc] < self.fa_limit:
                return
            step = self.evec1_vol[vox_loc]
        except IndexError:
            return
        angle_dot = dot(step, prev_step)
        if abs(angle_dot) < self.dot_limit:
            return
        if angle_dot > 0:
            return step
        else:
            return -step

def seeds_from_mask(mask, density, voxel_size=(1,1,1)):
    """Takes a binary mask and returns seeds in voxels != 0

    places evanly spaced points in nonzero voxels of mask, spaces the points
    based on density. For example if density is [1, 2, 3], there will be 6
    points in each voxel, at x=.5, y=[.25, .75] and z=[.166, .5, .833].
    density = a is the same as density = [a, a, a]

    """
    mask = atleast_3d(mask)
    if mask.ndim != 3:
        raise ValueError('mask cannot be more than 3d')
    density = asarray(density, 'int')
    sp = empty(3)
    sp[:] = 1./density

    voxels = mask.nonzero()
    mg = mgrid[0:1:sp[0], 0:1:sp[1], 0:1:sp[2]]

    seeds = []
    for ii, jj, kk in zip(voxels, mg, sp):
        s = ii[:,None] + jj.ravel() + kk/2
        seeds.append(s.ravel())

    seeds = array(seeds).T
    seeds *= voxel_size
    return seeds

def target(tracks, target_mask, voxel_size):
    """Retain tracks that pass though target_mask

    Parameters:
    -----------
    tracks
        A squence of streamlines. Each streamline should be a (N, 3) array,
        where N is the length of the streamline
    target_mask : array-like
        A mask used as a target
    voxel_size
        Size of the voxels in the target_mask

    Returns:
    tracks
        A sequence of streamlines that pass though target_mask
    """
    result_tracks = []
    for ii in tracks:
        ind = (ii/voxel_size).astype('int')
        i, j, k = ind.T
        try:
            state = target_mask[i, j, k]
        except IndexError:
            vol_dim = (i.max(), j.max(), k.max())
            msg = "traget_mask is too small, target_mask should be at least "+\
                  str(vol_dim)
            raise ValueError(msg)
        if state.any():
            result_tracks.append(ii)
    return result_tracks
