from nose import SkipTest


#############################################################################
# Remove this when the module becomes functional again
class ThisIsBroken(SkipTest):
    pass
raise ThisIsBroken("this module is undergoing a major overhaul as therefore "
                   "does not currently work")
#############################################################################


import pickle
import string
import os.path as path

import numpy as np
from scipy.ndimage import convolve

# Import traits as optional package
try:
    import traits.api as T
except ImportError:
    from ..utils.optpkg import OptionalImportError
    raise OptionalImportError("You must have traits to use this module")

import nibabel as nib
from nibabel.trackvis import write, empty_header

from ..reconst.shm import (SlowAdcOpdfModel, MonoExpOpdfModel, QballOdfModel,
                          normalize_data, ClosestPeakSelector,
                          ResidualBootstrapWrapper, hat, lcr_matrix,
                          bootstrap_data_array, NND_ClosestPeakSelector)
from ..reconst.interpolate import (TriLinearInterpolator,
                                  NearestNeighborInterpolator)
from ..tracking.integration import (BoundryIntegrator, FixedStepIntegrator,
                                   generate_streamlines)
from ..tracking.utils import (seeds_from_mask, target, merge_streamlines,
                             density_map)
from ..io.bvectxt import (read_bvec_file, orientation_to_string,
                         reorient_vectors)

nifti_file = T.File(filter=['Nifti Files', '*.nii.gz',
                            'Nifti Pair or Analyze Files', '*.img.gz',
                            'All Files', '*'])

def read_roi(file, threshold=0, shape=None):
    img = nib.load(file)
    if shape is not None:
        if shape != img.shape:
            raise IOError('The roi image does not have the right shape, '+
                          'expecting '+str(shape)+' got '+str(img.shape))
    img_data = img.get_data()
    if img_data.max() > 1:
        raise ValueError('this does not seem to be a mask')
    mask = img_data > threshold
    return mask


class InputData(T.HasTraits):
    dwi_images = nifti_file
    fa_file = nifti_file
    bvec_file = T.File(filter=['*.bvec'])
    bvec_orientation = T.String('IMG', minlen=3, maxlen=3)
    min_signal = T.Float(1)

    @T.on_trait_change('dwi_images')
    def update_files(self):
        dir, file = path.split(self.dwi_images)
        base = string.split(file, path.extsep, 1)[0]
        if self.fa_file == '':
            self.fa_file = path.join(dir, base+'_fa.nii.gz')
        if self.bvec_file == '':
            self.bvec_file = path.join(dir, base+'.bvec')

    def read_data(self):
        data_img = nib.load(self.dwi_images)
        affine = data_img.get_affine()
        voxel_size = data_img.get_header().get_zooms()
        voxel_size = voxel_size[:3]
        fa_img = nib.load(self.fa_file)
        assert data_img.shape[:-1] == fa_img.shape
        bvec, bval = read_bvec_file(self.bvec_file)
        data_ornt = nib.io_orientation(affine)
        if self.bvec_orientation != 'IMG':
            bvec = reorient_vectors(bvec, self.bvec_orientation, data_ornt)
        fa = fa_img.get_data()
        data = data_img.get_data()
        return data, voxel_size, affine, fa, bvec, bval

class GausianKernel(T.HasTraits):
    sigma = T.Float(1, label='sigma (in voxels)')
    shape = T.Array('int', shape=(3,), value=[1,1,1],
                    label='shape (in voxels)')
    def get_kernel(self):
        raise NotImplementedError
        #will get to this soon

class BoxKernel(T.HasTraits):
    shape = T.Array('int', shape=(3,), value=[3,3,3],
                    label='shape (in voxels)')

    def get_kernel(self):
        kernel = np.ones(self.shape)/self.shape.prod()
        kernel.shape += (1,)
        return kernel

def lazy_index(index):
    """Produces a lazy index

    Returns a slice that can be used for indexing an array, if no slice can be
    made index is returned as is.
    """
    index = np.asarray(index)
    assert index.ndim == 1
    if index.dtype == np.bool:
        index = index.nonzero()[0]
    if len(index) == 1:
        return slice(index[0], index[0] + 1)
    step = np.unique(np.diff(index))
    if len(step) != 1 or step[0] == 0:
        return index
    else:
        return slice(index[0], index[-1] + 1, step[0])

def closest_start(seeds, peak_finder, best_start):
    starts = np.empty(seeds.shape)
    best_start = np.asarray(best_start, 'float')
    best_start /= np.sqrt((best_start*best_start).sum())
    for i in xrange(len(seeds)):
        try:
            starts[i] = peak_finder.next_step(seeds[i], best_start)
        except StopIteration:
            starts[i] = best_start
    return starts

all_kernels = {None:None,'Box':BoxKernel,'Gausian':GausianKernel}
all_interpolators = {'NearestNeighbor':NearestNeighborInterpolator,
                     'TriLinear':TriLinearInterpolator}
all_shmodels = {'QballOdf':QballOdfModel, 'SlowAdcOpdf':SlowAdcOpdfModel,
                'MonoExpOpdf':MonoExpOpdfModel}
all_integrators = {'Boundry':BoundryIntegrator, 'FixedStep':FixedStepIntegrator}

class ShmTrackingInterface(T.HasStrictTraits):

    dwi_images = T.DelegatesTo('all_inputs')
    all_inputs = T.Instance(InputData, args=())
    min_signal = T.DelegatesTo('all_inputs')
    seed_roi = nifti_file
    seed_density = T.Array(dtype='int', shape=(3,), value=[1,1,1])

    smoothing_kernel_type = T.Enum(None, all_kernels.keys())
    smoothing_kernel = T.Instance(T.HasTraits)
    @T.on_trait_change('smoothing_kernel_type')
    def set_smoothing_kernel(self):
        if self.smoothing_kernel_type is not None:
            kernel_factory = all_kernels[self.smoothing_kernel_type]
            self.smoothing_kernel = kernel_factory()
        else:
            self.smoothing_kernel = None

    interpolator = T.Enum('NearestNeighbor', all_interpolators.keys())
    model_type = T.Enum('SlowAdcOpdf', all_shmodels.keys())
    sh_order = T.Int(4)
    Lambda = T.Float(0, desc="Smoothing on the odf")
    sphere_coverage = T.Int(5)
    min_peak_spacing = T.Range(0.,1,np.sqrt(.5), desc="as a dot product")
    min_relative_peak = T.Range(0.,1,.25)

    probabilistic = T.Bool(False, label='Probabilistic (Residual Bootstrap)')
    bootstrap_input = T.Bool(False)
    bootstrap_vector = T.Array(dtype='int', value=[])

    #integrator = Enum('Boundry', all_integrators.keys())
    seed_largest_peak = T.Bool(False, desc="Ignore sub-peaks and start follow "
                                           "the largest peak at each seed")
    start_direction = T.Array(dtype='float', shape=(3,), value=[0,0,1],
                              desc="Prefered direction from seeds when "
                                   "multiple directions are available. "
                                   "(Mostly) doesn't matter when 'seed "
                                   "largest peak' and 'track two directions' "
                                   "are both True",
                              label="Start direction (RAS)")
    track_two_directions = T.Bool(False)
    fa_threshold = T.Float(1.0)
    max_turn_angle = T.Range(0.,90,0)

    stop_on_target = T.Bool(False)
    targets = T.List(nifti_file, [])

    #will be set later
    voxel_size = T.Array(dtype='float', shape=(3,))
    affine = T.Array(dtype='float', shape=(4,4))
    shape = T.Tuple((0,0,0))

    #set for io
    save_streamlines_to = T.File('')
    save_counts_to = nifti_file

    #io methods
    def save_streamlines(self, streamlines, save_streamlines_to):
        trk_hdr = empty_header()
        voxel_order = orientation_to_string(nib.io_orientation(self.affine))
        trk_hdr['voxel_order'] = voxel_order
        trk_hdr['voxel_size'] = self.voxel_size
        trk_hdr['vox_to_ras'] = self.affine
        trk_hdr['dim'] = self.shape
        trk_tracks = ((ii,None,None) for ii in streamlines)
        write(save_streamlines_to, trk_tracks, trk_hdr)
        pickle.dump(self, open(save_streamlines_to + '.p', 'wb'))

    def save_counts(self, streamlines, save_counts_to):
        counts = density_map(streamlines, self.shape, self.voxel_size)
        if counts.max() < 2**15:
            counts = counts.astype('int16')
        nib.save(nib.Nifti1Image(counts, self.affine), save_counts_to)

    #tracking methods
    def track_shm(self, debug=False):
        if self.sphere_coverage > 7 or self.sphere_coverage < 1:
            raise ValueError("sphere coverage must be between 1 and 7")
        verts, edges, faces = create_half_unit_sphere(self.sphere_coverage)
        verts, pot = disperse_charges(verts, 10, .3)

        data, voxel_size, affine, fa, bvec, bval = self.all_inputs.read_data()
        self.voxel_size = voxel_size
        self.affine = affine
        self.shape = fa.shape

        model_type = all_shmodels[self.model_type]
        model = model_type(self.sh_order, bval, bvec, self.Lambda)
        model.set_sampling_points(verts, edges)

        data = np.asarray(data, dtype='float', order='C')
        if self.smoothing_kernel is not None:
            kernel = self.smoothing_kernel.get_kernel()
            convolve(data, kernel, out=data)

        normalize_data(data, bval, self.min_signal, out=data)
        dmin = data.min()
        data = data[..., lazy_index(bval > 0)]
        if self.bootstrap_input:
            if self.bootstrap_vector.size == 0:
                n = data.shape[-1]
                self.bootstrap_vector = np.random.randint(n, size=n)
            H = hat(model.B)
            R = lcr_matrix(H)
            data = bootstrap_data_array(data, H, R, self.bootstrap_vector)
            data.clip(dmin, out=data)

        mask = fa > self.fa_threshold
        targets = [read_roi(tgt, shape=self.shape) for tgt in self.targets]
        if self.stop_on_target:
            for target_mask in targets:
                mask = mask & ~target_mask

        seed_mask = read_roi(self.seed_roi, shape=self.shape)
        seeds = seeds_from_mask(seed_mask, self.seed_density, voxel_size)

        if self.interpolator == 'NearestNeighbor' and not self.probabilistic and not debug:
            using_optimze = True
            peak_finder = NND_ClosestPeakSelector(model, data, mask, voxel_size)
        else:
            using_optimze = False
            interpolator_type = all_interpolators[self.interpolator]
            interpolator = interpolator_type(data, voxel_size, mask)
            peak_finder = ClosestPeakSelector(model, interpolator)

        #Set peak_finder parameters for start steps
        peak_finder.angle_limit = 90
        model.peak_spacing = self.min_peak_spacing
        if self.seed_largest_peak:
            model.min_relative_peak = 1
        else:
            model.min_relative_peak = self.min_relative_peak

        data_ornt = nib.io_orientation(self.affine)
        best_start = reorient_vectors(self.start_direction, 'ras', data_ornt)
        start_steps = closest_start(seeds, peak_finder, best_start)

        if self.probabilistic:
            interpolator = ResidualBootstrapWrapper(interpolator, model.B,
                                                    min_signal=dmin)
            peak_finder = ClosestPeakSelector(model, interpolator)
        elif using_optimze and self.seed_largest_peak:
            peak_finder.reset_cache()

        #Reset peak_finder parameters for tracking
        peak_finder.angle_limit = self.max_turn_angle
        model.peak_spacing = self.min_peak_spacing
        model.min_relative_peak = self.min_relative_peak

        integrator = BoundryIntegrator(voxel_size, overstep=.1)
        streamlines = generate_streamlines(peak_finder, integrator, seeds,
                                           start_steps)
        if self.track_two_directions:
            start_steps = -start_steps
            streamlinesB = generate_streamlines(peak_finder, integrator, seeds,
                                                 start_steps)
            streamlines = merge_streamlines(streamlines, streamlinesB)

        for target_mask in targets:
            streamlines = target(streamlines, target_mask, voxel_size)

        return streamlines

