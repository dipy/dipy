import pickle
import string
import os.path as path
import numpy as np
import nibabel as nib

from scipy.ndimage import convolve
from traits.api import *
from nibabel.trackvis import write, empty_header

from dipy.reconst.shm import SlowAdcOpdfModel, MonoExpOpdfModel, \
        QballOdfModel, normalize_data, ClosestPeakSelector, \
        ResidualBootstrapWrapper, hat, lcr_matrix, bootstrap_data_array
from dipy.reconst.interpolate import TriLinearInterpolator, \
        NearestNeighborInterpolator
from dipy.core.triangle_subdivide import create_half_unit_sphere, \
        disperse_charges
from dipy.tracking.integration import FactIntegrator, FixedStepIntegrator, \
        generate_streamlines
from dipy.tracking.utils import seeds_from_mask, target, merge_streamlines, \
        streamline_counts
from dipy.io.bvectxt import read_bvec_file, orientation_to_string, \
        reorient_bvec

nifti_file = File(filter=['Nifti Files', '*.nii.gz',
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

class InputData(HasTraits):
    dwi_images = nifti_file
    fa_file = nifti_file
    bvec_file = File(filter=['*.bvec'])
    bvec_orientation = String('IMG', minlen=3, maxlen=3)
    min_signal = Float(1)

    @on_trait_change('dwi_images')
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
            bvec = reorient_bvec(bvec, self.bvec_orientation, data_ornt)
        fa = fa_img.get_data()
        data = data_img.get_data()
        return data, voxel_size, affine, fa, bvec, bval

class GausianKernel(HasTraits):
    sigma = Float(1, label='sigma (in voxels)')
    shape = Array('int', shape=(3,), value=[1,1,1], label='shape (in voxels)')
    def get_kernel(self):
        raise NotImplementedError
        #will get to this soon

class BoxKernel(HasTraits):
    shape = Array('int', shape=(3,), value=[3,3,3], label='shape (in voxels)')

    def get_kernel(self):
        kernel = np.ones(self.shape)/self.shape.prod()
        kernel.shape += (1,)
        return kernel

all_kernels = {None:None,'Box':BoxKernel,'Gausian':GausianKernel}
all_interpolators = {'NearestNeighbor':NearestNeighborInterpolator,
                     'TriLinear':TriLinearInterpolator}
all_shmodels = {'QballOdf':QballOdfModel, 'SlowAdcOpdf':SlowAdcOpdfModel,
                'MonoExpOpdf':MonoExpOpdfModel}
all_integrators = {'Fact':FactIntegrator, 'FixedStep':FixedStepIntegrator}

def _hack(mask):
    mask[[0,-1]] = 0
    mask[:,[0,-1]] = 0
    mask[:,:,[0,-1]] = 0

class EZTrackingInterface(HasStrictTraits):

    dwi_images = DelegatesTo('all_inputs')
    all_inputs = Instance(InputData, args=())
    min_signal = DelegatesTo('all_inputs')
    seed_roi = nifti_file
    seed_density = Array(dtype='int', shape=(3,), value=[1,1,1])

    smoothing_kernel_type = Enum(None, all_kernels.keys())
    smoothing_kernel = Instance(HasTraits)
    @on_trait_change('smoothing_kernel_type')
    def set_smoothing_kernel(self):
        if self.smoothing_kernel_type is not None:
            kernel_factory = all_kernels[self.smoothing_kernel_type]
            self.smoothing_kernel = kernel_factory()
        else:
            self.smoothing_kernel = None

    interpolator = Enum('NearestNeighbor', all_interpolators.keys())
    model_type = Enum('SlowAdcOpdf', all_shmodels.keys())
    sh_order = Int(4)
    Lambda = Float(0, desc="Smoothing on the odf")
    sphere_coverage = Int(5)
    min_peak_spacing = Range(0.,1,np.sqrt(.5), desc="as a dot product")
    min_relative_peak = Range(0.,1,.25)

    probabilistic = Bool(False, label='Probabilistic (Residual Bootstrap)')
    bootstrap_input = Bool(False)

    #integrator = Enum('Fact', all_integrators.keys())
    start_direction = Array(dtype='float', shape=(3,), value=[0,0,1],
                            desc="prefered direction from seeds when " +
                                 "multiple directions are available",
                            label="Start direction (RAS)")
    track_two_directions = Bool(False)
    fa_threshold = Float(1.0)
    max_turn_angle = Range(0.,90,0)

    stop_on_target = Bool(False)
    targets = List(nifti_file, [])

    #will be set later
    voxel_size = Array(dtype='float', shape=(3,))
    affine = Array(dtype='float', shape=(4,4))
    shape = Tuple((0,0,0))

    #set for io
    save_streamlines_to = File('')
    save_counts_to = nifti_file

    #io methods
    def save_streamlines(self, streamlines, save_streamlines_to):
        trk_hdr = empty_header()
        voxel_order = orientation_to_string(nib.io_orientation(self.affine))
        trk_hdr['voxel_order'] = voxel_order
        trk_hdr['voxel_size'] = self.voxel_size
        trk_hdr['dim'] = self.shape
        trk_hdr['voxel_to_ras'] = self.affine
        trk_tracks = ((ii,None,None) for ii in streamlines)
        write(save_streamlines_to, trk_tracks, trk_hdr)
        pickle.dump(self, open(save_streamlines_to + '.p', 'wb'))

    def save_counts(self, streamlines, save_counts_to):
        counts = streamline_counts(streamlines, self.shape, self.voxel_size)
        if counts.max() < 2**15:
            counts = counts.astype('int16')
        nib.save(nib.Nifti1Image(counts, self.affine), save_counts_to)

    #tracking methods
    def track_shm(self):

        data, voxel_size, affine, fa, bvec, bval = self.all_inputs.read_data()
        self.voxel_size = voxel_size
        self.affine = affine
        self.shape = fa.shape

        model_type = all_shmodels[self.model_type]
        model = model_type(self.sh_order, bval, bvec, self.Lambda)
        verts, edges, faces = create_half_unit_sphere(self.sphere_coverage)
        verts, pot = disperse_charges(verts, 40)
        model.set_sampling_points(verts, edges)

        if self.smoothing_kernel is not None:
            kernel = self.smoothing_kernel.get_kernel()
            data = np.asarray(data, 'float')
            convolve(data, kernel, data)

        data = normalize_data(data, bval, self.min_signal)
        if self.bootstrap_input:
            H = hat(model.B)
            R = lcr_matrix(H)
            dmin = data.min()
            data = bootstrap_data_array(data, H, R)
            data.clip(dmin, 1., data)

        mask = fa > self.fa_threshold
        _hack(mask)
        targets = [read_roi(tgt, shape=self.shape) for tgt in self.targets]
        if self.stop_on_target:
            for target_mask in targets:
                mask = mask & ~target_mask
        interpolator_type = all_interpolators[self.interpolator]
        interpolator = interpolator_type(data, voxel_size, mask)

        seed_mask = read_roi(self.seed_roi, shape=self.shape)
        seeds = seeds_from_mask(seed_mask, self.seed_density, voxel_size)

        peak_finder = ClosestPeakSelector(model, interpolator,
                            self.min_relative_peak, self.min_peak_spacing)
        peak_finder.angle_limit = 90
        start_steps = []
        data_ornt = nib.io_orientation(self.affine)
        ind = np.asarray(data_ornt[:,0], 'int')
        best_start = self.start_direction[ind]
        best_start *= data_ornt[:,1]
        best_start /= np.sqrt((best_start*best_start).sum())
        print best_start
        for ii in seeds:
            try:
                step = peak_finder.next_step(ii, best_start)
                start_steps.append(step)
            except StopIteration:
                start_steps.append(best_start)
        if self.probabilistic:
            interpolator = ResidualBootstrapWrapper(interpolator, model.B,
                                                    data.min())
        peak_finder = ClosestPeakSelector(model, interpolator,
                            self.min_relative_peak, self.min_peak_spacing)
        peak_finder.angle_limit = self.max_turn_angle
        integrator = FactIntegrator(voxel_size, overstep=.1)
        streamlines = generate_streamlines(peak_finder, integrator, seeds,
                                           start_steps)
        if self.track_two_directions:
            start_steps = [-ii for ii in start_steps]
            streamlinesB = generate_streamlines(peak_finder, integrator, seeds,
                                                 start_steps)
            streamlines = merge_streamlines(streamlines, streamlinesB)

        for target_mask in targets:
            streamlines = target(streamlines, target_mask, voxel_size)

        return streamlines

