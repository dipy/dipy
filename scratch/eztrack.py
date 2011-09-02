import os.path as path
import string
import numpy as np
import nibabel as nib
from scipy.ndimage import convolve
from traits.api import *
from nibabel.trackvis import write, empty_header
from dipy.reconst.shm import SlowAdcOpdfModel, MonoExpOpdfModel, \
        QballOdfModel, normalize_data, ClosestPeakSelector, \
        ResidualBootstrapWrapper, hat, lcr_matrix, bootstrap_data_array
from dipy.reconst.interpolate import LinearInterpolator, \
        NearestNeighborInterpolator
from dipy.core.triangle_subdivide import create_half_unit_sphere, \
        disperse_charges
from dipy.tracking.integration import FactIntegrator, FixedStepIntegrator, \
        generate_streamlines
from dipy.tracking.utils import seeds_from_mask, target, merge_streamlines, \
        streamline_counts
from dipy.io.bvectxt import read_bvec_file, orientation_to_string, \
        reorient_bvec
import pickle

nifti_file = File(filter=['*.nii.gz'])
def read_roi(file, threshold=0, shape=None):
    img = nib.load(file)
    if shape is not None:
        if shape != img.shape:
            raise IOError('The roi image does not have the right shape, '+
                          'expecting '+str(shape)+' got '+str(img.shape))
    mask = img.get_data() > threshold
    return mask

class InputData(HasTraits):
    dwi_images = nifti_file
    fa_file = nifti_file
    bvec_file = File(filter=['*.bvec'])
    bvec_orientation = String('', minlen=3, maxlen=3)
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
        if self.bvec_orientation is not '':
            data_ornt = nib.io_orientation(affine)
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
    shape = Array('int', shape=(3,), value=[1,1,1], label='shape (in voxels)')

    def get_kernel(self):
        kernel = np.ones(self.shape)/self.shape.prod()
        kernel.shape += (1,)
        return kernel

all_kernels = {None:None,'Box':BoxKernel,'Gausian':GausianKernel}
all_interpolators = {'NearestNeighbor':NearestNeighborInterpolator,
                     'TriLinear':LinearInterpolator}
all_shmodels = {'QballOdf':QballOdfModel, 'SlowAdcOpdf':SlowAdcOpdfModel,
                'MonoExpOpdf':MonoExpOpdfModel}
all_integrators = {'Fact':FactIntegrator, 'FixedStep':FixedStepIntegrator}

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
    sh_order = Int(6)
    Lambda = Float('')
    sphere_coverage = Int('')
    min_peak_spacing = Range(0.,90,90)
    min_relative_peak = Range(0.,1,1)

    probabilistic = Bool(False, label='Probabilistic (Residual Bootstrap)')
    bootstrap_input = Bool(False)

    #integrator = Enum('Fact', all_integrators.keys())
    direction = Array(dtype='float', shape=(3,), value=[1,0,0])
    track_two_directions = Bool(False)
    fa_threshold = Float(1.0)
    max_turn_angle = Range(0.,90,0)

    targets = List(nifti_file, [])

    #get set later
    voxel_size = Array(dtype='float', shape=(3,))
    affine = Array(dtype='float', shape=(4,4))
    shape = Tuple((0,0,0))

    def track_shm(self):

        data, voxel_size, affine, fa, bvec, bval = self.all_inputs.read_data()
        self.voxel_size = voxel_size
        self.affine = affine
        self.shape = fa.shape
        mask = fa > self.fa_threshold

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
            bootstrap_data_array(data, H, R, data.min())

        interpolator_type = all_interpolators[self.interpolator]
        interpolator = interpolator_type(data, voxel_size, mask)

        seed_mask = read_roi(self.seed_roi, shape=self.shape)
        seeds = seeds_from_mask(seed_mask, self.seed_density, voxel_size)

        peak_finder = ClosestPeakSelector(model, interpolator,
                            self.min_relative_peak, self.min_peak_spacing)
        peak_finder.angle_limit = 90
        start_steps = []
        best_start = self.direction.copy()
        best_start /= np.sqrt((best_start*best_start).sum())
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
        integrator = FactIntegrator(voxel_size, overstep=.01)
        streamlines = generate_streamlines(peak_finder, integrator, seeds,
                                           start_steps)
        if self.track_two_directions:
            start_steps = [-ii for ii in start_steps]
            streamlinesB = generate_streamlines(peak_finder, integrator, seeds,
                                                 start_steps)
            streamlines = merge_streamlines(streamlines, streamlinesB)

        for ii in self.targets:
            target_mask = read_roi(ii, shape=fa.shape)
            streamlines = target(streamlines, target_mask, voxel_size)

        return streamlines

