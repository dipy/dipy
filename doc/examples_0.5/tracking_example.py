from nibabel import load
from nibabel.trackvis import empty_header, write
from dipy.tracking.integration import BoundryIntegrator, generate_streamlines
from dipy.tracking.utils import seeds_from_mask, target
from dipy.reconst.shm import SlowAdcOpdfModel, ClosestPeakSelector, \
    normalize_data, ResidualBootstrapWrapper
from dipy.reconst.interpolate import NearestNeighborInterpolator
from dipy.core.triangle_subdivide import create_half_unit_sphere

from dipy.data import sample_hardi_data, sample_tracking_seedNtarget

def simple_tracking_function(data, fa, bval, bvec, seed_mask, start_steps,
                             voxel_size, density):
    """An example of a simple traking function using the tools in dipy

    This tracking function uses the SlowAdcOpdfModel to fit diffusion data. By
    using the ClosestPeakSelector, the function tracks along the peak of Opdf
    closest to the incoming direction. It also uses the BoundryIntegrator to
    integrate the streamlines and NearestNeighborInterpolator to interpolate
    the data. The ResidualBootstrap means the tracks are probabilistic, not
    deterministic.
    """

    #the interpolator allows us to index the dwi data in continous space
    data_mask = fa > .2
    normalized_data = normalize_data(data, bval)
    interpolator = NearestNeighborInterpolator(normalized_data, voxel_size,
                                               data_mask)

    #the model fits the dwi data, this model can resolve crossing fibers
    #see documentation of SlowAdcOpdfModel for more info
    model = SlowAdcOpdfModel(6, bval, bvec, .006)
    vert, edges, faces = create_half_unit_sphere(4)
    model.set_sampling_points(vert, edges)

    #this residual bootstrap wrapper returns a sample from the bootstrap
    #distribution istead of returning the raw data
    min_signal = normalized_data.min()
    B = model.B
    wrapped_interp = ResidualBootstrapWrapper(interpolator, B, min_signal)


    #the peakselector returns the closest peak to the incoming direction when
    #in voxels with multiple peaks
    peak_finder = ClosestPeakSelector(model, wrapped_interp)
    peak_finder.angle_limit = 60

    seeds = seeds_from_mask(seed_mask, density, voxel_size)

    #the propagator is used to integrate the streamlines
    propogator = BoundryIntegrator(voxel_size)
    tracks = generate_streamlines(peak_finder, propogator, seeds, start_steps)

    return tracks

def main():
    """Track example dataset"""
    data, fa, bvec, bval, voxel_size = sample_hardi_data()
    seed_mask, target_mask = sample_tracking_seedNtarget()

    density = [1, 1, 2]
    start_step = [-0.3, -0.7, -0.7]
    tracks = simple_tracking_function(data, fa, bval, bvec, seed_mask, start_step,
                                      voxel_size, density)
    tracks = list(tracks)
    targeted_tracks = target(tracks, target_mask, voxel_size)

    """
    Uncomment this to save tracks

    trk_tracks = ((streamline, None, None) for streamline in tracks)
    trgt_trk_tracks = ((streamline, None, None) for streamline in targeted_tracks)

    trk_hdr = empty_header()
    trk_hdr['voxel_order'] = 'LPI'
    trk_hdr['voxel_size'] = voxel_size
    trk_hdr['dim'] = fa.shape
    write('example_tracks_before_target.trk', trk_tracks, trk_hdr)
    write('example_tracks_after_target.trk', trgt_trk_tracks, trk_hdr)
    """

if __name__ == "__main__":
    main()

