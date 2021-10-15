import numpy as np
import numpy.testing as npt

from dipy.core.sphere import unit_octahedron
from dipy.reconst.shm import SphHarmFit, SphHarmModel
from dipy.direction import PTTDirectionGetter

def test_ptt(interactive=False):
    from dipy.io.image import load_nifti
    from dipy.reconst.shm import sh_to_sf
    from dipy.data import default_sphere
    from dipy.tracking.local_tracking import LocalTracking
    from dipy.tracking.stopping_criterion import BinaryStoppingCriterion
    from dipy.tracking.streamline import Streamlines
    from os.path import join as pjoin
    folder = "/Users/koudoro/Software/dipy/tmp"
    fod, affine = load_nifti(pjoin(folder, "FOD.nii"))
    seed_image, affine_seed = load_nifti(pjoin(folder, "seedImage.nii"))
    seed_coordinates = np.loadtxt(pjoin(folder, "seedCoordinates.txt"))

    # check basis (should be mrtrix3)
    sph = sh_to_sf(fod, default_sphere, basis_type='tournier07', sh_order=8)
    sph[sph <0] = 0
    sc = BinaryStoppingCriterion(np.ones(fod.shape[:3]))
    dg = PTTDirectionGetter.from_pmf(sph, sphere=default_sphere, max_angle=1)
    streamline_generator = LocalTracking(direction_getter=dg,
                                         step_size=1/20,
                                         stopping_criterion=sc,
                                         seeds=seed_coordinates,
                                         affine=affine)

    # import ipdb; ipdb.set_trace()
    streamlines = Streamlines(streamline_generator)

    if interactive:
        from fury import actor, window
        st = actor.streamtube(streamlines)
        scene = window.Scene()
        scene.add(st)
        window.show(scene)

    import ipdb; ipdb.set_trace()
    print(streamlines)



test_ptt()

def test_PTTDirectionGetter():
    # Test the constructors and errors of the PTTDirectionGetter

    class SillyModel(SphHarmModel):

        sh_order = 4

        def fit(self, data, mask=None):
            coeff = np.zeros(data.shape[:-1] + (15,))
            return SphHarmFit(self, coeff, mask=None)

    model = SillyModel(gtab=None)
    data = np.zeros((3, 3, 3, 7))

    # Test if the tracking works on different dtype of the same data.
    for dtype in [np.float32, np.float64]:
        fit = model.fit(data.astype(dtype))

        # Sample point and direction
        point = np.zeros(3)
        dir = unit_octahedron.vertices[0].copy()

        # make a dg from a fit
        dg = PTTDirectionGetter.from_shcoeff(fit.shm_coeff, 90,
                                             unit_octahedron)
        state = dg.get_direction(point, dir)
        npt.assert_equal(state, 1)

        # Make a dg from a pmf
        N = unit_octahedron.theta.shape[0]
        pmf = np.zeros((3, 3, 3, N))
        dg = PTTDirectionGetter.from_pmf(pmf, 90, unit_octahedron)
        state = dg.get_direction(point, dir)
        npt.assert_equal(state, 1)

        # pmf shape must match sphere
        bad_pmf = pmf[..., 1:]
        npt.assert_raises(ValueError, PTTDirectionGetter.from_pmf,
                          bad_pmf, 90, unit_octahedron)

        # pmf must have 4 dimensions
        bad_pmf = pmf[0, ...]
        npt.assert_raises(ValueError, PTTDirectionGetter.from_pmf,
                          bad_pmf, 90, unit_octahedron)
        # pmf cannot have negative values
        pmf[0, 0, 0, 0] = -1
        npt.assert_raises(ValueError, PTTDirectionGetter.from_pmf,
                          pmf, 90, unit_octahedron)

        # Check basis_type keyword
        dg = PTTDirectionGetter.from_shcoeff(fit.shm_coeff, 90,
                                             unit_octahedron,
                                             basis_type="tournier07")

        npt.assert_raises(ValueError,
                          PTTDirectionGetter.from_shcoeff,
                          fit.shm_coeff, 90, unit_octahedron,
                          basis_type="not a basis")


# test_PTTDirectionGetter()