
# Basic imports
import time
import numpy as np
import nibabel as nib

# Import tracking stuff
from dipy.tracking.localtrack import ThresholdTissueClassifier, local_tracker
from dipy.tracking.local import LocalTracking
from dipy.reconst.dg import ProbabilisticOdfWightedDirectionGetter
from dipy.tracking import utils

# Import a few different models as examples
from dipy.reconst.shm import CsaOdfModel
from dipy.reconst.peaks import peaks_from_model, default_sphere
from dipy.reconst.csdeconv import ConstrainedSphericalDeconvModel, auto_response

# Import data stuff
from dipy.segment.mask import median_otsu
from dipy.data import fetch_stanford_hardi, read_stanford_hardi, get_sphere


def clipMask(mask):
    """This is a hack until we fix the behaviour of the tracking objects
    around the edge of the image"""
    out = mask.copy()
    index = [slice(None)] * out.ndim
    for i in range(len(index)):
        idx = index[:]
        idx[i] = [0, -1]
        out[idx] = 0.
    return out


# Target number of seeds
N = 5000

fetch_stanford_hardi()
img, gtab = read_stanford_hardi()
data = img.get_data()
affine = img.get_affine()

sphere = default_sphere

_, mask = median_otsu(data, 3, 1, False, vol_idx=range(10, 50))
mask = clipMask(mask)

# Make header for saving trk files
hdr = nib.trackvis.empty_header()
hdr['dim'] = mask.shape
hdr['voxel_size'] = img.get_header().get_zooms()[:3]
hdr['voxel_order'] = 'ras'

# make trackvis affine
trackvis_affine = utils.affine_for_trackvis(hdr['voxel_size'])


def prob_tracking_example(model, data, mask, N, hdr, filename):
    # Fit data to model
    fit = model.fit(data, mask)

    # Create objects to be passed to tracker
    powdg = ProbabilisticOdfWightedDirectionGetter(fit, default_sphere, 45.)
    gfa = fit.gfa
    gfa = np.where(np.isnan(gfa), 0., gfa)
    ttc = ThresholdTissueClassifier(gfa, .2)

    # Create around N seeds
    seeds = utils.seeds_from_mask(gfa > .25, 2, affine=affine)
    seeds = seeds[::len(seeds) // N + 1]

    # Create streamline generator
    streamlines = LocalTracking(powdg, ttc, seeds, affine, .5, max_cross=1)
    trk_streamlines = utils.move_streamlines(streamlines,
                                         input_space=affine,
                                         output_space=trackvis_affine)

    trk = ((streamline, None, None) for streamline in trk_streamlines)
    # Save streamlines
    nib.trackvis.write(filename, trk, hdr)


def detr_tracking_example(model, data, mask, N, hdr, filename):
    csapeaks = peaks_from_model(model=csamodel,
                                data=data,
                                sphere=sphere,
                                relative_peak_threshold=.5,
                                min_separation_angle=45,
                                mask=mask,
                                return_odf=False,
                                normalize_peaks=True)
    gfa = csapeaks.gfa
    gfa = np.where(np.isnan(gfa), 0., gfa)
    ttc = ThresholdTissueClassifier(gfa, .2)

    # Create around N seeds
    seeds = utils.seeds_from_mask(gfa > .25, 2, affine=affine)
    seeds = seeds[::len(seeds) // N + 1]

    # Create streamline generator
    streamlines = LocalTracking(csapeaks, ttc, seeds, affine, .5, max_cross=1)
    trk_streamlines = utils.move_streamlines(streamlines,
                                             input_space=affine,
                                             output_space=trackvis_affine)
    trk = ((streamline, None, None) for streamline in trk_streamlines)

    # Save streamlines
    nib.trackvis.write(filename, trk, hdr)


# Constant Solid Angle Probableistic
csamodel = CsaOdfModel(gtab, 8)
start = time.time()
prob_tracking_example(csamodel, data, mask, N, hdr, "SolidAngle.trk")
print time.time() - start

# Deterministic tracking (eudx like)
start = time.time()
detr_tracking_example(csamodel, data, mask, N, hdr, "SolidAngle_Detr.trk")
print time.time() - start

# Constrained Spherical Deconv Probableistic
r, _ = auto_response(gtab, data)
csdmodel = ConstrainedSphericalDeconvModel(gtab, r, sh_order=10)
start = time.time()
prob_tracking_example(csdmodel, data, mask, N, hdr, "SphereDeconv.trk")
print time.time() - start


