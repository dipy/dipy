from warnings import warn

warn("The gui_tools module is very new and not well tested, please use it "
     "with care and help us make it better")

# Import traits as optional package
from ..utils.optional_traits import tapi, tuapi, have_traits, setup_module
# Import names to top level; done here in case we don't have actual traits but
# only the traits shell from the optional_traits module
File = tapi.File
Item, Group, View, ArrayEditor = (tuapi.Item, tuapi.Group, tuapi.View,
                                  tuapi.ArrayEditor)

from ..tracking.interfaces import InputData

if have_traits:
    I = InputData()
    iview = I.trait_view()
    iview.resizable = True
    iview.width = 600
    I.trait_view('traits_view', iview)

main_view = View(Group(Group(
                             Item( 'dwi_images' ),
                             Item( 'all_inputs' ),
                             Item( 'min_signal' ),
                             Item( 'seed_roi' ),
                             Item( 'seed_density', editor=ArrayEditor() ),
                             show_border=True),
                       Group(
                             Item( 'smoothing_kernel_type' ),
                             Item( 'smoothing_kernel' ),
                             show_border=True),
                       Group(
                             Item( 'interpolator' ),
                             Item( 'model_type' ),
                             Item( 'sh_order' ),
                             Item( 'Lambda' ),
                             Item( 'sphere_coverage' ),
                             Item( 'min_peak_spacing' ),
                             Item( 'min_relative_peak' ),
                             show_border=True),
                       Group(
                             Item( 'probabilistic' ),
                             show_border=True),
                       Group(
                             #Item( 'integrator' ),
                             Item( 'start_direction', editor=ArrayEditor()),
                             Item( 'track_two_directions'),
                             Item( 'fa_threshold' ),
                             Item( 'max_turn_angle' ),
                             show_border=True),
                       Group(
                             Item( 'stop_on_target' ),
                             Item( 'targets' ),
                             show_border=True),
                       Group(
                             Item( 'save_streamlines_to' ),
                             Item( 'save_counts_to' ),
                             show_border=True),
                       orientation = 'vertical'),
                buttons=['OK', 'Cancel'], resizable=True, width=600)

def gui_track(interface=None):
    if interface is None:
        interface = EZTrackingInterface()
    if not interface.configure_traits(view=main_view):
        return
    if interface.save_streamlines_to == '' and interface.save_counts_to == '':
        raise IOError('must provide filename where to save results')
    streamlines = list(interface.track_shm())
    if interface.save_streamlines_to != '':
        interface.save_streamlines(streamlines, interface.save_streamlines_to)
    if interface.save_counts_to != '':
        interface.save_counts(streamlines, interface.save_counts_to)

