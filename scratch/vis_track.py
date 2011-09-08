from eztrack import EZTrackingInterface
from traitsui.api import Item, Group, View, ArrayEditor
from traits.api import File, on_trait_change
import nibabel as nib
from nibabel.trackvis import write, empty_header
import numpy as np
from os import path
from dipy.io.bvectxt import read_bvec_file, orientation_to_string, \
                            reorient_bvec
from dipy.tracking.utils import streamline_counts
import pickle

class VizTrackingInterface(EZTrackingInterface):

    trait_view = View(Group(Group(
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
                        buttons=['OK', 'Cancel'], resizable=True)

    def gui_track(self):
        if not self.configure_traits():
            return
        if self.save_streamlines_to == '' and self.save_counts_to == '':
            raiseIOError('must provide filename where to save results')
        streamlines = list(self.track_shm())
        self.save_streamlines(streamlines)
        self.save_counts(streamlines)
        return streamlines

if __name__ == "__main__":

    b = VizTrackingInterface()
    b.gui_track()

