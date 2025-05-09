import logging
from pathlib import Path
import warnings

import numpy as np

from dipy.testing.decorators import warning_for_keywords
from dipy.utils.optpkg import optional_package

fury, has_fury, setup_module = optional_package("fury", min_version="0.10.0")

if has_fury:
    from fury import actor
    from fury.utils import apply_affine


class SlicesVisualizer:
    @warning_for_keywords()
    def __init__(
        self,
        interactor,
        scene,
        data,
        *,
        affine=None,
        world_coords=False,
        percentiles=(0, 100),
        rgb=False,
        fname=None,
    ):
        self._interactor = interactor
        self._scene = scene
        self._data = data
        self._affine = affine

        if not world_coords:
            self._affine = np.eye(4)

        self._slice_actors = [None] * 3

        if len(self._data.shape) == 4 and self._data.shape[-1] == 1:
            self._data = self._data[:, :, :, 0]

        self._data_ndim = self._data.ndim
        self._data_shape = self._data.shape
        self._rgb = False
        self._percentiles = percentiles

        vol_data = self._data

        if (
            self._data_ndim == 4 and rgb and self._data_shape[-1] == 3
        ) or self._data_ndim == 3:
            self._rgb = True and not self._data_ndim == 3
            self._int_range = np.percentile(vol_data, self._percentiles)
            _evaluate_intensities_range(self._int_range)
        else:
            if self._data_ndim == 4 and rgb and self._data_shape[-1] != 3:
                warnings.warn(
                    "The rgb flag is enabled but the color "
                    + "channel information is not provided",
                    stacklevel=2,
                )
            vol_data = self._volume_calculations(self._percentiles)

        self._vol_max = np.max(vol_data)
        self._vol_min = np.min(vol_data)

        self._resliced_vol = None
        if world_coords:
            np.set_printoptions(3, suppress=True)
            fname = "" if fname is None else Path(fname).name
            logging.info(f"-------------------{len(fname) * '-'}")
            logging.info(f"Applying affine to {fname}")
            logging.info(f"-------------------{len(fname) * '-'}")
            logging.info(f"Affine Native to RAS matrix \n{affine}")
            logging.info(f"Original shape: {self._data_shape}")
            self._create_and_resize_actors(vol_data, self._int_range)
            logging.info(f"Resized to RAS shape: {self._data_shape} \n")
            np.set_printoptions()
        else:
            logging.info(f"------------{len(fname) * '-'}")
            logging.info(f"Visualizing {fname}")
            logging.info(f"------------{len(fname) * '-'}")
            logging.info(f"Original shape: {self._data_shape}")
            self._create_and_resize_actors(vol_data, self._int_range)

        self._sel_slices = np.rint(np.asarray(self._data_shape[:3]) / 2).astype(int)

        self._add_slice_actors_to_scene(self._sel_slices)

        self._picker_callback = None
        self._picked_voxel_actor = None

    def _volume_calculations(self, percentiles):
        for i in range(self._data.shape[-1]):
            vol_data = self._data[..., i]
            self._int_range = np.percentile(vol_data, percentiles)
            if np.sum(np.diff(self._int_range)) != 0:
                break
            else:
                if i < self._data_shape[-1] - 1:
                    warnings.warn(
                        f"Volume N°{i} does not have any contrast. "
                        "Please, check the value ranges of your data. "
                        "Moving to the next volume.",
                        stacklevel=2,
                    )
                else:
                    _evaluate_intensities_range(self._int_range)
        return vol_data

    def _add_slice_actors_to_scene(self, visible_slices):
        self._slice_actors[0].display_extent(
            visible_slices[0],
            visible_slices[0],
            0,
            self._data_shape[1] - 1,
            0,
            self._data_shape[2] - 1,
        )

        self._slice_actors[1].display_extent(
            0,
            self._data_shape[0] - 1,
            visible_slices[1],
            visible_slices[1],
            0,
            self._data_shape[2] - 1,
        )

        self._slice_actors[2].display_extent(
            0,
            self._data_shape[0] - 1,
            0,
            self._data_shape[1] - 1,
            visible_slices[2],
            visible_slices[2],
        )

        for act in self._slice_actors:
            self._scene.add(act)

    def _create_and_resize_actors(self, vol_data, value_range):
        self._slice_actors[0] = actor.slicer(
            vol_data,
            affine=self._affine,
            value_range=value_range,
            interpolation="nearest",
        )

        self._resliced_vol = self._slice_actors[0].resliced_array()

        self._slice_actors[1] = self._slice_actors[0].copy()
        self._slice_actors[2] = self._slice_actors[0].copy()

        for slice_actor in self._slice_actors:
            slice_actor.AddObserver(
                "LeftButtonPressEvent", self._left_click_picker_callback, 1.0
            )

        if self._data_ndim == 4 and not self._rgb:
            self._data_shape = self._resliced_vol.shape + (self._data.shape[-1],)
        else:
            self._data_shape = self._resliced_vol.shape

    def _left_click_picker_callback(self, obj, event):
        # TODO: Find out why this is not triggered when opacity < 1
        event_pos = self._interactor.GetEventPosition()

        obj.picker.Pick(event_pos[0], event_pos[1], 0, self._scene)

        i, j, k = obj.picker.GetPointIJK()
        res = self._resliced_vol[i, j, k]

        try:
            message = f"{res:.2f}"
        except TypeError:
            message = f"{res[0]:.2f} {res[1]:.2f} {res[2]:.2f}"
        message = f"({i}, {j}, {k}) = {message}"
        self._picker_callback(message)
        # TODO: Fix this
        # self._replace_picked_voxel_actor(i, j, k)

    def _replace_picked_voxel_actor(self, x, y, z):
        if self._picked_voxel_actor:
            self._scene.rm(self._picked_voxel_actor)
        pnt = np.asarray([[x, y, z]])
        pnt = apply_affine(self._affine, pnt)
        self._picked_voxel_actor = actor.dot(pnt, colors=(0.9, 0.4, 0.0), dot_size=10)
        self._scene.add(self._picked_voxel_actor)

    def _adaptive_percentile(self, vol_data, intensity_ratios, idx):
        value_range = np.percentile(np.ravel(vol_data), intensity_ratios * 100)
        default_range = False

        if np.sum(np.diff(value_range)) == 0:
            warnings.warn(
                f"The selected intensity range have no contrast for Volume N°{idx}."
                "The selection of intensities will be ignored and changed to default.",
                stacklevel=2,
            )
            value_range = np.asarray((np.min(vol_data), np.max(vol_data)))
            default_range = True

        return (value_range, default_range)

    def change_volume(self, next_idx, intensity_ratios, visible_slices):
        vol_data = self._data[..., next_idx]
        value_range, default_range = self._adaptive_percentile(
            vol_data, intensity_ratios, next_idx
        )
        if np.sum(np.diff(value_range)) == 0:
            return False, default_range

        self._int_range = value_range
        self._vol_max = np.max(vol_data)
        self._vol_min = np.min(vol_data)

        for slice_actor in self._slice_actors:
            self._scene.rm(slice_actor)

        self._create_and_resize_actors(vol_data, self._int_range)

        self._add_slice_actors_to_scene(visible_slices)

        return True, default_range

    def register_picker_callback(self, callback):
        self._picker_callback = callback

    @property
    def data_shape(self):
        return self._data_shape

    @property
    def intensities_range(self):
        return self._int_range

    @property
    def selected_slices(self):
        return self._sel_slices

    @property
    def slice_actors(self):
        return self._slice_actors

    @property
    def volume_max(self):
        return self._vol_max

    @property
    def volume_min(self):
        return self._vol_min

    @property
    def rgb(self):
        return self._rgb


def _evaluate_intensities_range(intensities_range):
    if np.sum(np.diff(intensities_range)) == 0:
        raise ValueError(
            "Your data does not have any contrast. Please, check the "
            "value range of your data."
        )
