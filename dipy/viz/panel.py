import numpy as np
from dipy.utils.optpkg import optional_package
import itertools

fury, have_fury, setup_module = optional_package('fury')

if have_fury:
    from dipy.viz import actor, ui


def build_label(text, font_size=18, bold=False):
    """ Simple utility function to build labels

    Parameters
    ----------
    text : str
    font_size : int
    bold : bool

    Returns
    -------
    label : TextBlock2D
    """

    label = ui.TextBlock2D()
    label.message = text
    label.font_size = font_size
    label.font_family = 'Arial'
    label.justification = 'left'
    label.bold = bold
    label.italic = False
    label.shadow = False
    label.actor.GetTextProperty().SetBackgroundColor(0, 0, 0)
    label.actor.GetTextProperty().SetBackgroundOpacity(0.0)
    label.color = (1, 1, 1)

    return label


def slicer_panel(renderer, iren, data=None, affine=None, world_coords=False):
    """ Slicer panel with slicer included

    Parameters
    ----------
    renderer : Renderer
    iren : Interactor
    data : 3d ndarray
    affine : 4x4 ndarray
    world_coords : bool
        If True then the affine is applied.

    Returns
    -------
    panel : Panel

    """

    shape = data.shape
    ndim = data.ndim
    tmp = data
    if ndim == 4:
        if shape[-1] > 3:
            tmp = data[..., 0] 
            shape = shape[:3]

    if not world_coords:
        affine = np.eye(4)

    image_actor_z = actor.slicer(tmp, affine)

    slicer_opacity = 1.
    image_actor_z.opacity(slicer_opacity)

    image_actor_x = image_actor_z.copy()
    x_midpoint = int(np.round(shape[0] / 2))
    image_actor_x.display_extent(x_midpoint,
                                 x_midpoint, 0,
                                 shape[1] - 1,
                                 0,
                                 shape[2] - 1)

    image_actor_y = image_actor_z.copy()
    y_midpoint = int(np.round(shape[1] / 2))
    image_actor_y.display_extent(0,
                                 shape[0] - 1,
                                 y_midpoint,
                                 y_midpoint,
                                 0,
                                 shape[2] - 1)

    renderer.add(image_actor_z)
    renderer.add(image_actor_x)
    renderer.add(image_actor_y)

    line_slider_z = ui.LineSlider2D(min_value=0,
                                    max_value=shape[2] - 1,
                                    initial_value=shape[2] / 2,
                                    text_template="{value:.0f}",
                                    length=140)

    def _color_slider(slider):
        slider.default_color = (1, 0.5, 0)
        slider.track.color = (0.8, 0.3, 0)
        slider.active_color = (0.9, 0.4, 0)
        slider.handle.color = (1, 0.5, 0)

    _color_slider(line_slider_z)

    line_slider_x = ui.LineSlider2D(min_value=0,
                                    max_value=shape[0] - 1,
                                    initial_value=shape[0] / 2,
                                    text_template="{value:.0f}",
                                    length=140)

    _color_slider(line_slider_x)

    line_slider_y = ui.LineSlider2D(min_value=0,
                                    max_value=shape[1] - 1,
                                    initial_value=shape[1] / 2,
                                    text_template="{value:.0f}",
                                    length=140)

    _color_slider(line_slider_y)

    opacity_slider = ui.LineSlider2D(min_value=0.0,
                                     max_value=1.0,
                                     initial_value=slicer_opacity,
                                     length=140,
                                     text_template="{ratio:.0%}")

    _color_slider(opacity_slider)

    volume_slider = ui.LineSlider2D(min_value=0,
                                    max_value=data.shape[-1] - 1,
                                    initial_value=0,
                                    length=140,
                                    text_template="{value:.0f}", shape='square')

    _color_slider(volume_slider)

    # TODO ADD CheckBoxes to select between showing a slice or not.
    # Or make textboxes clickable.
    # Add double slider for selecting contrast range.

    def change_slice_x(slider):
        x = int(np.round(slider.value))
        change_volume.image_actor_x.display_extent(x, x, 0, shape[1] - 1, 0,
                                                   shape[2] - 1)
        change_slice_x.x = x

    def change_slice_y(slider):
        y = int(np.round(slider.value))
        change_volume.image_actor_y.display_extent(0, shape[0] - 1, y, y,
                                                   0, shape[2] - 1)
        change_slice_y.y = y

    def change_slice_z(slider):
        z = int(np.round(slider.value))
        change_volume.image_actor_z.display_extent(0, shape[0] - 1,
                                                   0, shape[1] - 1, z, z)
        change_slice_z.z = z

    def change_opacity(slider):
        slicer_opacity = slider.value
        change_volume.image_actor_z.opacity(slicer_opacity)
        change_volume.image_actor_x.opacity(slicer_opacity)
        change_volume.image_actor_y.opacity(slicer_opacity)

    def change_volume(istyle, obj, slider):
        vol_idx = int(np.round(slider.value))
        change_volume.vol_idx = vol_idx       
        renderer.rm(change_volume.image_actor_z)
        renderer.rm(change_volume.image_actor_x)
        renderer.rm(change_volume.image_actor_y)
        image_actor_z = actor.slicer(data[..., vol_idx],
                                     affine=affine)
        image_actor_z.display_extent(0, shape[0] - 1,
                                     0, shape[1] - 1,
                                     change_slice_z.z,
                                     change_slice_z.z)
    
        change_volume.image_actor_z = image_actor_z
        change_volume.image_actor_x = image_actor_z.copy()
        change_volume.image_actor_x.display_extent(change_slice_x.x,
                                                   change_slice_x.x, 0,
                                                   shape[1] - 1, 0,
                                                   shape[2] - 1)
        change_volume.image_actor_y = image_actor_z.copy()
        change_volume.image_actor_y.display_extent(0, shape[0] - 1,
                                                   change_slice_y.y,
                                                   change_slice_y.y,
                                                   0, shape[2] - 1)

        change_volume.image_actor_z.AddObserver('LeftButtonPressEvent',
                                                left_click_picker_callback,
                                                1.0)
        change_volume.image_actor_x.AddObserver('LeftButtonPressEvent',
                                                left_click_picker_callback,
                                                1.0)
        change_volume.image_actor_y.AddObserver('LeftButtonPressEvent',
                                                left_click_picker_callback,
                                                1.0)
        renderer.add(change_volume.image_actor_z)
        renderer.add(change_volume.image_actor_x)
        renderer.add(change_volume.image_actor_y)
        istyle.force_render()

    def left_click_picker_callback(obj, ev):
        ''' Get the value of the clicked voxel and show it in the panel.'''
       
        event_pos = iren.GetEventPosition()
        
        obj.picker.Pick(event_pos[0],
                        event_pos[1],
                        0,
                        renderer)

        i, j, k = obj.picker.GetPointIJK()
        if data.ndim == 4:
            message = '%.3f' % data[i, j, k, change_volume.vol_idx]
        if data.ndim == 3:
            message = '%.3f' % data[i, j, k]
        picker_label.message = '({}, {}, {})'.format(str(i), str(j), str(k)) + ' ' + message
        

    change_volume.vol_idx = 0
    change_volume.image_actor_x = image_actor_x
    change_volume.image_actor_y = image_actor_y
    change_volume.image_actor_z = image_actor_z

    change_volume.image_actor_x.AddObserver('LeftButtonPressEvent',
                                            left_click_picker_callback,
                                            1.0)
    change_volume.image_actor_y.AddObserver('LeftButtonPressEvent',
                                            left_click_picker_callback,
                                            1.0)
    change_volume.image_actor_z.AddObserver('LeftButtonPressEvent',
                                            left_click_picker_callback,
                                            1.0)


    change_slice_x.x = int(np.round(shape[0] / 2))
    change_slice_y.y = int(np.round(shape[1] / 2))
    change_slice_z.z = int(np.round(shape[2] / 2))
    
    line_slider_x.on_change = change_slice_x
    line_slider_y.on_change = change_slice_y
    line_slider_z.on_change = change_slice_z
    
    opacity_slider.on_change = change_opacity
    
    volume_slider.handle_events(volume_slider.handle.actor)
    volume_slider.on_left_mouse_button_released = change_volume

    # volume_slider.on_right_mouse_button_released = change_volume2

    line_slider_label_x = build_label(text="X Slice")
    line_slider_label_x.visibility = True
    x_counter = itertools.count()

    def label_callback_x(obj, event):
        line_slider_label_x.visibility = not line_slider_label_x.visibility
        line_slider_x.set_visibility(line_slider_label_x.visibility)
        cnt = next(x_counter)
        if line_slider_label_x.visibility and cnt > 0 :
            renderer.add(change_volume.image_actor_x)
        else:
            renderer.rm(change_volume.image_actor_x)
        iren.Render()
    
    line_slider_label_x.actor.AddObserver('LeftButtonPressEvent',
                                          label_callback_x,
                                          1.0)

    line_slider_label_y = build_label(text="Y Slice")
    line_slider_label_y.visibility = True
    y_counter = itertools.count()

    def label_callback_y(obj, event):
        line_slider_label_y.visibility = not line_slider_label_y.visibility
        line_slider_y.set_visibility(line_slider_label_y.visibility)
        cnt = next(y_counter)
        if line_slider_label_y.visibility and cnt > 0 :
            renderer.add(change_volume.image_actor_y)
        else:
            renderer.rm(change_volume.image_actor_y)
        iren.Render()
    
    line_slider_label_y.actor.AddObserver('LeftButtonPressEvent',
                                          label_callback_y,
                                          1.0)

    line_slider_label_z = build_label(text="Z Slice")
    line_slider_label_z.visibility = True
    z_counter = itertools.count()

    def label_callback_z(obj, event):
        line_slider_label_z.visibility = not line_slider_label_z.visibility
        line_slider_z.set_visibility(line_slider_label_z.visibility)
        cnt = next(z_counter)
        if line_slider_label_z.visibility and cnt > 0 :
            renderer.add(change_volume.image_actor_z)
        else:
            renderer.rm(change_volume.image_actor_z)
        iren.Render()
    
    line_slider_label_z.actor.AddObserver('LeftButtonPressEvent',
                                          label_callback_z,
                                          1.0)
    
    opacity_slider_label = build_label(text="Opacity")
    volume_slider_label = build_label(text="Volume")
    picker_label = build_label(text = '')
    
    if data.ndim == 4:
        panel_size = (400, 400)
    if data.ndim == 3:
        panel_size = (400, 300)
    
    panel = ui.Panel2D(size=panel_size,
                       position=(850, 110),
                       color=(1, 1, 1),
                       opacity=0.1,
                       align="right")

    ys = np.linspace(0, 1, 8)

    panel.add_element(line_slider_z, coords=(0.4, ys[1]))
    panel.add_element(line_slider_y, coords=(0.4, ys[2]))
    panel.add_element(line_slider_x, coords=(0.4, ys[3]))
    panel.add_element(opacity_slider, coords=(0.4, ys[4]))

    if data.ndim == 4:    
        panel.add_element(volume_slider, coords=(0.4, ys[6]))
    
    
    panel.add_element(line_slider_label_z, coords=(0.1, ys[1]))
    panel.add_element(line_slider_label_y, coords=(0.1, ys[2]))
    panel.add_element(line_slider_label_x, coords=(0.1, ys[3]))
    panel.add_element(opacity_slider_label, coords=(0.1, ys[4]))
    
    if data.ndim == 4:    
        panel.add_element(volume_slider_label, coords=(0.1, ys[6]))
    
    panel.add_element(picker_label, coords=(0.2, ys[5]))

    renderer.add(panel)
    return panel
