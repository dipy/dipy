import numpy as np
from dipy.utils.optpkg import optional_package

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


def slicer_panel(renderer, data=None, affine=None, world_coords=False):
    """ Slicer panel with slicer included

    Parameters
    ----------
    renderer : Renderer
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

    line_slider_x = ui.LineSlider2D(min_value=0,
                                    max_value=shape[0] - 1,
                                    initial_value=shape[0] / 2,
                                    text_template="{value:.0f}",
                                    length=140)

    line_slider_y = ui.LineSlider2D(min_value=0,
                                    max_value=shape[1] - 1,
                                    initial_value=shape[1] / 2,
                                    text_template="{value:.0f}",
                                    length=140)

    opacity_slider = ui.LineSlider2D(min_value=0.0,
                                     max_value=1.0,
                                     initial_value=slicer_opacity,
                                     length=140)

    volume_slider = ui.LineSlider2D(min_value=0,
                                    max_value=data.shape[-1] - 1,
                                    initial_value=0,
                                    length=140,
                                    text_template="{value:.0f}", shape='square')

    def change_slice_z(slider):
        z = int(np.round(slider.value))
        change_volume.image_actor_z.display_extent(0, shape[0] - 1,
                                                   0, shape[1] - 1, z, z)
        change_slice_z.z = z

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

    def change_opacity(slider):
        slicer_opacity = slider.value
        change_volume.image_actor_z.opacity(slicer_opacity)
        change_volume.image_actor_x.opacity(slicer_opacity)
        change_volume.image_actor_y.opacity(slicer_opacity)

    def change_volume(istyle, obj, slider):
        vol_idx = int(np.round(slider.value))        
       
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
        change_volume.istyle = istyle
        renderer.add(change_volume.image_actor_z)
        renderer.add(change_volume.image_actor_x)
        renderer.add(change_volume.image_actor_y)
        istyle.force_render()

    def left_click_picker_callback(obj, ev):
        ''' Get the value of the clicked voxel and show it in the panel.'''
        
        print(obj)
        event_pos = change_volume.istyle.GetInteractor().GetEventPosition()
        
        obj.picker.Pick(event_pos[0],
                        event_pos[1],
                        0,
                        renderer)

        i, j, k = obj.picker.GetPointIJK()
        # result_position.message = '({}, {}, {})'.format(str(i), str(j), str(k))
        # result_value.message = '%.8f' % data[i, j, k]
        message = '({}, {}, {})'.format(str(i), str(j), str(k))
        print(message)
        message = '%.8f' % data[i, j, k, 0]
        print(message)
        

    change_volume.image_actor_z = image_actor_z

    change_volume.image_actor_z.AddObserver('LeftButtonPressEvent',
                                            left_click_picker_callback,
                                            1.0)

    change_volume.image_actor_x = image_actor_x
    change_volume.image_actor_y = image_actor_y
    change_slice_z.z = int(np.round(shape[2] / 2))
    change_slice_x.x = int(np.round(shape[0] / 2))
    change_slice_y.y = int(np.round(shape[1] / 2))

    line_slider_z.on_change = change_slice_z
    line_slider_y.on_change = change_slice_y
    line_slider_x.on_change = change_slice_x
    opacity_slider.on_change = change_opacity
    
    volume_slider.handle_events(volume_slider.handle.actor)
    volume_slider.on_left_mouse_button_released = change_volume

    # volume_slider.on_right_mouse_button_released = change_volume2

    line_slider_label_z = build_label(text="Z Slice")
    line_slider_label_x = build_label(text="X Slice")
    line_slider_label_y = build_label(text="Y Slice")
    opacity_slider_label = build_label(text="Opacity")
    volume_slider_label = build_label(text="Volume")
    
    panel = ui.Panel2D(size=(400, 300),
                       position=(850, 110),
                       color=(1, 1, 1),
                       opacity=0.1,
                       align="right")


    panel.add_element(line_slider_x, coords=(0.4, 0.8))
    panel.add_element(line_slider_y, coords=(0.4, 0.6))
    panel.add_element(line_slider_z, coords=(0.4, 0.4))
    panel.add_element(opacity_slider, coords=(0.4, 0.2))
    panel.add_element(volume_slider, coords=(0.4, 0.0))
    
    
    panel.add_element(line_slider_label_x, coords=(0.1, 0.75))
    panel.add_element(line_slider_label_y, coords=(0.1, 0.55))
    panel.add_element(line_slider_label_z, coords=(0.1, 0.35))
    panel.add_element(opacity_slider_label, coords=(0.1, 0.15))
    panel.add_element(volume_slider_label, coords=(0.1, 0.0))
    

    renderer.add(panel)
    return panel
