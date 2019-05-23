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
        image_actor_z.display_extent(0, shape[0] - 1,
                                     0, shape[1] - 1, z, z)

    def change_slice_x(slider):
        x = int(np.round(slider.value))
        image_actor_x.display_extent(x, x, 0, shape[1] - 1, 0,
                                     shape[2] - 1)

    def change_slice_y(slider):
        y = int(np.round(slider.value))
        image_actor_y.display_extent(0, shape[0] - 1, y, y,
                                     0, shape[2] - 1)

    def change_opacity(slider):
        slicer_opacity = slider.value
        image_actor_z.opacity(slicer_opacity)
        image_actor_x.opacity(slicer_opacity)
        image_actor_y.opacity(slicer_opacity)

    def change_volume(iren, obj, slider):
        vol_idx = int(np.round(slider.value))        
        print(data.shape)
        print(vol_idx)
        
        renderer.rm(change_volume.image_actor_z)
        change_volume.image_actor_z = actor.slicer(data[..., vol_idx], affine=affine)
        renderer.add(change_volume.image_actor_z)
        iren.force_render()


    change_volume.image_actor_z = image_actor_z
    line_slider_z.on_change = change_slice_z
    line_slider_y.on_change = change_slice_y
    line_slider_x.on_change = change_slice_x
    opacity_slider.on_change = change_opacity
    # volume_slider.on_change = change_volume
    #volume_slider.add_callback(volume_slider.handle.actor, 'LeftButtonReleaseEvent', change_volume2)
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
