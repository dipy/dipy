from dipy.utils.optpkg import optional_package

fury, has_fury, _ = optional_package('fury')
if has_fury:
    from dipy.viz import window, actor


def show_bundles(bundles, interactive=True, view='sagital', colors=None,
                 linewidth=0.3, save_as=None):
    """ Render bundles to visualize them interactively or save them into a png.

    The function allows to just render the bundles in an interactive plot or
    to export them into a png file.

    Parameters
    ----------
    bundles : list
        Bundles to be rendered.
    interactive : boolean, optional
        If True a 3D interactive rendering is created. Default is True.
    view : str, optional
        Viewing angle. Supported options: 'sagital','axial' and 'coronal'.
        Default is 'sagital'.
    colors : list, optional
       Colors to be used for each bundle. If None default colors are used.
    linewidth : float, optional
        Width of each rendered streamline. Default is 0.3.
    save_as : str, optional
        If not None rendered scene is stored in a png file with that name.
        Default is None.

    """

    scene = window.Scene()
    scene.SetBackground(1., 1, 1)

    for i, bundle in enumerate(bundles):
        if colors is None:
            lines_actor = actor.streamtube(bundle, linewidth=linewidth)
        else:
            lines_actor = actor.streamtube(bundle, linewidth=linewidth,
                                           colors=colors[i])
        if view == 'sagital':
            lines_actor.RotateX(-90)
            lines_actor.RotateZ(90)
        elif view == 'axial':
            pass
        elif view == 'coronal':
            lines_actor.RotateX(-90)
        else:
            raise ValueError("Invalid view argument value. Use 'sagital'," +
                             "'axial' or 'coronal'.")

        scene.add(lines_actor)

    if interactive:
        window.show(scene)

    if save_as is not None:
        window.record(scene, n_frames=1, out_path=save_as, size=(900, 900))
