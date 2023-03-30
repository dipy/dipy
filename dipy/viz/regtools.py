import numpy as np
from dipy.utils.optpkg import optional_package
matplotlib, has_mpl, setup_module = optional_package("matplotlib")
plt, _, _ = optional_package("matplotlib.pyplot")


def _tile_plot(imgs, titles, **kwargs):
    """
    Helper function
    """
    # Create a new figure and plot the three images
    fig, ax = plt.subplots(1, len(imgs))
    for ii, a in enumerate(ax):
        a.set_axis_off()
        a.imshow(imgs[ii], **kwargs)
        a.set_title(titles[ii])

    return fig


def simple_plot(file_name, title, x, y, xlabel, ylabel):
    """ Saves the simple plot with given x and y values

    Parameters
    ----------
    file_name : string
        file name for saving the plot
    title : string
        title of the plot
    x : integer list
        x-axis values to be plotted
    y : integer list
        y-axis values to be plotted
    xlabel : string
        label for x-axis
    ylable : string
        label for y-axis

    """

    plt.plot(x, y)
    axes = plt.gca()
    axes.set_ylim([0, 4])
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)

    plt.savefig(file_name)
    plt.clf()


def overlay_images(img0, img1, title0='', title_mid='', title1='', fname=None, **fig_kwargs):
    r""" Plot two images one on top of the other using red and green channels.

    Creates a figure containing three images: the first image to the left
    plotted on the red channel of a color image, the second to the right
    plotted on the green channel of a color image and the two given images on
    top of each other using the red channel for the first image and the green
    channel for the second one. It is assumed that both images have the same
    shape. The intended use of this function is to visually assess the quality
    of a registration result.

    Parameters
    ----------
    img0 : array, shape(R, C)
        the image to be plotted on the red channel, to the left of the figure
    img1 : array, shape(R, C)
        the image to be plotted on the green channel, to the right of the
        figure
    title0 : string (optional)
        the title to be written on top of the image to the left. By default, no
        title is displayed.
    title_mid : string (optional)
        the title to be written on top of the middle image. By default, no
        title is displayed.
    title1 : string (optional)
        the title to be written on top of the image to the right. By default,
        no title is displayed.
    fname : string (optional)
        the file name to write the resulting figure. If None (default), the
        image is not saved.
    fig_kwargs: extra parameters for saving figure, e.g. `dpi=300`.
    """
    # Normalize the input images to [0,255]
    img0 = 255 * ((img0 - img0.min()) / (img0.max() - img0.min()))
    img1 = 255 * ((img1 - img1.min()) / (img1.max() - img1.min()))

    # Create the color images
    img0_red = np.zeros(shape=img0.shape + (3,), dtype=np.uint8)
    img1_green = np.zeros(shape=img0.shape + (3,), dtype=np.uint8)
    overlay = np.zeros(shape=img0.shape + (3,), dtype=np.uint8)

    # Copy the normalized intensities into the appropriate channels of the
    # color images
    img0_red[..., 0] = img0
    img1_green[..., 1] = img1
    overlay[..., 0] = img0
    overlay[..., 1] = img1

    fig = _tile_plot([img0_red, overlay, img1_green],
                     [title0, title_mid, title1])

    # If a file name was given, save the figure
    if fname is not None:
        fig.savefig(fname, bbox_inches='tight', **fig_kwargs)

    return fig


def draw_lattice_2d(nrows, ncols, delta):
    r"""Create a regular lattice of nrows x ncols squares.

    Creates an image (2D array) of a regular lattice of nrows x ncols squares.
    The size of each square is delta x delta pixels (not counting the
    separation lines). The lines are one pixel width.

    Parameters
    ----------
    nrows : int
        the number of squares to be drawn vertically
    ncols : int
        the number of squares to be drawn horizontally
    delta : int
        the size of each square of the grid. Each square is delta x delta
        pixels

    Returns
    -------
    lattice : array, shape (R, C)
        the image (2D array) of the segular lattice. The shape (R, C) of the
        array is given by
        R = 1 + (delta + 1) * nrows
        C = 1 + (delta + 1) * ncols
    """
    lattice = np.ndarray((1 + (delta + 1) * nrows,
                          1 + (delta + 1) * ncols),
                         dtype=np.float64)

    # Fill the lattice with "white"
    lattice[...] = 127

    # Draw the horizontal lines in "black"
    for i in range(nrows + 1):
        lattice[i * (delta + 1), :] = 0

    # Draw the vertical lines in "black"
    for j in range(ncols + 1):
        lattice[:, j * (delta + 1)] = 0

    return lattice


def plot_2d_diffeomorphic_map(mapping, delta=10, fname=None,
                              direct_grid_shape=None, direct_grid2world=-1,
                              inverse_grid_shape=None, inverse_grid2world=-1,
                              show_figure=True, **fig_kwargs):
    r"""Draw the effect of warping a regular lattice by a diffeomorphic map.

    Draws a diffeomorphic map by showing the effect of the deformation on a
    regular grid. The resulting figure contains two images: the direct
    transformation is plotted to the left, and the inverse transformation is
    plotted to the right.

    Parameters
    ----------
    mapping : DiffeomorphicMap object
        the diffeomorphic map to be drawn
    delta : int, optional
        the size (in pixels) of the squares of the regular lattice to be used
        to plot the warping effects. Each square will be delta x delta pixels.
        By default, the size will be 10 pixels.
    fname : string, optional
        the name of the file the figure will be written to. If None (default),
        the figure will not be saved to disk.
    direct_grid_shape : tuple, shape (2,), optional
        the shape of the grid image after being deformed by the direct
        transformation. By default, the shape of the deformed grid is the
        same as the grid of the displacement field, which is by default
        equal to the shape of the fixed image. In other words, the resulting
        deformed grid (deformed by the direct transformation) will normally
        have the same shape as the fixed image.
    direct_grid2world : array, shape (3, 3), optional
        the affine transformation mapping the direct grid's coordinates to
        physical space. By default, this transformation will correspond to
        the image-to-world transformation corresponding to the default
        direct_grid_shape (in general, if users specify a direct_grid_shape,
        they should also specify direct_grid2world).
    inverse_grid_shape : tuple, shape (2,), optional
        the shape of the grid image after being deformed by the inverse
        transformation. By default, the shape of the deformed grid under the
        inverse transform is the same as the image used as "moving" when
        the diffeomorphic map was generated by a registration algorithm
        (so it corresponds to the effect of warping the static image towards
        the moving).
    inverse_grid2world : array, shape (3, 3), optional
        the affine transformation mapping inverse grid's coordinates to
        physical space. By default, this transformation will correspond to
        the image-to-world transformation corresponding to the default
        inverse_grid_shape (in general, if users specify an inverse_grid_shape,
        they should also specify inverse_grid2world).
    show_figure : bool, optional
        if True (default), the deformed grids will be plotted using matplotlib,
        else the grids are just returned
    fig_kwargs: extra parameters for saving figure, e.g. `dpi=300`.

    Returns
    -------
    warped_forward : array
        Image with the grid showing the effect of transforming the moving image to
        the static image.  The shape will be `direct_grid_shape` if specified,
        otherwise the shape of the static image.
    warped_backward : array
        Image with the grid showing the effect of transforming the static image to
        the moving image.  Shape will be `inverse_grid_shape` if specified,
        otherwise the shape of the moving image.

    Notes
    -----
    The default value for the affine transformation is "-1" to handle the case
    in which the user provides "None" as input meaning "identity". If we used
    None as default, we wouldn't know if the user specifically wants to use
    the identity (specifically passing None) or if it was left unspecified,
    meaning to use the appropriate default matrix.
    """
    if mapping.is_inverse:
        # By default, direct_grid_shape is the codomain grid
        if direct_grid_shape is None:
            direct_grid_shape = mapping.codomain_shape
        if direct_grid2world == -1:
            direct_grid2world = mapping.codomain_grid2world

        # By default, the inverse grid is the domain grid
        if inverse_grid_shape is None:
            inverse_grid_shape = mapping.domain_shape
        if inverse_grid2world == -1:
            inverse_grid2world = mapping.domain_grid2world
    else:
        # Now by default, direct_grid_shape is the mapping's input grid
        if direct_grid_shape is None:
            direct_grid_shape = mapping.domain_shape
        if direct_grid2world == -1:
            direct_grid2world = mapping.domain_grid2world

        # By default, the output grid is the mapping's domain grid
        if inverse_grid_shape is None:
            inverse_grid_shape = mapping.codomain_shape
        if inverse_grid2world == -1:
            inverse_grid2world = mapping.codomain_grid2world

    # The world-to-image (image = drawn lattice on the output grid)
    # transformation is the inverse of the output affine
    world_to_image = None
    if inverse_grid2world is not None:
        world_to_image = np.linalg.inv(inverse_grid2world)

    # Draw the squares on the output grid
    lattice_out = draw_lattice_2d(
        (inverse_grid_shape[0] + delta) // (delta + 1),
        (inverse_grid_shape[1] + delta) // (delta + 1),
        delta)
    lattice_out = lattice_out[0:inverse_grid_shape[0], 0:inverse_grid_shape[1]]

    # Warp in the forward direction (sampling it on the input grid)
    warped_forward = mapping.transform(lattice_out, 'linear', world_to_image,
                                       direct_grid_shape, direct_grid2world)

    # Now, the world-to-image (image = drawn lattice on the input grid)
    # transformation is the inverse of the input affine
    world_to_image = None
    if direct_grid2world is not None:
        world_to_image = np.linalg.inv(direct_grid2world)

    # Draw the squares on the input grid
    lattice_in = draw_lattice_2d((direct_grid_shape[0] + delta) // (delta + 1),
                                 (direct_grid_shape[1] + delta) // (delta + 1),
                                 delta)
    lattice_in = lattice_in[0:direct_grid_shape[0], 0:direct_grid_shape[1]]

    # Warp in the backward direction (sampling it on the output grid)
    warped_backward = mapping.transform_inverse(
        lattice_in, 'linear', world_to_image, inverse_grid_shape,
        inverse_grid2world)

    # Now plot the grids
    if show_figure:
        plt.figure()

        plt.subplot(1, 2, 1).set_axis_off()
        plt.imshow(warped_forward, cmap=plt.cm.gray)
        plt.title('Direct transform')

        plt.subplot(1, 2, 2).set_axis_off()
        plt.imshow(warped_backward, cmap=plt.cm.gray)
        plt.title('Inverse transform')

        # Finally, save the figure to disk
        if fname is not None:
            plt.savefig(fname, bbox_inches='tight', **fig_kwargs)

    # Return the deformed grids
    return warped_forward, warped_backward


def plot_slices(V, slice_indices=None, fname=None, **fig_kwargs):
    r"""Plot 3 slices from the given volume: 1 sagittal, 1 coronal and 1 axial

    Creates a figure showing the axial, coronal and sagittal slices at the
    requested positions of the given volume. The requested slices are specified
    by slice_indices.

    Parameters
    ----------
    V : array, shape (S, R, C)
        the 3D volume to extract the slices from
    slice_indices : array, shape (3,) (optional)
        the indices of the sagittal (slice_indices[0]), coronal
        (slice_indices[1])
        and axial (slice_indices[2]) slices to be displayed. If None, the
        middle slices along each direction are displayed.
    fname : string (optional)
        the name of the file to save the figure to. If None (default), the
        figure is not saved to disk.
    fig_kwargs: extra parameters for saving figure, e.g. `dpi=300`.
    """
    if slice_indices is None:
        slice_indices = np.array(V.shape) // 2

    # Normalize the intensities to [0, 255]
    V = np.asarray(V, dtype=np.float64)
    V = 255 * (V - V.min()) / (V.max() - V.min())

    # Extract the middle slices
    axial = np.asarray(V[:, :, slice_indices[2]]).astype(np.uint8).T
    coronal = np.asarray(V[:, slice_indices[1], :]).astype(np.uint8).T
    sagittal = np.asarray(V[slice_indices[0], :, :]).astype(np.uint8).T

    fig = _tile_plot([axial, coronal, sagittal],
                     ['Axial', 'Coronal', 'Sagittal'],
                     cmap=plt.cm.gray, origin='lower')

    # Save the figure if requested
    if fname is not None:
        fig.savefig(fname, bbox_inches='tight', **fig_kwargs)

    return fig


def overlay_slices(L, R, slice_index=None, slice_type=1, ltitle='Left',
                   rtitle='Right', fname=None, **fig_kwargs):
    r"""Plot three overlaid slices from the given volumes.

    Creates a figure containing three images: the gray scale k-th slice of
    the first volume (L) to the left, where k=slice_index, the k-th slice of
    the second volume (R) to the right and the k-th slices of the two given
    images on top of each other using the red channel for the first volume and
    the green channel for the second one. It is assumed that both volumes have
    the same shape. The intended use of this function is to visually assess the
    quality of a registration result.

    Parameters
    ----------
    L : array, shape (S, R, C)
        the first volume to extract the slice from plotted to the left
    R : array, shape (S, R, C)
        the second volume to extract the slice from, plotted to the right
    slice_index : int (optional)
        the index of the slices (along the axis given by slice_type) to be
        overlaid. If None, the slice along the specified axis is used
    slice_type : int (optional)
        the type of slice to be extracted:
        0=sagittal, 1=coronal (default), 2=axial.
    ltitle : string (optional)
        the string to be written as the title of the left image. By default,
        no title is displayed.
    rtitle : string (optional)
        the string to be written as the title of the right image. By default,
        no title is displayed.
    fname : string (optional)
        the name of the file to write the image to. If None (default), the
        figure is not saved to disk.
    fig_kwargs: extra parameters for saving figure, e.g. `dpi=300`.
    """

    # Normalize the intensities to [0,255]
    sh = L.shape
    L = np.asarray(L, dtype=np.float64)
    R = np.asarray(R, dtype=np.float64)
    L = 255 * (L - L.min()) / (L.max() - L.min())
    R = 255 * (R - R.min()) / (R.max() - R.min())

    # Create the color image to draw the overlapped slices into, and extract
    # the slices (note the transpositions)
    if slice_type == 0:
        if slice_index is None:
            slice_index = sh[0] // 2
        colorImage = np.zeros(shape=(sh[2], sh[1], 3), dtype=np.uint8)
        ll = np.asarray(L[slice_index, :, :]).astype(np.uint8).T
        rr = np.asarray(R[slice_index, :, :]).astype(np.uint8).T
    elif slice_type == 1:
        if slice_index is None:
            slice_index = sh[1] // 2
        colorImage = np.zeros(shape=(sh[2], sh[0], 3), dtype=np.uint8)
        ll = np.asarray(L[:, slice_index, :]).astype(np.uint8).T
        rr = np.asarray(R[:, slice_index, :]).astype(np.uint8).T
    elif slice_type == 2:
        if slice_index is None:
            slice_index = sh[2] // 2
        colorImage = np.zeros(shape=(sh[1], sh[0], 3), dtype=np.uint8)
        ll = np.asarray(L[:, :, slice_index]).astype(np.uint8).T
        rr = np.asarray(R[:, :, slice_index]).astype(np.uint8).T
    else:
        print("Slice type must be 0, 1 or 2.")
        return

    # Draw the intensity images to the appropriate channels of the color image
    # The "(ll > ll[0, 0])" condition is just an attempt to eliminate the
    # background when its intensity is not exactly zero (the [0,0] corner is
    # usually background)
    colorImage[..., 0] = ll * (ll > ll[0, 0])
    colorImage[..., 1] = rr * (rr > rr[0, 0])

    fig = _tile_plot([ll, colorImage, rr],
                     [ltitle, 'Overlay', rtitle],
                     cmap=plt.cm.gray, origin='lower')

    # Save the figure to disk, if requested
    if fname is not None:
        fig.savefig(fname, bbox_inches='tight', **fig_kwargs)

    return fig
