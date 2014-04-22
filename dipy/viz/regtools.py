import numpy as np
import matplotlib.pyplot as plt


def overlay_images(img0, img1, title0='', title_mid='', title1='', fname=None):
    r"""
    Creates a figure containing three images: the first image to the left plotted
    on the red channel of a color image, the second to the right plotted on the 
    green channel of a color image and the two given images on top of each other
    using the red channel for the first image and the green channel for the 
    second one. It is assumed that both images have the same shape. The intended
    use of this function is to visually asses the quality of a registration
    result.

    Parameters
    ----------
    img0 : array, shape(R, C)
        the image to be plotted on the red channel
    img0 : array, shape(R, C)
        the image to be plotted on the green channel
    title0 : string
        the title to be written on top of the first gray-level image
    title_mid : string
        the title to be written on top of the middle, color image
    title1 : string
        the title to be written on top of the second gray-level image
    fname : string
        the file name to write the resulting figure. If None, the image is not
        written
    """
    #Normalize the input images to [0,255]
    img0 = 255*((img0 - img0.min()) / (img0.max() - img0.min()))
    img1 = 255*((img1 - img1.min()) / (img1.max() - img1.min()))

    #Create the color images
    img0_red=np.zeros(shape=(img0.shape) + (3,), dtype=np.uint8)
    img1_green=np.zeros(shape=(img0.shape) + (3,), dtype=np.uint8)
    overlay=np.zeros(shape=(img0.shape) + (3,), dtype=np.uint8)

    #Copy the normalized intensities into the appropriate channels of the
    #color images
    img0_red[..., 0] = img0
    img1_green[..., 1] = img1
    overlay[..., 0]=img0
    overlay[..., 1]=img1

    #Create a new figure and plot the three images
    plt.figure()
    plt.subplot(1, 3, 1).set_axis_off()
    plt.imshow(img0_red)
    plt.title(title0)
    plt.subplot(1, 3, 2).set_axis_off()
    plt.imshow(overlay)
    plt.title(title_mid)
    plt.subplot(1, 3, 3).set_axis_off()
    plt.imshow(img1_green)
    plt.title(title1)

    #If a file name was given, save the figure
    if fname is not None:
      from time import sleep
      sleep(1)
      plt.savefig(fname, bbox_inches='tight')


def draw_lattice_2d(nrows, ncols, delta):
    r"""
    Creates a figure of a regular lattice of nrows x ncols squares. The size of
    each square is delta x delta pixels (not counting the separation lines).
    The lines are one pixel width.

    Parameters
    ----------
    nrows : int
        the number of squares to be drawn vertically
    ncols : int
        the number of squares to be drawn horizontally
    delta : int
        the size of each square of the grid (delta x delta pixels) 
    """
    lattice=np.ndarray((1 + (delta + 1) * nrows, 
                        1 + (delta + 1) * ncols), 
                        dtype = np.float64)

    #Fill the lattice with "white"
    lattice[...] = 127

    #Draw the horizontal lines in "black"
    for i in range(nrows + 1):
        lattice[i*(delta + 1), :] = 0

    #Draw the vertical lines in "black"
    for j in range(ncols + 1):
        lattice[:, j * (delta + 1)] = 0

    return lattice


def plot_2d_diffeomorphic_map(mapping, delta = 10, fname = None,
                              input_shape=None, input_affine=-1,
                              output_shape=None, output_affine=-1,
                              show_figure = True):
    r"""
    Draws a diffeomorphic map by showing the effect of the deformation on a
    regular grid. The resulting figure contains two images: the direct
    transformation is plotted to the left, and the inverse transformation is 
    plotted to the right

    Parameters
    ----------
    mapping : DiffeomorphicMap object
        the diffeomorphic map to be drawn
    delta : int
        the size of the squares of the regular lattice to be used to plot the
        warping effects. Each square will be deta x delta pixels
    fname : string
        the name of the file the figure will be written to. If None, the figure
        will not be saved to disk
    input_shape : tuple, shape (2,)
        the shape of the grid to be used to sample the direct transformation (
        the shape of the grid will be output_shape, and the warped grid will be
        sampled on the input_shape grid)
    input_affine : array, shape (3, 3)
        the affine transformation mapping input grid's coordinates to physical
        space
    output_shape : tuple, shape (2,)
        the shape of the grid to be used to sample the inverse transformation (
        the shape of the grid will be input_shape, and the warped grid will be
        sampled on the output_shape grid)
    output_affine : array, shape (3, 3)
        the affine transformation mapping output grid's coordinates to physical
        space
    show_figure : Boolean
        if True, the deformed grids will be ploted using matplotlib, else the
        grids are just returned

    Note
    ----
    The default value for the affine transformation is "-1" to handle the case
    in which the user provides "None" as input meaning "identity". If we used
    None as default, we wouldn't know if the user specifically wants to use
    the identity (specifically passing None) or if it was left unspecified,
    meaning to use the apropriate default matrix

    """
    if mapping.is_inverse:
        #By default, the input grid is the domain grid, because it's inverse
        if input_shape is None:
            input_shape = mapping.domain_shape
        if input_affine is -1:
            input_affine = mapping.domain_affine

        #By default, the output grid is the mapping's input grid because it's 
        #inverse
        if output_shape is None:
            output_shape = mapping.input_shape
        if output_affine is -1:
            output_affine = mapping.input_affine
    else:
        #By default, the input grid is the mapping's input grid
        if input_shape is None:
            input_shape = mapping.input_shape
        if input_affine is -1:
            input_affine = mapping.input_affine

        #By default, the output grid is the mapping's domain grid
        if output_shape is None:
            output_shape = mapping.domain_shape
        if output_affine is -1:
            output_affine = mapping.domain_affine

    #The world-to-image (image = drawn lattice on the output grid)
    #transformation is the inverse of the output affine
    world_to_image = None if output_affine is None else np.linalg.inv(output_affine) 

    #Draw the squares on the output grid
    X1,X0 = np.mgrid[0:output_shape[0], 0:output_shape[1]]
    lattice_out=draw_lattice_2d((output_shape[0] + delta) / (delta + 1), 
                                 (output_shape[1] + delta) / (delta + 1), delta)
    lattice_out=lattice_out[0:output_shape[0], 0:output_shape[1]]

    #Warp in the forward direction (sampling it on the input grid)
    warped_forward = mapping.transform(lattice_out, 'lin', world_to_image,
                                       input_shape, input_affine)

    
    #Now, the world-to-image (image = drawn lattice on the input grid) 
    #transformation is the inverse of the input affine
    world_to_image = None if input_affine is None else np.linalg.inv(input_affine) 

    #Draw the squares on the input grid
    X1,X0 = np.mgrid[0:input_shape[0], 0:input_shape[1]]
    lattice_in=draw_lattice_2d((input_shape[0] + delta) / (delta + 1), 
                                 (input_shape[1] + delta) / (delta + 1), delta)
    lattice_in=lattice_in[0:input_shape[0], 0:input_shape[1]]

    #Warp in the backward direction (sampling it on the output grid)
    warped_backward = mapping.transform_inverse(lattice_in, 'lin', world_to_image,
                                       output_shape, output_affine)

    #Now plot the grids
    if show_figure:
        plt.figure()

        plt.subplot(1, 2, 1).set_axis_off()
        plt.imshow(warped_forward, cmap = plt.cm.gray)
        plt.title('Direct transform')

        plt.subplot(1, 2, 2).set_axis_off()
        plt.imshow(warped_backward, cmap = plt.cm.gray)
        plt.title('Inverse transform')

    #Finally, save the figure to disk
    if fname is not None:
        from time import sleep
        sleep(1)
        plt.savefig(fname, bbox_inches = 'tight')

    #Return the deformed grids
    return warped_forward, warped_backward


def plot_middle_slices(V, fname=None):
    r"""
    Creates a figure showing the axial, coronal and sagital middle slices of the
    given volume.

    Parameters
    ----------
    V : array, shape (S, R, C)
        the 3D volume to get the middle slices from
    fname : string
        the name of the file to save the figure to
    """
    #Normalize the intensities to [0, 255]
    sh=V.shape
    V = np.asarray(V, dtype = np.float64)
    V = 255 * (V - V.min()) / (V.max() - V.min())

    #Extract the middle slices
    axial = np.asarray(V[:, :, sh[2]//2]).astype(np.uint8).T
    coronal = np.asarray(V[:, sh[1]//2, :]).astype(np.uint8).T
    sagital = np.asarray(V[sh[0]//2, :, :]).astype(np.uint8).T
    
    #Plot the slices
    plt.figure()
    plt.subplot(1, 3, 1).set_axis_off()
    plt.imshow(axial, cmap = plt.cm.gray, origin='lower')
    plt.title('Axial')
    plt.subplot(1, 3, 2).set_axis_off()
    plt.imshow(coronal, cmap = plt.cm.gray, origin='lower')
    plt.title('Coronal')
    plt.subplot(1, 3, 3).set_axis_off()
    plt.imshow(sagital, cmap = plt.cm.gray, origin='lower')
    plt.title('Sagittal')
    
    #Save the figure if requested
    if fname is not None:
        from time import sleep
        sleep(1)
        plt.savefig(fname, bbox_inches='tight')


def overlay_middle_slices(L, R, slice_type=1, ltitle='Left', rtitle='Right', 
                          fname=None):
    r"""
    Creates a figure containing three images: the gray scale middle slice of
    the first volume (L) to the left, the middle slice of the second volume (R)
    to the right and the middle slices of the two given images on top of each other
    using the red channel for the first volume and the green channel for the 
    second one. It is assumed that both volumes have the same shape. The intended
    use of this function is to visually asses the quality of a registration
    result.

    Parameters
    ----------
    L : array, shape (S, R, C)
        the first volume to extract the middle slice from, plottet to the left
    R : array, shape (S, R, C)
        the second volume to extract the middle slice from, plotted to the right
    slice_type : int
        the type of slice to be extracted: 0=sagital, 1=coronal, 2=axial
    ltitle : string
        the string to be written as title of the left image
    rtitle : string
        the string to be written as title of the right image
    fname : string
        the name of the file to write the image to
    """

    #Normalize the intensities to [0,255]
    sh = L.shape
    L = np.asarray(L, dtype = np.float64)
    R = np.asarray(R, dtype = np.float64)
    L = 255 * (L - L.min()) / (L.max() - L.min())
    R = 255 * (R - R.min()) / (R.max() - R.min())
    
    #Create the color image to draw the overlapped slices into, and extract
    #the slices (note the transpositions)
    if slice_type is 0:
        colorImage = np.zeros(shape = (sh[2], sh[1], 3), dtype = np.uint8)
        ll = np.asarray(L[sh[0]//2, :, :]).astype(np.uint8).T
        rr = np.asarray(R[sh[0]//2, :, :]).astype(np.uint8).T
    elif slice_type is 1:
        colorImage = np.zeros(shape = (sh[2], sh[0], 3), dtype = np.uint8)
        ll = np.asarray(L[:, sh[1]//2, :]).astype(np.uint8).T
        rr = np.asarray(R[:, sh[1]//2, :]).astype(np.uint8).T
    elif slice_type is 2:
        colorImage = np.zeros(shape = (sh[1], sh[0], 3), dtype = np.uint8)
        ll = np.asarray(L[:, :, sh[2]//2]).astype(np.uint8).T
        rr = np.asarray(R[:, :, sh[2]//2]).astype(np.uint8).T
    else:
        print("Slice type must be 0, 1 or 2.")
        return

    #Draw the intensity images to the appropriate channels of the color image
    #The "(ll > ll[0, 0])" condition is just an attempt ro eliminate the 
    #background when its intensity is not exactly zero (the [0,0] corner is
    #usually background)
    colorImage[..., 0] = ll * (ll > ll[0, 0])
    colorImage[..., 1] = rr * (rr > rr[0, 0])

    #Create the figure
    plt.figure()
    plt.subplot(1, 3, 1).set_axis_off()
    plt.imshow(ll, cmap = plt.cm.gray, origin = 'lower')
    plt.title(ltitle)
    plt.subplot(1, 3, 2).set_axis_off()
    plt.imshow(colorImage, origin = 'lower')
    plt.title('Overlay')
    plt.subplot(1, 3, 3).set_axis_off()
    plt.imshow(rr, cmap = plt.cm.gray, origin = 'lower')
    plt.title(rtitle)

    #Save the figure to disk, if requested
    if fname is not None:
        from time import sleep
        sleep(1)
        plt.savefig(fname, bbox_inches = 'tight')
