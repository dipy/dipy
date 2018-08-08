"""
array2gif.core
~~~~~~~~~~~~~~

Defines the function `write_gif()`, with helper functions
to create the internal blocks required for the GIF 89a format.


Implements the definition of GIF 89a in:
https://www.w3.org/Graphics/GIF/spec-gif89a.txt,
which was made understandable by Matthew Flickinger's blog:
http://www.matthewflickinger.com/lab/whatsinagif/bits_and_bytes.asp
"""
from __future__ import division
import math
import struct
import warnings
from collections import Counter

import numpy

__title__ = 'array2gif'
__version__ = '1.0.4'
__author__ = 'Tanya Schlusser'
__license__ = 'BSD'
__copyright__ = 'Copyright 2016-2018 Tanya Schlusser'
__docformat__ = 'restructuredtext'

warnings.filterwarnings('default', module=__name__, message='.*')

BLOCK_TERMINATOR = b'\x00'
EXTENSION = b'\x21'
HEADER = b'GIF89a'
TRAILER = b'\x3b'
ZERO = b'\x00'


def check_dataset_range(dataset):
    """Confirm no rgb value is outside the range [0, 255]."""
    if dataset.max() > 255 or dataset.min() < 0:
        raise ValueError('The dataset has a value outside the range [0,255]')


def check_dataset_shape(dataset):
    """Confirm the dataset has shape 3 x nrows x ncols."""
    if len(dataset.shape) != 3:
        raise ValueError('Each image needs 3 dimensions: rgb, nrows, ncols')
    if dataset.shape[0] != 3:
        raise ValueError(
            'The dataset\'s first dimension must have all 3\n'
            'colors: red, green, and blue...in that order.'
        )


def check_dataset(dataset):
    """Confirm shape (3 colors x rows x cols) and values [0 to 255] are OK."""
    if isinstance(dataset, numpy.ndarray) and not len(dataset.shape) == 4:
        check_dataset_shape(dataset)
        check_dataset_range(dataset)
    else:  # must be a list of arrays or a 4D NumPy array
        for i, d in enumerate(dataset):
            if not isinstance(d, numpy.ndarray):
                raise ValueError(
                    'Requires a NumPy array (rgb x rows x cols) '
                    'with integer values in the range [0, 255].'
                )
            try:
                check_dataset_shape(d)
                check_dataset_range(d)
            except ValueError as err:
                raise ValueError(
                    '{}\nAt position {} in the list of arrays.'
                    .format(err, i)
                )


def try_fix_dataset(dataset):
    """Transpose the image data if it's in PIL format."""
    if isinstance(dataset, numpy.ndarray):
        if len(dataset.shape) == 3:  # NumPy 3D
            if dataset.shape[-1] == 3:
                return dataset.transpose((2, 0, 1))
        elif len(dataset.shape) == 4:  # NumPy 4D
            if dataset.shape[-1] == 3:
                return dataset.transpose((0, 3, 1, 2))
        # Otherwise couldn't fix it.
        return dataset
    # List of Numpy 3D arrays.
    for i, d in enumerate(dataset):
        if not isinstance(d, numpy.ndarray):
            return dataset
        if not (len(d.shape) == 3 and d.shape[-1] == 3):
            return dataset
        dataset[i] = d.transpose()
    return dataset


def get_image(dataset):
    """Convert the NumPy array to two nested lists with r,g,b tuples."""
    dim, nrow, ncol = dataset.shape
    uint8_dataset = dataset.astype('uint8')
    if not (uint8_dataset == dataset).all():
        message = (
            "\nYour image was cast to a `uint8` (`<img>.astype(uint8)`), "
            "but some information was lost.\nPlease check your gif and "
            "convert to uint8 beforehand if the gif looks wrong."
        )
        warnings.warn(message)
    image = [[
            struct.pack(
                'BBB',
                uint8_dataset[0, i, j],
                uint8_dataset[1, i, j],
                uint8_dataset[2, i, j]
            )
            for j in range(ncol)]
        for i in range(nrow)]
    return image


# -------------------------------- Logical Screen Descriptor --- #
def get_color_table_size(num_colors):
    """Total values in the color table is 2**(1 + int(result, base=2)).

    The result is a three-bit value (represented as a string with
    ones or zeros) that will become part of a packed byte encoding
    various details about the color table, used in the Logical
    Screen Descriptor block.
    """
    nbits = max(math.ceil(math.log(num_colors, 2)), 2)
    return '{:03b}'.format(int(nbits - 1))


def _get_logical_screen_descriptor(image, colors):
    height = len(image)
    width = len(image[0])
    global_color_table_flag = '1'
    # color resolution possibly doesn't do anything, because
    # the size of the colors in the global color table is always
    # 6 bytes (e.g. 0xFFFFFF) per color, and bits needed to express
    # the total number of colors is expressly stated at the beginning
    # of the LZW compression (later). Flickinger says to just use '001'
    color_resolution = '001'
    colors_sorted_flag = '0'  # even though I try to sort
    size_of_global_color_table = get_color_table_size(len(colors))
    packed_bits = int(
        global_color_table_flag +
        color_resolution +
        colors_sorted_flag +
        size_of_global_color_table,
        base=2
    )
    background_color_index = 0
    pixel_aspect_ratio = 0
    logical_screen_descriptor = struct.pack(
        '<HHBBB',
        width,
        height,
        packed_bits,
        background_color_index,
        pixel_aspect_ratio
    )
    return logical_screen_descriptor


# --------------------------------------- Global Color Table --- #
def get_colors(image):
    """Return a Counter containing each color and how often it appears.
    """
    colors = Counter(pixel for row in image for pixel in row)
    if len(colors) > 256:
        msg = (
            "The maximum number of distinct colors in a GIF is 256 but "
            "this image has {} colors and can't be encoded properly."
        )
        raise RuntimeError(msg.format(len(colors)))
    return colors


def _get_global_color_table(colors):
    """Return a color table sorted in descending order of count.
    """
    global_color_table = b''.join(c[0] for c in colors.most_common())
    full_table_size = 2**(1+int(get_color_table_size(len(colors)), 2))
    repeats = 3 * (full_table_size - len(colors))
    zeros = struct.pack('<{}x'.format(repeats))
    return global_color_table + zeros


# ------------------------------- Graphics Control Extension --- #
def _get_graphics_control_extension(delay_time=0):
    control_label = b'\xf9'
    block_size = 4
    disposal_method = '001'
    user_input_expected = '0'
    transparent_index_given = '0'
    packed_bits = int(
        '000' +
        disposal_method +
        user_input_expected +
        transparent_index_given,
        base=2
    )
    delay_time = delay_time
    transparency_index = 0
    graphics_control_extension = struct.pack(
        '<ccBBHBc',
        EXTENSION,
        control_label,
        block_size,
        packed_bits,
        delay_time,
        transparency_index,
        BLOCK_TERMINATOR
    )
    return graphics_control_extension


# ----------------------------------- Application Extension --- #
def _get_application_extension(loop_times=0):
    ANIMATION_LABEL = b'\xff'
    block_size = 11
    application_identifier = b'NETSCAPE2.0'
    data_length = 3
    loop_value = loop_times
    application_extension = struct.pack(
        '<ccB11sBcHc',
        EXTENSION,
        ANIMATION_LABEL,
        block_size,
        application_identifier,
        data_length,
        b'\x01',
        loop_value,
        BLOCK_TERMINATOR
    )
    return application_extension


# ============================================= Image Block ====== #
# --------------------------------------- Image Descriptor --- #
def _get_image_descriptor(image, left=0, top=0):
    image_separator = b'\x2c'
    image_left_position = left
    image_top_position = top
    image_width = len(image[0])
    image_height = len(image)
    local_color_table_exists = '0'
    interlaced_flag = '0'
    sort_flag = '0'
    reserved = '000'
    local_color_table_size = '000'
    packed_bits = int(
        local_color_table_exists +
        interlaced_flag +
        sort_flag +
        reserved +
        local_color_table_size,
        base=2
    )
    image_descriptor = struct.pack(
        '<cHHHHB',
        image_separator,
        image_left_position,
        image_top_position,
        image_width,
        image_height,
        packed_bits
    )
    return image_descriptor


# --------------------------------------------- Image Data --- #
def _lzw_encode(image, colors):
    MAX_COMPRESSION_CODE = 4095
    base_lookup = dict((c[0], i) for i, c in enumerate(colors.most_common()))
    lookup = base_lookup.copy()
    lzw_code_size = int(get_color_table_size(len(colors)), 2) + 1
    clear_code = 2**lzw_code_size
    end_code = clear_code + 1
    next_compression_code = end_code
    # Get the minimum number of bits needed for the next code.
    nbits = next_compression_code.bit_length()
    pixel_stream = [pixel for row in image for pixel in row]
    pixel_buffer = [pixel_stream.pop(0)]
    coded_bits = [(clear_code, nbits)]
    for pixel in pixel_stream:
        test_string = b''.join(pixel_buffer) + pixel
        if test_string in lookup:
            pixel_buffer.append(pixel)
        elif next_compression_code >= MAX_COMPRESSION_CODE:
            coded_bits.insert(0, (lookup[b''.join(pixel_buffer)], nbits))
            coded_bits.insert(0, (clear_code, nbits))
            pixel_buffer = [pixel]
            next_compression_code = end_code
            nbits = next_compression_code.bit_length()
            lookup = base_lookup.copy()
        else:
            code = lookup[b''.join(pixel_buffer)]
            coded_bits.insert(0, (code, nbits))
            pixel_buffer = [pixel]
            next_compression_code += 1
            nbits = next_compression_code.bit_length()
            lookup[test_string] = next_compression_code
    # Add the last content from the pixel buffer.
    coded_bits.insert(0, (lookup[b''.join(pixel_buffer)], nbits))
    coded_bits.insert(0, (end_code, nbits))
    return lzw_code_size, coded_bits


def _get_image_data(image, colors):
    """Performs the LZW compression as described by Matthew Flickinger.

    This isn't fast, but it works.
    http://www.matthewflickinger.com/lab/whatsinagif/lzw_image_data.asp
    """
    lzw_code_size, coded_bits = _lzw_encode(image, colors)
    coded_bytes = ''.join(
        '{{:0{}b}}'.format(nbits).format(val) for val, nbits in coded_bits)
    coded_bytes = '0' * ((8 - len(coded_bytes)) % 8) + coded_bytes
    coded_data = list(
        reversed([
            int(coded_bytes[8*i:8*(i+1)], 2)
            for i in range(len(coded_bytes) // 8)
        ])
    )
    output = [struct.pack('<B', lzw_code_size)]
    # Must output the data in blocks of length 255
    block_length = min(255, len(coded_data))
    while block_length > 0:
        block = struct.pack(
            '<{}B'.format(block_length + 1),
            block_length,
            *coded_data[:block_length]
        )
        output.append(block)
        coded_data = coded_data[block_length:]
        block_length = min(255, len(coded_data))
    return b''.join(output)


def _get_sub_image(image, colors, delay_time=0):
    graphics_control_extension = (
        _get_graphics_control_extension(delay_time=delay_time)
    )
    image_descriptor = _get_image_descriptor(image)
    image_data = _get_image_data(image, colors)
    return b''.join((
        graphics_control_extension,
        image_descriptor,
        image_data,
        BLOCK_TERMINATOR))


def _make_gif(dataset):
    image = get_image(dataset)
    colors = get_colors(image)
    yield _get_logical_screen_descriptor(image, colors)
    yield _get_global_color_table(colors)
    yield _get_sub_image(image, colors)


def _make_animated_gif(datasets, delay_time=10):
    images = [get_image(d) for d in datasets]
    color_sets = (get_colors(image) for image in images)
    colors = Counter()
    for color_set in color_sets:
        colors += color_set
    if len(colors) > 256:
        msg = (
            "The maximum number of distinct colors in a GIF is 256.\n"
            "Although each image has fewer than 256 colors, this library\n"
            "has not yet implemented the Local Color Table option, meaning\n"
            "the overall number of distinct colors in the animation has to\n"
            "be below 256 for now.\n"
            "This animation has {} distinct colors total...sorry."
        )
        raise RuntimeError(msg.format(len(colors)))
    yield _get_logical_screen_descriptor(images[0], colors)
    yield _get_global_color_table(colors)
    yield _get_application_extension()
    for image in images:
        yield _get_sub_image(image, colors, delay_time=delay_time)


def write_gif(dataset, filename, fps=10):
    """Write a NumPy array to GIF 89a format.

    Or write a list of NumPy arrays to an animation (GIF 89a format).

    - Positional arguments::

        :param dataset: A NumPy arrayor list of arrays with shape
                        rgb x rows x cols and integer values in [0, 255].
        :param filename: The output file that will contain the GIF image.
        :param fps: The (integer) frames/second of the animation (default 10).
        :type dataset: a NumPy array or list of NumPy arrays.
        :return: None

    - Example: a minimal array, with one red pixel, would look like this::

        import numpy as np
        one_red_pixel = np.array([[[255]], [[0]], [[0]]])
        write_gif(one_red_pixel, 'red_pixel.gif')

    ..raises:: ValueError
    """
    try:
        check_dataset(dataset)
    except ValueError as e:
        dataset = try_fix_dataset(dataset)
        check_dataset(dataset)
    delay_time = 100 // int(fps)

    def encode(d):
        four_d = isinstance(dataset, numpy.ndarray) and len(dataset.shape) == 4
        if four_d or not isinstance(dataset, numpy.ndarray):
            return _make_animated_gif(d, delay_time=delay_time)
        else:
            return _make_gif(d)

    with open(filename, 'wb') as outfile:
        outfile.write(HEADER)
        for block in encode(dataset):
            outfile.write(block)
        outfile.write(TRAILER)
