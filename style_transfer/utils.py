"""Summary.

Attributes:
    logger (TYPE): Description
"""
import io
import logging
import os

import PIL.Image
import numpy
from tensorflow.python.lib.io import file_io


logger = logging.getLogger('utils')


def load_image(
        filename,
        height,
        width,
        expand_dims=False):
    """Load an image and transform it to a specific size.

    Optionally, preprocess the image through the VGG preprocessor.

    Args:
        filename (TYPE): Description
        height (TYPE): Description
        width (TYPE): Description
        expand_dims (bool, optional): Description
        filename - an image file to load
        height - the height of the transformed image
        width - the width of the transformed image
        vgg_preprocess - if True, preprocess the image for a VGG network.
        expand_dims - Add an addition dimension (B, H, W, C), useful for
                      feeding models.

    Returns:
        img - a numpy array representing the image.
    """
    img = file_io.read_file_to_string(filename, binary_mode=True)
    img = PIL.Image.open(io.BytesIO(img))
    img = img.resize((width, height), resample=PIL.Image.BILINEAR)
    img = numpy.array(img)[:, :, :3]

    if expand_dims:
        img = numpy.expand_dims(img, axis=0)

    return img


def copy_file_from_gcs(file_path):
    """Copy a file from gcs to local machine.

    Args:
        file_path (str): a GCS url to download

    Returns:
        str: a local path to the file
    """
    logger.info('Downloading %s' % file_path)
    with file_io.FileIO(file_path, mode='rb') as input_f:
        basename = os.path.basename(file_path)
        with file_io.FileIO(basename, mode='w+') as output_f:
            output_f.write(input_f.read())
    return basename
