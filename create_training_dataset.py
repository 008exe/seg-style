import argparse
import logging
import io
import os
import sys
import urllib
import zipfile

import PIL.Image
import tensorflow as tf

logger = logging.getLogger('create_training_dataset')

_COCO_ZIP_URL = 'http://images.cocodataset.org/zips/train2014.zip'

try:
    raw_input          # Python 3
except NameError:
    raw_input = input  # Python 3


class DatasetCreator(object):
    """A class to preprocess images from the COCO training data.

    This does not apply any sort of normalization to images. It simply
    transforms and scales image sizes before packing them into an H5 dataset
    and saving them to disk.
    """

    allowed_formats = {'.jpg', '.jpeg', '.JPG', '.JPEG', '.png', '.PNG'}
    max_resize = 16

    @classmethod
    def _get_image_filenames(cls, input_dir, num_images):
        """Get a list of image filenames from a directory."""
        img_list = []
        for filename in os.listdir(input_dir):
            _, ext = os.path.splitext(filename)
            if ext in cls.allowed_formats:
                img_list.append(os.path.join(input_dir, filename))
                if num_images and len(img_list) > num_images:
                    break
        return img_list

    @staticmethod
    def _bytes_feature(value):
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

    @classmethod
    def process_images(
            cls,
            input_dir,
            output_filename,
            num_images=None,
            num_threads=1):
        """Process all images in a directory and create an H5 data set.

        Args:
            input_dir - a directory containing images
            output_filename - the name of the h5 file to write to
            num_images - the number of images to process. 'None' processes all
            num_threads - the number of threads to use. Default 1.
        """
        img_list = cls._get_image_filenames(input_dir, num_images)
        num_images = len(img_list)
        # Remove the h5 file if it exists
        try:
            os.remove(output_filename)
        except OSError:
            pass

        record_writer = tf.python_io.TFRecordWriter(output_filename)
        for idx, filename in enumerate(img_list):
            img = PIL.Image.open(filename)
            encoded_jpeg = io.BytesIO()
            img.save(encoded_jpeg, format='jpeg')
            encoded_jpeg.seek(0)

            example = tf.train.Example(features=tf.train.Features(
                feature={
                    'image/encoded': cls._bytes_feature(encoded_jpeg.read()),
                }))
            record_writer.write(example.SerializeToString())
        record_writer.close()


def download_coco_data(directory):
    """Download and extract the COCO image training data set.

    This file is very large (~13GB) so we check with the user to make
    sure that is ok.

    Args:
        dir - a directory to save the dataset to
    """
    # This is a really big file so ask the user if they are sure they want
    # to start the download.
    if not os.path.isdir(directory):
        logger.info('Creating directory: %s' % directory)
        os.makedirs(directory)

    answer = None
    while answer not in {'Y', 'n'}:
        answer = raw_input(
            'Are you sure you want to download the COCO dataset? [Y/n] '
        )

    if answer == 'n':
        sys.exit()

    logger.info('Downloading COCO image data set. This may take a while...')
    zip_save_path = os.path.join(directory, 'train2014.zip')
    urllib.urlretrieve(_COCO_ZIP_URL, zip_save_path)

    # Files are even bigger to unzip so ask again if they are fine to proceed.
    answer = None
    while answer not in {'Y', 'n'}:
        answer = raw_input(
            'Are you sure you want to unzip things? [Y/n] '
        )

    if answer == 'n':
        sys.exit()

    logger.info('Unzipping COCO image data set. This may take a while...')
    unzip = zipfile.ZipFile(zip_save_path, 'r')
    unzip.extractall(directory)
    unzip.close()
    # Delete the original zipfile
    os.remove(zip_save_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description=('Create a dataset to use when training the Fritz'
                     ' Style Transfer model.'))
    parser.add_argument(
        '--output', type=str, required=True,
        help='The name of the resulting dataset.')
    parser.add_argument(
        '--image-dir', type=str, required=True,
        help=('A directory containing images to turn into tfrecords')
    )
    parser.add_argument(
        '--download', action='store_true',
        help=('When present, download and extract the COCO image dataset.'
              'Note this is a huge download (~13GB).')
    )
    parser.add_argument(
        '--num-images', type=int, help='The number of images to process.'
    )

    args = parser.parse_args()
    image_directory = args.image_dir
    if args.download:
        download_coco_data(image_directory)
        image_directory = os.path.join(image_directory, 'train2014')

    image_directory = os.path.join(args.image_dir)
    DatasetCreator.process_images(
        image_directory,
        args.output,
        num_images=args.num_images
    )
