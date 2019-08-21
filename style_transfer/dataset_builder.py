import logging

import tensorflow as tf

logger = logging.getLogger('trainer')


class DatasetBuilder(object):
    """Build a TFRecord dataset for training."""

    @staticmethod
    def _resize_fn(images, image_size):
        return tf.image.resize_images(
            images,
            image_size,
            method=tf.image.ResizeMethod.BICUBIC
        )

    @staticmethod
    def _decode_example(example_proto):
        features = {
            "image/encoded": tf.FixedLenFeature(
                (), tf.string, default_value=""
            )
        }
        parsed_features = tf.parse_single_example(example_proto, features)
        image = tf.image.decode_jpeg(
            parsed_features["image/encoded"],
            channels=3)
        return image

    @classmethod
    def build(cls, filename, batch_size, image_size):
        """Build a TensorFlow dataset from images.

        Args:
            filename (str) - a filename of tfrecords to load
            batch_size (int) - the batch size for the iterator
            image_size ((int, int)) - resize all images to a single size

        Returns
            dataset - a tfrecord dataset
        """
        logger.info('Creating dataset from: %s' % filename)
        dataset = tf.data.TFRecordDataset(filename)
        dataset = dataset.map(cls._decode_example)
        dataset = dataset.map(lambda x: cls._resize_fn(x, image_size))
        dataset = dataset.batch(batch_size)
        dataset = dataset.repeat()  # Repeat forever
        return dataset
