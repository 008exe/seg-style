import argparse
import logging
import os
import sys
import tempfile

import keras
import tensorflow as tf

# TensorFlow Lite converters are only available in the most recent version
# of TensorFlow where the lite module has been moved out contrib and into the
# main folder.
try:
    assert(tf.lite)
except:
    raise Exception('You must use TensorFlow 1.13 or higher.')

from style_transfer import models
from style_transfer import tf_utils

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main(argv):

    parser = argparse.ArgumentParser(
        description='Convert a Fritz Style Transfer Model to TensorFlow Lite.'
    )
    parser.add_argument(
        '--keras-checkpoint', type=str, required=True,
        help='Weights from a trained Style Transfer Network.'
    )
    parser.add_argument(
        '--alpha', type=float, required=True,
        help='The width multiplier of the network.'
    )
    parser.add_argument(
        '--tflite-file', type=str, required=True,
        help='A path to the tflite model output'
    )
    parser.add_argument(
        '--image-size', type=str, default='640,480',
        help='The size of input and output of the final Core ML model: H,W'
    )
    parser.add_argument(
        '--use-small-network', action='store_true',
        help=('Use a very small network architecture that works in real time '
              'on some mobile devices using only CPU')
    )

    args = parser.parse_args(argv)

    image_size = [int(dim) for dim in args.image_size.split(',')]

    logger.info('Loading model weights from %s' % args.keras_checkpoint)

    # Set some keras params before loading the model
    keras.backend.clear_session()
    keras.backend.set_learning_phase(0)
    if args.use_small_network:
        keras_model = models.SmallStyleTransferNetwork.build(
            image_size,
            alpha=args.alpha,
            checkpoint_file=args.keras_checkpoint
        )
    else:
        keras_model = models.StyleTransferNetwork.build(
            image_size,
            alpha=args.alpha,
            checkpoint_file=args.keras_checkpoint
        )

    basename = os.path.basename(args.keras_checkpoint)
    # We need to extract a frozen, optimized tensorflow graph before we
    # we can convert it to TFLite. Conversion straight from Keras doesn't
    # work due to custom layers.

    # Freeze Graph to a temp dir.
    temp_dir = tempfile.mkdtemp()
    frozen_graph_filename = tf_utils.freeze_keras_model_graph(
        keras_model, basename, temp_dir)
    # Optimize Graph
    tf_utils.optimize_graph(frozen_graph_filename)

    # Convert the optmized graph to TFLite.
    converter = tf.lite.TFLiteConverter.from_frozen_graph(
        frozen_graph_filename,
        input_arrays=['input_1'],
        output_arrays=['deprocess_stylized_image_1/mul'],
        input_shapes={'input_1': [1, image_size[0], image_size[1], 3]}
    )
    tflite_model = converter.convert()
    open(args.tflite_file, "wb").write(tflite_model)
    logger.info('Saved .tflite model to: %s' % args.tflite_file)
    logger.info('Done.')


if __name__ == '__main__':
    main(sys.argv[1:])
