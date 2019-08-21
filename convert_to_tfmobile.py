import argparse
import logging
import os
import sys

import keras
import tensorflow as tf
from tensorflow.python.platform import gfile
from tensorflow.python.tools import freeze_graph
from tensorflow.python.tools import optimize_for_inference_lib
from tensorflow.python.framework import dtypes

from style_transfer import models

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('stylize_image')


def _freeze_graph(model, basename, output_dir):
    name, _ = os.path.splitext(basename)

    saver = tf.train.Saver()

    with keras.backend.get_session() as sess:
        checkpoint_filename = os.path.join(output_dir, '%s.ckpt' % name)
        output_graph_filename = os.path.join(output_dir, '%s_frozen.pb' % name)
        saver.save(sess, checkpoint_filename)
        tf.train.write_graph(
            sess.graph_def, output_dir, '%s_graph_def.pbtext' % name
        )

        freeze_graph.freeze_graph(
            input_graph=os.path.join(output_dir, '%s_graph_def.pbtext' % name),
            input_saver='',
            input_binary=False,
            input_checkpoint=checkpoint_filename,
            output_graph=output_graph_filename,
            output_node_names='deprocess_stylized_image_1/mul',
            restore_op_name="save/restore_all",
            filename_tensor_name="save/Const:0",
            clear_devices=True,
            initializer_nodes=None
        )
        logger.info('Saved frozen graph to: %s' % output_graph_filename)


def load_graph_def(filename):
    input_graph_def = tf.GraphDef()
    with gfile.FastGFile(filename, 'rb') as file:
        data = file.read()
        input_graph_def.ParseFromString(data)
    return input_graph_def


def _optimize_graph(basename, output_dir):
    name, _ = os.path.splitext(basename)
    frozen_graph_filename = os.path.join(output_dir, '%s_frozen.pb' % name)
    graph_def = load_graph_def(frozen_graph_filename)

    optimized_graph = optimize_for_inference_lib.optimize_for_inference(
        input_graph_def=graph_def,
        input_node_names=['input_1'],
        placeholder_type_enum=dtypes.float32.as_datatype_enum,
        output_node_names=['deprocess_stylized_image_1/mul'],
        toco_compatible=True
    )

    optimized_graph_filename = os.path.basename(
        frozen_graph_filename).replace('frozen', 'optimized')
    optimized_graph_filename = optimized_graph_filename
    tf.train.write_graph(
        optimized_graph, output_dir, optimized_graph_filename, as_text=False
    )
    logger.info('Saved optimized graph to: %s' %
                os.path.join(output_dir, optimized_graph_filename))


def main(argv):

    parser = argparse.ArgumentParser(
        description='Stylize an image using a trained model.'
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
        '--output-dir', type=str, required=True,
        help='A directory to save various tensorflow graphs to'
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
        model = models.SmallStyleTransferNetwork.build(
            image_size,
            alpha=args.alpha,
            checkpoint_file=args.keras_checkpoint
        )
    else:
        model = models.StyleTransferNetwork.build(
            image_size,
            alpha=args.alpha,
            checkpoint_file=args.keras_checkpoint
        )

    basename = os.path.basename(args.keras_checkpoint)
    # Freeze Graph
    _freeze_graph(model, basename, args.output_dir)
    # Optimize Graph
    _optimize_graph(basename, args.output_dir)


if __name__ == '__main__':
    main(sys.argv[1:])
