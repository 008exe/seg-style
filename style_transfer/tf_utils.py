import logging
import os

import keras
import tensorflow as tf
from tensorflow.python.platform import gfile
from tensorflow.python.tools import freeze_graph
from tensorflow.python.tools import optimize_for_inference_lib
from tensorflow.python.framework import dtypes

from style_transfer import models

logger = logging.getLogger(__name__)


def freeze_keras_model_graph(model, basename, output_dir):
    """Extract and freeze the tensorflow graph from a Keras model.

    Args:
        model (keras.models.Model): A Keras model.
        basename (str): the basename of the Keras model. E.g. starry_night.h5
        output_dir (str): a directory to output the frozen graph
    
    Returns:
        output_graph_filename (str): a path to the saved frozen graph.
    """
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
    return output_graph_filename


def optimize_graph(frozen_graph_filename, suffix='optimized'):
    """Optimize a TensorFlow graph for inference.

    Optimized graphs are saved to the same directory as the input frozen graph.

    Args:
        frozen_graph_filename (str): the filename of a frozen graph.
        suffix (optional, str): a suffix to append to the optimized graph file.
    
    Returns:
        optimized_graph_filename (str): a path to the saved optimized graph.
    """
    output_dir, basename = os.path.split(frozen_graph_filename)
    graph_def = load_graph_def(frozen_graph_filename)

    optimized_graph = optimize_for_inference_lib.optimize_for_inference(
        input_graph_def=graph_def,
        input_node_names=['input_1'],
        placeholder_type_enum=dtypes.float32.as_datatype_enum,
        output_node_names=['deprocess_stylized_image_1/mul'],
        toco_compatible=True
    )

    optimized_graph_filename = os.path.basename(
        frozen_graph_filename).replace('frozen', suffix)
    optimized_graph_filename = optimized_graph_filename
    tf.train.write_graph(
        optimized_graph, output_dir, optimized_graph_filename, as_text=False
    )
    logger.info('Saved optimized graph to: %s' %
                os.path.join(output_dir, optimized_graph_filename))
    return optimized_graph_filename


def load_graph_def(filename):
    """Load a graph_def file.

    Args:
        filename (str): a filename to load

    Returns:
        graph_def
    """
    input_graph_def = tf.GraphDef()
    with gfile.FastGFile(filename, 'rb') as file:
        data = file.read()
        input_graph_def.ParseFromString(data)
    return input_graph_def
