import argparse
import keras
import logging
import numpy
import PIL.Image

import keras_contrib

from style_transfer import layers
from style_transfer import utils

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('stylize_image')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Stylize an image using a trained model.'
    )

    parser.add_argument(
        '--input-image', type=str, required=True,
        help='An image to stylize.'
    )
    parser.add_argument(
        '--output-image', type=str, required=True,
        help='An output file for the stylized image.'
    )
    parser.add_argument(
        '--model-checkpoint', type=str, required=True,
        help='Checkpoint from a trained Style Transfer Network.'
    )

    args = parser.parse_args()

    logger.info('Loading model from %s' % args.model_checkpoint)
    custom_objects = {
        'InstanceNormalization':
            keras_contrib.layers.InstanceNormalization,
        'DeprocessStylizedImage': layers.DeprocessStylizedImage
    }
    transfer_net = keras.models.load_model(
        args.model_checkpoint,
        custom_objects=custom_objects
    )

    image_size = transfer_net.input_shape[1:3]

    inputs = [transfer_net.input, keras.backend.learning_phase()]
    outputs = [transfer_net.output]

    transfer_style = keras.backend.function(inputs, outputs)

    input_image = utils.load_image(
        args.input_image,
        image_size[0],
        image_size[1],
        expand_dims=True
    )
    output_image = transfer_style([input_image, 1])[0]
    output_image = PIL.Image.fromarray(numpy.uint8(output_image[0]))
    output_image.save(args.output_image)
