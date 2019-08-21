import argparse
import keras_contrib
import logging
import sys

from style_transfer import layer_converters
from style_transfer import layers
from style_transfer import models
from style_transfer.fritz_coreml_converter import FritzCoremlConverter

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('convert_to_coreml')


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
        '--coreml-model', type=str, required=True,
        help='A CoreML output file to save to'
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
    # Map custom layers to their custom coreml converters
    custom_layers = {
        keras_contrib.layers.InstanceNormalization: layer_converters.convert_instancenormalization,  # NOQA
        layers.DeprocessStylizedImage: layer_converters.convert_deprocessstylizedimage  # NOQA
    }

    logger.info('Loading model weights from %s' % args.keras_checkpoint)

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

    fritz_converter = FritzCoremlConverter()
    mlmodel = fritz_converter.convert_keras(
        model,
        input_names=['image'],
        image_input_names=['image'],
        output_names=['stylizedImage'],
        image_output_names=['stylizedImage'],
        custom_layers=custom_layers
    )
    logger.info('Saving .mlmodel to %s' % args.coreml_model)
    mlmodel.save(args.coreml_model)


if __name__ == '__main__':
    main(sys.argv[1:])
