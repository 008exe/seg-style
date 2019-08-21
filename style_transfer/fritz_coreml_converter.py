import coremltools
from coremltools.converters.keras._keras2_converter import * 
from coremltools.converters.keras._keras2_converter import _KERAS_LAYER_REGISTRY
from coremltools.converters.keras import _topology2
from coremltools.converters.keras._topology2 import _KERAS_SKIP_LAYERS
from coremltools.models.neural_network import NeuralNetworkBuilder as _NeuralNetworkBuilder
from coremltools.proto import FeatureTypes_pb2 as _FeatureTypes_pb2
from collections import OrderedDict as _OrderedDict
from coremltools.models import datatypes
from coremltools.models import MLModel as _MLModel
from coremltools.models.utils import save_spec as _save_spec
import keras as _keras
from coremltools._deps import HAS_KERAS2_TF as _HAS_KERAS2_TF
import PIL.Image
from six import string_types
from coremltools.proto import FeatureTypes_pb2 as ft

_IMAGE_SUFFIX = '_image'


class FritzCoremlConverter(object):
    """A class to convert keras models to coreml.

    This is converter is a modified version of the one that comes packaged with
    coremltools, but it allows the user to define custom layer mappings from
    keras to coreml.
    """

    @classmethod
    def _check_unsupported_layers(cls, model, supported_layers):
        """Check for any unsupported layers in the keras model.

        Args:
            model - a keras model
            supported_layers - a dictionary of supported layers. Keys are keras
                               layer classes and values are corresponding
                               coreml layer classes.
        """
        for i, layer in enumerate(model.layers):
            if (isinstance(layer, _keras.models.Sequential) or
                    isinstance(layer, _keras.models.Model)):
                cls._check_unsupported_layers(layer)
            else:
                if type(layer) not in supported_layers:
                    print(supported_layers)
                    raise ValueError(
                        "Keras layer '%s' not supported. " % str(type(layer))
                    )
                if isinstance(layer, _keras.layers.wrappers.TimeDistributed):
                    if type(layer.layer) not in supported_layers:
                        raise ValueError(
                            "Keras layer '%s' not supported. " %
                            str(type(layer.layer))
                        )
                if isinstance(layer, _keras.layers.wrappers.Bidirectional):
                    if not isinstance(layer.layer,
                                      _keras.layers.recurrent.LSTM):
                        raise ValueError(
                            'Keras bi-directional wrapper conversion supports '
                            'only LSTM layer at this time. ')

    @staticmethod
    def _get_layer_converter_fn(layer, supported_layers):
        """Get the right converter function for Keras.

        Args:
            layer - a keras layer
            supported_layers - a dictionary of supported layers. Keys are keras
                               layer classes and values are corresponding
                               coreml layer classes.
        Returns:
            layer - a coreml layer
        """
        layer_type = type(layer)
        if layer_type in supported_layers:
            return supported_layers[layer_type]
        else:
            raise TypeError(
                "Keras layer of type %s is not supported." % type(layer)
            )

    @staticmethod
    def _convert_multiarray_output_to_image(spec, feature_name, is_bgr=False):
        """Convert Core ML multiarray output to an image output.

        This modifies the core ml spec in place.

        spec - a Core ML spec protobuf object.
        feature_name - the name of the output feature to convert
        is_bgr - if true, assume image data is already in BGR mode.
                 Default False
        """
        for output in spec.description.output:
            if output.name != feature_name:
                continue
            if output.type.WhichOneof('Type') != 'multiArrayType':
                raise ValueError(
                    "{} is not a multiarray type".format(output.name,)
                )
            array_shape = tuple(output.type.multiArrayType.shape)
            if len(array_shape) == 2:
                height, width = array_shape
                output.type.imageType.colorSpace = \
                    ft.ImageFeatureType.ColorSpace.Value('GRAYSCALE')
            else:
                channels, height, width = array_shape

                if channels == 1:
                    output.type.imageType.colorSpace = \
                        ft.ImageFeatureType.ColorSpace.Value('GRAYSCALE')
                elif channels == 3:
                    if is_bgr:
                        output.type.imageType.colorSpace = \
                            ft.ImageFeatureType.ColorSpace.Value('BGR')
                    else:
                        output.type.imageType.colorSpace = \
                            ft.ImageFeatureType.ColorSpace.Value('RGB')
                else:
                    raise ValueError(
                        "Channel Value {} not supported for image inputs"
                        .format(channels,)
                    )

            output.type.imageType.width = width
            output.type.imageType.height = height

    @classmethod
    def convert_keras(
            cls,
            model,
            input_names=None,
            output_names=None,
            image_input_names=[],
            image_output_names=[],
            deprocessing_args={},
            is_bgr=False,
            is_grayscale=False,
            red_bias=0.0,
            green_bias=0.0,
            blue_bias=0.0,
            gray_bias=0.0,
            image_scale=1.0,
            class_labels=None,
            predicted_feature_name=None,
            custom_layers=None):
        """
        Convert a Keras model to a Core ML Model.

        model - a Keras model to convert
        input_names - names of input layers. Default None
        output_names - names of output layers. Default None
        image_input_names - a list of input names that are image datatypes
        image_output_names - a list of output names that are image datatypes
        preprocessing_args - a dictionary of arguments for input preprocessing
        class_labels - Class labels for outputs,
        predicted_feature_name - name for predicted features,
        custom_layers - a dictionary of custom layer conversions. Keys are
                        Keras layer classes, values are coreml layer functions

        Returns:
            mlmodel - a coreml model object.
        """
        if isinstance(model, string_types):
            model = _keras.models.load_model(model)
        elif isinstance(model, tuple):
            model = _load_keras_model(model[0], model[1])

        # Merge the custom layers with the Keras layer registry
        supported_layers = {}
        supported_layers.update(_KERAS_LAYER_REGISTRY)
        if custom_layers:
            supported_layers.update(custom_layers)

        # Check valid versions
        cls._check_unsupported_layers(model, supported_layers)

        # Build network graph to represent Keras model
        graph = _topology2.NetGraph(model)
        graph.build()
        graph.remove_skip_layers(_KERAS_SKIP_LAYERS)
        graph.insert_1d_permute_layers()
        graph.insert_permute_for_spatial_bn()
        graph.defuse_activation()
        graph.remove_internal_input_layers()
        graph.make_output_layers()

        # The graph should be finalized before executing this
        graph.generate_blob_names()
        graph.add_recurrent_optionals()

        inputs = graph.get_input_layers()
        outputs = graph.get_output_layers()

        # check input / output names validity
        if input_names is not None:
            if isinstance(input_names, string_types):
                input_names = [input_names]
        else:
            input_names = ['input' + str(i + 1) for i in range(len(inputs))]
        if output_names is not None:
            if isinstance(output_names, string_types):
                output_names = [output_names]
        else:
            output_names = ['output' + str(i + 1) for i in range(len(outputs))]

        if (image_input_names is not None and
                isinstance(image_input_names, string_types)):
            image_input_names = [image_input_names]

        graph.reset_model_input_names(input_names)
        graph.reset_model_output_names(output_names)

        # Keras -> Core ML input dimension dictionary
        # (None, None) -> [1, 1, 1, 1, 1]
        # (None, D) -> [D] or [D, 1, 1, 1, 1]
        # (None, Seq, D) -> [Seq, 1, D, 1, 1]
        # (None, H, W, C) -> [C, H, W]
        # (D) -> [D]
        # (Seq, D) -> [Seq, 1, 1, D, 1]
        # (Batch, Sequence, D) -> [D]

        # Retrieve input shapes from model
        if type(model.input_shape) is list:
            input_dims = [filter(None, x) for x in model.input_shape]
            unfiltered_shapes = model.input_shape
        else:
            input_dims = [filter(None, model.input_shape)]
            unfiltered_shapes = [model.input_shape]

        for idx, dim in enumerate(input_dims):
            unfiltered_shape = unfiltered_shapes[idx]
            dim = list(dim)
            if len(dim) == 0:
                # Used to be [None, None] before filtering; indicating
                # unknown sequence length
                input_dims[idx] = tuple([1])
            elif len(dim) == 1:
                s = graph.get_successors(inputs[idx])[0]
                if isinstance(graph.get_keras_layer(s),
                              _keras.layers.embeddings.Embedding):
                    # Embedding layer's special input (None, D) where D is
                    # actually sequence length
                    input_dims[idx] = (1,)
                else:
                    input_dims[idx] = dim  # dim is just a number
            elif len(dim) == 2:  # [Seq, D]
                input_dims[idx] = (dim[1],)
            elif len(dim) == 3:  # H,W,C
                if (len(unfiltered_shape) > 3):
                    # keras uses the reverse notation from us
                    input_dims[idx] = (dim[2], dim[0], dim[1])
                else:
                    # keras provided fixed batch and sequence length, so
                    # the input was (batch, sequence, channel)
                    input_dims[idx] = (dim[2],)
            else:
                raise ValueError(
                    'Input' + input_names[idx] + 'has input shape of length' +
                    str(len(dim)))

        # Retrieve output shapes from model
        if type(model.output_shape) is list:
            output_dims = [filter(None, x) for x in model.output_shape]
        else:
            output_dims = [filter(None, model.output_shape[1:])]

        for idx, dim in enumerate(output_dims):
            dim = list(dim)
            if len(dim) == 1:
                output_dims[idx] = dim
            elif len(dim) == 2:  # [Seq, D]
                output_dims[idx] = (dim[1],)
            elif len(dim) == 3:
                output_dims[idx] = (dim[2], dim[0], dim[1])

            input_types = [datatypes.Array(*dim) for dim in input_dims]
            output_types = [datatypes.Array(*dim) for dim in output_dims]

            # Some of the feature handling is sensitive about string vs unicode
            input_names = map(str, input_names)
            output_names = map(str, output_names)
            is_classifier = class_labels is not None
            if is_classifier:
                mode = 'classifier'
            else:
                mode = None

            # assuming these match
            input_features = list(zip(input_names, input_types))
            output_features = list(zip(output_names, output_types))

            builder = _NeuralNetworkBuilder(
                input_features, output_features, mode=mode
            )

        for iter, layer in enumerate(graph.layer_list):
            keras_layer = graph.keras_layer_map[layer]
            print("%d : %s, %s" % (iter, layer, keras_layer))
            if isinstance(keras_layer, _keras.layers.wrappers.TimeDistributed):
                keras_layer = keras_layer.layer

            converter_func = cls._get_layer_converter_fn(
                keras_layer, supported_layers
            )
            input_names, output_names = graph.get_layer_blobs(layer)
            converter_func(
                builder,
                layer,
                input_names,
                output_names,
                keras_layer
            )

        # Set the right inputs and outputs on the model description (interface)
        builder.set_input(input_names, input_dims)
        builder.set_output(output_names, output_dims)

        # Since we aren't mangling anything the user gave us, we only need to
        # update the model interface here
        builder.add_optionals(graph.optional_inputs, graph.optional_outputs)

        # Add classifier classes (if applicable)
        if is_classifier:
            classes_in = class_labels
            if isinstance(classes_in, string_types):
                import os
                if not os.path.isfile(classes_in):
                    raise ValueError(
                        "Path to class labels (%s) does not exist." %
                        classes_in
                    )
                with open(classes_in, 'r') as f:
                    classes = f.read()
                classes = classes.splitlines()
            elif type(classes_in) is list:  # list[int or str]
                classes = classes_in
            else:
                raise ValueError(
                    'Class labels must be a list of integers / '
                    'strings, or a file path'
                )

            if predicted_feature_name is not None:
                builder.set_class_labels(
                    classes,
                    predicted_feature_name=predicted_feature_name
                )
            else:
                builder.set_class_labels(classes)

        # Set pre-processing paramsters
        builder.set_pre_processing_parameters(
            image_input_names=image_input_names,
            is_bgr=is_bgr,
            red_bias=red_bias,
            green_bias=green_bias,
            blue_bias=blue_bias,
            gray_bias=gray_bias,
            image_scale=image_scale)

        # Convert the image outputs to actual image datatypes
        for output_name in output_names:
            if output_name in image_output_names:
                cls._convert_multiarray_output_to_image(
                    builder.spec, output_name, is_bgr=is_bgr
                )

        # Return the protobuf spec
        spec = builder.spec
        return _MLModel(spec)
