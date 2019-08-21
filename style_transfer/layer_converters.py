import numpy

"""Functions to convert custom Keras layers to equalivent Core ML Layers.

Each of these functions must conform to the spec set by apple here:
https://github.com/apple/coremltools/blob/master/coremltools/converters/keras/_layers2.py
"""


def convert_instancenormalization(
        builder,
        layer,
        input_names,
        output_names,
        keras_layer):
    """
    Convert InstanceNormalization layer from to coreml.

    This conforms to the Core ML layer spec.

    Parameters
    ----------
    keras_layer: layer
        A keras layer object.

    builder: NeuralNetworkBuilder
        A neural network builder object.
    """
    input_name, output_name = (input_names[0], output_names[0])
    nb_channels = keras_layer.get_weights()[0].shape[0]

    # Set parameters
    # Parameter arrangement in Keras: gamma, beta, mean, variance
    idx = 0
    gamma, beta = None, None
    if keras_layer.scale:
        gamma = keras_layer.get_weights()[idx]
        idx += 1
    if keras_layer.center:
        beta = keras_layer.get_weights()[idx]
        idx += 1

    epsilon = keras_layer.epsilon or 1e-5

    builder.add_batchnorm(
        name=layer,
        channels=nb_channels,
        gamma=gamma,
        beta=beta,
        compute_mean_var=True,
        instance_normalization=True,
        input_name=input_name,
        output_name=output_name,
        epsilon=epsilon
    )


def convert_deprocessstylizedimage(
        builder,
        layer,
        input_names,
        output_names,
        keras_layes):
    """Convert the DeprocessStylizedImage layer type to Core ML.

    This simply takes the output of the tanh activation layer and scales
    values to conform to typical image RGB values.
    """
    input_name, output_name = (input_names[0], output_names[0])

    # Apple's scale layer performs the following math
    # y = w * x + b
    # So to match the keras model's deprocessing layer y = (x + 1) * 127.5
    # We can set the following matrices
    scale = 127.5
    w = numpy.array([scale, scale, scale])
    b = numpy.array([scale, scale, scale])

    builder.add_scale(
        name=input_name,
        W=w,
        b=b,
        has_bias=True,
        shape_scale=w.shape,
        shape_bias=b.shape,
        input_name=input_name,
        output_name=output_name
    )
