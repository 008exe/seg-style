import logging
import time
import os

import keras
import numpy
from tensorflow.python.lib.io import file_io

from style_transfer import models
from style_transfer import layers
from style_transfer import utils
from style_transfer import dataset_builder

logger = logging.getLogger('trainer')


_log_statement = '''
Training Update:
    Step: {step}
    Avg. Total Loss: {avg_total_loss}
    Avg. Style Loss: {avg_style_loss}
    Avg. Content Loss: {avg_content_loss}
    Avg. Total Variantion Loss: {avg_total_variation_loss}
    Duration (s): {duration}
'''


def get_gram_matrix(x, norm_by_channels=False):
    """Compute the Gram matrix of the tensor x.

    This code was adopted from @robertomest
    https://github.com/robertomest/neural-style-keras/blob/master/training.py  # NOQA

    Args:
        x - a tensor
        norm_by_channels - if True, normalize the Gram Matrix by the number
        of channels.
    Returns:
        gram - a tensor representing the Gram Matrix of x
    """
    if keras.backend.ndim(x) == 3:
        features = keras.backend.batch_flatten(
            keras.backend.permute_dimensions(x, (2, 0, 1))
        )

        shape = keras.backend.shape(x)
        C, H, W = shape[0], shape[1], shape[2]

        gram = keras.backend.dot(
            features,
            keras.backend.transpose(features)
        )
    elif keras.backend.ndim(x) == 4:
        # Swap from (B, H, W, C) to (B, C, H, W)
        x = keras.backend.permute_dimensions(x, (0, 3, 1, 2))
        shape = keras.backend.shape(x)
        B, C, H, W = shape[0], shape[1], shape[2], shape[3]

        # Reshape as a batch of 2D matrices with vectorized channels
        features = keras.backend.reshape(
            x, keras.backend.stack([B, C, H * W])
        )
        # This is a batch of Gram matrices (B, C, C).
        gram = keras.backend.batch_dot(features, features, axes=2)
    else:
        raise ValueError(
            'The input tensor should be either a 3d (H, W, C) '
            'or 4d (B, H, W, C) tensor.'
        )
    # Normalize the Gram matrix
    if norm_by_channels:
        denominator = C * H * W  # Normalization from Johnson
    else:
        denominator = H * W  # Normalization from Google
    gram = gram / keras.backend.cast(denominator, x.dtype)

    return gram


def get_content_loss(
        transfer_image_outputs,
        original_image_outputs,
        layer_names):
    """Get content loss for each content layer.

    Args:
        transfer_image_outputs - the output at each layer of VGG16 for
            the stylized image
        original_image_outputs - the outputs at each layer of the VGG16
            for the original image
        layer_names - a list of layers to use for computing content loss

    Returns:
        content_loss - a list of content loss values for each content layer
    """
    return [
        get_content_layer_loss(
            transfer_image_outputs[layer_name],
            original_image_outputs[layer_name]
        )
        for layer_name in layer_names
    ]


def get_content_layer_loss(transfer_output, original_output):
    """Get content loss from a single content layer.

    Loss is defined as the L2 norm between the features of the
    original image and the stylized image.

    Args:
        transfer_output - an output tensor from a content layer
        original_output - an output tensor from a content layer

    Returns:
        loss - the content loss between the two layers
    """
    diffs = transfer_output - original_output
    return get_l2_norm_loss(diffs)


def get_l2_norm_loss(diffs):
    """Compute the l2 norm of diffs between layers.

    Args:
        diff - a tensor to compute the norm of

    Returns:
        norm - the L2 norm of the differences
    """
    axis = (1, 2, 3)
    if keras.backend.ndim(diffs) == 3:
        axis = (1, 2)

    return keras.backend.mean(
        keras.backend.square(diffs),
        axis=axis
    )


def get_style_loss(
        transfer_image_outputs,
        style_image_outputs,
        layer_names,
        norm_by_channels=False):
    """Get style loss for each style layer.

    Args:
        transfer_image_outputs - the output at each layer of VGG16 for
            the stylized image
        style_image_outputs - the outputs at each layer of the VGG16
            for the image style is being transfered from
        layer_names - a list of layers to use for computing style loss
        norm_by_channel - If True, normalize Gram Matrices by channel

    Returns:
        loss - a list of content loss values for each style layer
    """
    return [
        get_style_layer_loss(
            transfer_image_outputs[layer_name],
            style_image_outputs[layer_name],
            norm_by_channels=norm_by_channels
        )
        for layer_name in layer_names
    ]


def get_style_layer_loss(
        transfer_output,
        style_output,
        norm_by_channels=False):
    """Get style loss from a single content layer.

    Loss is defined as the L2 norm between the Gram Matrix of features
    between the stylized image and the original artistic style image.

    Args:
        transfer_output - an output tensor from a style layer
        style_output - an output tensor from a style layer

    Returns:
        loss - the style loss between the two layers
    """
    # TODO: We could improve efficiency by precomputing the Gram Matrices
    # for the style image as they remain the same for each image.
    style_gram = get_gram_matrix(
        style_output, norm_by_channels=norm_by_channels)
    transfer_gram = get_gram_matrix(
        transfer_output, norm_by_channels=norm_by_channels)

    diffs = style_gram - transfer_gram
    style_layer_loss = get_l2_norm_loss(diffs)
    return style_layer_loss


def get_total_variation_loss(output):
    """Compute the total variation loss of a tensor.

    The TV loss is a measure of how much adjacent tensor elements differ.
    A lower TV loss generally means the resulting image is smoother.

    Args:
        output - a tensor, usually representing an image.

    Returns:
        tv_loss - the total variation loss of the tensor.
    """
    width_var = keras.backend.square(
        output[:, :-1, :-1, :] - output[:, 1:, :-1, :]
    )
    height_var = keras.backend.square(
        output[:, :-1, :-1, :] - output[:, :-1, 1:, :]
    )
    return keras.backend.sum(
        keras.backend.pow(width_var + height_var, 1.25),
        axis=(1, 2, 3)
    )


def _get_inputs(
        tfrecord_filename,
        style_image_files,
        image_size,
        batch_size):
    # Get all of the images from the input dataset
    dataset = dataset_builder.DatasetBuilder.build(
        tfrecord_filename,
        batch_size,
        image_size
    )
    dataset_iterator = dataset.make_one_shot_iterator()

    # Load the style images.
    logger.info('Loading style images:\n%s' % '\n'.join(style_image_files))
    style_imgs = []
    for filename in style_image_files:
        # Note no preprocessing is done while loading.
        img = utils.load_image(filename, *image_size)
        style_imgs.append(img)
    style_imgs = numpy.array(style_imgs)

    return dataset_iterator, style_imgs


def _create_networks(
        image_size,
        alpha=1.0,
        input_tensor=None,
        fine_tune_checkpoint=None,
        use_small_network=False):
    if use_small_network:
        transfer_net = models.SmallStyleTransferNetwork.build(
            image_size,
            alpha=alpha,
            input_tensor=input_tensor,
            checkpoint_file=fine_tune_checkpoint
        )
    else:
        transfer_net = models.StyleTransferNetwork.build(
            image_size,
            alpha=alpha,
            input_tensor=input_tensor,
            checkpoint_file=fine_tune_checkpoint
        )
    # Create a VGG feature extractor for original image content
    original_content_in = layers.VGGNormalize()(transfer_net.input)
    content_net = models.IntermediateVGG(input_tensor=original_content_in)

    # Create a VGG feature extractor for style images.
    style_in = keras.layers.Input(shape=(image_size[0], image_size[1], 3))
    style_in = layers.VGGNormalize()(style_in)
    style_net = models.IntermediateVGG(
        prev_layer=style_in, input_tensor=style_in)

    # Finally, create a VGG feature extractor for the stylized images
    vgg_in = layers.VGGNormalize()(transfer_net.output)

    # Now create a VGG network and pass the output of the transfer network
    # as a previous layer.
    variable_net = models.IntermediateVGG(
        prev_layer=vgg_in,
        input_tensor=transfer_net.input
    )
    return transfer_net, variable_net, style_net, content_net, style_in


def _get_losses(
        transfer_net,
        variable_net,
        content_net,
        style_net,
        image_size,
        content_layers,
        style_layers,
        style_weight,
        content_weight,
        total_variation_weight,
        norm_by_channels=False):
    # Compute the losses.
    # Content Loss
    content_losses = get_content_loss(
        variable_net.layers, content_net.layers, content_layers)
    total_content_loss = sum(content_losses)
    weighted_total_content_loss = content_weight * total_content_loss

    # Style Loss
    style_losses = get_style_loss(
        variable_net.layers,
        style_net.layers,
        style_layers,
        norm_by_channels
    )
    total_style_loss = sum(style_losses)
    weighted_total_style_loss = style_weight * total_style_loss

    # Total Variation Loss
    total_variation_loss = get_total_variation_loss(
        transfer_net.output
    )
    weighted_total_variation_loss = (
        total_variation_weight * total_variation_loss
    )

    # Total all losses
    total_loss = keras.backend.variable(0.0)
    total_loss = (
        weighted_total_content_loss +
        weighted_total_style_loss +
        weighted_total_variation_loss
    )

    return (
        total_loss,
        weighted_total_content_loss,
        weighted_total_style_loss,
        weighted_total_variation_loss
    )


def _create_optimizer(transfer_net, total_loss, learning_rate):
    # Setup the optimizer
    params = transfer_net.trainable_weights
    constraints = {}  # There are none

    # Create an optimizer and updates
    optimizer = keras.optimizers.Adam(lr=learning_rate)
    updates = optimizer.get_updates(params, constraints, total_loss)
    return updates


def copy_file_to_gcs(job_dir, file_path):
    with file_io.FileIO(file_path, mode='rb') as input_f:
        with file_io.FileIO(
                os.path.join(job_dir, file_path), mode='w+') as output_f:
            output_f.write(input_f.read())


def train(
        tfrecord_filename,
        style_image_files,
        model_checkpoint_file,
        content_layers,
        style_layers,
        content_weight=1.0,
        style_weight=1e-4,
        total_variation_weight=1e-3,
        image_size=(256, 256),
        alpha=1.0,
        batch_size=4,
        num_iterations=1,
        norm_by_channels=False,
        learning_rate=0.001,
        log_interval=10,
        checkpoint_interval=10,
        fine_tune_checkpoint=None,
        gcs_bucket=None,
        use_small_network=False):
    """Train the Transfer Network.

    The training procedure consists of iterating over images in
    the COCO image training data set, transforimg the with the style
    transfer model, then computing the total loss across style and
    content layers.

    The default parameters are those suggested by Johnson et al.

    Args:
        tfrecord_filename - tfrecord data set containing training images
        style_image_file - a list of filenames of images that style will be
                           transfered from
        model_checkpoint_file -  a file to write model checkpoints
        content_layers - a list of layers used to compute content loss
        style_layers - a list of layers used to compute style loss
        content_weight - a weight factor for content loss. Default 1.0
        style_weight - a weight factor for style loss. Default 10
        total_variation_weight - a weight factor for total variation loss.
                                 default 1e-4
        img_height - the height of the input images. Default 256
        img_width - the width of the input images. Default 256
        batch_size - the batch size of inputs each iteration. Default 4
        num_iterations - the number of iterations
        norm_by_channels - bool to normalize Gram Matrices by the
                           number of channels. Default True
        learning_rate - the learning rate. Default 0.001
        log_interval -- the interval at which log statements are printed.
                        Default 10 iterations.
        checkpoint_interval - the interval at which to save model
                              checkpoints. Default 10
        fine_tune_checkpoint - a keras file to load first so a network can be
            fine tuned
        gcs_bucket - a toplevel bucket to save models to when using GCS
        use_small_network - if true, use a very small network architecture
    """
    # Get all of the images from the input dataset
    dataset_iterator, style_imgs = _get_inputs(
        tfrecord_filename,
        style_image_files,
        image_size,
        batch_size
    )

    transfer_net, variable_net, style_net, content_net, style_in = _create_networks(  # NOQA
        image_size,
        alpha=alpha,
        input_tensor=dataset_iterator.get_next(),
        fine_tune_checkpoint=fine_tune_checkpoint,
        use_small_network=use_small_network
    )

    logger.info('Setting up network for training.')
    logger.info('Content layers: %s' % ','.join(content_layers))
    logger.info('Style layers: %s' % ','.join(style_layers))

    total_loss, content_loss, style_loss, total_variation_loss = _get_losses(
        transfer_net,
        variable_net,
        content_net,
        style_net,
        image_size,
        content_layers,
        style_layers,
        style_weight,
        content_weight,
        total_variation_weight,
        norm_by_channels=norm_by_channels
    )

    updates = _create_optimizer(transfer_net, total_loss, learning_rate)

    # Define the training function. This takes images into the
    # transfer network as well as style images.
    # Technically the learning phase here is unneeded so long as we are
    # doing InstanceNormalization and not BatchNormalization. In the latter
    # case, be careful at which values you pass when evaluating the model.
    inputs = [
        style_in,
        keras.backend.learning_phase()
    ]

    # Output all of the losses.
    outputs = [
        total_loss,
        content_loss,
        style_loss,
        total_variation_loss,
    ]

    func_train = keras.backend.function(inputs, outputs, updates)

    start_time = time.time()
    for step in range(num_iterations):
        # perform the operations we defined earlier on batch

        out = func_train([style_imgs, 1.])

        if step % log_interval == 0:
            elapsed_time = time.time() - start_time
            start_time = time.time()
            log_msg = _log_statement.format(
                step=step,
                avg_total_loss=numpy.mean(out[0]),
                avg_content_loss=numpy.mean(out[1]),
                avg_style_loss=numpy.mean(out[2]),
                avg_total_variation_loss=numpy.mean(out[3]),
                duration=elapsed_time)
            logging.info(log_msg)
            # Save one more time.
            logger.info('Saving model to %s' % model_checkpoint_file)
            transfer_net.save(model_checkpoint_file)
            if gcs_bucket:
                copy_file_to_gcs(gcs_bucket, model_checkpoint_file)

    # Save one more time.
    logger.info('Saving model to %s' % model_checkpoint_file)
    transfer_net.save(model_checkpoint_file)
    if gcs_bucket:
        copy_file_to_gcs(gcs_bucket, model_checkpoint_file)
