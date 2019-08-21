# Hyperparameter Cheat Sheet

To help you choose the right set of hyperparameters to train the best style transfer models, we've created this cheat sheet with examples from ready-to-use styles and heuristics developed from our own experiments.

### Choosing a Style Image

Not all style images are equal. For the best results, pick images that:
* Have large geometric patterns on the scale of 10% - 33% of the images width
* Have bold, contrasting color palletes
* Have strong edges

Style images will be resized to match the value of the `--image-size` parameter discussed below. They are scaled to fit this side and thus may be skewed. If it's important that the aspect ratio of your style is preserved, we recommend cropping or resizing manually to match the `--image-size` parameter exactly.

If you're looking for inspiration, consider searching [Unslash](https://unsplash.com) or [Flickr](https://flickr.com) for terms like "abstract" or "geometric".

Make sure any images you use have an appropriate license.

### Image Size

The `--image-size` hyperparameter for training denotes the size of the style image and training images. Typical values are `256,256` or `512,512`.

For style images with small textural patterns like van Gogh's brush strokes, a small sizes like `256,256` may erase interesting details. For style images with large, geometric patterns, smaller values for `--image-sizes` can be used.

### Alpha and Model Size

By default, the style transfer networks produced here are roughly 7mb in size and contain 7 million parameters. They can create a stylized image in ~500ms on high end mobile phones, and 5s on lower end phones. To make the models faster, we've included a width-multiplier parameter similar to the one introduced by Google in their MobileNet architecture.

The value `--alpha` can be set between 0 and 1 and will control how many filters are included in each layer. Lower `--alpha` means fewer filters, fewer parameters, faster models, with slightly worse style transfer abilities. In testing, `--alpha=0.25` produced models that ran at 17fps on an iPhone X, while still transfering styles well.

Finally, for models that are intended to be used in real-time on a CPU only, you can use the `--model-size` flag to train a model architecture that has been heavily pruned. The style transfer itself isn't quite as good, but the results are usable and the models are incredible small. Using a small model with `alpha=0.3` and quantization, it is possible to [create a 17kb model](https://heartbeat.fritz.ai/creating-a-17kb-style-transfer-model-with-layer-pruning-and-quantization-864d7cc53693).

### Batch Size

Batch size is determined by the available GPU memory. Higher batch sizes generally result in faster training, but if it's too high, you'll crash the machine. The biggest impact on how high you can set the `--batch-size` is the `--image-size`. Larger images take up more space in memory.

In practice we've found the follow batch size limits.

| GPU | `--image-size` | Max `--batch-size` |
| --- | -------------- | -------------------|
|Nvidia K80 |`512,512` | 6 |
|Nvidia K80 |`256,256` | 12 |
|Nvidia V100 |`512,512` | 10 |
|Nvidia V100 |`256,256` | 24 |

### Learning Rate

By default the learning rate of `--lr=0.001` should train a reasonable style from scratch in 120,000 steps. It can also help to follow up this initial training with another 10,000 steps at a lower learning rate of `--lr=0.0001`. If you're style model is producing a stylized image with large black spots or holes, sometimes it's necessary to run training for a few thousand iterations at very high learning rate like `--lr=0.1`.

### Number of Steps

Training a style from scratch typically requires 120,000 individual images to be processed. This number is arrived at by multiplying the number of iterations by the batch size, `num_iterations * batch_size`.

If you're training a new style from a pre-trained checkpoint, only 40,000 individual images are needed.

### Loss Weights

Changing the relative magnitude of each weight in the loss function is the best way to change the aesthetic of your model.

There are four loss terms:
* `style-weight`: The higher this weight, the more the output images will resemble the style of your style image.
* `content-weight`: The higher this weight, the more your stylized image will retain the content of the original.
* `total-variation-weight`: The higher this value, the smoother your stylized images will be. This may wash out small textures.

Below are example configurations for a number of sample style images. All models were trained with `--image-size=512,512`.

| Name | Style Image | Output | content_weight | style_weight | total_variation_weight |
|------|-------------|--------|----------------|--------------|------------------------|
| bicentennial_print | ![alt text][bicentennial_print] | ![alt text][bicentennial_print_output] | 1.0 | 0.0005 | 0.00001 |
| femmes | ![alt text][femmes] | ![alt text][femmes_output] | 1.0 | 0.001 | 0.00001 |
| starry_night | ![alt text][starry_night] | ![alt text][starry_night_output] | 1.0 | 0.003 | 0.00001 |
| the_scream | ![alt text][the_scream] | ![alt text][the_scream_output] | 1.0 | 0.002 | 0.00001 |
| the_trial | ![alt text][the_trial] | ![alt text][the_trial_output] | 1.0 | 0.003 | 0.00001 |
| poppy_field | ![alt text][poppy_field] | ![alt text][poppy_field_output] | 3.0 | 0.003 | 0.00001 |
| ritmo_plastico | ![alt text][ritmo_plastico] | ![alt text][ritmo_plastico_output] | 1.0 | 0.003 | 0.00001 |
| head_of_clown | ![alt text][head_of_clown] | ![alt text][head_of_clown_output] | 1.0 | 0.003 | 0.00001 |
| horses_on_seashore | ![alt text][horses_on_seashore] | ![alt text][horses_on_seashore_output] | 1.0 | 0.003 | 0.00001 |
| kaleidoscope | ![alt text][kaleidoscope] | ![alt text][kaleidoscope_output] | 1.0 | 0.0005 | 0.00001 |
| pink_blue_rhombus | ![alt text][pink_blue_rhombus] | ![alt text][pink_blue_rhombus_output] | 1.0 | 0.0005 | 0.00001 |
| notre_dame | ![alt text][notre_dame] | ![alt text][notre_dame_output] | 1.0 | 0.0004 | 0.00001 |


[bicentennial_print]: style_images/bicentennial_print.jpg "bicentennial_print"
[femmes]: style_images/femmes.jpg "femmes"
[horses_on_seashore]: style_images/horses_on_seashore.jpg "horses_on_seashore"
[kaleidoscope]: style_images/kaleidoscope.jpg "kaleidoscope"
[head_of_clown]: style_images/head_of_clown.jpg "head_of_clown"
[notre_dame]: style_images/notre_dame.jpg "notre_dame"
[pink_blue_rhombus]: style_images/pink_blue_rhombus.jpg "pink_blue_rhombus"
[poppy_field]: style_images/poppy_field.jpg "poppy_field"
[ritmo_plastico]: style_images/ritmo_plastico.jpg "ritmo_plastico"
[starry_night]: style_images/starry_night.jpg "starry_night"
[the_scream]: style_images/the_scream.jpg "the_scream"
[the_trial]: style_images/the_trial.jpg "the_trial"

[bicentennial_print_output]: stylized_images/bicentennial_print_example.jpg "bicentennial_print"
[femmes_output]: stylized_images/femmes_example.jpg "femmes"
[head_of_clown_output]: stylized_images/head_of_clown_example.jpg "head_of_clown"
[horses_on_seashore_output]: stylized_images/horses_on_seashore_example.jpg "horses_on_seashore"
[kaleidoscope_output]: stylized_images/kaleidoscope_example.jpg "kaleidoscope"
[notre_dame_output]: stylized_images/notre_dame_example.jpg "notre_dame"
[pink_blue_rhombus_output]: stylized_images/pink_blue_rhombus_example.jpg "pink_blue_rhombus"
[poppy_field_output]: stylized_images/poppy_field_example.jpg "poppy_field"
[ritmo_plastico_output]: stylized_images/ritmo_plastico_example.jpg "ritmo_plastico"
[starry_night_output]: stylized_images/starry_night_example.jpg "starry_night"
[the_scream_output]: stylized_images/the_scream_example.jpg "the_scream"
[the_trial_output]: stylized_images/the_trial_example.jpg "the_trial"
