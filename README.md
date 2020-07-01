
```bash
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!! Warning! Not for public consumption yet.  Still actively developing!  !!
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
```
# PROGANomaly
The purpose of `PROGANomaly` is to learn a latent representation of a set of images. Learning such a representation
can be useful for tasks such as anomaly detection. This particular implementation combines the progressive learning 
originally described by 
[Karras et al., 2018](https://github.com/tkarras/progressive_growing_of_gans#preparing-datasets-for-training) combined 
with anomaly detection from [Berg et al., 2019](https://arxiv.org/abs/1905.11034).

## Background
In short, images are downscaled to 4x4x3, trained for a certain number of epochs in a GAN architecture, Generator and
 Discriminator are "grown" to accept/generate an image of double the size, and the new sizes are slowly faded into the 
 model. This multilayer architecture allows the learning of very large image sizes (1024x1024).

Additionally, [Berg et al., 2019](https://arxiv.org/abs/1905.11034) coupled the PROGAN architecture with simultaneous 
Encoder training to lean a mapping between image space and latent space. With this strategy, an image can be encoded 
into the latent space and remapped to image space via the Generator. The idea is that the reconstruction error of the 
`Image->Encoder->Generator` process should be larger in pixels that contain image content that was not present in the
original training set (i.e. an anomaly). The code for this "GANanomalyDetection" method is 
[here](https://github.com/amandaberg/GANanomalyDetection).

## Contribution
The [PROGAN](https://github.com/tkarras/progressive_growing_of_gan) architecture was written in Tensorflow 1.x. However,
since its publication, this coding style has been deprecated in favor of the Tensorflow 2.x style, which is more 
idiomatic (i.e. follows keras/scikit-learn-style APIs), efficient, and extensible for Tensor Processing Units 
([TPUs](https://en.wikipedia.org/wiki/Tensor_processing_unit)), which should dramatically improve the compute time 
needed to learn representation of large images and make it easier for others to tailor to their specific use cases.
 
## Getting Started
This assumes that you have created tfrecords at different level of detail 
as described by the original 
[PROGAN authors](https://github.com/tkarras/progressive_growing_of_gans#preparing-datasets-for-training). It should look 
something like this:
```bash
├── train_data-r02.tfrecords
├── train_data-r03.tfrecords
├── train_data-r04.tfrecords
├── train_data-r05.tfrecords
├── train_data-r06.tfrecords
├── train_data-r07.tfrecords
├── train_data-r08.tfrecords
└── train_data-r09.tfrecords
```
Each TFRecord should contain the following fields:
 * 'shape': tf.io.FixedLenFeature([3], tf.int64),
 * 'data': tf.io.FixedLenFeature([], tf.string)
		
To run `PROGANomaly`, review [cli.py](cli.py).  The only required parameter is the input directory.
```bash
usage: cli.py [-h] -d INPUT_DIR [-B BATCH_SIZES] [-E EPOCH_SIZES]
              [-L LATENT_DEPTHS] [-s IMG_SIZE] [-p TF_RECORD_PREFIX]
              [-x TF_RECORD_SUFFIX] [-S MAX_IMAGE_SIZE] [-e LATENT_DIM]
              [-l LEARNING_RATE] [-c COLOR_CHANNELS] [-i MAX_IMAGES]
              [-f NUMBER_OF_IMAGES_TO_FADE] [-C CHECKPOINT_NAME]
              [-V {DEBUG,INFO,WARNING,ERROR,CRITICAL}]

required arguments:
  -d INPUT_DIR, --data_input_dir INPUT_DIR
                        Where is the TFRecord file directory?

optional arguments:
  -h, --help            show this help message and exit
  -B BATCH_SIZES, --batch_size_dict BATCH_SIZES
                        Provide a dictionary of image size (key) and batch
                        size (value).
  -E EPOCH_SIZES, --epoch_size_dict EPOCH_SIZES
                        Provide a dictionary of image size (key) and epoch
                        size (value).
  -L LATENT_DEPTHS, --latent_size_dict LATENT_DEPTHS
                        Provide a dictionary of image size (key) and latent
                        size (value).
  -s IMG_SIZE, --min_image_size IMG_SIZE
                        What image size should I start with?
  -p TF_RECORD_PREFIX, --tf_record_prefix TF_RECORD_PREFIX
                        What is the string that preceeds the ## in the
                        tfrecord file name? ('.tfrecords' in the above example)
  -x TF_RECORD_SUFFIX, --tf_record_suffix TF_RECORD_SUFFIX
                        What is the string that comes after the ## in the
                        tfrecord file name? ('train_data-r' in the above example)
  -S MAX_IMAGE_SIZE, --max_image_size MAX_IMAGE_SIZE
                        What image size should I stop looking for bigger
                        images?
  -e LATENT_DIM, --encoder_dim LATENT_DIM
                        How big should the latent vector be in the encoder?
  -l LEARNING_RATE, --learning_rate LEARNING_RATE
                        What should the learning rate be?
  -c COLOR_CHANNELS, --channels COLOR_CHANNELS
                        Number of color channels (only works for n=3 now)
  -i MAX_IMAGES, --image_num MAX_IMAGES
                        Number of images to print out at each epoch
  -f NUMBER_OF_IMAGES_TO_FADE, --fade_num NUMBER_OF_IMAGES_TO_FADE
                        How many images should be faded during each resolution
                        stage
  -C CHECKPOINT_NAME, --checkpoint_name CHECKPOINT_NAME
                        What is the prefix to use for checkpoint files
  -V {DEBUG,INFO,WARNING,ERROR,CRITICAL}, --verbose {DEBUG,INFO,WARNING,ERROR,CRITICAL}
                        Set the logging level

To submit a dictionary input, please supply as "-B "{'key1': 'value1'}"".

```


# TODO
 * Add restart capability [main.py:41]
 * Add distributed training [main.py:43]
 * Add images to tensorboard [callbacks.py:38]