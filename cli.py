"""Console script for ANNOGAN."""
import argparse
import logging
import sys
import tensorflow as tf
import json
from main import ganomaly
import os

BATCH_SIZES = {'4': 512, '8': 300, '16': 60, '32': 15, '64': 4, '128': 2, '256': 4, '512': 2, '1024': 4}
EPOCH_SIZES = {'4': 1, '8': 1, '16': 1, '32': 1, '64': 1, '128': 5, '256': 5, '512': 5, '1024': 5}
LATENT_DEPTHS = {'4': 512, '8': 512, '16': 512, '32': 512, '64': 256, '128': 128, '256': 64, '512': 32, '1024': 16}


epilog = "To submit a dictionary input, please supply as \"-B \"{\'key1\': \'value1\'}\"\"."

def main():
    parser = argparse.ArgumentParser(epilog=epilog)

    parser.add_argument("-d", "--data_input_dir",
                        dest='input_dir',
                        required=True,
                        help="Where is the TFRecord file directory?")

    parser.add_argument("-B", "--batch_size_dict",
                        type=json.loads,
                        dest='BATCH_SIZES',
                        default=BATCH_SIZES,
                        help="Provide a dictionary of image size (key) and batch size (value).")

    parser.add_argument("-E", "--epoch_size_dict",
                        type=json.loads,
                        dest='EPOCH_SIZES',
                        default=EPOCH_SIZES,
                        help="Provide a dictionary of image size (key) and epoch size (value).")

    parser.add_argument("-L", "--latent_size_dict",
                        type=json.loads,
                        dest='LATENT_DEPTHS',
                        default=LATENT_DEPTHS,
                        help="Provide a dictionary of image size (key) and latent size (value).")

    parser.add_argument("-s", "--min_image_size",
                        dest='img_size',
                        default=4,
                        help="What image size should I start with?")

    parser.add_argument("-S", "--max_image_size",
                        dest='max_image_size',
                        default=1024,
                        help="What image size should I stop looking for bigger images?")

    parser.add_argument("-e", "--encoder_dim",
                        dest='latent_dim',
                        default=512,
                        help="How big should the latent vector be in the encoder?")

    parser.add_argument("-l", "--learning_rate",
                        dest='learning_rate',
                        default=0.001,
                        help="What should the learning rate be?")

    parser.add_argument("-c", "--channels",
                        dest='color_channels',
                        default=3,
                        help="Number of color channels (only works for n=3 now)")

    parser.add_argument("-i", "--image_num",
                        dest='max_images',
                        default=3,
                        help="Number of images to print out at each epoch")

    parser.add_argument("-f", "--fade_num",
                        dest='number_of_images_to_fade',
                        default=500000,
                        help="How many images should be faded during each resolution stage")

    parser.add_argument("-C", "--checkpoint_name",
                        dest='checkpoint_name',
                        default='training_checkpoints',
                        help="What is the prefix to use for checkpoint files")

    parser.add_argument("-V", "--verbose",
                        dest="logLevel",
                        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
                        default="INFO",
                        help="Set the logging level")

    args = parser.parse_args()
    logging.basicConfig(stream=sys.stderr, level=args.logLevel,
                        format='%(name)s (%(levelname)s): %(message)s')

    logger = logging.getLogger(__name__)
    logger.setLevel(args.logLevel)

    # input_dir = 'D:/PyCharm_Projects/Ganomaly/data/train_data'
    # input_dir = '/research/bsi/projects/PI/tertiary/Hart_Steven_m087494/s211408.DigitalPathology/Quincy/Data/train_data'
    assert os.path.exists(args.input_dir), "Please specify an input directory, not whatever this is: " + args.input_dir

    ganomaly(max_images=args.max_images,
             color_channels=args.color_channels,
             number_of_images_to_fade=args.number_of_images_to_fade,
             max_image_size=args.max_image_size,
             input_dir=args.input_dir,
             img_size=args.img_size,
             latent_dim=args.latent_dim,
             learning_rate=args.learning_rate,
             checkpoint_name=args.checkpoint_name,
             BATCH_SIZES=args.BATCH_SIZES,
             EPOCH_SIZES=args.EPOCH_SIZES,
             LATENT_DEPTHS=args.LATENT_DEPTHS
             )
    return 0


if __name__ == "__main__":
    print(tf.__version__)
    print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
    print(tf.config.experimental.list_physical_devices('GPU'))

    sys.exit(main())  # pragma: no cover