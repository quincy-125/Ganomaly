import logging
import os

import tensorflow as tf
from numpy import log2

from ganomaly.ANNOGAN import ANNOGAN
from ganomaly.callbacks import tbc, cpc, GANMonitor, AlphaUpdate
from ganomaly.datasets import get_dataset
from ganomaly.models import build_integrated_model


def ganomaly(max_images=4, color_channels=3, number_of_images_to_fade=500000, max_image_size=1024, input_dir=None,
             img_size=4, latent_dim=512, learning_rate=0.001, checkpoint_name='training_checkpoints',
             BATCH_SIZES={'4': 512, '8': 150, '16': 60, '32': 15, '64': 4, '128': 2, '256': 4, '512': 2, '1024': 4},
             EPOCH_SIZES={'4': 1, '8': 1, '16': 1, '32': 1, '64': 1, '128': 5, '256': 5, '512': 5, '1024': 5},
             LATENT_DEPTHS={'4': 512, '8': 512, '16': 512, '32': 512, '64': 256, '128': 128, '256': 64, '512': 32,
                            '1024': 16},
             tf_record_suffix=None, tf_record_prefix=None
             ):



    result_dir_components = [checkpoint_name, learning_rate, latent_dim]
    result_dir = '_'.join([str(x) for x in result_dir_components])
    checkpoint_prefix = os.path.join(result_dir, checkpoint_name)

    # ####################################################################
    # Initialize the models
    # ####################################################################
    logging.info('Initializing models')
    generator, discriminator, encoder = build_integrated_model(img_size, latent_dim, color_channels=color_channels)

    generator_optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate, beta_1=0.0, beta_2=0.99, epsilon=1e-08)
    discriminator_optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate, beta_1=0.0, beta_2=0.99, epsilon=1e-08)
    encoder_optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate, beta_1=0.0, beta_2=0.99, epsilon=1e-08)

    annogan = ANNOGAN(generator, discriminator, encoder, latent_dim)
    annogan.compile(generator_optimizer, discriminator_optimizer, encoder_optimizer)

    # ####################################################################
    # Run Training
    # ####################################################################

    # TODO: Add restart capability

    # TODO: Add distributed training
    # strategy = tf.distribute.MirroredStrategy()
    # with strategy.scope():

    for lod in range(2, int(log2(max_image_size))):
        img_size = int(2 ** lod)
        batch_size = int(BATCH_SIZES[str(img_size)])
        epochs = int(EPOCH_SIZES[str(img_size)])
        input = get_dataset(input_dir, batch_size, img_size, prefix=tf_record_prefix, suffix=tf_record_suffix)

        logging.info('Beginning {}x{}x3'.format(img_size, img_size))

        annogan.fit(x=input, y=None, verbose=1,
                    epochs=epochs,
                    #steps_per_epoch=5,
                    callbacks=[tbc(log_dir=result_dir),
                               cpc(log_dir=result_dir, prefix=checkpoint_prefix),
                               GANMonitor(num_img=max_images, latent_dim=latent_dim, log_dir=result_dir),
                               AlphaUpdate(batch_size, num_imgs_to_fade=number_of_images_to_fade)])

        logging.info('Updating model architecture')
        annogan.add_generator_block()
        annogan.add_discriminator_block(annogan.D, LATENT_DEPTHS, style='discriminator')
        annogan.add_discriminator_block(annogan.E, LATENT_DEPTHS, style='encoder')
        annogan.compile(generator_optimizer, discriminator_optimizer, encoder_optimizer)  # not sure this is necessary
        annogan.reset_alpha()
        logging.info('Completed Update')
