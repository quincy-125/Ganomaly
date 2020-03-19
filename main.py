import os
import re

import tensorflow as tf
from ganomaly.datasets import get_dataset
from ganomaly.models import define_discriminator, define_generator
from ganomaly.steps import train_step, summary_func

color_channels = 3
max_images = 9  # Tensorboard
save_frequency = 500  # After how many steps should we save a checkpoint and summary
number_of_images_to_fade = 350000  # How many images should be faded
# noinspection PyPep8
input_dir = 'D:\PyCharm_Projects\Ganomaly\data'
img_size = 4    # Beginning image size
n_blocks = 7    # how many doublings to do from size 4x4x3
latent_dim = 512  # for encoder
result_dir = 'results'
checkpoint_name = 'training_checkpoints'
overwrite = True
checkpoint_prefix = os.path.join(result_dir, checkpoint_name)
LATENT_DEPTHS = {'4': 512, '8': 512, '16': 512, '32': 256, '64': 256, '128': 128, '256': 64, '512': 32, '1024': 16}
BATCH_SIZES = {'4': 256, '8': 256, '16': 256, '32': 8, '64': 8, '128': 8, '256': 8, '512': 6, '1024': 3}
EPOCH_SIZES = {'4': 5, '8': 5, '16': 5, '32': 5, '64': 5, '128': 5, '256': 5, '512': 5, '1024': 5}

# ####################################################################
# Temporary cleaning function
# ####################################################################

if overwrite is True:
    for root, dirs, files in os.walk(result_dir):
        for file in filter(lambda x: re.match(checkpoint_name, x), files):
            print('Removing: {}'.format(os.path.join(root, file)))
            os.remove(os.path.join(root, file))
        for file in filter(lambda x: re.match('checkpoint', x), files):
            print('Removing: {}'.format(os.path.join(root, file)))
            os.remove(os.path.join(root, file))
        for file in filter(lambda x: re.match('events', x), files):
            print('Removing: {}'.format(os.path.join(root, file)))
            os.remove(os.path.join(root, file))

# ####################################################################
# Initialize the models
# ####################################################################
# Each of these returns a pair of models for each image size
# [[size1_full, size1_fade], [size2_full, size2_fade]]
discriminators = define_discriminator(n_blocks, latent_dim=latent_dim, input_shape=(4, 4, color_channels))
encoders = define_discriminator(n_blocks, latent_dim=latent_dim, input_shape=(4, 4, color_channels), style='encoder')
generators = define_generator(latent_dim, n_blocks)

generator_optimizer = tf.keras.optimizers.Adam(1e-4)
discriminator_optimizer = tf.keras.optimizers.Adam(1e-4)
encoder_optimizer = tf.keras.optimizers.Adam(1e-4)

writer = tf.summary.create_file_writer(result_dir)

# ####################################################################
# Begin training
# ####################################################################
global_step = -1
train_dict = dict()
m = tf.Module()
m.alpha = tf.Variable(0., trainable=False)

for lod in range(2, n_blocks):
    im_size = int(2 ** lod)

    print('Beginning {}x{}x3'.format(im_size, im_size))
    batch_size = int(BATCH_SIZES[str(im_size)])
    train_dict['batch_size'] = batch_size
    epochs = int(EPOCH_SIZES[str(im_size)])

    # Get models of appropriate size
    # These will now be [size1_full, size_2_fade]
    lod_generators = generators.pop(0)
    lod_discriminators = discriminators.pop(0)
    lod_encoders = encoders.pop(0)

    # Make sure sizes are the same
    assert lod_generators[0].output_shape == lod_discriminators[0].input_shape

    train_dict = {'generator_optimizer': generator_optimizer,
                  'discriminator_optimizer': discriminator_optimizer,
                  'encoder_optimizer': encoder_optimizer,
                  'generator_full': lod_generators[0],
                  'discriminator_full': lod_discriminators[0],
                  'encoder_full': lod_encoders[0],
                  'generator_fade': lod_generators[1],
                  'discriminator_fade': lod_discriminators[1],
                  'encoder_fade': lod_encoders[1],
                  'number_of_images_to_fade': number_of_images_to_fade,
                  'batch_size': batch_size,
                  'latent_dim': latent_dim
                  }

    checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
                                     discriminator_optimizer=discriminator_optimizer,
                                     encoder_optimizer=encoder_optimizer,
                                     generator_full=lod_generators[0],
                                     discriminator_full=lod_discriminators[0],
                                     encoder_full=lod_encoders[0],
                                     generator_fade=lod_generators[1],
                                     discriminator_fade=lod_discriminators[1],
                                     encoder_fade=lod_encoders[1]
                                     )

    scaled_data = get_dataset(input_dir, batch_size, im_size, epochs=epochs)
    # """
    result_dict = None
    i = None
    for i, images in enumerate(scaled_data):
        global_step += 1
        train_dict['num_images_so_far'] = i * batch_size

        m.alpha.assign(tf.math.minimum(train_dict['num_images_so_far'] / float(number_of_images_to_fade - 1), 1.0))
        train_dict['alpha'] = m.alpha.numpy()

        result_dict = train_step(images, train_dict)
        if i % save_frequency == 0:
            print('Images seen: {}\t'
                  'local_step: {}, global step: {}, gen_loss: {:06f}, disc_loss: {:06f}, enc_loss: {:06f}, alpha: {}'
                  .format(train_dict['num_images_so_far'],
                          i, global_step, result_dict['gen_loss'], result_dict['disc_loss'], result_dict['enc_loss'],
                          train_dict['alpha']),
                  flush=True)
            summary_func(writer, global_step, i, im_size, result_dict, alpha=train_dict['alpha'],
                         color_channels=color_channels, max_images=max_images, result_dir=result_dir)
            checkpoint.save(file_prefix=checkpoint_prefix)
        """
        # Temporarily limit the number of iterations for debugging purposes
        if i > 5:
            break
        """
    # Complete with this iteration
    summary_func(writer, global_step, i, im_size, result_dict, alpha=train_dict['alpha'],
                 color_channels=color_channels, max_images=max_images, result_dir=result_dir)
    checkpoint.save(file_prefix=checkpoint_prefix)
    print('Images seen: {}\t'
          'local_step: {}, global step: {}, gen_loss: {:06f}, disc_loss: {:06f}, enc_loss: {:06f}, alpha: {} '
          .format(train_dict['num_images_so_far'],
                  i, global_step, result_dict['gen_loss'], result_dict['disc_loss'], result_dict['enc_loss'],
                  train_dict['alpha']),
          flush=True)
