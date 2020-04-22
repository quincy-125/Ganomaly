import os

import tensorflow as tf
from ganomaly.datasets import get_dataset
from ganomaly.models import define_discriminator, define_generator
from ganomaly.steps import train_step, summary_func

color_channels = 3
max_images = 4  # Tensorboard
save_frequency = 100  # After how many steps should we save a checkpoint and summary
number_of_images_to_fade = 500000  # How many images should be faded
n_critic = 10
# noinspection PyPep8,PyPep8,PyPep8,PyPep8
input_dir = 'D:\PyCharm_Projects\Ganomaly\data'
# input_dir = '/research/bsi/projects/PI/tertiary/Hart_Steven_m087494/s211408.DigitalPathology/Quincy/Data/train_data'
img_size = 4  # Beginning image size
n_blocks = 7  # how many doublings to do from size 4x4x3
latent_dim = 128  # for encoder
result_dir = 'results'
checkpoint_name = 'training_checkpoints'
checkpoint_prefix = os.path.join(result_dir, checkpoint_name)
BATCH_SIZES = {'4': 1024, '8': 512, '16': 256, '32': 8, '64': 8, '128': 8, '256': 8, '512': 6, '1024': 3}
EPOCH_SIZES = {'4': 10, '8': 8, '16': 5, '32': 5, '64': 5, '128': 5, '256': 5, '512': 5, '1024': 5}
learning_rate = 0.0005
# ####################################################################
# Initialize the models
# ####################################################################
# Each of these returns a pair of models for each image size
# [[size1_full, size1_fade], [size2_full, size2_fade]]
discriminators = define_discriminator(n_blocks, latent_dim=latent_dim, input_shape=(4, 4, color_channels))
encoders = define_discriminator(n_blocks, latent_dim=latent_dim, input_shape=(4, 4, color_channels), style='encoder')
generators = define_generator(latent_dim, n_blocks)

generator_optimizer = tf.keras.optimizers.Adam(0.005)
discriminator_optimizer = tf.keras.optimizers.Adam(learning_rate)
encoder_optimizer = tf.keras.optimizers.Adam(learning_rate)

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

    print('\nBeginning {}x{}x3'.format(im_size, im_size))
    batch_size = int(BATCH_SIZES[str(im_size)])
    train_dict['batch_size'] = batch_size
    train_dict['n_critic'] = n_critic
    epochs = int(EPOCH_SIZES[str(im_size)])

    # Get models of appropriate size
    # These will now be [size1_full, size_2_fade]
    lod_generators = generators.pop(0)
    lod_discriminators = discriminators.pop(0)
    lod_encoders = encoders.pop(0)

    # Make sure sizes are the same
    assert lod_generators[0].output_shape == lod_discriminators[0].input_shape
    if im_size == 4:
        lod_generators.append(lod_generators[0])
    tf.keras.utils.plot_model(lod_generators[1], to_file='generator.png', show_shapes=True, expand_nested=True)
    tf.keras.utils.plot_model(lod_discriminators[1], to_file='discriminator.png', show_shapes=True, expand_nested=True)
    tf.keras.utils.plot_model(lod_encoders[1], to_file='encoder.png', show_shapes=True, expand_nested=True)

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
                  'latent_dim': latent_dim,
                  'n_critic': n_critic
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
    manager = tf.train.CheckpointManager(checkpoint, checkpoint_prefix, max_to_keep=3)
    scaled_data = get_dataset(input_dir, batch_size, im_size, epochs=epochs)
    # """
    result_dict = None
    i = None
    for i, images in enumerate(scaled_data):
        global_step += 1
        train_dict['step'] = i
        train_dict['num_images_so_far'] = i * batch_size

        m.alpha.assign(tf.math.minimum(train_dict['num_images_so_far'] / float(number_of_images_to_fade - 1), 1.0))
        train_dict['alpha'] = m.alpha.numpy()

        result_dict = train_step(images, train_dict)
        if i % save_frequency == 0 and i > 0:
            print('\rImages seen: {}\t'
                  'local_step: {}, global step: {}, gen_loss: {:.6f}, disc_loss: {:.6f}, enc_loss: {:.6f}, alpha:{:.3f}'
                  .format(train_dict['num_images_so_far'],
                          i, global_step, result_dict['generator_loss'], result_dict['discriminator_loss'],
                          result_dict['encoder_loss'],
                          result_dict['alpha']))
            summary_func(writer, global_step, i, im_size, result_dict, alpha=train_dict['alpha'],
                         max_images=max_images, result_dir=result_dir)
            manager.save()
        """
        # Temporarily limit the number of iterations for debugging purposes
        if i > 5 and im_size==4:
            break
        """
    # Complete with this iteration
    summary_func(writer, global_step, i, im_size, result_dict, alpha=train_dict['alpha'],
                 max_images=max_images, result_dir=result_dir)
    manager.save()
    lod_generators[0].save(os.path.join(result_dir, 'generator.h5'))
    lod_discriminators[0].save(os.path.join(result_dir, 'discriminator.h5'))
    lod_encoders[0].save(os.path.join(result_dir, 'encoder.h5'))
    print('\nImages seen: {}\t'
          'local_step: {}, global step: {}, gen_loss: {:06f}, disc_loss: {:06f}, enc_loss: {:06f}, alpha: {:03f} '
          .format(train_dict['num_images_so_far'],
                  i, global_step, result_dict['generator_loss'], result_dict['discriminator_loss'],
                  result_dict['encoder_loss'],
                  result_dict['alpha']))
