import tensorflow as tf
import os
import re
from ganomaly.models import *
from ganomaly.steps import *
from ganomaly.datasets import get_dataset
from tensorflow.keras.constraints import MinMaxNorm

color_channels = 3
max_images = 9  # Tensorboard
save_frequency = 1000  # After how many steps should we save a checkpoint and summary
# noinspection PyPep8
input_dir = 'D:\PyCharm_Projects\Ganomaly\data'
img_size = 32
EPOCHS = 1
latent_dim = 512
result_dir = 'results'
checkpoint_name = 'training_checkpoints'
overwrite = True
checkpoint_prefix = os.path.join(result_dir, checkpoint_name)
alpha_decrement = 0.001
LATENT_DEPTHS = {'4': 512, '8': 512, '16': 512, '32': 256, '64': 256, '128': 128, '256': 64, '512': 32, '1024': 16}
BATCH_SIZES = {'4': 128, '8': 64, '16': 32, '32': 8, '64': 8, '128': 8, '256': 8, '512': 6, '1024': 3}
EPOCH_SIZES = {'4': 1, '8': 1, '16': 3, '32': 4, '64': 5, '128': 6, '256': 7, '512': 8, '1024': 9}
const = MinMaxNorm(min_value=0.00000001, max_value=1.0)
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
discriminator = define_discriminator(LATENT_DEPTHS, input_shape=(4, 4, color_channels))
encoder = define_discriminator(LATENT_DEPTHS, style='encoder', latent_dim=latent_dim)
generator = define_generator(latent_dim, LATENT_DEPTHS, in_dim=4)

generator_optimizer = tf.keras.optimizers.Adam(1e-4)
discriminator_optimizer = tf.keras.optimizers.Adam(1e-4)
encoder_optimizer = tf.keras.optimizers.Adam(1e-4)

checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
                                 discriminator_optimizer=discriminator_optimizer,
                                 encoder_optimizer=encoder_optimizer,
                                 generator=generator,
                                 discriminator=discriminator,
                                 encoder=encoder)
writer = tf.summary.create_file_writer(result_dir)

# ####################################################################
# Begin training
# ####################################################################
prev_steps = 0

for im_size in range(2, int(np.log2(img_size * 2))):
    print('Beginning {}x{}x3'.format(2 ** im_size, 2 ** im_size))
    batch_size = int(BATCH_SIZES[str(2 ** im_size)])
    epochs = int(EPOCH_SIZES[str(2 ** im_size)])
    scaled_data = get_dataset(input_dir, batch_size, 2 ** im_size, epochs=epochs)
    alpha = tf.Variable(initial_value=0.99, constraint=const, trainable=False)
    # """
    i = 0
    result_dict = None
    for i, images in enumerate(scaled_data):
        alpha.assign_sub(alpha_decrement)
        result_dict = train_step(images,
                                 discriminator, generator, encoder,
                                 discriminator_optimizer, generator_optimizer, encoder_optimizer,
                                 batch_size=batch_size, latent_dim=latent_dim, alpha=alpha)
        if i % save_frequency == 0:
            print('Step {}:\tgen_loss: {:06f}, disc_loss: {:06f}, enc_loss: {:06f}'.format(i,
                                                                                             result_dict['gen_loss'],
                                                                                             result_dict['disc_loss'],
                                                                                             result_dict['enc_loss']),
                  flush=True)
            result_dict['alpha'] = alpha
            summary_func(writer, i + prev_steps, im_size, result_dict, color_channels=color_channels,
                         max_images=max_images)
            checkpoint.save(file_prefix=checkpoint_prefix)

        print('\rDone with {}'.format(i + prev_steps), end='', flush=True)

        # Temporarily limit the number of iterations for debugging purposes
        #if i > 5:
        #    break
    # Complete with this iteration
    print('\nCompleted {}x{}x3'.format(2 ** im_size, 2 ** im_size))

    summary_func(writer, i + prev_steps, im_size, result_dict, color_channels=color_channels, max_images=max_images)
    checkpoint.save(file_prefix=checkpoint_prefix)
    # """
    discriminator = add_discriminator_layer(discriminator, LATENT_DEPTHS, 2 ** im_size)
    encoder = add_discriminator_layer(encoder, LATENT_DEPTHS, 2 ** im_size, style='encoder', latent_dim=latent_dim)
    generator = add_generator_layer(generator, LATENT_DEPTHS, 2 ** im_size)
    prev_steps = i + prev_steps
