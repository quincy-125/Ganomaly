import os

import tensorflow as tf
from ganomaly.datasets import get_dataset
from ganomaly.models import define_discriminator, define_generator, update_fadein
from ganomaly.steps import train_step, train_generator, summary_func

color_channels = 3
max_images = 4  # Tensorboard
save_frequency = 100  # After how many steps should we save a checkpoint and summary
number_of_images_to_fade = 500000  # How many images should be faded
# noinspection PyPep8,PyPep8,PyPep8,PyPep8
input_dir = 'D:\PyCharm_Projects\Ganomaly\data\\train_data'
#input_dir = '/research/bsi/projects/PI/tertiary/Hart_Steven_m087494/s211408.DigitalPathology/Quincy/Data/train_data'
img_size = 4  # Beginning image size
n_blocks = 8  # how many doublings to do from size 4x4x3
latent_dim = 512  # for encoder
checkpoint_name = 'training_checkpoints'
BATCH_SIZES = {'4': 512, '8': 300, '16': 60, '32': 15, '64': 4, '128': 2, '256': 4, '512': 2, '1024': 4}
EPOCH_SIZES = {'4': 1, '8': 1, '16': 1, '32': 1, '64': 1, '128': 5, '256': 5, '512': 5, '1024': 5}
learning_rate = 0.001
label_flip_rate = 0.05
n_critic = 1

result_dir_components = ['results', learning_rate, label_flip_rate, latent_dim, n_critic]
result_dir = '_'.join([str(x) for x in result_dir_components])
checkpoint_prefix = os.path.join(result_dir, checkpoint_name)

# ####################################################################
# Initialize the models
# ####################################################################
# Each of these returns a pair of models for each image size
# [[size1_full, size1_fade], [size2_full, size2_fade]]
strategy = tf.distribute.MirroredStrategy()
with strategy.scope():
    discriminators = define_discriminator(n_blocks, latent_dim=latent_dim, input_shape=(4, 4, color_channels))
    encoders = define_discriminator(n_blocks, latent_dim=latent_dim, input_shape=(4, 4, color_channels), style='encoder')
    generators = define_generator(latent_dim, n_blocks)


generator_optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate, beta_1=0.0, beta_2=0.99, epsilon=1e-08)
discriminator_optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate, beta_1=0.0, beta_2=0.99, epsilon=1e-08)
encoder_optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate, beta_1=0.0, beta_2=0.99, epsilon=1e-08)

writer = tf.summary.create_file_writer(result_dir)

# ####################################################################
# Begin training
# ####################################################################
global_step = -1
train_dict = dict()
m = tf.Module()
m.alpha = tf.Variable(0., trainable=False)
result_dict = dict()

for lod in range(2, n_blocks):
    im_size = int(2 ** lod)

    print('\nBeginning {}x{}x3'.format(im_size, im_size))
    batch_size = int(BATCH_SIZES[str(im_size)])
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
    try:
        tf.keras.utils.plot_model(lod_generators[1], to_file='generator.png', show_shapes=True, expand_nested=True)
        tf.keras.utils.plot_model(lod_discriminators[1], to_file='discriminator.png', show_shapes=True, expand_nested=True)
        tf.keras.utils.plot_model(lod_encoders[1], to_file='encoder.png', show_shapes=True, expand_nested=True)
    except ImportError:
        print('Unable to print images of models')

    train_dict = {'generator_optimizer': generator_optimizer,
                  'discriminator_optimizer': discriminator_optimizer,
                  'encoder_optimizer': encoder_optimizer,
                  'generator_full': lod_generators[0],
                  'discriminator_full': lod_discriminators[0],
                  'encoder_full': lod_encoders[0],
                  'generator_fade': lod_generators[1],
                  'discriminator_fade': lod_discriminators[1],
                  'encoder_fade': lod_encoders[1]
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

    generator_optimizer = train_dict['generator_optimizer']
    discriminator_optimizer = train_dict['discriminator_optimizer']
    encoder_optimizer = train_dict['encoder_optimizer']

    # """
    result_dict = None
    i = 0
    for i, images in enumerate(scaled_data):
        global_step += 1
        train_dict['step'] = i
        num_images_so_far = i * batch_size

        m.alpha.assign(tf.math.minimum(num_images_so_far / float(number_of_images_to_fade - 1), 1.0))
        train_dict['alpha'] = m.alpha.numpy()

        # Determine whether fading should occur generator/discriminator/encoder
        if num_images_so_far < number_of_images_to_fade and images.shape[1] > 4:
            alpha = train_dict['generator_fade'].layers[-1].alpha.numpy()
            generator = train_dict['generator_fade']
            discriminator = train_dict['discriminator_fade']
            encoder = train_dict['encoder_fade']
            update_fadein([generator, discriminator, encoder], alpha=train_dict['alpha'])
        else:
            alpha = train_dict['alpha']
            generator = train_dict['generator_full']
            discriminator = train_dict['discriminator_full']
            encoder = train_dict['encoder_full']

        input_noise, disc_loss, real_score, fake_score, enc_loss, fake_images, gen_loss, fake_images_reconstructed = train_step(images,
                                                                                                     generator,
                                                                                                     discriminator,
                                                                                                     encoder,
                                                                                                     batch_size,
                                                                                                     latent_dim,
                                                                                                     generator_optimizer,
                                                                                                     discriminator_optimizer,
                                                                                                     encoder_optimizer,
                                                                                                     n_critic=n_critic,
                                                                                                     i=i)
        result_dict = {
            'input_noise': input_noise,
            'disc_loss': disc_loss,
            'real_score': real_score,
            'fake_score': fake_score,
            'enc_loss': enc_loss,
            'fake_images': fake_images,
            'real_images': images,
            'gen_loss': gen_loss,
            'reconstructed_images': fake_images_reconstructed,
            'alpha': alpha
        }
        try:
            print(
                f'\r{i:07d}, KImage: {int(num_images_so_far / 1000):07d}, '
                f'gen_loss: {gen_loss.numpy():.5f}, disc_loss: {disc_loss.numpy():.5f}, '
                f'enc_loss: {enc_loss.numpy():.5f}, alpha:{alpha:.3f}, '
                f'real_score:{real_score:.3f}, fake_score:{fake_score:.3f}', end='')
        except:
            pass
        if i % save_frequency == 0 and i > 0:
            if gen_loss is None:
                gen_loss, fake_images = train_generator(generator, discriminator, generator_optimizer, input_noise)
                result_dict['gen_loss'] = gen_loss

            print(
                f'\r{i:07d}, KImage: {int(num_images_so_far / 1000):07d}, '
                f'gen_loss: {gen_loss.numpy():.5f}, disc_loss: {disc_loss.numpy():.5f}, '
                f'enc_loss: {enc_loss.numpy():.5f}, alpha:{alpha:.3f}, '
                f'real_score:{real_score:.3f}, fake_score:{fake_score:.3f}')

            summary_func(writer, global_step, i, im_size, result_dict,
                         max_images=max_images, result_dir=result_dir)
            manager.save()

        #"""
        # Temporarily limit the number of iterations for debugging purposes
        if i > 5: #and im_size==4:
            break
        #"""
    if result_dict['gen_loss'] is None:
        gen_loss, fake_images = train_generator(generator, discriminator, generator_optimizer, input_noise)
        result_dict['gen_loss'] = gen_loss
        result_dict['fake_images'] = fake_images

    summary_func(writer, global_step, i, im_size, result_dict,
                 max_images=max_images, result_dir=result_dir)
    manager.save()
    lod_generators[0].save(os.path.join(result_dir, 'generator.h5'))
    lod_discriminators[0].save(os.path.join(result_dir, 'discriminator.h5'))
    lod_encoders[0].save(os.path.join(result_dir, 'encoder.h5'))
