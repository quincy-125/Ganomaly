from ganomaly.losses import *
from ganomaly.models import update_fadein
import numpy as np
import os
from matplotlib import pyplot
from math import sqrt
from PIL import Image

def train_step(images, train_dict):
    """
    Organize computation of each step

    :param images: real image matrix
    :param train_dict:
            'generator_optimizer' Optimizer for generator
            'discriminator_optimizer' Optimizer for discriminator
            'encoder_optimizer' Optimizer for encoder
            'generator_full' Generator without fade in
            'discriminator_full' Discriminator without fade in
            'encoder_full' Encoder without fade in
            'generator_fade' Generator with fade in
            'discriminator_fade' Discriminator with fade in
            'encoder_fade' Encoder with fade in
            'num_images_so_far' How many images have been seen for this size
            'number_of_images_to_fade' Number at which we need to run the full models

    :return:
    """
    batch_size = train_dict['batch_size']
    latent_dim = train_dict['latent_dim']
    noise = tf.random.normal([batch_size, latent_dim])

    generator_optimizer = train_dict['generator_optimizer']
    discriminator_optimizer = train_dict['discriminator_optimizer']
    encoder_optimizer = train_dict['encoder_optimizer']

    # Determine whether fading should occur generator/discriminator/encoder
    if train_dict['num_images_so_far'] > train_dict['number_of_images_to_fade'] and images.shape[1] > 4:
        generator = train_dict['generator_fade']
        discriminator = train_dict['discriminator_fade']
        encoder = train_dict['encoder_fade']
        update_fadein([generator, discriminator, encoder], alpha=train_dict['alpha'])
    else:
        generator = train_dict['generator_full']
        discriminator = train_dict['discriminator_full']
        encoder = train_dict['encoder_full']

    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape, tf.GradientTape() as enc_tape:

        generated_images = generator(noise, training=True)
        real_classification = discriminator(images, training=True)
        fake_classification = discriminator(generated_images, training=True)
        z_hat = encoder(generated_images, training=True)
        reconstructed_images = generator(z_hat, training=True)

        gen_loss = generator_loss(fake_classification)
        disc_loss = discriminator_loss(real_classification, fake_classification)
        enc_loss = encoder_loss(generated_images, reconstructed_images)

    tf.print('\rgen_loss: {:06f}'.format(gen_loss), end='\t')
    tf.print('disc_loss: {:06f}'.format(disc_loss), end='\t')
    tf.print('Avg_Fake_Score: {:06f}'.format(tf.reduce_mean(fake_classification)), end='\t')
    tf.print('Avg_Real_Score: {:06f}'.format(tf.reduce_mean(real_classification)), end='\t')
    tf.print('enc_loss: {:06f}'.format(enc_loss), end='\t')

    gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
    gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)
    gradients_of_encoder = enc_tape.gradient(enc_loss, encoder.trainable_variables)

    generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))
    encoder_optimizer.apply_gradients(zip(gradients_of_encoder, encoder.trainable_variables))
    return {'gen_loss': gen_loss, 'disc_loss': disc_loss, 'enc_loss': enc_loss,
            'fake_images': generated_images, 'fake_reconstructed_images': reconstructed_images, 'real_images': images,
            'real_classification_histogram': real_classification, 'fake_classification_histogram': fake_classification}


def summary_func(writer, i, local_step, im_size, result_dict, alpha=0, color_channels=3, max_images=25, result_dir='.'):

    def write_image_file(key_name=None, images=None, sub_img_size=800):
        square = int(sqrt(max_images))
        for q in range(max_images):
            pyplot.subplot(square, square, 1 + q)
            pyplot.axis('off')
            img = Image.fromarray(np.uint8(images[q])).resize((sub_img_size,sub_img_size))
            pyplot.imshow(img)
        # save plot to file
        fname = ['step_' + str(i), str(im_size) + 'x' + str(im_size), 'img.png']
        if not os.path.exists(os.path.join(result_dir, key_name)):
            os.makedirs(os.path.join(result_dir, key_name))
        filename1 = os.path.join(result_dir, key_name, '_'.join([str(x) for x in fname]))
        pyplot.savefig(filename1)
        pyplot.close()

    with writer.as_default():
        tf.summary.scalar('global_step', i, step=i)
        tf.summary.scalar('local_step', local_step, step=i)
        tf.summary.scalar('alpha', alpha, step=i)
        for k in result_dict.keys():
            if k.endswith('loss') or k.endswith('alpha'):
                tf.summary.scalar(k, result_dict[k], step=i)
            elif k.endswith('images'):
                images = result_dict[k].numpy()
                images = tf.math.divide(tf.math.multiply(tf.math.add(images, 1), 255), 2)
                images = np.reshape(images, (images.shape[0], im_size, im_size, color_channels))
                images = images.astype(int)
                tf.summary.image(k, images, step=i, max_outputs=max_images)
                if images.shape[0] > max_images:
                    images = images[:max_images, :, :, :]
                try:
                    write_image_file(k, images)
                except IndexError:
                    print('Can\'t write images ()'.format(images.shape))
            elif k.endswith('histogram'):
                tf.summary.histogram(k, result_dict[k], step=i)

        writer.flush()
