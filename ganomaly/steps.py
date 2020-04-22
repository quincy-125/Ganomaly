from ganomaly.losses import generator_loss, discriminator_loss, encoder_loss, gradient_penalty, img_loss
from ganomaly.models import update_fadein
import numpy as np
import os
from matplotlib import pyplot
from math import sqrt
from PIL import Image
import tensorflow as tf


def train_step(images, train_dict, lambda_param=10):
    """
    Organize computation of each step

    :param lambda_param: gradient penalty term
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
    prev_gen_loss = None
    active_critic = True

    batch_size = train_dict['batch_size']
    latent_dim = train_dict['latent_dim']

    gp = 0
    generator_optimizer = train_dict['generator_optimizer']
    discriminator_optimizer = train_dict['discriminator_optimizer']
    encoder_optimizer = train_dict['encoder_optimizer']

    # Determine whether fading should occur generator/discriminator/encoder
    if train_dict['num_images_so_far'] < train_dict['number_of_images_to_fade'] and images.shape[1] > 4:
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

    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape, tf.GradientTape() as enc_tape:
        input_noise = tf.random.normal([batch_size, latent_dim])

        fake_images = generator(input_noise, training=True)
        z_hat = encoder(fake_images, training=True)
        fake_images_reconstructed = generator(z_hat, training=True)

        real_classification = discriminator(images, training=True)
        fake_classification = discriminator(fake_images, training=True)

        # Get losses
        # Encoder loss is the delta of z scores (input_noise and z_hat) and reconstructed image loss
        enc_loss = img_loss(fake_images, fake_images_reconstructed)

        # Discriminator is based only on its calls relative to truth
        # randomly flip the labels 10% of the time
        if np.random.random() < 0.1:
            real_score, fake_score, disc_loss = discriminator_loss(real_classification, real_classification)
        else:
            real_score, fake_score, disc_loss = discriminator_loss(real_classification, fake_classification)

        disc_loss += gradient_penalty(discriminator, fake_images, images)
        gen_loss = generator_loss(fake_classification)
        gen_loss += img_loss(images, fake_images)

    gradients_of_encoder = enc_tape.gradient(enc_loss, encoder.trainable_variables)
    encoder_optimizer.apply_gradients(zip(gradients_of_encoder, encoder.trainable_variables))

    tf.print(f'\rMEAN:{np.mean(fake_images_reconstructed.numpy()):.2f} ', end='\t')
    tf.print(f'rMEAN: {np.mean(images.numpy()):.2f} ', end='\t')
    tf.print('gen_loss: {:.5f}'.format(gen_loss), end='\t')
    tf.print('disc_loss: {:.5f}'.format(disc_loss), end='\t')
    tf.print('enc_loss: {:.5f}'.format(enc_loss), end='\t')
    tf.print('Avg_Fake_Score: {:.5f}'.format(fake_score), end='\t')
    tf.print('Avg_Real_Score: {:.5f}'.format(real_score), end='\t')
    tf.print('Alpha: {:.3f}'.format(alpha), end='\t')
    tf.print('image_no.: {}'.format(train_dict['num_images_so_far']), end='\t')


    gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
    generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))

    if np.random.random() < 0.95:
        gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)
        discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))
    else:
        tf.print('Not Updating D', end='\t')


    return {'generator_loss': gen_loss, 'discriminator_loss': disc_loss, 'encoder_loss': enc_loss,
            'fake_images': fake_images, 'real_images': images, 'fake_avg': fake_score, 'real_avg': real_score,
            'alpha': alpha}


def summary_func(writer, i, local_step, im_size, result_dict, alpha=0, max_images=1, result_dir='.'):
    def write_image_file(key_name=None, images=None, sub_img_size=800):

        if images.shape[0] > max_images:
            images = images[:max_images, :, :, :]

        square = int(sqrt(max_images))
        for q in range(max_images):
            pyplot.subplot(square, square, 1 + q)
            pyplot.axis('off')
            img = Image.fromarray(np.uint8(images[q])).resize((sub_img_size, sub_img_size))
            pyplot.imshow(img)
        # save plot to file
        fname = ['step_' + str(i), str(im_size) + 'x' + str(im_size), 'img.png']
        if not os.path.exists(os.path.join(result_dir, key_name)):
            os.makedirs(os.path.join(result_dir, key_name))
        filename1 = os.path.join(result_dir, key_name, '_'.join([str(x) for x in fname]))
        pyplot.savefig(filename1, bbox_inches='tight',transparent=True, pad_inches=0)
        pyplot.close()

    with writer.as_default():
        tf.summary.scalar('global_step', i, step=i)
        tf.summary.scalar('local_step', local_step, step=i)
        tf.summary.scalar('alpha', alpha, step=i)

        for k in result_dict.keys():
            if k.endswith('loss') or k.endswith('alpha') or k.endswith('avg'):
                tf.summary.scalar(k, result_dict[k], step=i)
            elif k.endswith('images'):
                tf.summary.image(k, result_dict[k], step=i, max_outputs=max_images)
                imgs = tf.math.add(tf.math.multiply(result_dict[k].numpy(), 127.5), 127.5)
                imgs = tf.cast(imgs, tf.uint8)
                try:
                    write_image_file(key_name=k, images=imgs)
                except IndexError:
                    print('Can\'t write images {}'.format(imgs.shape))
            elif k.endswith('histogram'):
                tf.summary.histogram(k, result_dict[k], step=i)

        writer.flush()
