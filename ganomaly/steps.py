from ganomaly.losses import generator_loss, discriminator_loss, img_loss, epsilon_penalty, encoder_loss, interpolate_imgs
import numpy as np
import os
from matplotlib import pyplot
from math import sqrt
from PIL import Image
import tensorflow as tf

@tf.function
def train_generator(generator, discriminator, generator_optimizer, input_noise):
    with tf.GradientTape() as gen_tape:
        generated_imgs = generator(input_noise, training=True)
        generated_output = discriminator(generated_imgs, training=True)
        gen_loss = generator_loss(generated_output)

    grad_gen = gen_tape.gradient(gen_loss, generator.trainable_variables)
    generator_optimizer.apply_gradients(zip(grad_gen, generator.trainable_variables))

    return gen_loss


@tf.function
def train_discriminator(discriminator, generator, discriminator_optimizer, images, noise, batch_size):

    with tf.GradientTape() as disc_tape:
        generated_imgs = generator(noise, training=True)
        generated_output = discriminator(generated_imgs, training=True)
        real_output = discriminator(images, training=True)
        interpolated_img = interpolate_imgs(images, generated_imgs, batch_size)
        validity_interpolated = discriminator(interpolated_img, training=True)
        real_score, fake_score, disc_loss = discriminator_loss(real_output, generated_output, validity_interpolated, interpolated_img)

    grad_disc = disc_tape.gradient(disc_loss, discriminator.trainable_variables)
    discriminator_optimizer.apply_gradients(zip(grad_disc, discriminator.trainable_variables))

    return disc_loss, real_score, fake_score


def train_encoder(encoder, generator, encoder_optimizer, input_noise):

    with tf.GradientTape() as enc_tape:
        fake_images = generator(input_noise, training=True)
        z_hat = encoder(fake_images, training=True)
        fake_images_reconstructed = generator(z_hat, training=True)
        enc_loss = img_loss(fake_images, fake_images_reconstructed)

    grad_enc = enc_tape.gradient(enc_loss, encoder.trainable_variables)
    encoder_optimizer.apply_gradients(zip(grad_enc, encoder.trainable_variables))

    return enc_loss, fake_images, fake_images_reconstructed


def summary_func(writer, i, local_step, im_size, result_dict, max_images=1, result_dir='.'):
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

        for k in result_dict.keys():
            if k.endswith('loss') or k.endswith('alpha') or k.endswith('avg'):
                try:
                    tf.summary.scalar(k, result_dict[k], step=i)
                except ValueError:
                    print(f'Unable to save {k} with value of {result_dict[k]}')

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
