import tensorflow as tf
import numpy as np

bce = tf.keras.losses.BinaryCrossentropy(from_logits=True, label_smoothing=0.3)
# Real score should be 1.0
# Fake should be 0

def generator_loss(fake_classification):
    #gen_loss = -tf.math.reduce_mean(fake_classification)
    gen_loss = -bce(tf.ones_like(fake_classification), fake_classification)
    return gen_loss


def discriminator_loss(real_classification, fake_classification):
    fake_score = tf.reduce_mean(fake_classification)
    real_score = tf.reduce_mean(real_classification)

    cel_real = bce(tf.ones_like(real_classification), real_classification)
    cel_fake = bce(tf.zeros_like(fake_classification), fake_classification)
    disc_loss = tf.math.add(cel_real, cel_fake)

    return real_score, fake_score, disc_loss


def encoder_loss(noise_dim, z_hat):
    loss = tf.reduce_mean(tf.abs(noise_dim - z_hat))
    return loss

def img_loss(fake_images, fake_images_reconstructed):
    loss = tf.reduce_mean(tf.abs(fake_images - fake_images_reconstructed))
    return loss


def gp2(discriminator, x_tild=None, x=None, lamb=10):
    # https://mc.ai/gan-wasserstein-gan-wgan-gp/

    # e = Sample from 0,1.
    epsilon = np.random.random_sample()
    # x^ = (e*real) + ((1-e)*fake)
    x_hat = tf.math.multiply(epsilon, x) + tf.math.multiply((1 - epsilon), x_tild)
    # get partial from new fake img
    with tf.GradientTape() as t:
        t.watch(x_hat)
        pred = discriminator(x_hat)
    grad = t.gradient(tf.math.reduce_mean(pred), discriminator.trainable_variables)
    #   l2 norm grad
    grad = tf.math.l2_normalize(grad)
    # subtract 1
    grad = tf.math.subtract(1, grad)
    # square
    grad = tf.math.square(grad)
    # multiply by lamda
    return tf.math.multiply(lamb, grad)


def gradient_penalty(discriminator, batch_x, fake_image):
    batch_size = batch_x.shape[0] # [b, h, w, c]
    t = tf.random.uniform([batch_size, 1, 1, 1]) # [b, 1, 1, 1] => [b, h, w, c]
    t = tf.broadcast_to(t, batch_x.shape)
    interpolate = t * batch_x + (1 - t) * fake_image
    with tf.GradientTape() as tape:
        tape.watch([interpolate])
        d_interpolate_logits = discriminator(interpolate, training=True)
    grads = tape.gradient(d_interpolate_logits, interpolate) # grads:[b, h, w, c] => [b, -1]
    grads = tf.reshape(grads, [grads.shape[0], -1])
    gp = tf.norm(grads, axis=1) # [b]
    gp = tf.reduce_mean((gp - 1) ** 2)
    return gp
