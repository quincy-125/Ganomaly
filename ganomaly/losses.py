import tensorflow as tf
import functools
import numpy as np


# Real score should be 1.0
# Fake should be 0

def generator_loss(fake_classification):
    # Using ones here, because I want the fakes to be classified as close to 1 as possible
    # (opposite from discriminator)
    labels = tf.ones_like(fake_classification)
    predictions = fake_classification
    gen_loss = tf.keras.losses.binary_crossentropy(labels, predictions, label_smoothing=0.2)

    return tf.math.reduce_mean(gen_loss)


def discriminator_loss(real_classification, fake_classification, gp=0):
    # Assume that Fake is -1 and real is 1
    fake_loss = tf.reduce_mean(fake_classification)
    real_loss = tf.reduce_mean(real_classification)

    labels = tf.concat([tf.zeros_like(fake_classification), tf.ones_like(real_classification)], 0)
    predictions = tf.concat([fake_classification, real_classification], 0)
    disc_loss = tf.keras.losses.binary_crossentropy(labels, predictions, label_smoothing=0.2)
    # gp2 = gp2(discriminator, x_tild=fake_img, x=real, lamb=10)
    # disc_loss += tf.math.multiply(gp, disc_loss)    #  Add gradient penalty
    return real_loss, fake_loss, tf.math.reduce_mean(disc_loss)


def encoder_loss(real_images, real_images_reconstructed):
    loss = tf.reduce_mean(tf.abs(real_images - real_images_reconstructed))
    return loss



def gradient_penalty(f, real, fake):
    # https://github.com/LynnHo/DCGAN-LSGAN-WGAN-GP-DRAGAN-Tensorflow-2/blob/master/tf2gan/loss.py

    def _gradient_penalty(f, real, fake=None):

        def _interpolate(a, b=None):
            if b is None:   # interpolation in DRAGAN
                beta = tf.random.uniform(shape=tf.shape(a), minval=0., maxval=1.)
                b = a + 0.5 * tf.math.reduce_std(a) * beta
            shape = [tf.shape(a)[0]] + [1] * (a.shape.ndims - 1)
            alpha = tf.random.uniform(shape=shape, minval=0., maxval=1.)
            inter = a + alpha * (b - a)
            inter.set_shape(a.shape)
            return inter

        x = _interpolate(real, fake)
        with tf.GradientTape() as t:
            t.watch(x)
            pred = f(x)
        grad = t.gradient(pred, x)
        norm = tf.norm(tf.reshape(grad, [tf.shape(grad)[0], -1]), axis=1)
        gp = tf.reduce_mean((norm - 1.)**2)

        return gp

    gp = _gradient_penalty(f, real, fake)
    return gp


def gp2(discriminator, x_tild=None, x=None, lamb=10):
    #https://mc.ai/gan-wasserstein-gan-wgan-gp/

    # e = Sample from 0,1.
    epsilon = np.random.random_sample()
    # x^ = (e*real) + ((1-e)*fake)
    x_hat = tf.math.multiply(epsilon, x) + tf.math.multiply((1-epsilon), x_tild)
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