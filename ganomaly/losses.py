import tensorflow as tf
import functools


# Real score should be 1.0
# Fake should be 0.0

def generator_loss(fake_classification):
    return - tf.reduce_mean(fake_classification)


def discriminator_loss(real_classification, fake_classification):
    fake_loss = - tf.reduce_mean(real_classification)
    real_loss = tf.reduce_mean(fake_classification)
    disc_loss = tf.math.add(fake_loss, real_loss)
    return real_loss, fake_loss, disc_loss


def encoder_loss(fake_images_out, fake_images_reconstructed):
    loss = tf.reduce_mean(tf.abs(fake_images_out - fake_images_reconstructed))
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