import tensorflow as tf


# calculate wasserstein loss    # TODO: Change to other losses
def wasserstein_loss(y_pred, y_true):
    return tf.reduce_mean(tf.math.multiply(y_pred, y_true))


def generator_loss(y_pred):
    loss = tf.reduce_mean(y_pred)
    return loss


def discriminator_loss(x_real_d_logit, x_fake_d_logit, epsilon=0.0000001):
    real_loss = wasserstein_loss(tf.ones_like(x_real_d_logit), x_real_d_logit)
    fake_loss = tf.reduce_mean(x_fake_d_logit)
    total_loss = real_loss + fake_loss + epsilon
    return total_loss


def encoder_loss(fake_images_out, fake_images_reconstructed):
    loss = tf.reduce_mean(tf.abs(fake_images_out - fake_images_reconstructed))
    return loss
