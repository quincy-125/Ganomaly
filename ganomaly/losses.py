import tensorflow as tf

cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True, label_smoothing=0.1)


def generator_loss(y_pred):
    loss = cross_entropy(tf.zeros_like(y_pred), y_pred)
    return loss


def discriminator_loss(x_real_d_logit, x_fake_d_logit, epsilon=0.0000001):
    real_loss = cross_entropy(tf.ones_like(x_real_d_logit), x_real_d_logit)
    fake_loss = cross_entropy(tf.zeros_like(x_fake_d_logit), x_fake_d_logit)
    total_loss = real_loss + fake_loss + epsilon
    return total_loss


def encoder_loss(fake_images_out, fake_images_reconstructed):
    loss = tf.reduce_mean(tf.abs(fake_images_out - fake_images_reconstructed))
    return loss
