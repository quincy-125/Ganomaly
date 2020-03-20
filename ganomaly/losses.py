import tensorflow as tf


# calculate wasserstein loss
def wasserstein_loss(y_true, y_pred, generator=False):
    if generator is False:
        # Return a negative value for queries that should return 0 when perfect (i.e. from the generator)
        # In other words we want the generator to produce more realistic (0) images
        return -tf.reduce_mean(y_true * y_pred)
    else:
        return tf.reduce_mean(y_true * y_pred)


def generator_loss(fake_classification):
    loss = tf.subtract(1.0, fake_classification)
    # 0 means real, 1 is fake
    # if prediction = 1, then the total loss would be (1*-1=-1)
    # if prediction is 0, then the loss would be small (0*-1=0)
    return tf.reduce_mean(loss)


def discriminator_loss(real_classification, fake_classification, epsilon=0.0000001):
    fake_loss = generator_loss(fake_classification)
    real_loss = tf.reduce_mean(real_classification)
    total_loss = tf.math.add(fake_loss, real_loss, epsilon)
    return total_loss


def encoder_loss(fake_images_out, fake_images_reconstructed):
    loss = tf.reduce_mean(tf.abs(fake_images_out - fake_images_reconstructed))
    return loss
