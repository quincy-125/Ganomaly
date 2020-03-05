import tensorflow as tf

cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)

def generator_loss(y_pred):
    return tf.reduce_mean( y_pred  * 0.9)

def discriminator_loss(x_real_d_logit, x_fake_d_logit):
    total_loss = tf.reduce_mean(tf.reduce_mean(tf.abs(x_real_d_logit)) + tf.reduce_mean(tf.abs(x_fake_d_logit)) + 0.001)
    return total_loss

def encoder_loss(fake_images_out, fake_images_reconstructed):
    loss = tf.reduce_mean(tf.abs(fake_images_out - fake_images_reconstructed))
    return loss