import tensorflow as tf

# This method returns a helper function to compute cross entropy loss
cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)

def generator_loss(y_true, y_pred):
    return cross_entropy(tf.ones_like(y_true), y_pred)

def discriminator_loss(x_real_d_logit, x_fake_d_logit):
    total_loss = x_real_d_logit + x_fake_d_logit + 0.001
    return total_loss

def encoder_loss(fake_images_out, fake_images_reconstructed):
    loss = tf.reduce_mean(tf.abs(fake_images_out - fake_images_reconstructed))
    return loss