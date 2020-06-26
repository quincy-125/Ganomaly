import tensorflow as tf
import numpy as np

bce = tf.keras.losses.BinaryCrossentropy(from_logits=False, label_smoothing=0.3)
mse = tf.keras.losses.MeanSquaredError()

class GeneratorLoss(tf.keras.losses.Loss):
    def call(self, misleading_labels, predictions):
        #gen_loss = -tf.math.reduce_mean(fake_classification) + 1e-8
        #return gen_loss
        return bce(misleading_labels, predictions)

class DiscriminatorLoss(tf.keras.losses.Loss):
    def call(self, true, pred):
        print(f'D labels: {true}\nD predictions: {pred}')
        exit()
        return bce(true, pred)


class EncoderLoss(tf.keras.losses.Loss):

    def call(self, true, pred):
        return mse(tf.abs(true - pred))




def generator_loss(truth, pred):
    #gen_loss = -tf.math.reduce_mean(fake_classification) + 1e-8
    return gen_loss


@tf.function
def gradient_penalty_loss(gradients):
    gradients_sqr = tf.square(gradients)
    gradients_sqr_sum = tf.reduce_sum(gradients_sqr, axis=np.arange(1, len(gradients_sqr.shape)))
    gradients_l2_norm = tf.sqrt(gradients_sqr_sum)
    gradient_penalty = tf.square(gradients_l2_norm)
    return tf.reduce_mean(gradient_penalty)


def discriminator_loss(real_classification, fake_classification):
    fake_score = tf.reduce_mean(fake_classification)
    real_score = tf.reduce_mean(real_classification)

    real_loss = -tf.reduce_mean(real_classification)
    fake_loss = tf.reduce_mean(fake_classification)
    disc_loss = real_loss + fake_loss
    return real_score, fake_score, disc_loss



def encoder_loss(fake_images, fake_images_reconstructed):
    loss = tf.math.reduce_mean(tf.abs(fake_images - fake_images_reconstructed))
    loss = tf.clip_by_value(loss, -1e12, 1e12)  # Remove possible nan
    return loss


def epsilon_penalty(real_score, epsilon=0.001):
    return tf.math.square(real_score) * epsilon


def lerp(a, b, t):
    return a + (b - a) * t


def interpolate_imgs(real_imgs, fake_imgs,batch_size):
    #epsilon = np.random.uniform(0, 1, size=(batch_size, 1, 1, 1))
    epsilon = tf.random.uniform(shape=(batch_size, 1, 1, 1), minval=0, maxval=1)
    interpolation = epsilon * real_imgs + (1 - epsilon) * fake_imgs
    return interpolation
