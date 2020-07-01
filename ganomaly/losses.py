import numpy as np
import tensorflow as tf


def generator_loss(generated_output):
	gen_loss = -tf.math.reduce_mean(generated_output) + 1e-8
	return gen_loss


def encoder_loss(real_images, images_reconstructed):
	loss = tf.math.reduce_mean(tf.abs(real_images - images_reconstructed))
	return loss


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


def interpolate_imgs(real_imgs, fake_imgs, batch_size):
	epsilon = tf.random.uniform(shape=(batch_size, 1, 1, 1), minval=0, maxval=1)
	interpolation = epsilon * real_imgs + (1 - epsilon) * fake_imgs
	return interpolation
