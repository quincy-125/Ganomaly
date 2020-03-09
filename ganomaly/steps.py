from ganomaly.losses import *
import tensorflow as tf
# Notice the use of `tf.function`
# This annotation causes the function to be "compiled".

@tf.function
def train_step(images, discriminator, generator, encoder, discriminator_optimizer, generator_optimizer,
               encoder_optimizer, batch_size=32, latent_dim=512):
    noise = tf.random.normal([batch_size, latent_dim])
    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape, tf.GradientTape() as enc_tape:

      generated_images = generator(noise, training=True)
      real_classification = discriminator(images, training=True)
      fake_classification = discriminator(generated_images, training=True)

      z_hat = encoder(generated_images, training=True)
      reconstructed_images = generator(z_hat, training=True)

      gen_loss = generator_loss(fake_classification)
      disc_loss = discriminator_loss(real_classification, fake_classification)
      enc_loss = encoder_loss(generated_images, reconstructed_images)

    gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
    gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)
    gradients_of_encoder = enc_tape.gradient(enc_loss, encoder.trainable_variables)

    generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))
    encoder_optimizer.apply_gradients(zip(gradients_of_encoder, encoder.trainable_variables))
    return gen_loss, disc_loss, enc_loss
