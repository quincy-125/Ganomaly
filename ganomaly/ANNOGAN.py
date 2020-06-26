import tensorflow as tf
from ganomaly.losses import interpolate_imgs, gradient_penalty_loss, discriminator_loss

class ANNOGAN(tf.keras.Model):
    def __init__(self, G, D, E, latent_dim):
        super(ANNOGAN, self).__init__()
        self.G = G
        self.D = D
        self.E = E
        self.latent_dim = latent_dim
        self.generator_optimizer = None
        self.discriminator_optimizer = None
        self.encoder_optimizer = None

    def compile(self, generator_optimizer, discriminator_optimizer, encoder_optimizer, **kwargs):
        super(ANNOGAN, self).compile()
        self.generator_optimizer = generator_optimizer
        self.discriminator_optimizer = discriminator_optimizer
        self.encoder_optimizer = encoder_optimizer

    def train_step(self, real_images, _lambda=10):
        # Sample random points in the latent space
        batch_size = tf.shape(real_images)[0]
        random_latent_vectors = tf.random.normal(shape=(self.latent_dim, batch_size))

        #############################################################################
        ## Train Generator
        #############################################################################

        with tf.GradientTape() as gen_tape:
            generated_imgs = self.G(random_latent_vectors, training=True)
            generated_output = self.D(generated_imgs, training=False)
            gen_loss = -tf.math.reduce_mean(generated_output)

        grad_gen = gen_tape.gradient(gen_loss, self.G.trainable_variables)
        self.generator_optimizer.apply_gradients(zip(grad_gen, self.G.trainable_variables))

        #############################################################################
        ## Train Encoder
        #############################################################################
        with tf.GradientTape() as enc_tape:
            z_hat = self.E(real_images, training=True)
            images_reconstructed = self.G(z_hat, training=False)
            enc_loss = tf.math.reduce_mean(tf.abs(real_images - images_reconstructed))

        grad_enc = enc_tape.gradient(enc_loss, self.E.trainable_variables)
        self.encoder_optimizer.apply_gradients(zip(grad_enc, self.E.trainable_variables))

        #############################################################################
        ## Train Discriminator
        #############################################################################
        with tf.GradientTape() as disc_tape:
            generated_imgs = self.G(random_latent_vectors, training=False)
            generated_output = self.D(generated_imgs, training=True)
            real_output = self.D(real_images, training=True)

            with tf.GradientTape() as interp_tape:
                interp_tape.watch(real_output)
                interp_tape.watch(generated_output)
                interp_tape.watch(generated_imgs)

                interpolated_img = interpolate_imgs(real_images, generated_imgs, batch_size)
                validity_interpolated = self.D(interpolated_img, training=True)
                real_score, fake_score, disc_loss = discriminator_loss(real_output, generated_output)

                grads = interp_tape.gradient(validity_interpolated, interpolated_img)
            grad_penalty = gradient_penalty_loss(grads)
            disc_loss += 0.5 * _lambda * grad_penalty

        grad_disc = disc_tape.gradient(disc_loss, self.D.trainable_variables)
        self.discriminator_optimizer.apply_gradients(zip(grad_disc, self.D.trainable_variables))

        return {'gen_loss': gen_loss, 'enc_loss': enc_loss, 'disc_loss': disc_loss}
