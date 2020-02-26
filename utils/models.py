import tensorflow as tf


def create_networks(z_dim):
    G = build_generator(z_dim)
    D = build_discriminator()
    E = build_encoder(z_dim)

    G.summary()
    D.summary()
    E.summary()


# ==============================================================================
# =                                  networks                                  =
# ==============================================================================
def pixel_norm(z, epsilon=1e-8):
    return z * tf.math.rsqrt(tf.reduce_mean(tf.math.square(z), axis=1, keepdims=True) + epsilon)

def build_generator(z_dim, epsilon=1e-8):
    """
    Build a 4x4 generator from a

    :param z_dim: latent vector size
    :param max_dim: number of nodes in first layer
    :param normalize_latents: perform pixel normalization on z first
    :param epsilon: prevents agains divide by 0 errors
    :return: generator of size (None, 3, 4, 4) # Channels first
    """

    in_latents = tf.keras.Input(shape=(z_dim), name='z')
    model = pixel_norm(in_latents, epsilon=epsilon)
    model = tf.keras.layers.Dense(512 * 4 * 4)(model)
    model = tf.reshape(model, (-1, 512, 4, 4), name='4x4/Dense')

    model = tf.keras.layers.Conv2D(512, 4, padding='same', data_format='channels_first')(model)
    model = pixel_norm(model, epsilon=epsilon)
    model = tf.keras.layers.LeakyReLU(alpha=0.2)(model)

    # To RGB
    model = tf.keras.layers.Conv2D(3, 4, padding='same', data_format='channels_first')(model)

    generator = tf.keras.Model(in_latents, model)
    generator.summary()
    return generator