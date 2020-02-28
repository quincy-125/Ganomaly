import tensorflow as tf
from .layers import WeightedSum, PixelNormalization, MinibatchStdev
from tensorflow.keras.constraints import MinMaxNorm


def create_networks(z_dim, limit):
    G = build_generator(z_dim)
    D = build_descriminator(model_type='discriminator', limit=z_dim)
    E = build_descriminator(model_type='encoder', limit=z_dim)
    return G, D, E

def grow_networks(G, D, E, limit=512):
    G_new, G_fade = upsample_generator(G, limit=limit)
    D_new, D_fade = grow_discriminator(D, limit=limit)
    E_new, E_fade = grow_discriminator(E, limit=limit)

    return G_new, G_fade, D_new, D_fade, E_new, E_fade

def lerp_clip(a, b, t): return a + (b - a) * tf.clip_by_value(tf.cast(t, dtype='float32'), 0.0, 1.0)

# ==============================================================================
# =                                  generator                                 =
# ==============================================================================
def build_generator(z_dim,
                    resolution=32,
                    lod_in=0,
                    FEATURE_SIZE = {'1024': 16, '512': 32, '256': 64, '128': 128, '64': 256, '32': 512, '16': 512, '8':512, '4': 512}
                    ):
    """
	Build a 4x4 generator from a z_dim vector

	:param z_dim: latent vector size
	:param resolution: Output resolution
	:param lod_in: lod initial resolution
	:return: tensor graph of model before adding RGB
	"""
    resolution_log2 = int(np.log2(resolution))  # number of log2's
    assert resolution == 2 ** resolution_log2 and resolution >= 4

    in_latents = tf.keras.Input(shape=z_dim, name='z')

    # Building blocks.
    def block(in_latents, res):  # res = 2..resolution_log2
        limit = FEATURE_SIZE[str(2 ** res)]
        if res == 2:  # 4x4
            x = PixelNormalization()(in_latents)
            # Dense
            x = tf.keras.layers.Dense(limit * 4 * 4, name='4x4/Dense')(x)
            x = tf.reshape(x, (-1, 4, 4, limit))
            # Conv
            x = tf.keras.layers.Conv2D(limit, 3, padding='same',
                                           kernel_constraint=MinMaxNorm(min_value=-1., max_value=1.),
                                           activation=None,
                                           use_bias=False)(x)
            x = PixelNormalization()(x)
            x = tf.keras.layers.LeakyReLU(alpha=0.2)(x)

        else:  # 8x8 and up
            x = tf.keras.layers.UpSampling2D(2)(in_latents)
            # conv 3x3
            x = tf.keras.layers.Conv2D(limit, 3, padding='same',
                                           kernel_constraint=MinMaxNorm(min_value=-1., max_value=1.),
                                           activation=None,
                                           use_bias=False,
                                       name='Conv0_' + str(2**res) + 'x' + str(2**res))(x)
            x = PixelNormalization()(x)
            x = tf.keras.layers.LeakyReLU(alpha=0.2)(x)
            # conv 3x3
            x = tf.keras.layers.Conv2D(limit, 3, padding='same',
                                           kernel_constraint=MinMaxNorm(min_value=-1., max_value=1.),
                                           activation=None,
                                           use_bias=False,
                                       name='Conv1_' + str(2**res) + 'x' + str(2**res))(x)
            x = PixelNormalization()(x)
            x = tf.keras.layers.LeakyReLU(alpha=0.2)(x)

        return x

    def torgb(x, res):  # res = 2..resolution_log2

        x = tf.keras.layers.Conv2D(3, 3, padding='same',
                                       kernel_constraint=MinMaxNorm(min_value=-1., max_value=1.),
                                       activation=None,
                                       use_bias=True,
                                name='RGB_' + str(2**res) + 'x' + str(2**res))(x)
        return x

    x = block(in_latents, 2)
    images_out = torgb(x, 2)

    for res in range(3, resolution_log2 + 1):
        lod = resolution_log2 - res
        x = block(x, res)
        img = torgb(x, res)
        images_out = tf.keras.layers.UpSampling2D(2)(images_out)
        images_out = lerp_clip(img, images_out, lod_in - lod)

    images_out = tf.identity(images_out, name='images_out')
    model = tf.keras.Model(in_latents, images_out)
    return model

# ==============================================================================
# =                                  discriminator                             =
# ==============================================================================

def build_descriminator(model_type='discriminator', limit=1024):
    """
    Build a 4x4 discriminator

    :param limit:
    :return:
    """
    image_in = tf.keras.Input(shape=[4, 4, 3], name='image_in')
    # from RGB
    model = tf.keras.layers.Conv2D(limit, 1,
                                   padding='same',
                                   kernel_constraint=MinMaxNorm(min_value=-1., max_value=1.),
                                   activation=None,
                                   use_bias=False)(image_in)
    model = tf.keras.layers.LeakyReLU(alpha=0.2)(model)
    model = MinibatchStdev()(model)

    # conv 4x4
    model = tf.keras.layers.Conv2D(limit, 4, padding='same',
                                   kernel_constraint=MinMaxNorm(min_value=-1., max_value=1.),
                                   activation=None,
                                   use_bias=False)(model)
    model = tf.keras.layers.LeakyReLU(alpha=0.2)(model)
    model = MinibatchStdev()(model)

    # conv 3x3
    model = tf.keras.layers.Conv2D(limit, 3, padding='same',
                                   kernel_constraint=MinMaxNorm(min_value=-1., max_value=1.),
                                   activation=None,
                                   use_bias=False)(model)
    model = tf.keras.layers.LeakyReLU(alpha=0.2)(model)
    model = MinibatchStdev()(model)

    model = tf.keras.layers.Flatten()(model)
    if model_type == 'discriminator':
        prediction = tf.keras.layers.Dense(1)(model)
    else:
        prediction = tf.keras.layers.Dense(z_dim)(model)
    discriminator = tf.keras.Model(image_in, prediction)

    return discriminator  # Y or z_hat


def grow_discriminator(initial_model, limit=512):

new_input_shape = (initial_model.input.shape[-2]*2, initial_model.input.shape[-2]*2, 3)
in_image = tf.keras.Input(shape=new_input_shape, name='image_in_' + str(new_input_shape[0]))

# conv 3x3
model = tf.keras.layers.Conv2D(limit, 3, padding='same',
                               kernel_constraint=MinMaxNorm(min_value=-1., max_value=1.),
                               activation=None,
                               use_bias=False)(in_image)
model = tf.keras.layers.LeakyReLU(alpha=0.2)(model)
model = MinibatchStdev()(model)
# conv 3x3
model = tf.keras.layers.Conv2D(limit, 3, padding='same',
                               kernel_constraint=MinMaxNorm(min_value=-1., max_value=1.),
                               activation=None,
                               use_bias=False)(model)
model = tf.keras.layers.LeakyReLU(alpha=0.2)(model)
model = MinibatchStdev()(model)

# Fading
fade_model = tf.keras.layers.Conv2D(limit, 3, padding='same',
                           kernel_constraint=MinMaxNorm(min_value=-1., max_value=1.),
                           activation=None,
                           use_bias=False)(in_image)
fade_model = MinibatchStdev()(fade_model)

fade_model = WeightedSum()([model, fade_model])
####################################################
# Need to somehow not loose the 4x4 when I go to 8x8
####################################################
#  rather than stopping at Fade model, I need to add initial_model
#
####################################################
full_model = tf.keras.Model(in_image, model)
full_input = tf.keras.Input(shape=full_model.input_shape[1:])
tf.keras.Model(full_model, model)
fade_model = tf.keras.Model(in_image, fade_model)

    #only tack on the bottom of the predictor
    if new_input_shape[-2] == 4:
        # new out
        if model_type == 'discriminator':
            prediction = tf.keras.layers.Dense(1)(model)
        else:
            prediction = tf.keras.layers.Dense(z_dim)(model)
        discriminator_new = tf.keras.Model(in_latents, prediction)

        # fade out
        if model_type == 'discriminator':
            prediction = tf.keras.layers.Dense(1)(fade_model)
        else:
            prediction = tf.keras.layers.Dense(z_dim)(fade_model)
        discriminator_fade = tf.keras.Model(in_latents, prediction)

        return discriminator_new, discriminator_fade

    return full_model, fade_model