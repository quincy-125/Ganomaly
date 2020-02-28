import tensorflow as tf
from .layers import WeightedSum, PixelNormalization, MinibatchStdev
from tensorflow.keras.constraints import MinMaxNorm
import numpy as np

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

def build_discriminator(z_dim, resolution=32, model_type='discriminator', lod_in=0,
                    FEATURE_SIZE = {'1024': 16, '512': 32, '256': 64, '128': 128, '64': 256, '32': 512, '16': 512, '8': 512, '4': 512}
                    ):

    resolution_log2 = int(np.log2(resolution))  # number of log2's
    assert resolution == 2 ** resolution_log2 and resolution >= 4
    img = tf.keras.Input(shape=(resolution, resolution, 3), name='img')

    # Building blocks.
    def fromrgb(x, res):  # res = 2..resolution_log2
        x = tf.keras.layers.Conv2D(1, 1, padding='same',
                                   kernel_constraint=MinMaxNorm(min_value=-1., max_value=1.),
                                   activation=None,
                                   use_bias=True,
                                   name='from_RGB_' + str(2 ** res) + 'x' + str(2 ** res))(x)
        return x

    def block(x, res, model_type=model_type, z_dim=z_dim):  # res = 2..resolution_log2
        limit = FEATURE_SIZE[str(2 ** res)]
        if res >= 3:  # 8x8 and up
            x = tf.keras.layers.Conv2D(limit, 3, padding='same',
                                           kernel_constraint=MinMaxNorm(min_value=-1., max_value=1.),
                                           activation=None,
                                           use_bias=False)(x)
            x = tf.keras.layers.LeakyReLU(alpha=0.2)(x)
            x = tf.keras.layers.Conv2D(limit, 3, padding='same',
                                          kernel_constraint=MinMaxNorm(min_value=-1., max_value=1.),
                                          activation=None,
                                          use_bias=False)(x)
            x = tf.keras.layers.LeakyReLU(alpha=0.2)(x)
            x = tf.keras.layers.AveragePooling2D()(x)
        else:  # 4x4
            x = tf.keras.layers.Conv2D(limit, 3, padding='same',
                                           kernel_constraint=MinMaxNorm(min_value=-1., max_value=1.),
                                           activation=None,
                                           use_bias=False,
                                            name='d_Conv0_' + str(2**res) + 'x' + str(2**res))(x)
            x = tf.keras.layers.LeakyReLU(alpha=0.2)(x)

            x = MinibatchStdev()(x)
            x = tf.keras.layers.Flatten()(x)
            x = tf.keras.layers.Dense(z_dim)(x)
            if model_type == 'discriminator':
                x = tf.keras.layers.Dense(1)(x)
        return x
    img_ = img
    x = fromrgb(img_, resolution_log2)
    for res in range(resolution_log2, 2, -1):
        lod = resolution_log2 - res
        x = block(x, res)
        img_ = tf.nn.avg_pool(img_, ksize=2, strides=2, padding='VALID')
        y = fromrgb(img_, res - 1)
        x = lerp_clip(x, y, lod_in - lod)
    combo_out = block(x, 2)

    if model_type == 'discriminator':
        scores_out = tf.identity(combo_out[:, :1], name='scores_out')
        labels_out = tf.identity(combo_out[:, 1:], name='labels_out')
        model = tf.keras.Model(img, [scores_out, labels_out])
    else:
        model = tf.keras.Model(img,combo_out)
    return model