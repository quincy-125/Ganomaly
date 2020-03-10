from tensorflow.keras.constraints import MinMaxNorm
from tensorflow.keras.initializers import RandomNormal
from tensorflow.keras.layers import Conv2D, Conv2DTranspose
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.layers import Reshape
from tensorflow.keras.models import Model
from tensorflow.keras.models import Sequential

from .layers import *


def define_discriminator(LATENT_DEPTHS=None, input_shape=(4, 4, 3), style='discriminator', latent_dim=512, im_size=4):
    # weight initialization
    init = RandomNormal(stddev=0.02)
    # weight constraint
    const = MinMaxNorm(min_value=-1.0, max_value=1.0)
    # base model input
    in_image = Input(shape=input_shape)
    latent_depth = int(LATENT_DEPTHS[str(im_size)])

    # conv 1x1
    d = Conv2D(latent_depth, (1, 1), padding='same', kernel_initializer=init, kernel_constraint=const,
               data_format='channels_last')(in_image)
    d = LeakyReLU(alpha=0.2)(d)
    # conv 3x3 (output block)
    d = MinibatchStdev()(d)
    d = Conv2D(latent_depth, (3, 3), padding='same', kernel_initializer=init, kernel_constraint=const,
               data_format='channels_last')(d)
    d = LeakyReLU(alpha=0.2)(d)
    # conv 4x4
    d = Conv2D(latent_depth, (4, 4), padding='same', kernel_initializer=init, kernel_constraint=const,
               data_format='channels_last')(d)
    d = LeakyReLU(alpha=0.2)(d)
    # dense output layer
    d = Flatten()(d)
    if style == 'discriminator':
        out_class = Dense(1, activation='sigmoid')(d)
    else:
        out_class = Dense(latent_dim)(d)
    # define model
    model = Model(in_image, out_class)
    return model


def define_generator(latent_dim, LATENT_DEPTHS, in_dim=4, im_size=4, color_channels=3):
    init = RandomNormal(stddev=0.02)
    # weight constraint
    const = MinMaxNorm(min_value=-1.0, max_value=1.0)
    latent_depth = int(LATENT_DEPTHS[str(im_size)])

    model = Sequential()
    model.add(Dense(latent_dim * in_dim * in_dim, kernel_initializer=init, kernel_constraint=const,
                    input_shape=(latent_dim,)))
    model.add(Reshape((in_dim, in_dim, latent_dim)))
    model.add(Conv2D(latent_depth, (4, 4), padding='same', kernel_initializer=init, kernel_constraint=const,
                     data_format='channels_last'))
    model.add(PixelNormalization())
    model.add(LeakyReLU(alpha=0.2))
    model.add(Conv2D(latent_depth, (3, 3), padding='same', kernel_initializer=init, kernel_constraint=const,
                     data_format='channels_last'))
    model.add(Conv2D(color_channels, (1, 1), padding='same', kernel_initializer=init, kernel_constraint=const,
                     data_format='channels_last'))
    return model


def add_discriminator_layer(discriminator, LATENT_DEPTHS, im_size, color_channels=3):
    strides = 2
    # weight initialization
    init = RandomNormal(stddev=0.02)
    # weight constraint
    const = MinMaxNorm(min_value=-1.0, max_value=1.0)
    # Add to the beginning of the model so it assumes a larger input
    # Skip the RGB layers
    grown_discriminator = discriminator.layers[2:]
    latent_depth = int(LATENT_DEPTHS[str(im_size * 2)])
    # base model input
    in_image = Input(shape=(im_size * 2, im_size * 2, color_channels))
    # From RGB
    d = Conv2D(latent_depth, (1, 1), padding='same', kernel_initializer=init, kernel_constraint=const,
               data_format='channels_last')(in_image)
    d = LeakyReLU(alpha=0.2)(d)

    # conv 3x3
    d = Conv2D(latent_depth, (3, 3), padding='same', kernel_initializer=init, kernel_constraint=const, strides=strides,
               data_format='channels_last')(
        d)
    d = LeakyReLU(alpha=0.2)(d)
    # conv 3x3 (output block)
    d = MinibatchStdev()(d)
    # Nex step's latent depth
    latent_depth2 = int(LATENT_DEPTHS[str(im_size)])
    d = Conv2D(latent_depth2, (3, 3), padding='same', kernel_initializer=init, kernel_constraint=const,
               data_format='channels_last')(d)
    d = LeakyReLU(alpha=0.2)(d)

    model2 = Model(in_image, d)
    model = Sequential()
    for layer in model2.layers:
        model.add(layer)
    for layer in grown_discriminator:
        model.add(layer)
    tf.keras.utils.plot_model(model, to_file='new_discriminator.png', show_shapes=True, expand_nested=True)
    return model


def add_generator_layer(generator, LATENT_DEPTHS, im_size):
    # weight initialization
    init = RandomNormal(stddev=0.02)
    # weight constraint
    const = MinMaxNorm(min_value=-1.0, max_value=1.0)

    # Add to the end of the model so it makes a larger output
    latent_depth = int(LATENT_DEPTHS[str(im_size)])

    # Chop off the "toRGB" layer
    num_layers = generator.layers.__len__() - 2  # 1 for being out of index range, 2 for the RGB layer
    model = Sequential()
    for layer in generator.layers[:num_layers]:
        model.add(layer)
    model.add(Conv2DTranspose(latent_depth, (3, 3), strides=2, padding='same', kernel_initializer=init,
                              kernel_constraint=const, data_format='channels_last'))
    model.add(PixelNormalization())
    model.add(LeakyReLU(alpha=0.2))
    tf.keras.utils.plot_model(model, to_file='new_generator.png', show_shapes=True, expand_nested=True)
    return model
