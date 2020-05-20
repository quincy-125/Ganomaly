from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Reshape
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import UpSampling2D
from tensorflow.keras.layers import AveragePooling2D
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras import backend
from ganomaly.layers import WeightedSum, PixelNormalization, MinibatchStdev
from ganomaly.layers import WScaleConv2DLayer as WScaleLayer
import tensorflow as tf

init = tf.keras.initializers.VarianceScaling()  # weight initialization
# const = tf.keras.constraints.MinMaxNorm(min_value=-1.0, max_value=1.0)          # weight constraint
const = None
LATENT_DEPTHS = {'4': 512, '8': 512, '16': 512, '32': 512, '64': 256, '128': 128, '256': 64, '512': 32, '1024': 16}


# update the alpha value on each instance of WeightedSum
def update_fadein(models, alpha=0):
    # update the alpha for each model
    for model in models:
        for layer in model.layers:
            if isinstance(layer, WeightedSum):
                backend.set_value(layer.alpha, alpha)


# add a discriminator block
# noinspection DuplicatedCode
def add_discriminator_block(old_model, n_input_layers=3):
    # get shape of existing model
    in_shape = list(old_model.input.shape)
    # define new input shape as double the size
    input_shape = (in_shape[-2] * 2, in_shape[-2] * 2, in_shape[-1])
    in_image = Input(shape=input_shape)
    latent_depth = int(LATENT_DEPTHS[str(input_shape[-2])])
    latent_depth_next = int(LATENT_DEPTHS[str(in_shape[-2])])
    # define new input processing layer
    # FROM RGB
    d = WScaleLayer(latent_depth, (1, 1), padding='same', kernel_initializer=init, kernel_constraint=const)(in_image)
    d = LeakyReLU(alpha=0.2)(d)
    # define new block
    d = WScaleLayer(latent_depth, (3, 3), padding='same', kernel_initializer=init, kernel_constraint=const)(d)
    d = LeakyReLU(alpha=0.2)(d)
    d = WScaleLayer(latent_depth_next, (3, 3), padding='same', kernel_initializer=init, kernel_constraint=const)(d)
    d = LeakyReLU(alpha=0.2)(d)
    d = AveragePooling2D()(d)
    block_new = d

    # skip the input, 1x1 and activation for the old model
    for i in range(n_input_layers, len(old_model.layers)):
        d = old_model.layers[i](d)

    model1 = Model(in_image, d)

    # Downsample and convert from RGB
    downsample = AveragePooling2D()(in_image)
    block_old = old_model.layers[1](downsample)
    block_old = old_model.layers[2](block_old)
    d = WeightedSum()([block_old, block_new])

    # skip the input, 1x1 and activation for the old model
    for i in range(n_input_layers, len(old_model.layers)):
        d = old_model.layers[i](d)

    model2 = Model(in_image, d)
    return [model1, model2]


# define the discriminator models for each image resolution
# noinspection DuplicatedCode
def define_discriminator(n_blocks, latent_dim=None, input_shape=(4, 4, 3), style='discriminator'):
    latent_depth = int(LATENT_DEPTHS[str(input_shape[-2])])
    model_list = list()
    # base model input
    in_image = Input(shape=input_shape)
    # conv 1x1
    d = WScaleLayer(latent_depth, (1, 1), padding='same', kernel_initializer=init, kernel_constraint=const)(in_image)
    d = LeakyReLU(alpha=0.2)(d)
    # conv 3x3 (output block)
    d = WScaleLayer(latent_depth, (3, 3), padding='same', kernel_initializer=init, kernel_constraint=const)(d)
    d = LeakyReLU(alpha=0.2)(d)
    # conv 4x4
    d = WScaleLayer(latent_depth, (4, 4), padding='same', kernel_initializer=init, kernel_constraint=const)(d)
    # d = WScaleLayer()(d)
    d = LeakyReLU(alpha=0.2)(d)
    # dense output layer
    d = MinibatchStdev()(d)
    d = Flatten()(d)
    if style == 'discriminator':
        out_class = Dense(1, activation=None, kernel_initializer=init, kernel_constraint=const)(d)
    else:
        out_class = Dense(latent_dim)(d)
    # define model
    model = Model(in_image, out_class)

    # store model
    model_list.append([model, model])
    # create submodels
    for i in range(1, n_blocks):
        # get prior model without the fade-on
        old_model = model_list[i - 1][0]
        # create new model for next resolution
        models = add_discriminator_block(old_model)
        # store model
        model_list.append(models)
    return model_list


# add a generator block
# noinspection DuplicatedCode
def add_generator_block(old_model):
    # get the end of the last block
    block_end = old_model.layers[-2].output
    # upsample, and define new block
    upsampling = UpSampling2D()(block_end)
    latent_depth = int(LATENT_DEPTHS[str(upsampling.shape[-2])])
    g = WScaleLayer(latent_depth, (3, 3), padding='same', kernel_initializer=init, kernel_constraint=const)(upsampling)
    g = PixelNormalization()(g)
    g = LeakyReLU(alpha=0.2)(g)
    g = WScaleLayer(latent_depth, (3, 3), padding='same', kernel_initializer=init, kernel_constraint=const)(g)
    g = PixelNormalization()(g)
    g = LeakyReLU(alpha=0.2)(g)
    # add new output layer
    out_image = Conv2D(3, (1, 1), padding='same', kernel_initializer=init, kernel_constraint=const)(g)
    # define model
    model1 = Model(old_model.input, out_image)
    # get the output layer from old model
    out_old = old_model.layers[-1]
    # connect the upsampling to the old output layer
    out_image2 = out_old(upsampling)
    # define new output image as the weighted sum of the old and new models
    merged = WeightedSum()([out_image2, out_image])
    # define model
    model2 = Model(old_model.input, merged)
    return [model1, model2]


# define generator models
# noinspection DuplicatedCode
def define_generator(latent_dim, n_blocks, in_dim=4):
    latent_depth = int(LATENT_DEPTHS['4'])

    model_list = list()
    # base model latent input
    in_latent = Input(shape=(latent_dim,))
    g = PixelNormalization()(in_latent)
    # linear scale up to activation maps
    g = Dense(latent_depth * in_dim * in_dim, kernel_initializer=init, kernel_constraint=const)(g)
    g = Reshape((in_dim, in_dim, latent_depth))(g)
    # conv 4x4, input block
    g = WScaleLayer(latent_depth, (3, 3), padding='same', kernel_initializer=init, kernel_constraint=const)(g)
    g = PixelNormalization()(g)
    g = LeakyReLU(alpha=0.2)(g)
    # conv 3x3
    g = WScaleLayer(latent_depth, (3, 3), padding='same', kernel_initializer=init, kernel_constraint=const)(g)
    g = PixelNormalization()(g)
    g = LeakyReLU(alpha=0.2)(g)
    # conv 1x1, output block
    out_image = Conv2D(3, (1, 1), padding='same', activation='tanh', kernel_initializer=init, kernel_constraint=const)(
        g)

    # define model
    model = Model(in_latent, out_image)
    # store model
    model_list.append([model, model])
    # create submodels
    for i in range(1, n_blocks):
        # get prior model without the fade-on
        old_model = model_list[i - 1][0]
        # create new model for next resolution
        models = add_generator_block(old_model)
        # store model
        model_list.append(models)
    return model_list
