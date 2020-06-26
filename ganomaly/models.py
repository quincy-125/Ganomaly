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
def define_discriminator(latent_dim=None, input_shape=(4, 4, 3), style='discriminator'):
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
        out_class = Dense(1, activation=None, kernel_initializer=init, kernel_constraint=const, name='d_out')(d)
        name='disc'
    else:
        out_class = Dense(latent_dim, kernel_initializer=init, kernel_constraint=const, name='e_out')(d)
        name='enc'
    model = Model(in_image, out_class, name=name)
    return model


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
    merged = WeightedSum(name='g_out')([out_image2, out_image])
    # define model
    model2 = Model(old_model.input, merged)
    return [model1, model2]


# define generator models
# noinspection DuplicatedCode
def define_generator(latent_dim, in_dim=4):
    # base model latent input
    in_latent = Input(shape=(latent_dim))
    g = PixelNormalization()(in_latent)
    # linear scale up to activation maps
    g = Dense(latent_dim * in_dim * in_dim, kernel_initializer=init, kernel_constraint=const)(g)
    g = Reshape((in_dim, in_dim, latent_dim))(g)
    # conv 4x4, input block
    g = WScaleLayer(latent_dim, (3, 3), padding='same', kernel_initializer=init, kernel_constraint=const)(g)
    g = PixelNormalization()(g)
    g = LeakyReLU(alpha=0.2)(g)
    # conv 3x3
    g = WScaleLayer(latent_dim, (3, 3), padding='same', kernel_initializer=init, kernel_constraint=const)(g)
    g = PixelNormalization()(g)
    g = LeakyReLU(alpha=0.2)(g)
    # conv 1x1, output block
    out_image = Conv2D(3, (1, 1), padding='same', activation='tanh', kernel_initializer=init, kernel_constraint=const,
                       name='g_out')(g)
    model = Model(in_latent, out_image, name='gen')
    return model


def build_integrated_model(img_size, latent_dim, color_channels=3):
    input_shape = (img_size,img_size,color_channels)
    G = define_generator(latent_dim)
    D = define_discriminator(latent_dim=latent_dim, input_shape=input_shape, style='discriminator')
    E = define_discriminator(latent_dim=latent_dim, input_shape=input_shape,       style='encoder')
    return G, D, E


class Generator(tf.keras.layers.Layer):

    def __init__(self, latent_dim, name='gen', **kwargs):
        super(Generator, self).__init__(name=name, **kwargs)
        self.generator = define_generator(latent_dim)

    def call(self, latent_vector, training=None):
        return self.generator(latent_vector)


class Discriminator(tf.keras.layers.Layer):
    def __init__(self, latent_dim, input_shape, name='disc', **kwargs):
        super(Discriminator, self).__init__(name=name, **kwargs)
        self.discriminator = define_discriminator(latent_dim=latent_dim, input_shape=input_shape, style='discriminator')

    def call(self, input, training=None):
        return self.discriminator(input)

class Encoder(tf.keras.layers.Layer):
    def __init__(self, latent_dim, input_shape, name='enc', **kwargs):
        super(Encoder, self).__init__(name=name, **kwargs)
        self.encoder = define_discriminator(latent_dim=latent_dim, input_shape=input_shape, style='encoder')

    def call(self, image, training=None):
        encoded_images = self.encoder(image)
        recoded_images = self.generator(encoded_images)
        return recoded_images


class AnnoGAN(tf.keras.Model):
    # https://keras.io/examples/generative/dcgan_overriding_train_step/
    # https://www.tensorflow.org/guide/keras/custom_layers_and_models
    # https: // www.tensorflow.org / guide / keras / customizing_what_happens_in_fit
    def __init__(self, img_size, latent_dim, loss_fn, color_channels=3, name='annogan', **kwargs):
        super(AnnoGAN, self).__init__(name=name, **kwargs)
        input_shape = (img_size,img_size, color_channels)
        self.discriminator = Discriminator(latent_dim, input_shape)
        self.encoder = Encoder(latent_dim, input_shape)
        self.generator = Generator(latent_dim)
        self.loss_fn = loss_fn
        self.latent_dim = latent_dim

    def call(self, real_images, training=None):

        if isinstance(real_images, tuple):
            real_images = real_images[0]

        loss_fn = self.loss_fn
        # Sample random points in the latent space
        batch_size = tf.shape(real_images)[0]
        random_latent_vectors = tf.random.normal(shape=(self.latent_dim, batch_size))

        # Get fake images
        generated_images = self.generator(random_latent_vectors)

        # Combine them with real images
        combined_images = tf.concat([generated_images, real_images], axis=0)

        # Assemble labels discriminating real from fake images
        labels = tf.concat([tf.ones((batch_size, 1)), tf.zeros((batch_size, 1))], axis=0)
        # Add random noise to the labels - important trick!
        labels += 0.05 * tf.random.uniform(tf.shape(labels))

        # Make and encoder loop
        recoded_images = self.encoder(real_images)

        # Train the discriminator
        with tf.GradientTape() as tape:
            predictions = self.discriminator(combined_images)
            d_loss = loss_fn['d_out'](labels, predictions)
        grads = tape.gradient(d_loss, self.discriminator.trainable_weights)
        self.d_optimizer.apply_gradients(
            zip(grads, self.discriminator.trainable_weights)
        )

        # Sample random points in the latent space
        random_latent_vectors = tf.random.normal(shape=(batch_size, self.latent_dim))

        # Assemble labels that say "all real images"
        misleading_labels = tf.zeros((batch_size, 1))

        # Train the generator (note that we should *not* update the weights
        # of the discriminator)!
        with tf.GradientTape() as tape:
            predictions = self.discriminator(self.generator(random_latent_vectors))
            _loss =loss_fn['g_out']
            g_loss = _loss(misleading_labels, predictions)
        grads = tape.gradient(g_loss, self.generator.trainable_weights)
        self.g_optimizer.apply_gradients(zip(grads, self.generator.trainable_weights))

        # Train the encoder, but dont' update the weights of either the generator or discriminator
        with tf.GradientTape() as tape:
            _loss = loss_fn['e_out']
            e_loss = _loss(real_images, recoded_images)
        grads = tape.gradient(e_loss, self.encoder.trainable_weights)
        self.e_optimizer.apply_gradients(zip(grads, self.encoder.trainable_weights))

        return {"d_loss": d_loss, "g_loss": g_loss, "e_loss": e_loss}