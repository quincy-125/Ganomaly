from .layers import *
from tensorflow.keras.constraints import MinMaxNorm
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Sequential
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Reshape
from tensorflow.keras.layers import Conv2D, Conv2DTranspose
from tensorflow.keras.layers import UpSampling2D
from tensorflow.keras.layers import AveragePooling2D
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.initializers import RandomNormal
from numpy.random import randn
from numpy.random import randint

def define_discriminator(input_shape=(4, 4, 3), style='discriminator', latent_dim=512):
    # weight initialization
    init = RandomNormal(stddev=0.02)
    # weight constraint
    const = MinMaxNorm(min_value=-1.0, max_value=1.0)
    model_list = list()
    # base model input
    in_image = Input(shape=input_shape)
    # conv 1x1
    d = Conv2D(128, (1, 1), padding='same', kernel_initializer=init, kernel_constraint=const)(in_image)
    d = LeakyReLU(alpha=0.2)(d)
    # conv 3x3 (output block)
    d = MinibatchStdev()(d)
    d = Conv2D(128, (3, 3), padding='same', kernel_initializer=init, kernel_constraint=const)(d)
    d = LeakyReLU(alpha=0.2)(d)
    # conv 4x4
    d = Conv2D(128, (4, 4), padding='same', kernel_initializer=init, kernel_constraint=const)(d)
    d = LeakyReLU(alpha=0.2)(d)
    # dense output layer
    d = Flatten()(d)
    if style == 'discriminator':
        out_class = Dense(1)(d)
    else:
        out_class = Dense(latent_dim)(d)
    # define model
    model = Model(in_image, out_class)
    return model


def define_generator(latent_dim, in_dim=4):
    init = RandomNormal(stddev=0.02)
    # weight constraint
    const = MinMaxNorm(min_value=-1.0, max_value=1.0)
    """
    model = Sequential([
        Dense(128 * in_dim * in_dim, kernel_initializer=init, kernel_constraint=const, input_shape=(latent_dim,)),
        Reshape((in_dim, in_dim, 128)),
        Conv2D(128, (3, 3), padding='same', kernel_initializer=init, kernel_constraint=const),
        PixelNormalization(),
        LeakyReLU(alpha=0.2),
        Conv2D(3, (1, 1), padding='same', kernel_initializer=init, kernel_constraint=const)
    ])
    """
    model = Sequential()
    model.add(Dense(latent_dim * in_dim * in_dim, kernel_initializer=init, kernel_constraint=const, input_shape=(latent_dim,)))
    model.add(Reshape((in_dim, in_dim, latent_dim)))
    model.add(Conv2D(128, (4, 4), padding='same', kernel_initializer=init, kernel_constraint=const))
    model.add(PixelNormalization())
    model.add(LeakyReLU(alpha=0.2))
    model.add(Conv2D(128, (3, 3), padding='same', kernel_initializer=init, kernel_constraint=const))
    model.add(Conv2D(3, (1, 1), padding='same', kernel_initializer=init, kernel_constraint=const))
    return model


# generate points in latent space as input for the generator
def generate_latent_points(latent_dim, n_samples):
	# generate points in the latent space
	x_input = randn(latent_dim * n_samples)
	# reshape into a batch of inputs for the network
	x_input = x_input.reshape(n_samples, latent_dim)
	return x_input

# use the generator to generate n fake examples, with class labels
def generate_fake_samples(generator, latent_dim, n_samples):
	# generate points in latent space
	x_input = generate_latent_points(latent_dim, n_samples)
	# predict outputs
	X = generator.predict(x_input)
	# create class labels
	y = -ones((n_samples, 1))
	return X, y