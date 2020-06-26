import os

import tensorflow as tf
from ganomaly.datasets import get_dataset
from ganomaly.models import build_integrated_model
from ganomaly.calbacks import tbc, cpc, GANMonitor
from ganomaly.ANNOGAN import ANNOGAN

print(tf.__version__)
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
print(tf.config.experimental.list_physical_devices('GPU'))

color_channels = 3
max_images = 4  # Tensorboard
save_frequency = 100  # After how many steps should we save a checkpoint and summary
number_of_images_to_fade = 500000  # How many images should be faded
# noinspection PyPep8,PyPep8,PyPep8,PyPep8
input_dir = 'D:\PyCharm_Projects\Ganomaly\data\\train_data'
# input_dir = '/research/bsi/projects/PI/tertiary/Hart_Steven_m087494/s211408.DigitalPathology/Quincy/Data/train_data'
img_size = 4  # Beginning image size
n_blocks = 8  # how many doublings to do from size 4x4x3
latent_dim = 512  # for encoder
checkpoint_name = 'training_checkpoints'
BATCH_SIZES = {'4': 512, '8': 300, '16': 60, '32': 15, '64': 4, '128': 2, '256': 4, '512': 2, '1024': 4}
EPOCH_SIZES = {'4': 1, '8': 1, '16': 1, '32': 1, '64': 1, '128': 5, '256': 5, '512': 5, '1024': 5}
learning_rate = 0.001
label_flip_rate = 0.05

result_dir_components = ['results', learning_rate, label_flip_rate, latent_dim]
result_dir = '_'.join([str(x) for x in result_dir_components])
checkpoint_prefix = os.path.join(result_dir, checkpoint_name)

# ####################################################################
# Initialize the models
# ####################################################################
batch_size = int(BATCH_SIZES[str(img_size)])
epochs = int(EPOCH_SIZES[str(img_size)])
lod = 2
im_size = int(2 ** lod)

input = get_dataset(input_dir, batch_size, im_size)
print('\nBeginning {}x{}x3'.format(im_size, im_size))

generator, discriminator, encoder = build_integrated_model(img_size, latent_dim, color_channels=3)
print('Models built')

generator_optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate, beta_1=0.0, beta_2=0.99, epsilon=1e-08)
discriminator_optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate, beta_1=0.0, beta_2=0.99, epsilon=1e-08)
encoder_optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate, beta_1=0.0, beta_2=0.99, epsilon=1e-08)


# TODO: Add distributed training
#strategy = tf.distribute.MirroredStrategy()
#with strategy.scope():
annogan = ANNOGAN(generator, discriminator, encoder, latent_dim)
annogan.compile(generator_optimizer, discriminator_optimizer, encoder_optimizer )
annogan.fit(x=input, y=None, verbose=1, callbacks=[tbc, cpc, GANMonitor(num_img=3, latent_dim=latent_dim)])

# TODO: Add progression
