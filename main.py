from ganomaly.datasets import get_dataset
import tensorflow as tf
from ganomaly.models import *
from ganomaly.steps import *

batch_size = 32
input_dir = 'D:\PyCharm_Projects\Ganomaly\data'
img_size = 4
EPOCHS = 1
latent_dim = 512
result_dir = 'results'
scaled_data = get_dataset(input_dir, batch_size, img_size, epochs=EPOCHS)

discriminator = define_discriminator()
encoder = define_discriminator(style='encoder', latent_dim=latent_dim)
generator = define_generator(latent_dim, in_dim=4)


generator_optimizer = tf.keras.optimizers.Adam(1e-4)
discriminator_optimizer = tf.keras.optimizers.Adam(1e-4)
encoder_optimizer = tf.keras.optimizers.Adam(1e-4)

for i, images in enumerate(scaled_data):
    gen_loss, disc_loss, enc_loss = train_step(images, discriminator, generator, encoder,
                                               discriminator_optimizer, generator_optimizer, encoder_optimizer,
                                               batch_size=batch_size, latent_dim=latent_dim)
    if i % 100 == 0:
        print('Step {}:\tgen_loss: {}, disc_loss: {}, enc_loss: {}'.format(i, gen_loss, disc_loss, enc_loss))
