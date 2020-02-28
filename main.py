from ganomaly.models import build_generator, build_discriminator
from ganomaly.train_steps import train_step
from ganomaly.losses import generator_loss, discriminator_loss, encoder_loss
from ganomaly.datasets import get_dataset
import tensorflow as tf
import numpy as np

import os

def run(args):
	z_dim = 1024
	max_parameter_size = 1024
	max_image_size = 1024
	BATCH_SIZE = {'1024': 3, '512': 6, '256': 14, '128': 16, '64': 16, '32': 16, '16': 16, '8':16, '4': 16}
	output_dir = 'results'
	input_dir='data'
	epochs = 5
	number_of_doublings = int(np.log2(max_image_size))
	# Build the Networks
	G = build_generator(z_dim, resolution=max_image_size)
	D = build_discriminator(z_dim, resolution=max_image_size, model_type='discriminator')
	E = build_discriminator(z_dim, resolution=max_image_size, model_type='encoder')
	G_optimizer = tf.keras.optimizers.Adam(1e-4)
	D_optimizer = tf.keras.optimizers.Adam(1e-4)
	E_optimizer = tf.keras.optimizers.Adam(1e-4)

	# Set up checkpoints
	if not os.path.exists(output_dir):
		os.mkdir(output_dir)
	checkpoint_prefix = os.path.join(output_dir, "ckpt")
	checkpoint = tf.train.Checkpoint(G=G, D=D, E=E,
									 G_optimizer=G_optimizer, D_optimizer=D_optimizer, E_optimizer=E_optimizer)

	# Get 4x4 data
	for img_size in [2**x for x in range(2, number_of_doublings + 1)]:
		batch_size = int(BATCH_SIZE[str(img_size)])
		datasets = get_dataset(input_dir, batch_size, img_size, epochs=epochs)
		counter = 0
		for image_batch in datasets:
			counter += batch_size
			print(image_batch)
			train_step(image_batch,
				   G, D, E,
				   G_optimizer, D_optimizer, E_optimizer,
				   generator_loss, discriminator_loss, encoder_loss,
				   z_dim=z_dim, batch_size=batch_size)
			if counter % 1000 == 0:
				checkpoint.save(file_prefix=checkpoint_prefix)

		checkpoint.save(file_prefix=checkpoint_prefix)





args=None

if __name__ == "__main__":
	run(args)
