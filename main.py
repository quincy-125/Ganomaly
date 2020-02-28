from ganomaly.models import create_networks, add_rgb
from ganomaly.train_steps import train_step
import tensorflow as tf
import numpy as np
import os

def run(args):
	z_dim = 1024
	max_parameter_size = 1024
	max_image_size = 1024
	BATCH_SIZE = {'1024': 3, '512': 6, '256': 14, '128': 16, '64': 16, '32': 16, '16': 16, '8':16, '4': 16}
	output_dir = 'results'
	epochs = 5
	fade_in = 800	# number of thousand to fade in

	number_of_doublings = np.log2(max_image_size) + 1

	# Build the Networks
	G_new, D_new, E_new = create_networks(max_parameter_size)
	G_optimizer = tf.keras.optimizers.Adam(1e-4)
	D_optimizer = tf.keras.optimizers.Adam(1e-4)
	E_optimizer = tf.keras.optimizers.Adam(1e-4)
	G_fade, D_fade, E_fade = None, None, None

	# Set up checkpoints
	if not os.path.exists(output_dir):
		os.mkdir(output_dir)
	checkpoint_prefix = os.path.join(output_dir, "ckpt")
	checkpoint = tf.train.Checkpoint(G_new=G_new, G_fade=G_fade, D_new=D_new, D_fade=D_fade, E_new=E_new, E_fade=E_fade,
									 G_optimizer=G_optimizer, D_optimizer=D_optimizer, E_optimizer=E_optimizer)

	# Get 4x4 data
	for img_size in [2**x for x in range(2, number_of_doublings):
		datasets = get_dataset(x)
		counter = 0
		for image_batch in datasets:
			counter += image_batch
			if counter < fade_in * 1000 and img_size != 4:
				train_step(image_batch,
						   G_fade, D_fade, E_fade,
						   G_optimizer, D_optimizer, E_optimizer,
						   g_loss_fn, d_loss_fn, e_loss_fn,
						   z_dim=z_dim, batch_size=BATCH_SIZE[img_size])
			else:
				train_step(image_batch,
					   G_new, D_new, E_new,
					   G_optimizer, D_optimizer, E_optimizer,
					   g_loss_fn, d_loss_fn, e_loss_fn,
					   z_dim=z_dim, batch_size=BATCH_SIZE[img_size])
			if counter % 1000 == 0:
				checkpoint.save(file_prefix=checkpoint_prefix)

		# After iterating through 4x4, upgrade the models to the next size
		checkpoint.save(file_prefix=checkpoint_prefix)
		G_new, G_fade, D_new, D_fade, E_new, E_fade = grow_networks(G, D, E, limit=max_parameter_size)





args=None

if __name__ == "__main__":
	run(args)
