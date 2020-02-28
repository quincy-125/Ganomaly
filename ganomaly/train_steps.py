import tensorflow as tf
from tensorflow.keras import backend


# train the generator and discriminator
def train(g_models, d_models, e_models, gan_models, dataset, latent_dim, e_norm, e_fadein, n_batch):
	# fit the baseline model
	g_normal, d_normal, e_model, gan_normal = g_models[0][0], d_models[0][0], e_model[0][0], gan_models[0][0]
	# scale dataset to appropriate size
	gen_shape = g_normal.output_shape
	scaled_data = scale_dataset(dataset, gen_shape[1:])
	print('Scaled Data', scaled_data.shape)
	# train normal or straight-through models
	train_epochs(g_normal, d_normal, gan_normal, scaled_data, e_norm[0], n_batch[0])
	# process each level of growth
	for i in range(1, len(g_models)):
		# retrieve models for this level of growth
		[g_normal, g_fadein] = g_models[i]
		[d_normal, d_fadein] = d_models[i]
		[gan_normal, gan_fadein] = gan_models[i]
		[e_normal, e_fadein] = e_models[i]
		# scale dataset to appropriate size
		gen_shape = g_normal.output_shape
		scaled_data = scale_dataset(dataset, gen_shape[1:])
		print('Scaled Data', scaled_data.shape)
		# train fade-in models for next level of growth
		train_epochs(g_fadein, d_fadein, gan_fadein, e_fadein, scaled_data, e_fadein[i], n_batch[i], True)
		# train normal or straight-through models
		train_epochs(g_normal, d_normal, gan_normal, e_normal, scaled_data, e_norm[i], n_batch[i])

# train a generator and discriminator
def train_epochs(g_model, d_model, gan_model, e_model, dataset, n_epochs, n_batch, fadein=False):
	# calculate the number of batches per training epoch
	bat_per_epo = int(dataset.shape[0] / n_batch)
	# calculate the number of training iterations
	n_steps = bat_per_epo * n_epochs
	# calculate the size of half a batch of samples
	half_batch = int(n_batch / 2)
	# manually enumerate epochs
	for i in range(n_steps):
		# update alpha for all WeightedSum layers when fading in new blocks
		if fadein:
			update_fadein([g_model, d_model, gan_model, e_model], i, n_steps)
		# prepare real and fake samples
		z_input = generate_latent_points(latent_dim, n_samples)
		X_real, y_real = generate_real_samples(dataset, half_batch)
		X_fake, y_fake = z_to_img(g_model, n_samples, z_input)
		z_hat = img_to_z(e_model, X_real)
		X_reconstructed, _ = z_to_img(g_model, n_samples, z_hat)

		# update discriminator model
		d_loss1 = d_model.train_on_batch(X_real, y_real)
		d_loss2 = d_model.train_on_batch(X_fake, y_fake)

		# update the generator via the discriminator's error
		y_real2 = ones((n_batch, 1))
		g_loss = gan_model.train_on_batch(z_input, y_real2)

		# update the encoder based on the real z
		e_loss = e_model.train_on_batch(X_reconstructed, z_input)

		# summarize loss on this batch
		if step % 100 == 0:
			tf.summary.scalar('d_loss_real', d_loss1)
			tf.summary.scalar('d_loss_fake', d_loss2)
			tf.summary.scalar('g_loss', g_loss)
			tf.summary.scalar('e_loss', e_loss)


# update the alpha value on each instance of WeightedSum
def update_fadein(models, step, n_steps):
	# calculate current alpha (linear from 0 to 1)
	alpha = step / float(n_steps - 1)
	# update the alpha for each model
	for model in models:
		for layer in model.layers:
			if isinstance(layer, WeightedSum):
				backend.set_value(layer.alpha, alpha)


# generate points in latent space as input for the generator
def generate_latent_points(latent_dim, n_samples):
	# generate points in the latent space
	x_input = randn(latent_dim * n_samples)
	# reshape into a batch of inputs for the network
	x_input = x_input.reshape(n_samples, latent_dim)
	return x_input


# use the generator to generate n fake examples, with class labels
def z_to_img(generator, n_samples, x_input):
	# predict outputs
	X = generator.predict(x_input)
	# create class labels
	y = -ones((n_samples, 1))
	return X, y

def img_to_z(encoder, img):
	# generate points in latent space
	X = encoder.predict(img)
	return X


