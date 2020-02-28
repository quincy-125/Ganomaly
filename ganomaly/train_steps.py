
@tf.function
def train_step(images,
			   G, D, E,
			   G_optimizer, D_optimizer, E_optimizer,
			   g_loss_fn, d_loss_fn, e_loss_fn,
			   z_dim=512, batch_size=16):
	z = tf.random.normal(shape=(batch_size, 1, 1, z_dim))
	with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape, tf.GradientTape() as enc_tape:
		x_fake = G(z, training=True)
		x_fake_d_logit = D(x_fake, training=True)
		x_real_d_logit = D(x_real, training=True)
		x_fake_g_img = G(x_fake_d_logit, training=True)

		G_loss = g_loss_fn(x_fake_d_logit)
		D_loss = d_loss_fn(x_real_d_logit, x_fake_d_logit)
		E_loss = e_loss_fn(x_real, x_fake_g_img)

	G_grad = gen_tape.gradient(G_loss, G.trainable_variables)
	G_optimizer.apply_gradients(zip(G_grad, G.trainable_variables))
	D_grad = disc_tape.gradient(D_loss, D.trainable_variables)
	D_optimizer.apply_gradients(zip(D_grad, D.trainable_variables))
	E_grad = enc_tape.gradient(E_loss, E.trainable_variables)
	E_optimizer.apply_gradients(zip(E_grad, E.trainable_variables))
