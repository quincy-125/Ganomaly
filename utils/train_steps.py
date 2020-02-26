import tensorflow as tf

@tf.function
def train_G(G, G_optimizer, g_loss_fn, batch_size, z_dim):
    with tf.GradientTape() as t:
        z = tf.random.normal(shape=(batch_size, 1, 1, z_dim))
        x_fake = G(z, training=True)
        x_fake_d_logit = D(x_fake, training=True)

        G_loss = g_loss_fn(x_fake_d_logit)

    G_grad = t.gradient(G_loss, G.trainable_variables)
    G_optimizer.apply_gradients(zip(G_grad, G.trainable_variables))

    return {'g_loss': G_loss, z: z, 'fake_img': x_fake}


@tf.function
def train_D(G, D, D_optimizer, z, x_real, d_loss_fn):
    with tf.GradientTape() as t:
        x_fake = G(z, training=True)
        x_real_d_logit = D(x_real, training=True)
        x_fake_d_logit = D(x_fake, training=True)

        D_loss = d_loss_fn(x_real_d_logit, x_fake_d_logit)

    D_grad = t.gradient(D_loss, D.trainable_variables)
    D_optimizer.apply_gradients(zip(D_grad, D.trainable_variables))

    return {'d_loss': D_loss}


@tf.function
def train_E(E, G, E_optimizer, x_fake, e_loss_fn):
    with tf.GradientTape() as t:
        x_fake_d_logit = E(x_fake, training=True)
        x_fake_g_img = G(x_fake_d_logit, training=True)

        E_loss = e_loss_fn(x_real, x_fake_g_img)

    E_grad = t.gradient(E_loss, E.trainable_variables)
    E_optimizer.apply_gradients(zip(E_grad, E.trainable_variables))

    return {'e_loss': E_loss}
