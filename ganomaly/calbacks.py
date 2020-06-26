import tensorflow as tf
import os


def tbc(log_dir='logs', hist_freq=0, write_graph=True, write_images=False, update_freq='batch'):
    return tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=hist_freq, write_graph=write_graph,
                                          write_images=write_images, update_freq=update_freq)


def cpc(log_dir='logs', prefix='ckpt', save_weights_only=True):
    return tf.keras.callbacks.ModelCheckpoint(filepath=os.path.join(log_dir, prefix),
                                              save_weights_only=save_weights_only)


class GANMonitor(tf.keras.callbacks.Callback):
    def __init__(self, num_img=3, latent_dim=512, log_dir='logs'):
        super().__init__()
        self.num_img = num_img
        self.latent_dim = latent_dim
        self.log_dir = log_dir

    def on_epoch_end(self, epoch, logs=None):
        random_latent_vectors = tf.random.normal(shape=(self.num_img, self.latent_dim))
        generated_images = self.model.G(random_latent_vectors)
        latent_space = self.model.E(generated_images)
        regenerated_images = self.model.G(latent_space)

        generated_images *= 255
        generated_images.numpy()
        regenerated_images *= 255
        regenerated_images.numpy()

        for i in range(self.num_img):
            img = tf.keras.preprocessing.image.array_to_img(generated_images[i])
            img.save(os.path.join(self.log_dir, "generated_img_" + str(i) + "_" + str(epoch) + ".png"))
            img = tf.keras.preprocessing.image.array_to_img(regenerated_images[i])
            img.save(os.path.join(self.log_dir, "regenerated_img_" + str(i) + "_" + str(epoch) + ".png"))


class AlphaUpdate(tf.keras.callbacks.Callback):
    def __init__(self, batch_size, num_imgs_to_fade=500000, **kwargs):
        super().__init__()
        self.batch_size = batch_size
        self.num_imgs_to_fade = num_imgs_to_fade

    def on_batch_end(self, batch, **kwargs):
        num_imgs_seen = batch * self.batch_size
        # Only update if I am on an image bigger than 4x4 (which has self.model.G_fade=None)
        if num_imgs_seen > self.num_imgs_to_fade and self.model.G_fade is not None:
            self.model.alpha = 1
        else:
            self.model.alpha = num_imgs_seen // self.num_imgs_to_fade
