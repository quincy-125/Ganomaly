import tensorflow as tf
import os

def tbc(log_dir='logs', hist_freq=0, write_graph=True, write_images=False, update_freq='batch'):
    return tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=hist_freq, write_graph=write_graph, write_images=write_images, update_freq=update_freq)

def cpc(log_dir='logs', save_weights_only=True)
    tf.keras.callbacks.ModelCheckpoint(filepath=os.path.join(log_dir, 'ckpt'), save_weights_only=save_weights_only)

class GANMonitor(tf.keras.callbacks.Callback):
    def __init__(self, num_img=3, latent_dim=128):
        self.num_img = num_img
        self.latent_dim = latent_dim

    def on_epoch_end(self, epoch, logs=None):
        random_latent_vectors = tf.random.normal(shape=(self.num_img, self.latent_dim))
        generated_images = self.ANNOGAN.G(random_latent_vectors)
        generated_images *= 255
        generated_images.numpy()
        for i in range(self.num_img):
            img = tf.keras.preprocessing.image.array_to_img(generated_images[i])
            img.save("generated_img_{i}_{epoch}.png".format(i=i, epoch=epoch))


