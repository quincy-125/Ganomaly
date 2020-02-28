import tensorflow as tf
import numpy as np


def equalize_learning_rate(shape, gain=2, fan_in=None):
    """
	Adjust the weights of each layer by the constant from He, to limit the range of weights [-1,1]

	:param shape: layer dimensions [kernel_size, kernel_size, number_of_filters, feature_maps]
	:param gain: typically sqrt(2)
	:param fan_in: adjustment for the number of incoming connections as per Xavier's / He's initialization

	:return: equalized weights
	"""

    if fan_in is None:
        fan_in = np.prod(shape[:-1])
    std = tf.math.sqrt(gain) / tf.math.sqrt(fan_in)
    wscale = tf.constant(std, name='wscale', dtype=np.float32)
    adjusted_weights = tf.keras.backend.get_value('layer', shape=shape,
                                                  initializer=tf.keras.initializers.random_normal()) * wscale
    return adjusted_weights
