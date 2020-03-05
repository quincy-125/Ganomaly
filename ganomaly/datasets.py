import tensorflow as tf
import numpy as np
import os


def get_dataset(input_dir, batch_size, img_size, epochs=1, prefix='custom-r', suffix='.tfrecords'):
    size_num = int(np.log2(img_size))
    size_num = f"{size_num:02}"  # makes it '02' and '10'
    tfrecord_file = os.path.join(input_dir, prefix + size_num + suffix)
    dataset = tf.data.TFRecordDataset(tfrecord_file)
    dataset = dataset.map(parse_tfrecord_tf)
    dataset = dataset.batch(batch_size)
    dataset = dataset.repeat(epochs)
    return dataset


def parse_tfrecord_tf(record):
    features = tf.io.parse_single_example(record, features={
        'shape': tf.io.FixedLenFeature([3], tf.int64),
        'data': tf.io.FixedLenFeature([], tf.string)})
    image = tf.io.decode_raw(features['data'], tf.uint8)
    image = tf.reshape(image, [4, 4, 3])
    #image = tf.transpose(image, [2, 0, 1])
    image = tf.cast(image, 'float32')
    image = (image - 127.5) / 127.5  # normalize to [-1, 1] instead of 0-255

    return image


batch_size = 25
input_dir = 'D:\PyCharm_Projects\Ganomaly\data'
img_size = 4
EPOCHS = 1

scaled_data = get_dataset(input_dir, batch_size, img_size, epochs=EPOCHS)
