import os

import numpy as np
import tensorflow as tf

def get_dataset(input_dir, batch_size, img_size, epochs=1, prefix='train_data-r', suffix='.tfrecords',
                color_channels=3):
    size_num = int(np.log2(img_size))
    size_num = f"{size_num:02}"  # makes it '02' and '10'
    tfrecord_file = os.path.join(input_dir, prefix + size_num + suffix)
    assert os.path.exists(tfrecord_file), f"Cannot find file: {tfrecord_file}"
    dataset = tf.data.TFRecordDataset(tfrecord_file)
    dataset = dataset.map(lambda x: parse_tfrecord_tf(x))

    dataset = dataset.batch(batch_size, drop_remainder=True)
    dataset = dataset.repeat(epochs)

    return dataset


def parse_tfrecord_tf(record, batch_size=512, latent_dim=512):
    features = tf.io.parse_single_example(record, features={
        'shape': tf.io.FixedLenFeature([3], tf.int64),
        'data': tf.io.FixedLenFeature([], tf.string)})
    data = tf.io.decode_raw(features['data'], tf.uint8)
    data = tf.reshape(data, features['shape'])
    # Convert to channels last format
    data = tf.transpose(data, [1, 2, 0])
    data = tf.cast(data, tf.float32)
    data = tf.math.divide(tf.math.subtract(data, 127.5), 127.5)  # Scale from 0-255 to [-1,-1]
    return data

