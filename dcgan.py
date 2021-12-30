import tensorflow as tf
from tensorflow import keras


def load_mnist(buffer_size, batch_size, input_shape=(28, 28, 1)):
    (images, labels), (_, _) = keras.datasets.mnist.load_data()

    images = images.reshape(images.shape[0], *input_shape).astype('float32')
    images = (images - 127.5) / 127.5  # normalize images to [-1, 1]

    return tf.data.Dataset.from_tensor_slices(images)\
        .shuffle(buffer_size)\
        .batch(batch_size)
