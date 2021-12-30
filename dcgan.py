import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


def load_mnist(buffer_size, batch_size, input_shape=(28, 28, 1)):
    (images, labels), (_, _) = keras.datasets.mnist.load_data()

    images = images.reshape(images.shape[0], *input_shape).astype('float32')
    images = (images - 127.5) / 127.5  # normalize images to [-1, 1]

    return tf.data.Dataset.from_tensor_slices(images)\
        .shuffle(buffer_size)\
        .batch(batch_size)


def mnist_generator():
    model = keras.Sequential(name='mnist_generator')

    model.add(layers.Dense(7 * 7 * 256, use_bias=False, input_shape=(100,)))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Reshape((7, 7, 256)))
    assert model.output_shape == (None, 7, 7, 256)  # None is the batch size

    model.add(layers.Conv2DTranspose(
        128, (5, 5), strides=(1, 1), padding='same', use_bias=False))
    assert model.output_shape == (None, 7, 7, 128)
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Conv2DTranspose(
        64, (5, 5), strides=(2, 2), padding='same', use_bias=False))
    assert model.output_shape == (None, 14, 14, 64)
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Conv2DTranspose(
        1, (5, 5), strides=(2, 2), padding='same', use_bias=False, activation='tanh'))
    assert model.output_shape == (None, 28, 28, 1)

    return model
