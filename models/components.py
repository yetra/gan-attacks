"""
Author: [A_K_Nain](https://twitter.com/A_K_Nain)
Taken from: https://keras.io/examples/generative/cyclegan/ (slightly modified)
"""

import tensorflow as tf
from keras_contrib.layers import InstanceNormalization

from tensorflow.keras import layers


kernel_init = tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.02)
gamma_init = tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.02)


def residual_block(
        x,
        activation,
        kernel_initializer=kernel_init,
        kernel_size=(3, 3),
        strides=(1, 1),
        padding="same",
        gamma_initializer=gamma_init,
        use_bias=False,
):
    dim = x.shape[-1]
    input_tensor = x

    x = layers.Conv2D(
        dim,
        kernel_size,
        strides=strides,
        kernel_initializer=kernel_initializer,
        padding=padding,
        use_bias=use_bias,
    )(x)
    x = InstanceNormalization(gamma_initializer=gamma_initializer)(x)
    x = activation(x)

    x = layers.Conv2D(
        dim,
        kernel_size,
        strides=strides,
        kernel_initializer=kernel_initializer,
        padding=padding,
        use_bias=use_bias,
    )(x)
    x = InstanceNormalization(gamma_initializer=gamma_initializer)(x)
    x = layers.add([input_tensor, x])

    return x


def downsample(
        x,
        filters,
        activation,
        kernel_initializer=kernel_init,
        kernel_size=(3, 3),
        strides=(2, 2),
        padding="same",
        gamma_initializer=gamma_init,
        use_bias=False,
):
    x = layers.Conv2D(
        filters,
        kernel_size,
        strides=strides,
        kernel_initializer=kernel_initializer,
        padding=padding,
        use_bias=use_bias,
    )(x)
    x = InstanceNormalization(gamma_initializer=gamma_initializer)(x)
    if activation:
        x = activation(x)

    return x


def upsample(
        x,
        filters,
        activation,
        kernel_size=(3, 3),
        strides=(2, 2),
        padding="same",
        kernel_initializer=kernel_init,
        gamma_initializer=gamma_init,
        use_bias=False,
):
    x = layers.Conv2DTranspose(
        filters,
        kernel_size,
        strides=strides,
        padding=padding,
        kernel_initializer=kernel_initializer,
        use_bias=use_bias,
    )(x)
    x = InstanceNormalization(gamma_initializer=gamma_initializer)(x)
    if activation:
        x = activation(x)

    return x


def get_resnet_generator(
        input_img_size,
        filters=64,
        kernel_size=(7, 7),
        num_downsampling_blocks=2,
        num_residual_blocks=9,
        num_upsampling_blocks=2,
        gamma_initializer=gamma_init,
        name=None,
):
    img_input = layers.Input(shape=input_img_size, name=name + "_img_input")
    x = layers.Conv2D(filters, kernel_size, kernel_initializer=kernel_init, use_bias=False)(img_input)
    x = InstanceNormalization(gamma_initializer=gamma_initializer)(x)
    x = layers.Activation("relu")(x)

    # Downsampling
    for _ in range(num_downsampling_blocks):
        filters *= 2
        x = downsample(x, filters=filters, activation=layers.Activation("relu"))

    # Residual blocks
    for _ in range(num_residual_blocks):
        x = residual_block(x, activation=layers.Activation("relu"))

    # Upsampling
    for _ in range(num_upsampling_blocks):
        filters //= 2
        x = upsample(x, filters, activation=layers.Activation("relu"))

    # Final block
    x = layers.Conv2D(input_img_size[-1], kernel_size, padding="same")(x)
    x = layers.Activation("tanh")(x)

    return tf.keras.Model(img_input, x, name=name)


def get_discriminator(
        input_img_size,
        filters=64,
        num_downsampling_blocks=3,
        kernel_initializer=kernel_init,
        name=None
):
    img_input = layers.Input(shape=input_img_size, name=name + "_img_input")
    x = layers.Conv2D(
        filters,
        (4, 4),
        strides=(2, 2),
        padding="same",
        kernel_initializer=kernel_initializer,
    )(img_input)
    x = layers.LeakyReLU(0.2)(x)

    num_filters = filters
    for num_block in range(num_downsampling_blocks):
        num_filters *= 2

        if num_block < 2:
            x = downsample(
                x,
                filters=num_filters,
                activation=layers.LeakyReLU(0.2),
                kernel_size=(4, 4),
                strides=(2, 2),
            )

        else:
            x = downsample(
                x,
                filters=num_filters,
                activation=layers.LeakyReLU(0.2),
                kernel_size=(4, 4),
                strides=(1, 1),
            )

    # x = layers.Conv2D(
    #     1, (4, 4), strides=(1, 1), padding="same", kernel_initializer=kernel_initializer
    # )(x)

    x = layers.Flatten()(x)
    x = layers.Dense(1)(x)

    return tf.keras.Model(img_input, x, name=name)


def get_speech_commands_adv_generator(input_audio_size):
    audio_input = layers.Input(shape=input_audio_size)

    e1 = layers.Conv1D(16, kernel_size=32, strides=2, activation='PReLU')(audio_input)
    e2 = layers.Conv1D(32, kernel_size=32, strides=2, activation='PReLU')(e1)
    e3 = layers.Conv1D(32, kernel_size=32, strides=2, activation='PReLU')(e2)
    e4 = layers.Conv1D(64, kernel_size=32, strides=2, activation='PReLU')(e3)
    e5 = layers.Conv1D(64, kernel_size=32, strides=2, activation='PReLU')(e4)
    e6 = layers.Conv1D(128, kernel_size=32, strides=2, activation='PReLU')(e5)
    e7 = layers.Conv1D(128, kernel_size=32, strides=2, activation='PReLU')(e6)
    e8 = layers.Conv1D(256, kernel_size=32, strides=2, activation='PReLU')(e7)

    s = layers.add([e8, e7])  # skip connection
    d8 = layers.Conv1DTranspose(128, kernel_size=32, strides=2, activation='PReLU')(s)
    s = layers.add([d8, e6])  # skip connection
    d7 = layers.Conv1DTranspose(128, kernel_size=32, strides=2, activation='PReLU')(s)
    s = layers.add([d7, e5])  # skip connection
    d6 = layers.Conv1DTranspose(64, kernel_size=32, strides=2, activation='PReLU')(s)
    s = layers.add([d6, e4])  # skip connection
    d5 = layers.Conv1DTranspose(64, kernel_size=32, strides=2, activation='PReLU')(s)
    s = layers.add([d5, e3])  # skip connection
    d4 = layers.Conv1DTranspose(32, kernel_size=32, strides=2, activation='PReLU')(s)
    s = layers.add([d4, e2])  # skip connection
    d3 = layers.Conv1DTranspose(32, kernel_size=32, strides=2, activation='PReLU')(s)
    s = layers.add([d3, e1])  # skip connection
    d2 = layers.Conv1DTranspose(16, kernel_size=32, strides=2, activation='PReLU')(s)

    d1 = layers.Conv1DTranspose(1, kernel_size=32, strides=2, activation='tanh')(d2)

    return tf.keras.Model(audio_input, d1)

