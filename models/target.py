import tensorflow as tf
from tensorflow.keras import layers

from components import Spectrogram


class MNISTConvTarget(tf.keras.Model):
    """A CNN-based model for classifying MNIST images."""

    def __init__(self):
        super().__init__()

        self.model = tf.keras.Sequential([
            layers.Conv2D(32, (3, 3), input_shape=(28, 28, 1)),
            layers.ReLU(),
            layers.Conv2D(32, (3, 3)),
            layers.ReLU(),
            layers.MaxPooling2D(),

            layers.Conv2D(64, (3, 3)),
            layers.ReLU(),
            layers.Conv2D(64, (3, 3)),
            layers.ReLU(),
            layers.MaxPooling2D(),

            layers.Flatten(),

            layers.Dense(128),
            layers.ReLU(),
            layers.Dropout(0.4),
            layers.Dense(10),
            layers.Softmax()])

    def call(self, inputs, training=None, mask=None):
        return self.model(inputs)


class SCConvTarget(tf.keras.Model):
    """
    A CNN-based model for classifying 2D transformations of SpeechCommands
    audio samples.

    References:

    * Simple audio recognition: Recognizing keywords
      (https://www.tensorflow.org/tutorials/audio/simple_audio)
    """

    def __init__(self, input_shape, num_labels=10):
        super().__init__()

        self.model = tf.keras.Sequential([
            layers.Input(shape=input_shape),
            layers.Resizing(32, 32),
            layers.BatchNormalization(),

            layers.Conv2D(32, 3),
            layers.ReLU(),
            layers.Conv2D(64, 3),
            layers.ReLU(),
            layers.MaxPooling2D(),
            layers.Dropout(0.25),

            layers.Flatten(),

            layers.Dense(128),
            layers.ReLU(),
            layers.Dropout(0.5),

            layers.Dense(num_labels),
        ])

    def call(self, inputs, training=None, mask=None):
        return self.model(inputs)
