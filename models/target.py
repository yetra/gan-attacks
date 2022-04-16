import tensorflow as tf
import tensorflow_io as tfio

from tensorflow.keras import layers


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
            layers.Softmax()
        ])

    def call(self, inputs, training=None, mask=None):
        return self.model(inputs)


def to_spectrogram(audio):
    """Maps a 1D audio sample to a spectrogram."""
    spectrogram = tfio.audio.spectrogram(
        audio,
        nfft=None,
        window=255,
        stride=128
    )

    # add channels axis
    spectrogram = spectrogram[..., tf.newaxis]

    return spectrogram


class Spectrogram(layers.Layer):
    """A custom layer for converting audio inputs to spectrograms."""

    def __init__(self):
        super().__init__()

    def call(self, inputs, *args, **kwargs):
        return tf.map_fn(to_spectrogram, inputs)


class SCConvTarget(tf.keras.Model):
    """
    A CNN-based model for classifying 2D transformations of SpeechCommands
    audio samples.

    References:

    * Simple audio recognition: Recognizing keywords
      (https://www.tensorflow.org/tutorials/audio/simple_audio)
    """

    def __init__(self, num_labels=10):
        super().__init__()

        self.model = tf.keras.Sequential([
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
