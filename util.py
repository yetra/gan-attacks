import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow import keras


class GANMonitor(keras.callbacks.Callback):
    def __init__(self, num_images, latent_dim):
        super().__init__()

        self.noise = tf.random.normal(shape=(num_images, latent_dim))

    def on_epoch_end(self, epoch, logs=None):
        predictions = self.model.generator(self.noise, training=False)

        for i in range(predictions.shape[0]):
            plt.subplot(4, 4, i + 1)
            plt.imshow(predictions[i, :, :, 0] * 127.5 + 127.5, cmap='gray')
            plt.axis('off')

        plt.savefig(f'epoch_{epoch + 1:03d}.png')
        plt.show()
