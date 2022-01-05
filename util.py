import glob

import imageio
import matplotlib.pyplot as plt
import tensorflow as tf


class GANMonitor(tf.keras.callbacks.Callback):
    def __init__(self, num_images, latent_dim, filename):
        super().__init__()

        self.filename = filename
        self.noise = tf.random.normal(shape=(num_images, latent_dim))

    def on_epoch_end(self, epoch, logs=None):
        predictions = self.model.generator(self.noise, training=False)

        for i in range(predictions.shape[0]):
            plt.subplot(4, 4, i + 1)
            plt.imshow(predictions[i, :, :, 0] * 127.5 + 127.5, cmap='gray')
            plt.axis('off')

        plt.savefig(f'{self.filename}_{epoch + 1:03d}.png')
        plt.show()


def create_gif(image_pattern, gif_path):
    """
    Creates a GIF from the images located at `image_pattern`.

    :param image_pattern: filename pattern of the images (GIF frames)
    :param gif_path: path to the GIF
    """
    with imageio.get_writer(gif_path, mode='I') as writer:
        filenames = glob.glob(image_pattern)
        filenames = sorted(filenames)

        for filename in filenames:
            image = imageio.imread(filename)
            writer.append_data(image)

        image = imageio.imread(filename)
        writer.append_data(image)
