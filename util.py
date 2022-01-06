import glob

import imageio
import matplotlib.pyplot as plt
import tensorflow as tf


class GANMonitor(tf.keras.callbacks.Callback):
    """
    A callback for monitoring a GAN's progress. On epoch end, the generator
    is used to create a specified number of images, which are then saved to
    the specified location.
    """
    def __init__(self, num_images, latent_dim, path_prefix):
        """
        Inits the `GANMonitor`.

        :param num_images: the number of images to generate
        :param latent_dim: the size of the noise vector
        :param path_prefix: prefix of the path at which to save the images
        """
        super().__init__()

        self.path_prefix = path_prefix
        self.noise = tf.random.normal(shape=(num_images, latent_dim))

    def on_epoch_end(self, epoch, logs=None):
        predictions = self.model.generator(self.noise, training=False)

        for i in range(predictions.shape[0]):
            plt.subplot(4, 4, i + 1)
            plt.imshow(predictions[i, :, :, 0] * 127.5 + 127.5, cmap='gray')
            plt.axis('off')

        plt.savefig(f'{self.path_prefix}_{epoch + 1:03d}.png')
        plt.show()


def images_to_gif(image_pattern, gif_path):
    """
    Creates a GIF from the images specified with `image_pattern`.

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
