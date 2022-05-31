import glob
import os

import imageio

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf

from IPython.display import display, Audio


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


class ImageAdvGANCallback(tf.keras.callbacks.Callback):
    """
    A callback for creating and saving a specified number of adversarial images
    after each epoch.
    """

    def __init__(self, real_images, path_prefix, latent_dim=None, perturb_bound=None):
        """
        Inits the `ImageAdvGANCallback`.

        :param real_images: a sample of real images
        :param path_prefix: prefix of the path at which to save the images
        :param latent_dim: size of the latent space vector (for DCGAN-based generators)
        :param perturb_bound: L-infinity norm of the perturbations
        """
        super().__init__()

        self.real_images = real_images
        self.num_images = len(real_images)
        self.path_prefix = path_prefix

        if not latent_dim:
            self.inputs = real_images
        else:
            self.inputs = tf.random.normal(shape=(self.num_images, latent_dim))

        self.perturb_bound = perturb_bound

    def on_epoch_end(self, epoch, logs=None):
        perturbations = self.model.generator(self.inputs, training=False)

        if self.perturb_bound:
            perturbations = tf.clip_by_value(
                perturbations,
                -self.perturb_bound,
                self.perturb_bound
            )

        _, ax = plt.subplots(self.num_images, 3, figsize=(8, 8))

        for i, real_image in enumerate(self.real_images):
            real_image = real_image[0, :, :, 0]
            perturbation = perturbations[i, :, :, 0]

            adv_image = tf.clip_by_value(real_image + perturbation, -1.0, 1.0)

            for j, image in enumerate([real_image, perturbation, adv_image]):
                ax[i, j].imshow(
                    image * 127.5 + 127.5,
                    cmap='gray',
                    vmin=0,
                    vmax=255
                )
                ax[i, j].axis('off')

            ax[i, 0].set_title('real image')
            ax[i, 1].set_title('perturbation')
            ax[i, 2].set_title('adversarial image')

        plt.tight_layout()
        plt.savefig(f'{self.path_prefix}_{epoch + 1:03d}.png')
        plt.close()


class AudioAdvGANCallback(tf.keras.callbacks.Callback):
    """
    A callback for creating and displaying a sample of adversarial audio examples
    after each epoch.
    """

    def __init__(self, real_audios, sample_rate, path_prefix, latent_dim=None, perturb_bound=None):
        """
        Inits the `AudioAdvGANCallback`.

        :param real_audios: a sample of real audio examples
        :param sample_rate: audio sample rate
        :param path_prefix: prefix of the path at which to save the examples
        :param latent_dim: size of the latent space vector (for WaveGAN-based generators)
        :param perturb_bound: L-infinity norm of the perturbations
        """
        self.real_audios = real_audios
        self.num_audios = len(real_audios)
        self.sample_rate = sample_rate
        self.path_prefix = path_prefix

        if not latent_dim:
            self.inputs = real_audios
        else:
            self.inputs = tf.random.normal(shape=(self.num_audios, latent_dim))

        self.perturb_bound = perturb_bound

    def on_epoch_end(self, epoch, logs=None):
        perturbations = self.model.generator(self.inputs, training=False)

        if self.perturb_bound:
            perturbations = tf.clip_by_value(
                perturbations,
                -self.perturb_bound,
                self.perturb_bound
            )

        for i, real_audio in enumerate(self.real_audios):
            adv_audio = real_audio + perturbations[i]
            adv_audio = tf.clip_by_value(adv_audio, -1.0, 1.0)
            adv_audio = tf.squeeze(adv_audio, axis=-1).numpy()

            display(Audio(adv_audio, rate=self.sample_rate))


def images_to_gif(image_pattern, gif_path, delete_frames=False):
    """
    Creates a GIF from the images specified with `image_pattern`.

    :param image_pattern: filename pattern of the images (GIF frames)
    :param gif_path: path to the GIF
    :param delete_frames: if `True`, deletes frames after creating the GIF
    """
    filenames = sorted(glob.glob(image_pattern))

    with imageio.get_writer(gif_path, mode='I') as writer:
        for filename in filenames:
            image = imageio.imread(filename)
            writer.append_data(image)

        image = imageio.imread(filename)
        writer.append_data(image)

    if delete_frames:
        for filename in filenames:
            os.remove(filename)


def plot_confusion_matrix(true_labels, predicted_labels, classes, ignore_idx=None):
    """
    Plots a confusion matrix based on the given labels and predictions.

    :param true_labels: the true labels
    :param predicted_labels: the model's predictions
    :param classes: a collection of all the possible label values
    :param ignore_idx: index of the true label(s) to ignore in computing the matrix
    """
    cm = tf.math.confusion_matrix(
        labels=true_labels,
        predictions=predicted_labels
    ).numpy()

    cm_norm = np.around(cm / cm.sum(axis=1)[:, None], decimals=2)

    if ignore_idx is not None:
        cm_norm[ignore_idx, :] = np.nan

    cm_df = pd.DataFrame(cm_norm, index=classes, columns=classes)

    plt.figure(figsize=(8, 8))

    sns.set(rc={'axes.facecolor': '#03051A'})
    sns.heatmap(cm_df, annot=True, vmin=0, vmax=1)

    plt.tight_layout()
    plt.ylabel('true label')
    plt.xlabel('predicted label')
    plt.show()
