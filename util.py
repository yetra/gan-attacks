import glob
import os

import imageio

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
import tensorflow_io as tfio

from IPython.display import display, Audio


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


def plot_image_results(generator, target, images, target_label, latent_dim=None, perturb_bound=None):
    """
    Plots original images with their adversarial counterparts, and displays
    their classification results.

    :param generator: the model for generating adversarial perturbations
    :param target: the model for classifying images
    :param images: the original images
    :param target_label: the target label of the adversarial attack
    :param latent_dim: size of the latent space vector (for DCGAN-based generators)
    :param perturb_bound: L-infinity norm of the perturbations
    """
    if not latent_dim:
        inputs = images
    else:
        inputs = tf.random.normal(shape=(len(images), latent_dim))

    perturbations = generator(inputs, training=False)

    if perturb_bound:
        perturbations = tf.clip_by_value(
            perturbations,
            -perturb_bound,
            perturb_bound
        )

    adv_images = tf.clip_by_value(images + perturbations, -1.0, 1.0)

    _, ax = plt.subplots(len(images), 2, figsize=(8, 8))

    for i, zipped_images in enumerate(zip(images, adv_images)):
        for j, image in enumerate(zipped_images):
            probs = target.predict(image)

            ax[i, j].imshow(
                image * 127.5 + 127.5,
                cmap='gray',
                vmin=0,
                vmax=255
            )

            ax[i, j].set_title(
                f'target: {target_label} ({probs[0][target_label]:.4f})'
                f'\nassigned: {probs.argmax()} ({probs.max():.4f})'
            )

            ax[i, j].axis('off')

    plt.tight_layout()
    plt.show()


def to_spectrogram(inputs):
    """Computes spectrograms for the given audio inputs."""
    spectrograms = tfio.audio.spectrogram(
        tf.squeeze(inputs, axis=-1),
        nfft=None,
        window=165,
        stride=65
    )

    return tf.expand_dims(spectrograms, axis=-1)


def plot_audio_results(
        generator,
        target,
        audios,
        sample_rate,
        target_label,
        label2idx,
        idx2label,
        latent_dim=None,
        perturb_bound=None
):
    """
    Plots original audio examples with their adversarial counterparts, and
    displays their classification results.

    :param generator: the model for generating adversarial perturbations
    :param target: the model for classifying images
    :param audios: the original audio examples
    :param sample_rate: sample rate of the audio examples
    :param target_label: the target label of the adversarial attack
    :param label2idx: maps label names to indices
    :param idx2label: maps indices to label names
    :param latent_dim: size of the latent space vector (for WaveGAN-based generators)
    :param perturb_bound: L-infinity norm of the perturbations
    """
    if not latent_dim:
        inputs = audios
    else:
        inputs = tf.random.normal(shape=(len(audios), latent_dim))

    perturbations = generator(inputs, training=False)

    if perturb_bound:
        perturbations = tf.clip_by_value(
            perturbations,
            -perturb_bound,
            perturb_bound
        )

    adv_audios = tf.clip_by_value(audios + perturbations, -1.0, 1.0)

    _, axes = plt.subplots(len(audios), 4, figsize=(16, 20))

    for i, zipped_audios in enumerate(zip(audios, adv_audios)):
        for j, audio in enumerate(zipped_audios):
            probs = target.predict(audio)

            display_audio = tf.squeeze(audio).numpy()
            display(Audio(display_audio, rate=sample_rate))

            ax = axes[i][j]
            ax.plot(display_audio)
            ax.set_yticks(np.arange(-1.0, 1.2, 0.2))
            ax.set_title(
                f'\ntarget: {target_label} ({probs[0][label2idx(target_label)]:.4f})'
                f'\nassigned: {idx2label(probs.argmax())} ({probs.max():.4f})'
            )

            spectrogram = tf.squeeze(to_spectrogram(audio))

            ax = axes[i][j + 2]
            ax.imshow(tf.math.log(tf.transpose(spectrogram)).numpy())
            ax.set_title(
                f'\ntarget: {target_label} ({probs[0][label2idx(target_label)]:.4f})'
                f'\nassigned: {idx2label(probs.argmax())} ({probs.max():.4f})'
            )

    plt.tight_layout()
    plt.show()


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
