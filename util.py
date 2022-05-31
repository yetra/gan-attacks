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


def generate_adv_examples(
        generator,
        original_examples,
        latent_dim=None,
        perturb_bound=None
):
    """
    Generates adversarial examples for the given inputs.

    :param generator: the model for generating adversarial perturbations
    :param original_examples: the original examples
    :param latent_dim: size of the latent space vector
    (for DCGAN- or WaveGAN-based generators)
    :param perturb_bound: L-infinity norm of the perturbations
    :return: the adversarial examples and corresponding perturbations
    """
    if not latent_dim:
        inputs = original_examples
    else:
        inputs = tf.random.normal(shape=(len(original_examples), latent_dim))

    perturbations = generator(inputs, training=False)

    if perturb_bound:
        perturbations = tf.clip_by_value(
            perturbations,
            -perturb_bound,
            perturb_bound
        )

    adv_examples = tf.clip_by_value(original_examples + perturbations, -1.0, 1.0)

    return adv_examples, perturbations


def plot_image_results(
        orig_images,
        adv_images,
        target,
        target_label,
        label2idx=lambda x: x,
        idx2label=lambda x: x,
):
    """
    Plots original images with their adversarial counterparts, and displays
    their classification results.

    :param orig_images: the original images
    :param adv_images: the adversarial images
    :param target: the model for classifying images
    :param target_label: the target label of the adversarial attack
    :param label2idx: maps label names to indices
    :param idx2label: maps indices to label names
    """
    _, ax = plt.subplots(len(orig_images), 2, figsize=(8, 8))

    for i, zipped_images in enumerate(zip(orig_images, adv_images)):
        for j, image in enumerate(zipped_images):
            probs = target.predict(image[tf.newaxis, :])

            ax[i, j].imshow(
                image * 127.5 + 127.5,
                cmap='gray',
                vmin=0,
                vmax=255
            )

            ax[i, j].set_title(
                f'target: {target_label} ({probs[0][label2idx(target_label)]:.4f})'
                f'\nassigned: {idx2label(probs.argmax())} ({probs.max():.4f})'
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
        orig_audios,
        adv_audios,
        sample_rate,
        target,
        target_label,
        label2idx=lambda x: x,
        idx2label=lambda x: x,
):
    """
    Plots original audio examples with their adversarial counterparts, and
    displays their classification results.

    :param orig_audios: the original audio examples
    :param adv_audios: the adversarial audio examples
    :param sample_rate: sample rate of the audio examples
    :param target: the model for classifying images
    :param target_label: the target label of the adversarial attack
    :param label2idx: maps label names to indices
    :param idx2label: maps indices to label names
    """
    _, axes = plt.subplots(len(orig_audios), 4, figsize=(16, 20))

    for i, zipped_audios in enumerate(zip(orig_audios, adv_audios)):
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
