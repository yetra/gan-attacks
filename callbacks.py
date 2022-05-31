import matplotlib.pyplot as plt
import tensorflow as tf

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
