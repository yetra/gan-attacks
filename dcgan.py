import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

import matplotlib.pyplot as plt


def load_mnist(buffer_size, batch_size, input_shape=(28, 28, 1)):
    (images, labels), (_, _) = keras.datasets.mnist.load_data()

    images = images.reshape(images.shape[0], *input_shape).astype('float32')
    images = (images - 127.5) / 127.5  # normalize images to [-1, 1]

    return tf.data.Dataset.from_tensor_slices(images)\
        .shuffle(buffer_size)\
        .batch(batch_size)


def mnist_generator():
    model = keras.Sequential(name='mnist_generator')

    model.add(layers.Dense(7 * 7 * 256, use_bias=False, input_shape=(100,)))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Reshape((7, 7, 256)))

    model.add(layers.Conv2DTranspose(
        128, (5, 5), strides=(1, 1), padding='same', use_bias=False))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Conv2DTranspose(
        64, (5, 5), strides=(2, 2), padding='same', use_bias=False))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Conv2DTranspose(
        1, (5, 5), strides=(2, 2), padding='same', use_bias=False, activation='tanh'))

    return model


def mnist_discriminator():
    model = keras.Sequential(name='mnist_discriminator')
    model.add(layers.Conv2D(
        64, (5, 5), strides=(2, 2), padding='same', input_shape=[28, 28, 1]))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))

    model.add(layers.Conv2D(
        128, (5, 5), strides=(2, 2), padding='same'))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))

    model.add(layers.Flatten())
    model.add(layers.Dense(1, activation='sigmoid'))

    return model


class DCGAN(keras.Model):
    def __init__(self, discriminator, generator, latent_dim):
        super().__init__()

        self.discriminator = discriminator
        self.generator = generator
        self.latent_dim = latent_dim

    def compile(self, d_optimizer, g_optimizer, loss_fn):
        super().compile()

        self.d_optimizer = d_optimizer
        self.g_optimizer = g_optimizer
        self.loss_fn = loss_fn
        self.d_loss_metric = keras.metrics.Mean(name='d_loss')
        self.g_loss_metric = keras.metrics.Mean(name='g_loss')

    @property
    def metrics(self):
        return [self.d_loss_metric, self.g_loss_metric]

    def train_step(self, real_images):
        # sample random noise in the latent space
        batch_size = tf.shape(real_images)[0]
        noise = tf.random.normal(shape=(batch_size, self.latent_dim))

        # decode noise as generated images
        generated_images = self.generator(noise, training=True)
        misleading_labels = tf.ones((batch_size, 1))  # assume all are real

        # combine generated and real images for the discriminator
        combined_images = tf.concat([generated_images, real_images], axis=0)
        labels = tf.concat([
            tf.zeros((batch_size, 1)),  # 0 for generated images
            tf.ones((batch_size, 1))  # 1 for real images
        ], axis=0)

        # train the discriminator and the generator (separately)
        # discriminator training - max log(D(x)) + log(1 - D(G(z)))
        # generator training - max log(D(G(z)))
        with tf.GradientTape() as d_tape, tf.GradientTape() as g_tape:
            predictions = self.discriminator(combined_images, training=True)
            d_loss = self.loss_fn(labels, predictions)

            predictions = self.discriminator(self.generator(noise, training=True), training=True)
            g_loss = self.loss_fn(misleading_labels, predictions)

        d_grads = d_tape.gradient(d_loss, self.discriminator.trainable_variables)
        g_grads = g_tape.gradient(g_loss, self.generator.trainable_variables)

        self.d_optimizer.apply_gradients(
            zip(d_grads, self.discriminator.trainable_variables))
        self.g_optimizer.apply_gradients(
            zip(g_grads, self.generator.trainable_variables))

        # Update metrics
        self.d_loss_metric.update_state(d_loss)
        self.g_loss_metric.update_state(g_loss)

        return {
            'd_loss': self.d_loss_metric.result(),
            'g_loss': self.g_loss_metric.result(),
        }


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
