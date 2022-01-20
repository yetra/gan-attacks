"""
DCGAN (Deep Convolutional Generative Adversarial Network) implementation.

References:

* Keras documentation: DCGAN to generate face images
  (https://keras.io/examples/generative/dcgan_overriding_train_step/)
* TensorFlow tutorials: Deep Convolutional Generative Adversarial Network
  (https://www.tensorflow.org/tutorials/generative/dcgan)
* Unsupervised representation learning with Deep Convolutional Generative
  Adversarial Networks (https://arxiv.org/pdf/1511.06434.pdf%C3)
"""

import tensorflow as tf
from tensorflow import keras


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

    def generator_loss_fn(self, fake_output):
        return self.loss_fn(tf.ones_like(fake_output), fake_output)

    def discriminator_loss_fn(self, real_output, fake_output):
        real_loss = self.loss_fn(tf.ones_like(real_output), real_output)
        fake_loss = self.loss_fn(tf.zeros_like(fake_output), fake_output)

        return real_loss + fake_loss

    def train_step(self, real_images):
        # sample random noise in the latent space
        batch_size = tf.shape(real_images)[0]
        noise = tf.random.normal(shape=(batch_size, self.latent_dim))

        # train the discriminator and the generator (separately)
        with tf.GradientTape() as d_tape, tf.GradientTape() as g_tape:
            generated_images = self.generator(noise, training=True)

            real_output = self.discriminator(real_images, training=True)
            fake_output = self.discriminator(generated_images, training=True)

            d_loss = self.discriminator_loss_fn(real_output, fake_output)
            g_loss = self.generator_loss_fn(fake_output)

        # calculate the gradients for the generators and discriminators
        d_grads = d_tape.gradient(d_loss, self.discriminator.trainable_variables)
        g_grads = g_tape.gradient(g_loss, self.generator.trainable_variables)

        # apply the gradients to the optimizers
        self.d_optimizer.apply_gradients(
            zip(d_grads, self.discriminator.trainable_variables))
        self.g_optimizer.apply_gradients(
            zip(g_grads, self.generator.trainable_variables))

        return {
            'd_loss': d_loss,
            'g_loss': g_loss,
        }
