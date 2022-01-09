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

    def generator_loss(self, fake_output):
        # assume generated images are real
        return self.loss_fn(tf.ones_like(fake_output), fake_output)

    def discriminator_loss(self, real_output, fake_output):
        # 1s for real images
        real_loss = self.loss_fn(tf.ones_like(real_output), real_output)

        # 0s for generated images
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

            d_loss = self.discriminator_loss(real_output, fake_output)
            g_loss = self.generator_loss(fake_output)

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
