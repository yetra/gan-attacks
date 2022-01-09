import tensorflow as tf

from dcgan import DCGAN


def carlini_wagner_loss_fn(target_output, target_labels):
    max_probs = tf.reduce_max(target_output, axis=1)
    target_label_probs = tf.reduce_max(target_output * target_labels, axis=1)

    return tf.reduce_sum(tf.maximum(max_probs - target_label_probs, 0))


class AdvDCGAN(DCGAN):
    def __init__(self, target, discriminator, generator, latent_dim):
        super().__init__(discriminator, generator, latent_dim)

        self.target = target

    def compile(self, d_optimizer, g_optimizer, loss_fn, adv_loss_fn, lambda_adv):
        super().compile(d_optimizer, g_optimizer, loss_fn)

        self.adv_loss_fn = adv_loss_fn
        self.lambda_adv = lambda_adv

    def train_step(self, inputs):
        real_images, target_labels = inputs

        # sample random noise in the latent space
        batch_size = tf.shape(real_images)[0]
        noise = tf.random.normal(shape=(batch_size, self.latent_dim))

        # train the discriminator and the generator (separately)
        with tf.GradientTape() as d_tape, tf.GradientTape() as g_tape:
            adv_images = self.generator(noise, training=True)

            target_output = self.target(adv_images)
            adv_loss = self.adv_loss_fn(target_output, target_labels)

            real_output = self.discriminator(real_images, training=True)
            fake_output = self.discriminator(adv_images, training=True)

            d_loss = self.discriminator_loss(real_output, fake_output)
            g_loss = self.generator_loss(fake_output) + adv_loss * self.lambda_adv

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
