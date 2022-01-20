import tensorflow as tf

from dcgan import DCGAN


class AdvDCGAN(DCGAN):
    def __init__(self, target, discriminator, generator, latent_dim):
        super().__init__(discriminator, generator, latent_dim)

        self.target = target

    def compile(self, d_optimizer, g_optimizer, loss_fn, adv_loss_fn,
                perturb_loss_fn, lambda_adv, lambda_perturb):
        super().compile(d_optimizer, g_optimizer, loss_fn)

        self.adv_loss_fn = adv_loss_fn
        self.perturb_loss_fn = perturb_loss_fn

        self.lambda_adv = lambda_adv
        self.lambda_perturb = lambda_perturb

    def train_step(self, real_images):
        # sample random noise in the latent space
        batch_size = tf.shape(real_images)[0]
        noise = tf.random.normal(shape=(batch_size, self.latent_dim))

        # train the discriminator and the generator (separately)
        with tf.GradientTape() as d_tape, tf.GradientTape() as g_tape:
            perturbations = self.generator(noise, training=True)
            perturbations = tf.clip_by_value(perturbations, -0.3, 0.3)
            adv_images = tf.clip_by_value(real_images + perturbations, -1, 1)

            target_output = self.target(adv_images)
            adv_loss = self.adv_loss_fn(target_output)

            perturb_loss = self.perturb_loss_fn(perturbations)

            real_output = self.discriminator(real_images, training=True)
            fake_output = self.discriminator(adv_images, training=True)

            d_loss = self.discriminator_loss(real_output, fake_output)

            g_gan_loss = self.generator_loss(fake_output)
            g_loss = g_gan_loss + adv_loss * self.lambda_adv + perturb_loss * self.lambda_perturb

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
            'g_gan_loss': g_gan_loss,
            'g_adv_loss': adv_loss,
            'g_perturb_loss': perturb_loss
        }
