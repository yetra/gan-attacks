import tensorflow as tf


class CycleGAN(tf.keras.Model):
    def __init__(self, discriminator_x, discriminator_y, generator_g, generator_f):
        super().__init__()

        self.discriminator_x = discriminator_x
        self.discriminator_y = discriminator_y
        self.generator_g = generator_g
        self.generator_f = generator_f

    def compile(self, disc_x_optimizer, disc_y_optimizer, gen_g_optimizer,
                gen_f_optimizer, loss_fn, loss_lambda):
        super().compile()

        self.disc_x_optimizer = disc_x_optimizer
        self.disc_y_optimizer = disc_y_optimizer
        self.gen_g_optimizer = gen_g_optimizer
        self.gen_f_optimizer = gen_f_optimizer

        self.loss_fn = loss_fn

        self.disc_x_loss_metric = tf.keras.metrics.Mean(name='disc_x_loss')
        self.disc_y_loss_metric = tf.keras.metrics.Mean(name='disc_y_loss')
        self.gen_g_loss_metric = tf.keras.metrics.Mean(name='gen_g_loss')
        self.gen_f_loss_metric = tf.keras.metrics.Mean(name='gen_f_loss')

        self.loss_lambda = loss_lambda

    @property
    def metrics(self):
        return [self.d_loss_metric, self.g_loss_metric]

    def generator_loss(self, fake_output):
        # assume generated images are real
        return self.loss_fn(tf.ones_like(fake_output), fake_output)

    def discriminator_loss(self, real_output, fake_output):
        # 1s for real images
        real_loss = self.loss_fn(tf.ones_like(real_output), real_output)

        # 0s for generated images
        fake_loss = self.loss_fn(tf.zeros_like(fake_output), fake_output)

        loss = real_loss + fake_loss

        return loss * 0.5

    def cycle_consistency_loss(self, real_image, cycled_image):
        loss = tf.reduce_mean(tf.abs(real_image - cycled_image))

        return self.loss_lambda * loss

    def identity_loss(self, real_image, same_image):
        loss = tf.reduce_mean(tf.abs(real_image - same_image))

        return self.loss_lambda * 0.5 * loss
