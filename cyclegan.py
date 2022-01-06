import tensorflow as tf


class CycleGAN(tf.keras.Model):
    def __init__(self, disc_x, disc_y, gen_g, gen_f):
        super().__init__()

        self.disc_x = disc_x
        self.disc_y = disc_y
        self.gen_g = gen_g
        self.gen_f = gen_f

    def compile(self, disc_x_optimizer, disc_y_optimizer, gen_g_optimizer,
                gen_f_optimizer, loss_fn, lambda_cyc, lambda_id):
        super().compile()

        self.disc_x_optimizer = disc_x_optimizer
        self.disc_y_optimizer = disc_y_optimizer
        self.gen_g_optimizer = gen_g_optimizer
        self.gen_f_optimizer = gen_f_optimizer

        self.loss_fn = loss_fn
        self.cycle_consistency_loss_fn = tf.keras.losses.MeanAbsoluteError()
        self.identity_loss_fn = tf.keras.losses.MeanAbsoluteError()

        self.disc_x_loss_metric = tf.keras.metrics.Mean(name='disc_x_loss')
        self.disc_y_loss_metric = tf.keras.metrics.Mean(name='disc_y_loss')
        self.gen_g_loss_metric = tf.keras.metrics.Mean(name='gen_g_loss')
        self.gen_f_loss_metric = tf.keras.metrics.Mean(name='gen_f_loss')

        self.lambda_cyc = lambda_cyc
        self.lambda_id = lambda_id

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

    def train_step(self, real_images):
        real_x, real_y = real_images

        with tf.GradientTape(persistent=True) as tape:
            # G : X -> Y
            # F : Y -> X

            # X -> G(X) -> F(G(X))
            fake_y = self.gen_g(real_x, training=True)
            cycled_x = self.gen_f(fake_y, training=True)

            # Y -> F(Y) -> G(F(Y))
            fake_x = self.gen_f(real_y, training=True)
            cycled_y = self.gen_g(fake_x, training=True)

            # for the identity loss:
            # if the generators get an image from their target domain as input
            # they should output the same image (or something similar to it)
            same_x = self.gen_f(real_x, training=True)  # X -> F(X)
            same_y = self.gen_g(real_y, training=True)  # Y -> G(Y)

            # get predictions from the discriminators
            disc_real_x = self.disc_x(real_x, training=True)
            disc_real_y = self.disc_y(real_y, training=True)

            disc_fake_x = self.disc_x(fake_x, training=True)
            disc_fake_y = self.disc_y(fake_y, training=True)

            # generator loss components
            adv_gen_g_loss = self.generator_loss(disc_fake_y)
            adv_gen_f_loss = self.generator_loss(disc_fake_x)

            cyc_gen_g_loss = self.cycle_consistency_loss_fn(real_y, cycled_y) * self.lambda_cyc
            cyc_gen_f_loss = self.cycle_consistency_loss_fn(real_x, cycled_x) * self.lambda_cyc

            id_gen_g_loss = self.identity_loss_fn(real_y, same_y) * 0.5 * self.lambda_id
            id_gen_f_loss = self.identity_loss_fn(real_x, same_x) * 0.5 * self.lambda_id

            # generator loss = adversarial loss + cycle consistency loss + identity loss
            gen_g_loss = adv_gen_g_loss + cyc_gen_g_loss + id_gen_g_loss
            gen_f_loss = adv_gen_f_loss + cyc_gen_f_loss + id_gen_f_loss

            # discriminator loss
            disc_x_loss = self.discriminator_loss(disc_real_x, disc_fake_x)
            disc_y_loss = self.discriminator_loss(disc_real_y, disc_fake_y)

        # calculate the gradients for the generators and discriminators
        gen_g_grads = tape.gradient(gen_g_loss, self.gen_g.trainable_variables)
        gen_f_grads = tape.gradient(gen_f_loss, self.gen_f.trainable_variables)

        disc_x_grads = tape.gradient(disc_x_loss, self.disc_x.trainable_variables)
        disc_y_grads = tape.gradient(disc_y_loss, self.disc_y.trainable_variables)

        # apply the gradients to the optimizers
        self.gen_g_optimizer.apply_gradients(
            zip(gen_g_grads, self.gen_g.trainable_variables))
        self.gen_f_optimizer.apply_gradients(
            zip(gen_f_grads, self.gen_f.trainable_variables))

        self.disc_x_optimizer.apply_gradients(
            zip(disc_x_grads, self.disc_x.trainable_variables))
        self.disc_y_optimizer.apply_gradients(
            zip(disc_y_grads, self.disc_y.trainable_variables))

        # update metrics
        self.disc_x_loss_metric.update_state(disc_x_loss)
        self.disc_y_loss_metric.update_state(disc_y_loss)
        self.gen_g_loss_metric.update_state(gen_g_loss)
        self.gen_f_loss_metric.update_state(gen_f_loss)

        return {
            'disc_x_loss': self.disc_x_loss_metric.result(),
            'disc_y_loss': self.disc_y_loss_metric.result(),
            'gen_g_loss': self.gen_g_loss_metric.result(),
            'gen_f_loss': self.gen_f_loss_metric.result(),
        }
