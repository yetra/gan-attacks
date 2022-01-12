import tensorflow as tf


class AdvGAN(tf.keras.Model):
    def __init__(self, target, discriminator, generator):
        super().__init__()

        self.target = target
        self.discriminator = discriminator
        self.generator = generator

    def compile(self, d_optimizer, g_optimizer, loss_fn, adv_loss_fn,
                perturb_loss_fn, lambda_gan, lambda_perturb):
        super().compile()

        self.d_optimizer = d_optimizer
        self.g_optimizer = g_optimizer

        self.loss_fn = loss_fn

        self.adv_loss_fn = adv_loss_fn
        self.perturb_loss_fn = perturb_loss_fn

        self.lambda_gan = lambda_gan
        self.lambda_perturb = lambda_perturb
