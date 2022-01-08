import tensorflow as tf

from dcgan import DCGAN


class AdvDCGAN(DCGAN):
    def __init__(self, target, discriminator, generator, latent_dim):
        super().__init__(discriminator, generator, latent_dim)

        self.target = target

    def compile(self, d_optimizer, g_optimizer, loss_fn, adv_loss_fn):
        super().compile(d_optimizer, g_optimizer, loss_fn)

        self.adv_loss_fn = adv_loss_fn
