import torch.nn as nn
import torch
from torchvision.models import vgg19, VGG19_Weights


# ----------------------------- VGG perceptual ----------------------
class VGGFeatureExtractor(nn.Module):
    def __init__(self, layer_idx: int = 35):  # relu5_4
        super().__init__()
        vgg = vgg19(weights=VGG19_Weights.DEFAULT).features
        self.body = nn.Sequential(*list(vgg.children())[:layer_idx]).eval()
        for p in self.body.parameters():
            p.requires_grad_(False)

    def forward(self, x):
        return self.body(x)


# ----------------------------- GAN loss ----------------------------
class LSGANLoss(nn.Module):
    """
    Implements least-squares GAN loss (MSE to targets 0 / 1).
    """

    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()

    # Discriminator loss
    def d_loss(self, real_out, fake_out):
        real_loss = 0.5 * self.mse(real_out, torch.ones_like(real_out))
        fake_loss = 0.5 * self.mse(fake_out, torch.zeros_like(fake_out))
        return real_loss + fake_loss

    # Generator adversarial component
    def g_loss(self, fake_out):
        return 0.5 * self.mse(fake_out, torch.ones_like(fake_out))


# ----------------------------- SRGAN total -------------------------
class SRGANLoss(nn.Module):
    """
    Final loss used to train the generator:
        L_G = L_content + λ_adv * L_adv
    """

    def __init__(self, adv_weight: float = 1e-3):
        super().__init__()
        self.perceptual = VGGFeatureExtractor()
        self.pixel_mse = nn.MSELoss()
        self.adv = LSGANLoss()
        self.lambda_adv = adv_weight

    def content_loss(self, sr, hr):
        return self.pixel_mse(self.perceptual(sr), self.perceptual(hr))

    def g_total(self, sr, hr, d_fake):
        c = self.content_loss(sr, hr)
        a = self.adv.g_loss(d_fake)
        return c + self.lambda_adv * a, c, a
