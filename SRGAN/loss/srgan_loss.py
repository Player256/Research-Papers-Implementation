import torch
import torch.nn as nn
from torchvision.models import vgg19, VGG19_Weights


class VGGFeatureExtractor(nn.Module):
    def __init__(self, layer_index=35):
        super().__init__()
        vgg = vgg19(weights=VGG19_Weights.DEFAULT).features
        self.features_extractor = nn.Sequential(*list(vgg.children())[:layer_index])
        for param in self.features_extractor.parameters():
            param.requires_grad = False

    def forward(self, img):
        return self.features_extractor(img)


class SRGANLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.adversarial_loss = nn.BCELoss()
        self.content_loss = nn.MSELoss()
        self.vgg = VGGFeatureExtractor().eval()

    def forward(self, sr_image, hr_image, disc_pred_fake, is_generator=True):
        content_loss = self.content_loss(self.vgg(sr_image), self.vgg(hr_image))

        if is_generator:
            adversarial_loss = self.adversarial_loss(
                disc_pred_fake, torch.ones_like(disc_pred_fake)
            )
            total_loss = content_loss + 0.001 * adversarial_loss
            return total_loss, content_loss, adversarial_loss
        else:
            real_loss = self.adversarial_loss(
                disc_pred_fake, torch.zeros_like(disc_pred_fake)
            )
            return real_loss
