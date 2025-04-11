import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class CLIP(nn.Module):
    def __init__(
        self,
        vision_encoder,
        text_encoder,
        embed_dim=512,
        temperature=0.07,
        device="cuda"
    ):
        super().__init__()
        self.vision_encoder = vision_encoder
        self.text_encoder = text_encoder
        self.embed_dim = embed_dim
        self.logit_scale = nn.Parameter(
            torch.ones([]) * math.log(1/temperature)
        )
        self.device = device

    def encode_image(self,image):
        return self.vision_encoder(image)

    def encode_text(self, text,attention_mask=None):
        return self.text_encoder(text,attention_mask)

    def forward(self,image,text,attention_mask=None):
        image_features = self.encode_image(image)

        text_features = self.encode_text(text,attention_mask)

        image_features = F.normalize(image_features,dim=-1)
        text_features = F.normalize(text_features, dim=-1)

        logit_scale = self.logit_scale.exp()
        logits_per_image = logit_scale * (image_features @ text_features.T)

        logits_per_text = logits_per_image.T

        labels = torch.arange(image_features.shape[0],device=torch.device(self.device))

        loss_i = F.cross_entropy(logits_per_image,labels)
        loss_t = F.cross_entropy(logits_per_text,labels)
        loss = (loss_i + loss_t)/2.0

        return {
            'loss' : loss,
            'logits_per_image': logits_per_image,
            'logits_per_text': logits_per_text,
            'image_features': image_features,
            'text_features': text_features
        }
