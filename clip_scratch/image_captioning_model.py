import torch
import torch.nn as nn

class ImageCaptioningModel(nn.Module):
    def __init__(
        self,
        clip_model,
        text_decoder
    ):
        super().__init__()
        self.clip_model = clip_model
        self.text_decoder = text_decoder
        
    def forward(self,image,captions=None,attention_mask=None):
        image_features = self.clip_model.encode(image)
        
        if captions is not None:
            logits = self.text_decoder(image_features,captions,attention_mask)
            return logits

        else:
            generated_ids = self.text_decoder(image_features)
            return generated_ids
        