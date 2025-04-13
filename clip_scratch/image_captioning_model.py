import torch
import torch.nn as nn


class ImageCaptioningModel(nn.Module):
    def __init__(self, clip_model, text_decoder):
        super().__init__()

        self.clip_model = clip_model
        self.text_decoder = text_decoder

    def forward(
        self,
        pixel_values=None,
        input_ids=None,
        attention_mask=None,
        image=None,
        captions=None,
    ):
        """
        Accepts standard Trainer keys ('pixel_values', 'input_ids')
        and also allows passing 'image' and 'captions' directly for flexibility.
        """

        if pixel_values is not None:
            image_input = pixel_values
        elif image is not None:
            image_input = image
        else:
            raise ValueError("Image input missing. Provide 'pixel_values' or 'image'.")

        if input_ids is not None:
            captions_input = input_ids

            attention_mask_input = attention_mask
        elif captions is not None:
            captions_input = captions

            attention_mask_input = attention_mask
        else:

            captions_input = None
            attention_mask_input = None

        image_features = self.clip_model.encode_image(image_input)

        if captions_input is not None:

            logits = self.text_decoder(
                image_features=image_features,
                captions=captions_input,
                attention_mask=attention_mask_input,
            )

            return {"logits": logits}
        else:

            generated_ids = self.text_decoder(image_features=image_features)

            return {"generated_ids": generated_ids}
