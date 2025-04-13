import torch
import torch.nn.functional as F
from transformers import Trainer, CLIPProcessor

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class CLIPTrainer(Trainer):
    """Custom Trainer for CLIP Model"""

    def compute_loss(
        self,
        model,
        inputs,
        return_outputs=False,
        **kwargs
    ):
        images = inputs.get("pixel_values")
        captions = inputs.get("input_ids")
        attention_mask = inputs.get("attention_mask", None)

        outputs = model(pixel_values=images, input_ids=captions, attention_mask=attention_mask)

        logits_per_image = outputs["logits_per_image"]
        logits_per_text = outputs["logits_per_text"]

        batch_size = images.shape[0]
        labels = torch.arange(
            batch_size, dtype=torch.long, device=logits_per_image.device
        )

        loss_i = F.cross_entropy(logits_per_image, labels)
        loss_t = F.cross_entropy(logits_per_text, labels)
        loss = (loss_i + loss_t) / 2

        return (loss, outputs) if return_outputs else loss


class CaptioningTrainer(Trainer):
    """Custom Trainer for Captioning Model"""

    def compute_loss(
        self,
        model,
        inputs,
        return_outputs=False,
        **kwargs
    ):
        images = inputs.get("pixel_values")
        captions = inputs.get("input_ids")
        attention_mask = inputs.get("attention_mask", None)
        
        processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch16")
        self.pad_token_id = processor.tokenizer.pad_token_id
        
        decoder_input_ids = captions[:, :-1]
        decoder_attention_mask = (
            attention_mask[:, :-1] if attention_mask is not None else None
        )
        labels = captions[:, 1:].clone().contiguous()

        model_output = model(
            image=images,
            captions=decoder_input_ids,
            attention_mask=decoder_attention_mask,
        )
        
        logits = model_output["logits"]
        
        labels = labels.to(DEVICE)
        labels[labels == self.pad_token_id] = -100  

        loss_fct = torch.nn.CrossEntropyLoss(ignore_index=-100)
        loss = loss_fct(logits.reshape(-1, logits.size(-1)), labels.reshape(-1))

        return (loss,{"loss":loss}) if return_outputs else loss
