import os
import logging
import torch
from transformers import (
    HfArgumentParser,
    TrainingArguments as HfTrainingArguments,
    CLIPProcessor,
    set_seed
)
from arguments import ModelArguments, DataArguments, TrainingArguments
from dataset import load_flickr
from trainer import CLIPTrainer, CaptioningTrainer
from text_encoder import TextEncoder
from vision_encoder import VisionEncoder
from text_decoder import TextDecoder
from clip import CLIP
from image_captioning_model import ImageCaptioningModel

logger = logging.getLogger(__name__)

def train():
    parser = HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments)
    )

    model_args, data_args, training_args, _ = parser.parse_args_into_dataclasses(return_remaining_strings=True)

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        level=logging.INFO,
        datefmt="%m/%d/%Y %H:%M:%S",
    )
    set_seed(training_args.seed)

    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch16")

    datasets = load_flickr(
        processor,
        dataset_name=data_args.dataset_name,
    )

    vision_encoder = VisionEncoder(
        d_model=model_args.vision_hidden_dim,
        img_size=224,
        patch_size=16,
        n_channels=3,
        n_heads=model_args.vision_num_heads,
        n_layers=model_args.vision_num_layers,
        emb_dim=model_args.vision_hidden_dim,
    )

    text_encoder = TextEncoder(
        vocab_size=48048,
        d_model=model_args.text_hidden_dim,
        max_seq_len=model_args.max_seq_len,
        n_layers=model_args.text_num_layers,
        n_heads=model_args.text_num_heads,
        emb_dim=model_args.embed_dim,
    )

    clip_model = CLIP(
        vision_encoder=vision_encoder,
        text_encoder=text_encoder,
        temperature=model_args.temperature,
        embed_dim=model_args.embed_dim,
    )

    hf_training_args = HfTrainingArguments(
        output_dir=training_args.output_dir,
        per_device_train_batch_size=training_args.per_device_train_batch_size,
        per_device_eval_batch_size=training_args.per_device_eval_batch_size,
        learning_rate=training_args.learning_rate,
        weight_decay=training_args.weight_decay,
        num_train_epochs=training_args.num_train_epochs,
        evaluation_strategy=training_args.evaluation_strategy,
        save_strategy=training_args.save_strategy,
        logging_steps=training_args.logging_steps,
        fp16=training_args.fp16,
        seed=training_args.seed,
    )

    clip_trainer = CLIPTrainer(
        model=clip_model,
        args=hf_training_args,
        train_dataset=datasets["train"],
        eval_dataset=datasets["validation"]
    )

    logger.info("*** Starting CLIP Model Training ***")
    clip_trainer.train()

    clip_trainer.save_model(training_args.output_dir)

    if training_args.training_captioning_model:
        logger.info("*** Starting Captioning Model Training ***")

        text_decoder = TextDecoder(
            vocab_size=49408,
            max_seq_len=model_args.max_seq_len,
            embed_dim=model_args.embed_dim,
            hidden_dim=768,
            n_layers=6,
            n_heads=8
        )

        captioning_model = ImageCaptioningModel(
            clip_model=clip_model,
            text_decoder=text_decoder,
        )

        captioning_trainer = CaptioningTrainer(
            model=captioning_model,
            args=hf_training_args,
            train_dataset=datasets["train"],
            eval_dataset=datasets["validation"],
        )

        captioning_trainer.train()
        captioning_output_dir = os.path.join(training_args.output_dir, "captioning")
        os.makedirs(captioning_output_dir, exist_ok=True)
        captioning_trainer.save_model(captioning_output_dir)

if __name__ == "__main__":
    train()