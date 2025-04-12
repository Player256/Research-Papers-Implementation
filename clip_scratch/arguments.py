from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ModelArguments:
    """Arguments for model configuration."""

    embed_dim: int = field(
        default=512, metadata={"help": "Joint embedding dimension for CLIP"}
    )
    vision_hidden_dim: int = field(
        default=768, metadata={"help": "Hidden dimension for vision encoder"}
    )
    vision_num_layers: int = field(
        default=12, metadata={"help": "Number of transformer layers for vision encoder"}
    )
    vision_num_heads: int = field(
        default=12, metadata={"help": "Number of attention heads for vision encoder"}
    )
    text_hidden_dim: int = field(
        default=512, metadata={"help": "Hidden dimension for text encoder"}
    )
    text_num_layers: int = field(
        default=6, metadata={"help": "Number of transformer layers for text encoder"}
    )
    text_num_heads: int = field(
        default=8, metadata={"help": "Number of attention heads for text encoder"}
    )
    temperature: float = field(
        default=0.07, metadata={"help": "Temperature parameter for contrastive loss"}
    )
    max_seq_len: int = field(
        default=77, metadata={"help": "Maximum sequence length for text"}
    )


@dataclass
class DataArguments:
    """Arguments for data configuration."""

    dataset_name: str = field(
        default="flickr8k", metadata={"help": "Dataset to use (flickr8k or flickr30k)"}
    )


@dataclass
class TrainingArguments:
    """Arguments for training configuration."""

    batch_size: int = field(
        default=32, metadata={"help": "Batch size for training and evaluation"}
    )
    output_dir: str = field(
        default="./output",
        metadata={"help": "Output directory for model and checkpoints"}
    )
    per_device_train_batch_size: int = field(
        default=32, metadata={"help": "Batch size per device for training"}
    )
    per_device_eval_batch_size: int = field(
        default=32, metadata={"help": "Batch size per device for evaluation"}
    )
    learning_rate: float = field(default=5e-5, metadata={"help": "Learning rate"})
    weight_decay: float = field(default=0.01, metadata={"help": "Weight decay"})
    num_train_epochs: int = field(
        default=10, metadata={"help": "Number of training epochs"}
    )
    evaluation_strategy: str = field(
        default="epoch",
        metadata={"help": "Evaluation strategy: 'no', 'steps', 'epoch'"},
    )
    save_strategy: str = field(
        default="epoch",
        metadata={"help": "Checkpoint save strategy: 'no', 'steps', 'epoch'"},
    )
    logging_steps: int = field(default=100, metadata={"help": "Log every X steps"})
    fp16: bool = field(default=False, metadata={"help": "Use mixed precision training"})
    seed: int = field(default=42, metadata={"help": "Random seed"})
    train_captioning_model: bool = field(
        default=False, metadata={"help": "Whether to train captioning model after CLIP"}
    )
