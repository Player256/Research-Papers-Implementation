import os
import torch
from torch.utils.data import Dataset
from PIL import Image
from datasets import load_dataset
import random


class FlickrDataset(Dataset):
    def __init__(self, dataset, processor, max_len=77, split="train"):
        self.dataset = dataset[split]
        self.processor = processor
        self.max_len = max_len

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        image_path = item["image_filename"]

        captions = item["captions"]
        caption = random.choice(captions)
        image_path = os.path.join("./data/Flicker8k_Dataset", image_path)
        image = Image.open(image_path).convert("RGB")

        processed = self.processor(
            text=caption,
            images=image,
            return_tensors="pt",
            padding="max_length",
            max_length=self.max_len,
            truncation=True,
        )

        for k, v in processed.items():
            processed[k] = v.squeeze(0)

        return processed


def load_flickr(processor):
    dataset = load_dataset("tsystems/flickr8k", trust_remote_code=True)
    split_dataset = dataset["train"].train_test_split(test_size=0.2, seed=42)
    test_valid_dataset = split_dataset["test"].train_test_split(test_size=0.5, seed=42)

    dataset = {
        "train": split_dataset["train"],
        "validation": test_valid_dataset["train"],
        "test": test_valid_dataset["test"],
    }

    train_dataset = FlickrDataset(dataset, processor, split="train")
    val_dataset = FlickrDataset(dataset, processor, split="validation")
    test_dataset = FlickrDataset(dataset, processor, split="test")

    return {"train": train_dataset, "validation": val_dataset, "test": test_dataset}
