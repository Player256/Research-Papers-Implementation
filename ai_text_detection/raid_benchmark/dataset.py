from datasets import load_dataset
from transformers import AutoTokenizer

choices = [
    "human","chatgpt","gpt4","gpt3","gpt2","llama-chat","mistral","mistral-chat",
    "mpt","mpt-chat","cohere","cohere-chat"
]
label2id = {k: i for i, k in enumerate(choices)}

def preprocess_dataset(sample):
    return {
        "text": sample["generation"],
        "label": label2id[sample["model"]],
    }

def load_and_tokenize_dataset(dataset_name, tokenizer_name, max_length=384, num_proc=16):
    dataset = load_dataset(dataset_name, split="train")

    dataset = dataset.map(preprocess_dataset, num_proc=num_proc)

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name,use_fast=True,trust_remote_code=True )

    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            max_length=max_length
        )

    print("Tokenizing...")
    dataset = dataset.map(tokenize_function, batched=True, num_proc=num_proc)

    print("Creating train/val/test split...")
    datasets = dataset.train_test_split(test_size=0.2, seed=42)
    train_dataset = datasets["train"]
    split_dataset = datasets["test"]
    temp_dataset = split_dataset.train_test_split(test_size=0.5, seed=42)
    test_dataset = temp_dataset["train"]
    val_dataset = temp_dataset["test"]

    print("Saving train dataset...")
    train_dataset.save_to_disk("./dataset/train")

    print("Saving val dataset...")
    val_dataset.save_to_disk("./dataset/val")

    print("Saving test dataset...")
    test_dataset.save_to_disk("./dataset/test")

   
if __name__ == "__main__":
    load_and_tokenize_dataset(
        dataset_name="liamdugan/raid",
        tokenizer_name="microsoft/deberta-v3-large",
        num_proc=16
    )

