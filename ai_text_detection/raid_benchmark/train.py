import os 
import numpy as np
from datasets import load_from_disk
from transformers import (
    AutoTokenizer,
    AutoConfig,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding,
)
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, f1_score, roc_curve
from model import DebertaV3RaidModel
from dataset import choices

label2id = {k: i for i,k in enumerate(choices)}
id2label = {i: k for i,k in enumerate(choices)}

def tpr_at_fpr(labels, logits, target_fpr=0.01):
    probs = logits.softmax(axis=-1)[:, 0]
    prob_ai = 1 - probs

    labels_binary = (labels != 0).astype(int)

    fpr, tpr, thresholds = roc_curve(labels_binary, prob_ai, pos_label=1)
    idx = np.where(fpr <= target_fpr)[0]
    if len(idx) == 0:
        return 0.0
    return tpr[idx[-1]]

def compute_metrics(pred):
    logits, labels = pred
    preds = np.argmax(logits, axis=-1)

    acc = accuracy_score(labels, preds)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average="macro", zero_division=0)
    tpr_1_percent_fpr = tpr_at_fpr(labels, logits, target_fpr=0.01)
    tpr_5_percent_fpr = tpr_at_fpr(labels, logits, target_fpr=0.05)
    return {
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tpr_1%_fpr": tpr_1_percent_fpr,
        "tpr_5%_fpr": tpr_5_percent_fpr,
    }

def train():
    model_name = "microsoft/deberta-v3-large"

    train_dataset = load_from_disk("./dataset/train")
    eval_dataset = load_from_disk("./dataset/val")

    tokenizer = AutoTokenizer.from_pretrained(
        pretrained_model_name_or_path=model_name,
        use_fast=True,  
        trust_remote_code=True
    )

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer, pad_to_multiple_of=8)

    config = AutoConfig.from_pretrained(
        model_name,
        id2label=id2label,
        label2id=label2id,
    )

    model = DebertaV3RaidModel(config)

    model.gradient_checkpointing_enable()  

    # for name, param in model.named_parameters():
    #     if not name.startswith("classifier"):
    #         param.requires_grad = False

    output_dir = "./ai_text_detector"

    training_args = TrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=True,
        gradient_checkpointing=True,
        do_train=True,
        do_eval=True,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=3,
        logging_steps=100,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        gradient_accumulation_steps=4,
        num_train_epochs=1,
        learning_rate=2e-5,
        warmup_ratio=0.1,
        weight_decay=0.01,
        fp16=False,
        bf16=True,
        deepspeed="deepspeed.json", 
        report_to=["none"],  
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    trainer.train()
    trainer.save_model(output_dir)

if __name__ == "__main__":
    train()
    

    
