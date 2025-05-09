#!/bin/bash

DATASET=${1:-"atasoglu/flickr8k-dataset"}
BATCH_SIZE=${2:-64}
EPOCHS=${3:-10}
TRAIN_CAPTIONING=${4:-True}
OUTPUT_DIR="./clip_${DATASET}_model"
EVALUATION_STRATEGY=${5:-"epoch"}
TRAIN_CLIP=${6:-False}

python train.py \
    --dataset_name $DATASET \
    --batch_size $BATCH_SIZE \
    --num_train_epochs $EPOCHS \
    --output_dir $OUTPUT_DIR \
    --per_device_train_batch_size $BATCH_SIZE \
    --per_device_eval_batch_size $BATCH_SIZE \
    --learning_rate 5e-5 \
    --weight_decay 0.01 \
    --evaluation_strategy $EVALUATION_STRATEGY \
    --logging_steps 100 \
    --fp16 \
    --train_captioning_model $TRAIN_CAPTIONING \
    --data_dir "./data" \
    --text_num_heads 8 \
    --text_hidden_dim 512 \
    --train_clip_model $TRAIN_CLIP \