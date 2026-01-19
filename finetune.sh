#!/bin/bash

# Distributed training configuration
MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
MASTER_PORT=${MASTER_PORT:-$(shuf -i 20001-29999 -n 1)}
NNODES=${WORLD_SIZE:-1}
NPROC_PER_NODE=3

# Training entry point
deepspeed_config="/home/wsj/Desktop/code/VLA/robot/config/deepspeed/zero3.json"
DATA_PATHS="/home/wsj/Desktop/code/VLA/robot/datasets/libero_10,"\
"/home/wsj/Desktop/code/VLA/robot/datasets/libero_goal"

# Training arguments
args="
    --dataset_path ${DATA_PATHS} \
    --modality_id "libero_panda" \
    --model_path "/home/wsj/Downloads/weights/qwen3-vl-2b" \
    --per_device_train_batch_size 2 \
    --num_gpus ${NPROC_PER_NODE} \
"

# Launch training
torchrun --nproc_per_node=${NPROC_PER_NODE} \
         --master_addr=${MASTER_ADDR} \
         --master_port=${MASTER_PORT} \
         benchmark_finetune.py ${args}