#!/usr/bin/env bash
# Copyright @2023 GUANGNIANAI Inc. (guannianai.com)
# @author: Q.Y.Duan <duanqiyuan@meituan.com>
# @date: 2023/11/22

SELF_DIR=$(cd "$(dirname "$0")" || exit 1; pwd)
PROJECT_ROOT_DIR=${SELF_DIR}/..
cd $PROJECT_ROOT_DIR


MODEL_PATH=/share_nfs/duanqiyuan/models/trained_models/hf/2.6B2.6TB
OUTPUT_MODEL_PATH=/share_nfs/duanqiyuan/models/trained_models/hf/sft_experiment_models/2.6B2.6TB
mkdir -p ${OUTPUT_MODEL_PATH}

CUDA_VISIBLE_DEVICES=0 python ${PROJECT_ROOT_DIR}/src/train_bash.py \
    --stage sft \
    --model_name_or_path ${MODEL_PATH} \
    --do_train \
    --overwrite_output_dir \
    --flash_attn \
    --dataset alpaca_gpt4_zh \
    --template default \
    --finetuning_type full \
    --output_dir ${OUTPUT_MODEL_PATH} \
    --overwrite_cache \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 4 \
    --lr_scheduler_type cosine \
    --logging_steps 10 \
    --save_steps 1000 \
    --learning_rate 5e-5 \
    --num_train_epochs 3.0 \
    --plot_loss \
    --fp16