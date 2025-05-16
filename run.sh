#!/bin/bash
export CUDA_VISIBLE_DEVICES=1,2

base_save_dir="./results/reflect/"
mkdir -p $base_save_dir
python -m scripts.inference_hunyuandit\
  --save_dir $base_save_dir \
  --device cuda \
  --num_images_per_prompt 1 \
  --num_inference_steps 50 \
  --model_path Tencent-Hunyuan/HunyuanDiT-v1.2-Diffusers \
  --vl_model_path Qwen/Qwen2.5-VL-7B-Instruct \
  --seed 42 \
  --process_steps 1 \
  --process_steps_interval 5 \
  --process_start 0.1 \
  --process_end 0.9 \
  --use_self_reflection True
