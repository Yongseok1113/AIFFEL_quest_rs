#!/bin/bash
export MODEL_NAME="CompVis/stable-diffusion-v1-4"
export INSTANCE_DIR="/home/ysoh1113/workspace/projects/AIFFEL_quest_rs/GoingDeeper/Generative AI/diffusers_git/examples/dreambooth/dog"
export CLASS_DIR="/home/ysoh1113/workspace/projects/AIFFEL_quest_rs/GoingDeeper/Generative AI/diffusers_git/examples/dreambooth/dog"
export OUTPUT_DIR="/home/ysoh1113/workspace/projects/AIFFEL_quest_rs/GoingDeeper/Generative AI/diffusers_git/examples/dreambooth/data"

echo "$MODEL_NAME"

accelerate launch "/home/ysoh1113/workspace/projects/AIFFEL_quest_rs/GoingDeeper/Generative AI/diffusers_git/examples/dreambooth/train_dreambooth.py" \
  --pretrained_model_name_or_path="$MODEL_NAME" \
  --instance_data_dir="$INSTANCE_DIR" \
  --class_data_dir="$CLASS_DIR" \
  --output_dir="$OUTPUT_DIR" \
  --instance_prompt="a photo of sks dog" \
  --class_prompt="a photo of dog" \
  --resolution=512 \
  --train_batch_size=1 \
  --with_prior_preservation --prior_loss_weight=1.0 \
  --gradient_accumulation_steps=1 --gradient_checkpointing \
  --use_8bit_adam \
  --enable_xformers_memory_efficient_attention \
  --set_grads_to_none \
  --learning_rate=2e-6 \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --num_class_images=5 \
  --max_train_steps=100
