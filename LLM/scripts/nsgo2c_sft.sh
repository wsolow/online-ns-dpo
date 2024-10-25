#!bin/bash

USE_WANDB=false
WANDB_KEY=SET_VALUE
WANDB_ENTITY=SET_VALUE

MODEL_NAME=tiny-mistral
# MODEL_NAME=llama2-7b-chat-hf
EXP_NAME=nsgo2c-sft-$MODEL_NAME
TARGET_COUNTRY=Germany

python3 train.py model=$MODEL_NAME datasets=[nsgo-2c] \
    loss=sft exp_name=$EXP_NAME gradient_accumulation_steps=2 \
    batch_size=24 eval_batch_size=24 trainer=BasicTrainer sample_during_eval=false \
    ++wandb.key=$WANDB_KEY ++wandb.enabled=$USE_WANDB \
    ++wandb.entity=$WANDB_ENTITY ++wandb.project=nsdpo ++test_dataset=false \
    ++dataset.timesteps=100 ++dataset.force_new=true ++dataset.sample_to_size=10000 \
    ++dataset.coef_shift=1.0 ++dataset.threshold_high=0.99 ++dataset.threshold_low=0.01 \
    ++dataset.country2=$TARGET_COUNTRY ++dataset.min_diff=0.2
