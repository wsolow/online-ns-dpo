#!/bin/bash

DIRECTORY_SFT=FIND_AFTER_TRAINING_SFT

USE_WANDB=false
WANDB_KEY=SET_VALUE
WANDB_ENTITY=SET_VALUE

MODEL_NAME=tiny-mistral
# MODEL_NAME=llama2-7b-chat-hf
EXP_NAME_ORIG=nsgo2c-sft-$MODEL_NAME
TARGET_COUNTRY=Germany

for SEED in 2021 2022 2023 2024 2025
do
    EXP_NAME="${EXP_NAME_ORIG}_${SEED}"
    python3 train.py model=$MODEL_NAME datasets=[nsgo-2c] \
        seed=$SEED \
        loss=ns_dpo loss.gamma=0.95 loss.current_time=100 loss.beta=0.1 \
        model.archive=.cache/$DIRECTORY_SFT/LATEST/policy.pt \
        exp_name=$EXP_NAME gradient_accumulation_steps=2 eval_every=1000 \
        batch_size=24 eval_batch_size=12 trainer=BasicTrainer sample_during_eval=false \
        ++wandb.key=$WANDB_KEY ++wandb.enabled=$USE_WANDB \
        ++wandb.entity=$WANDB_ENTITY ++wandb.project=nsdpo ++test_dataset=false \
        ++dataset.timesteps=100 ++dataset.force_new=true ++dataset.sample_to_size=10000 \
        ++dataset.coef_shift=1.0 ++dataset.threshold_high=0.99 ++dataset.threshold_low=0.01 \
        ++dataset.country2=$TARGET_COUNTRY ++dataset.min_diff=0.2
done