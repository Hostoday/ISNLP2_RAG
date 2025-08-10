#!/bin/sh

conda activate rag

python -m run.train_grpo_batch \
    --train_input ../data/train.json \
    --eval_input ../data/dev_with_examples.json \
    --output grpo_train_ckpt/lastmodel \
    --model_id K-intelligence/Midm-2.0-Base-Instruct \
    --batch_size 1 \
    --epochs 3 \
    --lr 2e-5 \
    --max_new_tokens 256 \
    --rank 32 \
    --lora_alpha 64 \
    --gradient_ckpt \
    --num_candidates 3 \
    --diversity_penalty 0.1 \
    --wandb_run_name retrieval_examples_max_256_cand3_r32_a64 \
    --train \
    --eval \
    --device cuda

python -m run.prediction \
    --input ../data/test_with_examples.json \
    --output test_wr_wr_ep3_wss.json \
    --is_quant \
    --model_id K-intelligence/Midm-2.0-Base-Instruct \
    --model_path grpo_train_ckpt/lastmodel/epoch_3 \
