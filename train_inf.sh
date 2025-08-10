conda activate rag

python -m run.prediction \
    --input ../data/test_with_examples.json \
    --output test_wr_wr_ep3_wss.json \
    --is_quant \
    --model_id K-intelligence/Midm-2.0-Base-Instruct \
    --model_path grpo_train_ckpt/retrieval_examples_2025-07-29-1224.37/epoch_3 \
