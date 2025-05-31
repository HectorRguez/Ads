
CUDA_VISIBLE_DEVICES=1 nohup deepspeed train.py \
    --model_name "/workspace/IterativeMCTS/ads/models/LLM-Research/Mistral-7B-v0.3" \
    --dataset_name "/workspace/IterativeMCTS/ads/dpo_dataset.json" \
    --output_dir "output/mistral-7b" \
    --wandb_project "dpo" \
    --batch_size 32 \
    --num_epochs 5 \
    --learning_rate 1e-5 \
    --save_strategy "epoch" \
    --save_steps 500 \
    --ds_config ds_config.json \
> train.log 2>&1 &