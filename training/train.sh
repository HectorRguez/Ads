deepspeed /home/hector/Ads/training/train.py \
  --model_name "/data/hector/models/mistral-7b-original" \
  --dataset_name "/home/hector/Ads/datasets/improved_dpo_dataset.json" \
  --output_dir "/data/hector/models/mistral-7b-original-DPO" \
  --batch_size 64 \
  --num_epochs 5 \
  --learning_rate 5e-5 \
  --save_strategy "epoch" \
  --save_steps 1000 \
  --ds_config /home/hector/Ads/training/ds_config.json
