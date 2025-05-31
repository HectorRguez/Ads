from trl import DPOTrainer, DPOConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from datasets import load_dataset
import torch
import argparse
import os
from peft import LoraConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

def enable_input_gradients(model):
    """Enable gradients for input embeddings"""
    def make_inputs_require_grad(module, input, output):
        for inp in input:
            if isinstance(inp, torch.Tensor) and inp.dtype in [torch.float16, torch.bfloat16, torch.float32]:
                inp.requires_grad_(True)
    
    # Apply to embedding layer
    if hasattr(model, 'get_input_embeddings'):
        model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)
    
    return model


def parse_args():
    parser = argparse.ArgumentParser(description="Train a model using DPO with catastrophic forgetting prevention")
    # general parameters
    parser.add_argument("--model_name", type=str, required=True, help="Name of the pre-trained model")
    parser.add_argument("--dataset_name", type=str, required=True, help="Name of the dataset to use")
    parser.add_argument("--output_dir", type=str, default="./output", help="Directory to save the model")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size per device for training")
    parser.add_argument("--num_epochs", type=int, default=3, help="Number of epochs to train (reduced from 5)")
    parser.add_argument("--learning_rate", type=float, default=1e-6, help="Learning rate for training (reduced from 1e-5)")
    parser.add_argument("--local_rank", type=int, default=-1, help="Local rank for distributed training")
    
    # New arguments for catastrophic forgetting prevention
    parser.add_argument("--beta", type=float, default=0.1, help="DPO beta parameter (KL penalty strength)")
    parser.add_argument("--lora_rank", type=int, default=16, help="LoRA rank (reduced for stability)")
    parser.add_argument("--lora_alpha", type=int, default=16, help="LoRA alpha parameter")
    parser.add_argument("--lora_dropout", type=float, default=0.1, help="LoRA dropout rate")
    parser.add_argument("--warmup_ratio", type=float, default=0.1, help="Warmup ratio for learning rate scheduler")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay for regularization")
    parser.add_argument("--max_grad_norm", type=float, default=1.0, help="Maximum gradient norm for clipping")
    
    # checkpointing
    parser.add_argument("--save_strategy", type=str, default="steps", choices=["no", "epoch", "steps"], help="Strategy to save the model")
    parser.add_argument("--save_steps", type=int, default=250, help="Number of steps between model saves (reduced for monitoring)")
    parser.add_argument("--eval_steps", type=int, default=250, help="Number of steps between evaluations")
    parser.add_argument("--resume_from_checkpoint", type=str, default=None, help="Path to a checkpoint to resume training from")
    
    # Early stopping and monitoring
    parser.add_argument("--early_stopping_patience", type=int, default=3, help="Early stopping patience")
    parser.add_argument("--load_best_model_at_end", action="store_true", help="Load best model at end of training")
    
    return parser.parse_args()


def main():
    args = parse_args()

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    
    # More conservative quantization config (8-bit instead of 4-bit for better stability)
    bnb_config = BitsAndBytesConfig(
        load_in_8bit=True,
        llm_int8_threshold=6.0,
        llm_int8_has_fp16_weight=False,
        llm_int8_enable_fp32_cpu_offload=False, 
        bnb_8bit_compute_dtype=torch.float16,  # Add this
    )

    # Load model with quantization applied
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name, 
        local_files_only=True,
        quantization_config=bnb_config,  # Apply quantization
        torch_dtype=torch.float16,
        use_cache=False,
        device_map="auto", 
        low_cpu_mem_usage=True, 
    )

    model = prepare_model_for_kbit_training(model)

    

    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, local_files_only=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'left'  
    tokenizer.truncation_side = 'left' 
    
    # Load dataset
    dataset = load_dataset("json", data_files=args.dataset_name)
    
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    
    # More conservative LoRA configuration
    peft_config = LoraConfig(
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        r=args.lora_rank,
        bias="none",
        target_modules="all-linear",
        task_type="CAUSAL_LM",
        inference_mode=False,  # Ensure training mode
        modules_to_save=None,  # Don't save full modules
    )
    # model = get_peft_model(model, peft_config)

    model.train()

    # Enable gradients for LoRA parameters explicitly
    for name, param in model.named_parameters():
        if 'lora' in name:
            param.requires_grad = True

    # More conservative DPO configuration
    dpo_config = DPOConfig(
        beta=args.beta,                  # Increased from 0.1 for stronger KL penalty
        learning_rate=args.learning_rate,  # Reduced learning rate
        num_train_epochs=args.num_epochs,  # Reduced epochs
        logging_dir=os.path.join(args.output_dir, "logs"),
        output_dir=args.output_dir,
        remove_unused_columns=True,
        logging_steps=10,
        save_strategy=args.save_strategy,
        save_steps=args.save_steps,
        eval_strategy="no",           # Add evaluation strategy
        per_device_train_batch_size=args.batch_size,
        report_to=None,
        # Learning rate scheduler with warmup
        warmup_ratio=args.warmup_ratio,
        lr_scheduler_type="cosine",      # Cosine decay for stable training
        
        # Regularization
        weight_decay=args.weight_decay,
        max_grad_norm=args.max_grad_norm,  # Gradient clipping
        
        # Model selection and early stopping
        load_best_model_at_end=args.load_best_model_at_end,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        
        # Monitoring
        disable_tqdm=False,
        bf16=False,              # Disable bfloat16
        fp16=True,               # Use float16 instead
        dataloader_pin_memory=False,
        dataloader_num_workers=0,
        ddp_find_unused_parameters=False,
        save_only_model=True,
        
        # More frequent logging for monitoring
        logging_first_step=True,
        save_total_limit=3,  # Keep only last 3 checkpoints
    )

    model = get_peft_model(model, peft_config)
    
    # Initialize trainer
    trainer = DPOTrainer(
        model=model,
        ref_model=None,  # Will create a copy automatically
        peft_config=peft_config,
        args=dpo_config,
        train_dataset=dataset["train"],
        processing_class=tokenizer
    )
    
    # Clear cache before training
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    # Start training with monitoring
    print("🚀 Starting DPO training with catastrophic forgetting prevention...")
    print(f"📊 Configuration:")
    print(f"   LoRA rank: {args.lora_rank}")
    print(f"   Learning rate: {args.learning_rate}")
    print(f"   Beta (KL penalty): {args.beta}")
    print(f"   Epochs: {args.num_epochs}")
    print(f"   Batch size: {args.batch_size}")

    print("Checking trainable parameters:")
    trainable_count = 0
    for name, param in trainer.model.named_parameters():
        if param.requires_grad:
            trainable_count += 1
            print(f"✅ {name}: {param.shape}")
        else:
            print(f"❌ {name}: {param.shape} (frozen)")

    print(f"Total trainable parameters: {trainable_count}")
    if trainable_count == 0:
        print("🚨 ERROR: No trainable parameters found!")
    
    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    
    # Post-training evaluation
    if local_rank == 0:
        # Save final model
        trainer.save_model(os.path.join(args.output_dir, "final_model"))
        print(f"💾 Final model saved to: {os.path.join(args.output_dir, 'final_model')}")
        

if __name__ == "__main__":
    main()