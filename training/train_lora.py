from trl import DPOTrainer, DPOConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from datasets import load_dataset
import torch
import argparse
import os
import wandb
from peft import LoraConfig



def parse_args():
    parser = argparse.ArgumentParser(description="Train a model using DPO")
    # general parameters
    parser.add_argument("--model_name", type=str, required=True, help="Name of the pre-trained model")
    parser.add_argument("--dataset_name", type=str, required=True, help="Name of the dataset to use")
    parser.add_argument("--ds_config", type=str, default="ds_config.json", help="Path to the DeepSpeed configuration file")
    parser.add_argument("--output_dir", type=str, default="./output", help="Directory to save the model")
    parser.add_argument("--wandb_project", type=str, default="dpo_training", help="WandB project name")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size per device for training")
    parser.add_argument("--num_epochs", type=int, default=5, help="Number of epochs to train")
    parser.add_argument("--learning_rate", type=float, default=1e-5, help="Learning rate for training")
    parser.add_argument("--local_rank", type=int, default=-1, help="Local rank for distributed training")
    # checkpointing
    parser.add_argument("--save_strategy", type=str, default="steps", choices=["no", "epoch", "steps"], help="Strategy to save the model")
    parser.add_argument("--save_steps", type=int, default=500, help="Number of steps between model saves")
    parser.add_argument("--resume_from_checkpoint", type=str, default=None, help="Path to a checkpoint to resume training from")
    return parser.parse_args()

def main():
    args = parse_args()
    
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    if local_rank == 0:
        wandb.require("service")
        wandb.init(project=args.wandb_project, config=vars(args))
    
    #quantization config
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )

    # Load model with better memory management
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name, 
        local_files_only=True,
        torch_dtype=torch.bfloat16,
        use_cache=False,
        device_map="auto", 
        low_cpu_mem_usage=True, 
    )
    

    model.gradient_checkpointing_enable()
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, local_files_only=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'left'  
    tokenizer.truncation_side = 'left' 
    
    # Load dataset
    dataset = load_dataset("json", data_files=args.dataset_name)
    

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
        

    peft_config = LoraConfig(
        lora_alpha=64,
        lora_dropout=0.05,
        r=128, 
        bias="none",
        target_modules="all-linear",
        task_type="CAUSAL_LM", 
    )

    dpo_config = DPOConfig(
        beta=0.1,
        deepspeed=args.ds_config,
        learning_rate=args.learning_rate,
        num_train_epochs=args.num_epochs,
        logging_dir=os.path.join(args.output_dir, "logs"),
        output_dir=args.output_dir,
        remove_unused_columns=True,
        report_to=["wandb"] if local_rank == 0 else [],
        logging_steps=10,
        save_strategy=args.save_strategy,
        save_steps=args.save_steps,
        per_device_train_batch_size=args.batch_size,
        load_best_model_at_end=False,
        metric_for_best_model=None,
        greater_is_better=False,
        disable_tqdm=False,
        bf16=True,
        dataloader_pin_memory=False,
        dataloader_num_workers=0,
        ddp_find_unused_parameters=False,
        save_only_model=True,
    )
    
    # Initialize trainer
    trainer = DPOTrainer(
        model=model,
        ref_model=None,
        peft_config=peft_config,
        args=dpo_config,
        train_dataset=dataset["train"],
        processing_class=tokenizer
    )
    
    # Clear cache before training
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    # Start training
    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    
    # Save final model
    if local_rank == 0:
        trainer.save_model("./final_model")
        wandb.finish()

if __name__ == "__main__":
    main()