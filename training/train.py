from trl import DPOTrainer, DPOConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
import torch
import argparse
import os
import wandb


def parse_args():
    parser = argparse.ArgumentParser(description="Train a model using DPO")
    
    #genereal parameters
    parser.add_argument("--model_name", type=str, required=True, help="Name of the pre-trained model")
    parser.add_argument("--dataset_name", type=str, required=True, help="Name of the dataset to use")
    parser.add_argument("--ds_config", type=str, default="ds_config.json", help="Path to the DeepSpeed configuration file")
    parser.add_argument("--output_dir", type=str, default="./output", help="Directory to save the model")
    parser.add_argument("--wandb_project", type=str, default="dpo_training", help="WandB project name")

    #training parameters
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for training")
    parser.add_argument("--num_epochs", type=int, default=5, help="Number of epochs to train")
    parser.add_argument("--learning_rate", type=float, default=5e-5, help="Learning rate for training")
    parser.add_argument("--local_rank", type=int, default=-1, help="Local rank for distributed training")


    #checkpointing
    parser.add_argument("--save_strategy", type=str, default="steps", choices=["no", "epoch", "steps"], help="Strategy to save the model")
    parser.add_argument("--save_steps", type=int, default=500, help="Number of steps between model saves")
    parser.add_argument("--resume_from_checkpoint", type=str, default=None, help="Path to a checkpoint to resume training from")
    return parser.parse_args()




def main():
    # the dataset has to be structured: prompt, chosen response, rejected response

    args = parse_args()
    
    # Initialize WandB
    wandb.require("service")
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    if local_rank == 0:
        wandb.init(project=args.wandb_project, config=vars(args))
    

    model = AutoModelForCausalLM.from_pretrained(args.model_name, local_files_only=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    dataset = load_dataset(args.dataset_name)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    model.config.pad_token_id = tokenizer.pad_token_id
    ref_model = AutoModelForCausalLM.from_pretrained(args.model_name, local_files_only=True)


    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
        
    # split dataset into train and validation sets
    if "test" not in dataset:
        dataset = dataset["train"].train_test_split(test_size=0.2, seed=42)


    dpo_config = DPOConfig(
        model_name=args.model_name,
        beta=0.1,
        deepspeed = args.ds_config,
        learning_rate=args.learning_rate,
        num_train_epochs=args.num_epochs,
        logging_dir=os.path.join(args.output_dir, "logs"),
        output_dir=args.output_dir,
        remove_unused_columns=False,
        report_to=["wandb"],
        logging_steps=10,
        save_strategy=args.save_strategy,
        save_steps=args.save_steps,
        evaluation_strategy="epoch",
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False
    )

    trainer = DPOTrainer(
        model=model,
        ref_model=ref_model,
        args=dpo_config,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        tokenizer=tokenizer
    )

    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    trainer.save_model(args.output_dir)
    if local_rank==0:
        wandb.finish()  


if __name__ == "__main__":
    main()



