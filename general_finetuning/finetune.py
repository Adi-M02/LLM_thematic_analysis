import sys
import argparse
import os
import csv
import json  # Added for JSON serialization

# Parse command-line arguments before importing CUDA-dependent libraries
def parse_args():
    parser = argparse.ArgumentParser(description="Train LoRA adapter with a specified model, data, output directory, and number of epochs.")
    parser.add_argument("--data_dir", type=str, required=True, help="Directory containing train.jsonl and validation.jsonl")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save outputs, logs, and LoRA model")
    parser.add_argument("--cuda_device", type=str, default=None, help="Set which CUDA device to use (e.g., '0' or '1'). If not set, uses the environment default.")
    parser.add_argument("--num_epochs", type=int, default=3, help="Set number of epochs")
    return parser.parse_args()

args = parse_args()

# Set CUDA_VISIBLE_DEVICES if requested
if args.cuda_device is not None:
    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_device

# Now import CUDA-dependent libraries
import torch
from unsloth import FastLanguageModel
from unsloth import is_bfloat16_supported
from unsloth.trainer import UnslothVisionDataCollator
from transformers import TrainingArguments, DataCollatorForSeq2Seq, TrainerCallback
from trl import SFTTrainer
from datasets import load_dataset, Value
from unsloth.chat_templates import get_chat_template, standardize_sharegpt, train_on_responses_only

def ensure_directory_exists(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

# Callback to log training and eval losses
class MetricsLoggerCallback(TrainerCallback):
    def __init__(self, output_csv):
        self.output_csv = output_csv
        ensure_directory_exists(os.path.dirname(self.output_csv))
        if not os.path.exists(self.output_csv):
            with open(self.output_csv, mode='w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(["step", "train_loss", "eval_loss"])

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is None:
            return
        step = state.global_step
        train_loss = logs.get("loss", None)
        eval_loss = logs.get("eval_loss", None)
        if train_loss is not None or eval_loss is not None:
            with open(self.output_csv, mode='a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([step, train_loss, eval_loss])

if __name__ == "__main__":
    # Ensure base output directory exists
    ensure_directory_exists(args.output_dir)

    # Create a subdirectory within output_dir named '{num_epochs}_epochs'
    combined_output_dir = os.path.join(args.output_dir, f"{args.num_epochs}_epochs")
    ensure_directory_exists(combined_output_dir)

    # Set cache_dir to the specified fixed path
    cache_dir = "/data/not_backed_up/amukundan/semantic_analysis/finetuning/models_cache"

    # Name of the base model
    model_name = "meta-llama/Llama-3.3-70B-Instruct"
    max_seq_length = 4096
    dtype = None
    load_in_4bit = True

    # Load model in 4-bit for QLoRA with automatic device mapping
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=max_seq_length,
        dtype=dtype,
        load_in_4bit=load_in_4bit,
        device_map="auto",
        cache_dir=cache_dir
    )

    # Convert to a PEFT QLoRA model
    model = FastLanguageModel.get_peft_model(
        model,
        r=16,  # Consider lowering this if overfitting occurs
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        lora_alpha=16,
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=3407,
        use_rslora=False,
        loftq_config=None,
    )

    # Prepare data files
    train_file = os.path.join(args.data_dir, "train.jsonl")
    val_file = os.path.join(args.data_dir, "validation.jsonl")
    data_files = {
        "train": train_file,
        "validation": val_file
    }

    raw_datasets = load_dataset("json", data_files=data_files)

    # Cast label column to string for both train and validation
    raw_datasets["train"] = raw_datasets["train"].cast_column("label", Value("string"))
    raw_datasets["validation"] = raw_datasets["validation"].cast_column("label", Value("string"))

    def create_conversation(examples):
        convs = []
        for txt, lbl in zip(examples["text"], examples["label"]):
            # Serialize the label dictionary to a JSON string
            lbl_str = json.dumps(lbl)
            conv = [
                {"from": "human", "value": txt},
                {"from": "gpt", "value": lbl_str},
            ]
            convs.append(conv)
        return {"conversations": convs}

    # Process training dataset
    train_dataset = raw_datasets["train"]
    train_dataset = train_dataset.map(create_conversation, batched=True)
    train_dataset = standardize_sharegpt(train_dataset)

    # Apply the llama-3.1 chat template
    tokenizer = get_chat_template(
        tokenizer,
        chat_template="llama-3.1",
    )

    def formatting_prompts_func(examples):
        convos = examples["conversations"]
        texts = [
            tokenizer.apply_chat_template(convo, tokenize=False, add_generation_prompt=False)
            for convo in convos
        ]
        return {"text": texts}

    train_dataset = train_dataset.map(formatting_prompts_func, batched=True)

    # Process validation dataset with the same steps
    val_dataset = raw_datasets["validation"]
    val_dataset = val_dataset.map(create_conversation, batched=True)
    val_dataset = standardize_sharegpt(val_dataset)
    val_dataset = val_dataset.map(formatting_prompts_func, batched=True)

    # Define training arguments
    training_args = TrainingArguments(
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        warmup_steps=5,
        num_train_epochs=args.num_epochs,
        learning_rate=2e-4,
        fp16=not is_bfloat16_supported(),
        bf16=is_bfloat16_supported(),
        logging_steps=1,
        optim="adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="linear",
        seed=3407,
        output_dir=combined_output_dir,  # Set to combined_output_dir
        report_to="none",
        evaluation_strategy="steps",
        eval_steps=100,  # Adjusted to 100 steps
        save_strategy="no",
        # save_steps=100,
    )

    # Initialize the trainer
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        dataset_text_field="text",
        max_seq_length=max_seq_length,
        data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer),
        dataset_num_proc=2,
        packing=False,
        args=training_args,
    )

    # Apply additional training modifications
    trainer = train_on_responses_only(
        trainer,
        instruction_part="<|start_header_id|>user<|end_header_id|>\n\n",
        response_part="<|start_header_id|>assistant<|end_header_id|>\n\n"
    )

    # Add the metrics logging callback
    metrics_log_path = os.path.join(combined_output_dir, "metrics_log.csv")
    trainer.add_callback(MetricsLoggerCallback(metrics_log_path))

    # Optional: Print GPU stats before training
    try:
        gpu_stats = torch.cuda.get_device_properties(0)
        start_gpu_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
        max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)
        print(f"GPU = {gpu_stats.name}. Max memory = {max_memory} GB.")
        print(f"{start_gpu_memory} GB of memory reserved.")
    except Exception as e:
        print(f"Couldn't get GPU stats: {e}")

    # Start training
    trainer_stats = trainer.train()

    # Optional: Print GPU stats after training
    try:
        used_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
        used_memory_for_lora = round(used_memory - start_gpu_memory, 3)
        used_percentage = round(used_memory / max_memory * 100, 3)
        lora_percentage = round(used_memory_for_lora / max_memory * 100, 3)
        print(f"{trainer_stats.metrics['train_runtime']} seconds used for training.")
        print(f"{round(trainer_stats.metrics['train_runtime']/60, 2)} minutes used for training.")
        print(f"Peak reserved memory = {used_memory} GB.")
        print(f"Peak reserved memory for training = {used_memory_for_lora} GB.")
        print(f"Peak reserved memory % of max memory = {used_percentage} %.")
        print(f"Peak reserved memory for training % of max memory = {lora_percentage} %.")
    except Exception as e:
        print(f"Issue with stats: {e}")

    # Save LoRA adapters
    lora_model_dir = os.path.join(combined_output_dir, "lora_model")
    ensure_directory_exists(lora_model_dir)
    model.save_pretrained(lora_model_dir)
    tokenizer.save_pretrained(lora_model_dir)
    print(f"LoRA model saved to {lora_model_dir}")