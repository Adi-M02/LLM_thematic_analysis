import os
import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    TrainingArguments,
    pipeline,
)
from peft import LoraConfig, PeftModel
from trl import SFTTrainer
import subprocess

# Paths to training and validation data
train_file = "./train.jsonl"
validation_file = "./validation.jsonl"

# Model and output paths
model_name = "llama3.2-vision:11b-instruct-q8_0"
new_model = "./finetuned_model"
gguf_model_path = "./finetuned_model.gguf"

# QLoRA parameters
lora_r = 64
lora_alpha = 16
lora_dropout = 0.1

# BitsAndBytes Configuration
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=False,
)

# Training arguments
training_arguments = TrainingArguments(
    output_dir="./results",
    num_train_epochs=5,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=1,
    optim="paged_adamw_32bit",
    save_steps=0,
    logging_steps=50,
    learning_rate=2e-4,
    fp16=False,
    bf16=False,
    max_grad_norm=0.3,
    weight_decay=0.001,
    lr_scheduler_type="cosine",
    warmup_ratio=0.03,
    group_by_length=True,
    max_steps=-1,
    report_to="tensorboard",
)

# Load dataset from JSONL files
dataset = load_dataset("json", data_files={"train": train_file, "validation": validation_file})

# Combine post title and content into a single text field
def combine_title_content(examples):
    examples["text"] = examples["post_title"] + " " + examples["post_content"]
    return examples

dataset = dataset.map(combine_title_content)

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

# Load model with QLoRA configuration
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    device_map={"": 0},
)
model.config.use_cache = False

# LoRA configuration
peft_config = LoraConfig(
    lora_alpha=lora_alpha,
    lora_dropout=lora_dropout,
    r=lora_r,
    bias="none",
    task_type="CAUSAL_LM",
)

# Initialize the trainer
trainer = SFTTrainer(
    model=model,
    train_dataset=dataset["train"],
    eval_dataset=dataset["validation"],
    peft_config=peft_config,
    dataset_text_field="text",  # Field containing combined title and content
    max_seq_length=512,
    args=training_arguments,
    tokenizer=tokenizer,
    packing=False,
)

# Start training
trainer.train()

# Save the fine-tuned Hugging Face model
trainer.model.save_pretrained(new_model)
tokenizer.save_pretrained(new_model)

# Convert the fine-tuned model to GGUF format
def convert_to_gguf(hf_model_path, output_gguf_path):
    llama_cpp_dir = "./llama.cpp"  # Path to the llama.cpp repo
    converter_script = os.path.join(llama_cpp_dir, "convert-hf-to-gguf.py")

    # Run the conversion command
    subprocess.run(
        [
            "python", converter_script,
            "--model", hf_model_path,
            "--outfile", output_gguf_path,
        ],
        check=True,
    )
    print(f"Model successfully converted to GGUF format: {output_gguf_path}")

# Run the GGUF conversion
convert_to_gguf(new_model, gguf_model_path)

# Quantize the GGUF model (optional)
def quantize_gguf(gguf_model_path, quantized_model_path, quantization_type="q4_0"):
    llama_cpp_dir = "./llama.cpp"  # Path to the llama.cpp repo
    quantize_script = os.path.join(llama_cpp_dir, "quantize")

    # Run the quantization command
    subprocess.run(
        [
            quantize_script,
            gguf_model_path,
            quantized_model_path,
            quantization_type,
        ],
        check=True,
    )
    print(f"Model successfully quantized: {quantized_model_path}")

quantized_model_path = gguf_model_path.replace(".gguf", f"-{quantization_type}.gguf")
quantize_gguf(gguf_model_path, quantized_model_path, quantization_type="q4_0")
