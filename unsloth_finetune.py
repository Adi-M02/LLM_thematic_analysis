import os
import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    TrainingArguments,
)
from peft import LoraConfig, PeftModel
from trl import SFTTrainer
import subprocess

torch.cuda.empty_cache()

# Set environment variables
os.environ['HF_TOKEN'] = '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'
os.environ['CUDA_VISIBLE_DEVICES'] = "0"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# Function to ensure a directory exists
def ensure_directory_exists(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

# Paths to training and validation data
train_file = "finetuning_data/withdrawal/subs_method/train.jsonl"
validation_file = "finetuning_data/withdrawal/subs_method/validation.jsonl"

# Model and output paths
model_name = "meta-llama/Llama-3.2-11B-Vision-Instruct"
adapter_model_path = "unsloth_finetuned/hf_models"  # LoRA adapter output
full_model_path = "unsloth_finetuned/full_model"    # Merged full model
gguf_model_path = "unsloth_finetuned/gguf_models/llama-3.2-11B-Vision-Instruct-q4km.gguf"

# Ensure output directories exist
ensure_directory_exists(os.path.dirname(adapter_model_path))
ensure_directory_exists(os.path.dirname(gguf_model_path))
ensure_directory_exists(full_model_path)

# QLoRA parameters
lora_r = 64
lora_alpha = 16
lora_dropout = 0.1

# BitsAndBytes Configuration
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,  # Ensure weights are loaded in 4-bit
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
)

# Training arguments
training_arguments = TrainingArguments(
    output_dir="finetuned_models",
    num_train_epochs=5,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
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

# Ensure training output directory exists
ensure_directory_exists(training_arguments.output_dir)

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

# Load base model (disable quantization for merging)
base_model = AutoModelForCausalLM.from_pretrained(
    model_name,
    trust_remote_code=True,
    device_map=None,  # Load entirely on CPU to avoid meta tensors
)
base_model.gradient_checkpointing_enable()

# Initialize the LoRA adapter
peft_config = LoraConfig(
    lora_alpha=lora_alpha,
    lora_dropout=lora_dropout,
    r=lora_r,
    bias="none",
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    task_type="CAUSAL_LM",
)

# Initialize the trainer
trainer = SFTTrainer(
    model=base_model,  # Use the base model
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
# trainer.train()

# # Save the LoRA adapter
# trainer.model.save_pretrained(adapter_model_path)
# tokenizer.save_pretrained(adapter_model_path)

print("Merging LoRA adapter with the base model...")

# Load the LoRA adapter into the base model
lora_model = PeftModel.from_pretrained(base_model, adapter_model_path)

# Resolve meta tensor issues by initializing and moving to the correct device
for param in lora_model.parameters():
    if param.device == torch.device("meta"):
        print(f"Initializing {param.name} on CPU...")
        param.data = torch.zeros_like(param, device="cpu")

# Move the fully initialized model to GPU
lora_model = lora_model.to("cuda")

# Force initialization
dummy_input = tokenizer("Hello", return_tensors="pt").input_ids.to("cuda")
_ = lora_model.generate(dummy_input)

# Save the merged model
lora_model.save_pretrained(full_model_path)
tokenizer.save_pretrained(full_model_path)
print(f"Full model saved to: {full_model_path}")

# Convert the fine-tuned model to GGUF format
def convert_to_gguf(hf_model_path, output_gguf_path):
    llama_cpp_dir = "./llama.cpp"  # Path to the llama.cpp repo
    converter_script = os.path.join(llama_cpp_dir, "convert-hf-to-gguf.py")

    # Ensure the GGUF model directory exists
    ensure_directory_exists(os.path.dirname(output_gguf_path))

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
convert_to_gguf(full_model_path, gguf_model_path)

# Quantize the GGUF model (optional)
def quantize_gguf(gguf_model_path, quantized_model_path, quantization_type="q4_0"):
    llama_cpp_dir = "./llama.cpp"  # Path to the llama.cpp repo
    quantize_script = os.path.join(llama_cpp_dir, "quantize")

    # Ensure the quantized model directory exists
    ensure_directory_exists(os.path.dirname(quantized_model_path))

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

quantized_model_path = gguf_model_path.replace(".gguf", "-q4_0.gguf")
quantize_gguf(gguf_model_path, quantized_model_path, quantization_type="q4_0")