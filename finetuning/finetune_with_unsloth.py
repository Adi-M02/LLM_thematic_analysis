import torch
from unsloth import FastLanguageModel
from unsloth import is_bfloat16_supported
from unsloth.trainer import UnslothVisionDataCollator
from transformers import TextStreamer
from trl import SFTTrainer
from transformers import TrainingArguments, DataCollatorForSeq2Seq
from datasets import load_dataset, Value
from unsloth.chat_templates import get_chat_template, standardize_sharegpt, train_on_responses_only
import os
import csv
from transformers import TrainerCallback

def ensure_directory_exists(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

# Callback to log training and eval losses
class MetricsLoggerCallback(TrainerCallback):
    def __init__(self, output_csv="metrics_log.csv"):
        self.output_csv = output_csv
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

# Name of the base model (4bit or 16bit version)
model_name = "meta-llama/Llama-3.3-70B-Instruct"
max_seq_length = 4096
dtype = None
load_in_4bit = True

# Load model in 4-bit for QLoRA
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name  = model_name,
    max_seq_length = max_seq_length,
    dtype = dtype,
    load_in_4bit=load_in_4bit,
    cache_dir="/data/not_backed_up/amukundan/semantic_analysis/finetuning/models_cache"
)

# Convert to a PEFT QLoRA model
model = FastLanguageModel.get_peft_model(
    model,
    r = 16,
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj",],
    lora_alpha = 16,
    lora_dropout = 0,
    bias = "none",
    use_gradient_checkpointing = "unsloth",
    random_state = 3407,
    use_rslora = False,
    loftq_config = None,
)

# Load datasets
data_files = {
    "train": "/data/not_backed_up/amukundan/semantic_analysis/finetuning_data/withdrawal/subs_method/train.jsonl",
    "validation": "/data/not_backed_up/amukundan/semantic_analysis/finetuning_data/withdrawal/subs_method/validation.jsonl"
}
raw_datasets = load_dataset("json", data_files=data_files)

# Cast label column to string for both train and validation
raw_datasets["train"] = raw_datasets["train"].cast_column("label", Value("string"))
raw_datasets["validation"] = raw_datasets["validation"].cast_column("label", Value("string"))

# Instruction
instruction = "You are a researcher in an academic research study focused on posts about opiate use on social media. Your task is to analyze the addiction state language in post text and respond with a label"

def create_conversation(examples):
    convs = []
    for txt, lbl in zip(examples["text"], examples["label"]):
        conv = [
            {"from": "human", "value": txt},
            {"from": "gpt", "value": lbl},
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
    chat_template = "llama-3.1",
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

# Ensure output directory exists
ensure_directory_exists("outputs")

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,  # Include validation dataset in the trainer
    dataset_text_field = "text",
    max_seq_length = max_seq_length,
    data_collator = DataCollatorForSeq2Seq(tokenizer = tokenizer),
    dataset_num_proc = 2,
    packing = False,
    args = TrainingArguments(
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        warmup_steps=5,
        num_train_epochs=3,
        learning_rate=2e-4,
        fp16 = not is_bfloat16_supported(),
        bf16 = is_bfloat16_supported(),
        logging_steps=1,
        optim="adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="linear",
        seed=3407,
        output_dir="outputs",
        report_to="none",
        evaluation_strategy="steps",    # Evaluate periodically
        eval_steps=100,                  # Evaluate every 50 steps
        save_steps=100,
    ),
)

trainer = train_on_responses_only(
    trainer, 
    instruction_part = "<|start_header_id|>user<|end_header_id|>\n\n",
    response_part = "<|start_header_id|>assistant<|end_header_id|>\n\n"
)

# Add the metrics logging callback
trainer.add_callback(MetricsLoggerCallback("metrics_log.csv"))

try:
    gpu_stats = torch.cuda.get_device_properties(0)
    start_gpu_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
    max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)
    print(f"GPU = {gpu_stats.name}. Max memory = {max_memory} GB.")
    print(f"{start_gpu_memory} GB of memory reserved.")
except:
   print("couldnt get stats")

trainer_stats = trainer.train()
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
except:
    print("issue with stats")

# save LoRA adapters
ensure_directory_exists("lora_model")
model.save_pretrained("lora_model")
tokenizer.save_pretrained("lora_model")
