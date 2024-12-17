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

def ensure_directory_exists(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

# Name of the base model (4bit or 16bit version). If you have a 4-bit quantized base, use that path.
model_name = "meta-llama/Llama-3.3-70B-Instruct"
max_seq_length = 4096
dtype = None
load_in_4bit = True

# Load model in 4-bit for QLoRA (if supported)
# If the model you have isn't pre-quantized, QLoRA can still quantize on the fly,
# but ensure you have the right environment setup (bitsandbytes etc.).
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
    r = 16, # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj",],
    lora_alpha = 16,
    lora_dropout = 0, # Supports any, but = 0 is optimized
    bias = "none",
    use_gradient_checkpointing = "unsloth", # True or "unsloth" for very long context
    random_state = 3407,
    use_rslora = False,  # We support rank stabilized LoRA
    loftq_config = None, # And LoftQ
)

# Load your dataset from local jsonl files created by get_training_data()
# They should have entries like: {"text": "example text of post", "label": "expected label of post"}
data_files = {
    "train": "/data/not_backed_up/amukundan/semantic_analysis/finetuning_data/withdrawal/subs_method/train.jsonl",
    "validation": "/data/not_backed_up/amukundan/semantic_analysis/finetuning_data/withdrawal/subs_method/validation.jsonl"
}
raw_datasets = load_dataset("json", data_files=data_files)
raw_datasets["train"] = raw_datasets["train"].cast_column("label", Value("string"))
raw_datasets["validation"] = raw_datasets["validation"].cast_column("label", Value("string"))
dataset = raw_datasets["train"]

# Define an instruction for the text classification:
instruction = "You are a researcher in an academic research study focused on posts about opiate use on social media. Your task is to analyze the addiction state language in post text and respond with a label"

# Convert the dataset to ShareGPT style
def create_conversation(examples):
    convs = []
    for txt, lbl in zip(examples["text"], examples["label"]):
        conv = [
            {"from": "human", "value": txt},
            {"from": "gpt", "value": lbl},
        ]
        convs.append(conv)
    return {"conversations": convs}
dataset = dataset.map(create_conversation, batched=True)

# Convert it from ShareGPT style to Hugging Face format using standardize_sharegpt.
dataset = standardize_sharegpt(dataset)

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

dataset = dataset.map(formatting_prompts_func, batched=True)

# Ensure output directory exists
ensure_directory_exists("outputs")

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    # eval_dataset=val_dataset,
    dataset_text_field = "text",
    max_seq_length = max_seq_length,
    data_collator = DataCollatorForSeq2Seq(tokenizer = tokenizer),
    dataset_num_proc = 2,
    packing = False,
    args = TrainingArguments(
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        warmup_steps=5,
        # max_steps=30,
        num_train_epochs=3, # For a longer training run, comment out max_steps and set epochs
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
    ),
)

trainer = train_on_responses_only(
    trainer, 
    instruction_part = "<|start_header_id|>user<|end_header_id|>\n\n",
    response_part = "<|start_header_id|>assistant<|end_header_id|>\n\n"
)
# verify masking
# print(tokenizer.decode(trainer.train_dataset[5]["input_ids"]))
# space = tokenizer(" ", add_special_tokens = False).input_ids[0]
# print(tokenizer.decode([space if x == -100 else x for x in trainer.train_dataset[5]["labels"]]))
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
    used_percentage = round(used_memory         /max_memory*100, 3)
    lora_percentage = round(used_memory_for_lora/max_memory*100, 3)
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
