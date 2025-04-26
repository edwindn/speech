from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from datasets import load_dataset
from dotenv import load_dotenv
import os
from huggingface_hub import login as hf_login
import wandb
from transformers import TrainingArguments, Trainer, default_data_collator
import accelerate # install trl

load_dotenv()

hf_login(os.getenv("HF_TOKEN_EDWIN"))

USE_WANDB = True
MAX_SEQ_LENGTH = 8192
CPU_COUNT = os.cpu_count()
TRAIN_BATCH_SIZE = 1

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

tokenizer = AutoTokenizer.from_pretrained("canopylabs/orpheus-3b-0.1-pretrained")
model = AutoModelForCausalLM.from_pretrained("canopylabs/orpheus-3b-0.1-pretrained")

model = model.to(device)

ds = load_dataset("edwindn/orpheus-3b-maya-finetune-v2", split="train")
ds.shuffle(seed=42)

if USE_WANDB:
    wandb.login(key=os.getenv("WANDB_API_KEY"))
    wandb.init(
        project="orpheus-3b",
        name="finetuning-run",
        config={
            "model_name": "canopylabs/orpheus-3b-maya-finetune",
            "max_seq_length": MAX_SEQ_LENGTH,
            "batch_size": TRAIN_BATCH_SIZE,
            "learning_rate": 2e-5,
            "epochs": 1
        }
    )

# ---------------------- #

llama_token_end = 128256
snac_vocab_size = 4096
start_of_text = 128000
end_of_text = 128009

start_of_human = llama_token_end + 3
end_of_human = llama_token_end + 4

start_of_gpt = llama_token_end + 5
end_of_gpt = llama_token_end + 6

start_of_audio = llama_token_end + 1
end_of_audio = llama_token_end + 2

pad_token = llama_token_end + 7

audio_token_start = llama_token_end + 10

start = [start_of_human]
middle = [end_of_text, end_of_human, start_of_gpt, start_of_audio]
end = [end_of_audio, end_of_gpt]

# ---------------------- #

prompt = "Once upon a time"
inputs = tokenizer(prompt, return_tensors="pt").to(device)
with torch.no_grad():
    out = model.generate(**inputs, max_new_tokens=20)
print("\n→", tokenizer.decode(out[0], skip_special_tokens=True))


training_args = TrainingArguments(
    output_dir="orpheus-3b-voiceFinetune-0.2",
    per_device_train_batch_size=1,
    gradient_accumulation_steps=1,
    num_train_epochs=1,
    learning_rate=5e-5,
    gradient_checkpointing=False,
    bf16=True,
    logging_steps=5,
    eval_steps=100,
    ddp_find_unused_parameters=False,
    ddp_timeout=1800,
    report_to="wandb" if USE_WANDB else None,
    dataloader_num_workers=4,
    remove_unused_columns=False,
    optim="adamw_torch_fused",
)

def collate_fn(batch):
    max_length = min(MAX_SEQ_LENGTH, max(len(item["input_ids"]) for item in batch))
    
    truncated_count = 0
    total_tokens_removed = 0
    
    input_ids = []
    attention_mask = []
    
    for item in batch:
        original_length = len(item["input_ids"])
        if original_length > max_length:
            truncated_count += 1
            total_tokens_removed += original_length - max_length
        
        input_ids.append(item["input_ids"][:max_length])
        attention_mask.append(item["attention_mask"][:max_length])
    
    if truncated_count > 0:
        print(f"Batch stats: {truncated_count} sequences truncated, {total_tokens_removed} tokens removed")
    
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": input_ids.copy()
    }

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=ds,
    tokenizer=tokenizer,
    #data_collator=default_data_collator,
    data_collator=collate_fn,
)

print('training')
trainer.train()

print('pushing to hub')
trainer.push_to_hub("edwindn/orpheus-3b-voiceFinetune-0.2")
