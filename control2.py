from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer, default_data_collator, AutoConfig, PreTrainedModel
import torch
import torch.nn as nn
from pyannote.audio import Model, Inference
from dotenv import load_dotenv
import os
from huggingface_hub import login as hf_login, snapshot_download
import librosa
from snac import SNAC
import multiprocessing
import wandb

load_dotenv()

SPEAKER_EMBEDDING_DIM = 192
LLAMA_EMBEDDING_DIM = 3072
AUDIO_EMBEDDING_SR = 16000
NUM_WORKERS = os.cpu_count()
MAX_SEQ_LENGTH = 2048

# DE-DUPLICATE CODES

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

start, middle, end = torch.tensor(start), torch.tensor(middle), torch.tensor(end)

# ---------------------- #

model_name = "canopylabs/orpheus-3b-0.1-pretrained"

# Only initialize wandb on the master GPU
if int(os.environ.get("LOCAL_RANK", -1)) in [-1, 0]:
    wandb.init(project="speaker-embedding")

# hf_login(os.getenv("HF_TOKEN_AMUVARMA"))

repo_id = "amuvarma/snac_and_embs" # codes_list, speaker_embedding, text
snapshot_download(
    repo_id=repo_id,
    repo_type="dataset",
    revision="main",        
    max_workers=NUM_WORKERS,
) 
dataset = load_dataset(repo_id, split="train")

# dataset = dataset.select(range(len(dataset) // 10))
dataset = dataset.shuffle(seed=42)
dataset = dataset.select(range(10000))
print(f'len dataset: {len(dataset)}')

# hf_login(os.getenv("HF_TOKEN_EDWIN"))

tokenizer = AutoTokenizer.from_pretrained(model_name)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

def map_fn(batch):
    text = batch["text"]
    text_tokens = tokenizer(text).input_ids
    batch["text"] = text_tokens
    return batch

dataset = dataset.map(map_fn, num_proc=NUM_WORKERS, batched=False)

model = AutoModelForCausalLM.from_pretrained(model_name)


training_args = TrainingArguments(
    output_dir="llama-voice-cloning",
    per_device_train_batch_size=1,
    gradient_accumulation_steps=1,
    learning_rate=5e-6,
    num_train_epochs=1,
    bf16=True,
    logging_dir="logs",
    logging_steps=1,
    remove_unused_columns=False,
    report_to="wandb" if int(os.environ.get("LOCAL_RANK", -1)) in [-1, 0] else None,
    dataloader_num_workers=4,
    optim="adamw_torch_fused",
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    #data_collator=collate_fn,
)

print("training")
trainer.train()