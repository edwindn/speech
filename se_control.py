from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer, default_data_collator
import torch
import torch.nn as nn
from pyannote.audio import Model, Inference
from dotenv import load_dotenv
import os
from huggingface_hub import login as hf_login, snapshot_download
import librosa

load_dotenv()

SPEAKER_EMBEDDING_DIM = 256
LLAMA_EMBEDDING_DIM = 3072
AUDIO_EMBEDDING_SR = 16000
NUM_WORKERS = min(os.cpu_count(), 64)

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

hf_login(os.getenv("HF_TOKEN_AMUVARMA"))
repo_id = "amuvarma/em-EN"
snapshot_download(
    repo_id=repo_id,
    repo_type="dataset",
    revision="main",        
    max_workers=NUM_WORKERS,
) 
dataset = load_dataset(repo_id, split="train")

hf_login(os.getenv("HF_TOKEN_EDWIN"))

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-3B")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Using device: {device}")




