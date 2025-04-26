from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
import torch
import torch.nn as nn
from pyannote.audio import Model, Inference
from dotenv import load_dotenv
import os
from huggingface_hub import login as hf_login, snapshot_download
import librosa

load_dotenv()
hf_login(os.getenv("HF_TOKEN"))


SPEAKER_EMBEDDING_DIM = 256
LLAMA_EMBEDDING_DIM = 3072
AUDIO_EMBEDDING_SR = 16000
NUM_WORKERS = min(os.cpu_count(), 10)


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

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-3B")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Using device: {device}")

class GatedMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = self.fc1(x)
        x = torch.relu(x)
        x = self.fc2(x)
        return x
    

model = Model.from_pretrained("pyannote/wespeaker-voxceleb-resnet34-LM")
inference = Inference(model, window="whole")

def embed_speaker(audio_path):
    embedding = inference(audio_path)
    return embedding


class LlamaForSpeakerModeling(AutoModelForCausalLM):
    def __init__(self, config):
        super().__init__(config)
        
        self.projection = GatedMLP(SPEAKER_EMBEDDING_DIM, 768, LLAMA_EMBEDDING_DIM).to(device)
        self.llama = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.1-3B").to(device)
        self.tokenizer = tokenizer

    def forward(self, input_ids, attention_mask):
        return


def map_fn(batch):
    audio_tokens = batch["codes_list"]
    text = batch["text"]
    text_tokens = tokenizer(text).input_ids

    sr = batch["audio"]["sampling_rate"]

    if sr != AUDIO_EMBEDDING_SR:
        audio = librosa.resample(
            y=audio,
            orig_sr=sr,
            target_sr=AUDIO_EMBEDDING_SR
        )

    embedding = embed_speaker(audio)

    return {
        "text_tokens": text_tokens,
        "audio_tokens": audio_tokens,
        "speaker_embedding": embedding,
    }


def collate_fn(features):
    return




repo_id = "amuvarma/em-EN"
snapshot_download(
    repo_id=repo_id,
    repo_type="dataset",
    revision="main",        
    max_workers=NUM_WORKERS,
) 
dataset = load_dataset(repo_id, split="train")
dataset = dataset.map(map_fn, batched=False, num_proc=NUM_WORKERS)


training_args = TrainingArguments(
    output_dir="llama-voice-cloning",
    per_device_train_batch_size=1,
    num_train_epochs=1,
    logging_dir="logs",
    logging_steps=10,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    data_collator=collate_fn,
)

print("training")
trainer.train()

print("saving")
trainer.push_to_hub("edwindn/llama-voice-cloning")

