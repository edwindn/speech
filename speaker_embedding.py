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
# import wandb

load_dotenv()

SPEAKER_EMBEDDING_DIM = 192
LLAMA_EMBEDDING_DIM = 3072
AUDIO_EMBEDDING_SR = 16000
NUM_WORKERS = os.cpu_count()

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

# ---------------------- #

model_name = "canopylabs/orpheus-3b-0.1-pretrained"

# wandb.init(project="speaker-embedding")

hf_login(os.getenv("HF_TOKEN_AMUVARMA"))
repo_id = "amuvarma/snac_and_embs" # codes_list, speaker_embedding, text
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

# snac = SNAC.from_pretrained("hubertsiuzdak/snac_24khz").eval()
# snac = snac.to(device)

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
    

# speaker_embedding_model = Model.from_pretrained("pyannote/embedding")
# embed_speaker = Inference(speaker_embedding_model, window="whole")


def detokenize_codes(tokens):
    assert len(tokens) % 7 == 0, "Token length must be divisible by 7"
    tokens = torch.tensor(tokens, device=device).reshape(-1, 7) - audio_token_start

    snac0 = tokens[:, 0].unsqueeze(0)
    snac1 = torch.stack([
        tokens[:, 1] - snac_vocab_size,
        tokens[:, 4] - snac_vocab_size * 4
    ], dim=1).flatten().unsqueeze(0)
    snac2 = torch.stack([
        tokens[:, 2] - snac_vocab_size * 2,
        tokens[:, 3] - snac_vocab_size * 3,
        tokens[:, 5] - snac_vocab_size * 5,
        tokens[:, 6] - snac_vocab_size * 6
    ], dim=1).flatten().unsqueeze(0)

    codes = [snac0, snac1, snac2]

    assert all(c < snac_vocab_size for c in codes[0][0]), "snac0 must be less than snac_vocab_size"
    assert all(c < snac_vocab_size for c in codes[1][0]), "snac1 must be less than snac_vocab_size"
    assert all(c < snac_vocab_size for c in codes[2][0]), "snac2 must be less than snac_vocab_size"

    return codes


class SpeakerModelingLM(PreTrainedModel):
    config_class = AutoConfig
    base_model_prefix = ""

    def __init__(self, config, model):
        super().__init__(config)

        self.model = model
        self.tokenizer = tokenizer
        self.speaker_projection = GatedMLP(SPEAKER_EMBEDDING_DIM, 768, LLAMA_EMBEDDING_DIM)
        self.embedding_layer = self.model.get_input_embeddings()
        print(f'embedding_layer: {self.embedding_layer.weight.shape}')
        # post init

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, **kwargs):
        model = AutoModelForCausalLM.from_pretrained(pretrained_model_name_or_path, **kwargs)
        return cls(model.config, model)

    def forward(
            self,
            input_ids: torch.Tensor,
            speaker_embedding: torch.Tensor,
            text: str,
            **kwargs
        ):
        # input_ids = text + audio
        input_ids, speaker_embedding = input_ids.to(device), speaker_embedding.to(device)
        labels = tokenizer(text, return_tensors="pt")["input_ids"].to(device)

        B, A = input_ids.size()
        _, T = labels.size()

        print(f'input_ids: {input_ids.shape}')

        speaker_embedding = self.speaker_projection(speaker_embedding)
        audio_embedding = self.embedding_layer(input_ids)
        pad_tensor = torch.ones((B, 1), dtype=torch.long, device=device) * pad_token
        print(f'pad_tensor: {pad_tensor.shape}')
        print(f'speaker_embedding: {speaker_embedding.shape}')
        print(f'audio_embedding: {audio_embedding.shape}')
        model_inputs = torch.cat([audio_embedding, pad_tensor, speaker_embedding], dim=1) # can remove pad tensor
        print(f'model_inputs: {model_inputs.shape}')

        audio_mask = torch.ones((B, A), dtype=torch.long, device=device)
        pad_mask = torch.ones((B, 1), dtype=torch.long, device=device)
        text_mask = torch.ones((B, T), dtype=torch.long, device=device)
        attention_mask = torch.cat([audio_mask, pad_mask, text_mask], dim=1)
        print(f'attention mask: {attention_mask.shape}')

        ignore_audio = torch.full_like(model_inputs, -100, device=device, dtype=labels.dtype)
        labels_padded = torch.cat([ignore_audio, labels], dim=1)
        print(f'labels: {labels_padded.shape}')

        out = self.model(inputs_embeds=model_inputs, attention_mask=attention_mask, labels=labels_padded, return_dict=True)

        return out.loss, out.logits

SpeakerModelingLM.register_for_auto_class("AutoModelForCausalLM")
model = SpeakerModelingLM.from_pretrained(model_name).to(device)

def collate_fn(batch):
    coll = default_data_collator(batch)
    coll["input_ids"] = torch.stack([torch.tensor(b["codes_list"]) for b in batch], dim=0)
    coll["speaker_embedding"] = torch.stack([torch.tensor(b["speaker_embedding"]) for b in batch], dim=0)
    coll["text"] = [b["text"] for b in batch]
    return coll

training_args = TrainingArguments(
    output_dir="llama-voice-cloning",
    per_device_train_batch_size=1,
    num_train_epochs=1,
    logging_dir="logs",
    logging_steps=10,
    remove_unused_columns=False,
    report_to="wandb",
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

