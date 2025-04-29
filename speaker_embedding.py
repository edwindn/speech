from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer, default_data_collator
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

SPEAKER_EMBEDDING_DIM = 0
LLAMA_EMBEDDING_DIM = 3072
AUDIO_EMBEDDING_SR = 16000
NUM_WORKERS = 1

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
    max_workers=os.cpu_count(),
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


class LlamaForSpeakerModeling(AutoModelForCausalLM):
    def __init__(self, config):
        super().__init__(config)        
        self.tokenizer = tokenizer

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, **kwargs):
        model = super().from_pretrained(pretrained_model_name_or_path, **kwargs)
        model.speaker_projection = GatedMLP(SPEAKER_EMBEDDING_DIM, 768, LLAMA_EMBEDDING_DIM).to(device)
        model.forward = cls.forward._get_(model, type(model))
        return model

    def forward(
            self,
            input_ids: torch.Tensor,
            speaker_embedding: torch.Tensor,
            text: str,
        ):
        # input_ids = text + audio
        labels = tokenizer(text).input_ids

        B, A = input_ids.size()
        _, T = labels.size()

        speaker_embedding = self.speaker_projection(speaker_embedding)
        audio_embedding = self.llama.embed_tokens(input_ids)
        pad_tensor = torch.ones((B, 1), dtype=torch.long) * pad_token
        model_inputs = torch.cat([audio_embedding, pad_tensor, speaker_embedding], dim=1) # can remove pad tensor
        print(f'model_inputs: {model_inputs.shape}')

        audio_mask = torch.ones((B, A), dtype=torch.long, device=audio_embedding.device)
        pad_mask = torch.ones((B, 1), dtype=torch.long, device=audio_embedding.device)
        text_mask = torch.ones((B, T), dtype=torch.long, device=audio_embedding.device)
        attention_mask = torch.cat([audio_mask, pad_mask, text_mask], dim=1)
        print(f'attention mask: {attention_mask.shape}')

        ignore_audio = torch.full_like(model_inputs, -100, device=model_inputs.device, dtype=labels.dtype)
        labels_padded = torch.cat([ignore_audio, labels], dim=1)
        print(f'labels: {labels_padded.shape}')

        out = self.llama(inputs_embeds=model_inputs, attention_mask=attention_mask, labels=labels_padded, return_dict=True)

        return out.loss, out.logits


model = LlamaForSpeakerModeling.from_pretrained(model_name)


training_args = TrainingArguments(
    output_dir="llama-voice-cloning",
    per_device_train_batch_size=1,
    num_train_epochs=1,
    logging_dir="logs",
    logging_steps=10,
    #report_to="wandb",
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    data_collator=default_data_collator,
)

print("training")
trainer.train()

print("saving")
trainer.push_to_hub("edwindn/llama-voice-cloning")

