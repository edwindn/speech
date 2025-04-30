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

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-3B")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

def map_fn(batch):
    text = batch["text"]
    text_tokens = tokenizer(text).input_ids
    batch["text"] = text_tokens
    return batch

dataset = dataset.map(map_fn, num_proc=NUM_WORKERS, batched=False)

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


class SpeakerModelingLM_OLD(PreTrainedModel):
    config_class = AutoConfig
    base_model_prefix = ""

    def __init__(self, config, model):
        super().__init__(config)

        self.model = model
        self.tokenizer = tokenizer
        self.speaker_projection = GatedMLP(SPEAKER_EMBEDDING_DIM, 768, LLAMA_EMBEDDING_DIM)
        self.embedding_layer = self.model.get_input_embeddings()

        self.register_buffer("start_tokens", torch.tensor(start, dtype=torch.long).unsqueeze(0))
        self.register_buffer("middle_tokens", torch.tensor(middle, dtype=torch.long).unsqueeze(0))
        self.register_buffer("end_tokens", torch.tensor(end, dtype=torch.long).unsqueeze(0))

        self.register_buffer("start_embedding", self.embedding_layer(self.start_tokens))
        self.register_buffer("middle_embedding", self.embedding_layer(self.middle_tokens))
        self.register_buffer("end_embedding", self.embedding_layer(self.end_tokens))
        # post init

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, **kwargs):
        model = AutoModelForCausalLM.from_pretrained(pretrained_model_name_or_path, **kwargs)
        return cls(model.config, model)

    def forward(
            self,
            codes_list: torch.Tensor, # audio
            speaker_embedding: torch.Tensor,
            text: torch.Tensor,
            **kwargs
        ):

        device = self.start_tokens.device
        print(f'forward using device: {device}')
        codes_list, speaker_embedding, text = codes_list.to(device), speaker_embedding.to(device), text.to(device)
    
        B, _ = codes_list.size()

        speaker_embedding = self.speaker_projection(speaker_embedding).unsqueeze(1).to(device)
        audio_embedding = self.embedding_layer(codes_list).to(device)
        text_embedding = self.embedding_layer(text).to(device)

        model_inputs = torch.cat([self.start_embedding, text_embedding, self.middle_embedding, speaker_embedding, audio_embedding, self.end_embedding], dim=1)

        # attention_mask = torch.ones_like(model_inputs)
        attention_mask = torch.ones(model_inputs.size(0), model_inputs.size(1), dtype=torch.long, device=model_inputs.device)

        start_gpu = self.start_tokens.repeat(B, 1)
        middle_gpu = self.middle_tokens.repeat(B, 1)
        end_gpu = self.end_tokens.repeat(B, 1)
        pad_idx = torch.full((B, 1), -100, dtype=text.dtype, device=model_inputs.device)
        labels_padded = torch.cat([start_gpu, text, middle_gpu, pad_idx, codes_list, end_gpu], dim=1)

        out = self.model(inputs_embeds=model_inputs, attention_mask=attention_mask, labels=labels_padded, return_dict=True)

        return out.loss, out.logits
    



class SpeakerModelingLM(PreTrainedModel):
    config_class = AutoConfig
    base_model_prefix = ""

    def __init__(self, config, model):
        super().__init__(config)
        self.model = model

        self.speaker_projection = GatedMLP(
            input_dim=SPEAKER_EMBEDDING_DIM,
            hidden_dim=768,
            output_dim=LLAMA_EMBEDDING_DIM,
        )

        self.embedding_layer = self.model.get_input_embeddings()

        self.register_buffer(
            "start_tokens",
            torch.tensor(start, dtype=torch.long).unsqueeze(0),
        )
        self.register_buffer(
            "middle_tokens",
            torch.tensor(middle, dtype=torch.long).unsqueeze(0),
        )
        self.register_buffer(
            "end_tokens",
            torch.tensor(end, dtype=torch.long).unsqueeze(0),
        )

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, **kwargs):
        model = AutoModelForCausalLM.from_pretrained(
            pretrained_model_name_or_path, **kwargs
        )
        config = model.config
        return cls(config, model)

    def forward(
        self,
        codes_list: torch.Tensor,       # audio token IDs
        speaker_embedding: torch.Tensor,
        text: torch.Tensor,
        **kwargs
    ):
        # figure out exactly where the Trainer put our model
        device = next(self.parameters()).device

        # move inputs onto that device
        codes_list        = codes_list.to(device)
        speaker_embedding = speaker_embedding.to(device)
        text              = text.to(device)

        B, _ = codes_list.size()

        # recompute the special embeddings on-the-fly
        start_emb  = self.embedding_layer(self.start_tokens.to(device))
        middle_emb = self.embedding_layer(self.middle_tokens.to(device))
        end_emb    = self.embedding_layer(self.end_tokens.to(device))

        # project speaker and embed audio/text
        spk_proj       = self.speaker_projection(speaker_embedding).unsqueeze(1)
        audio_emb      = self.embedding_layer(codes_list)
        text_emb       = self.embedding_layer(text)

        # concatenate everything
        model_inputs = torch.cat([
            start_emb,
            text_emb,
            middle_emb,
            spk_proj,
            audio_emb,
            end_emb,
        ], dim=1)

        start_ids  = self.start_tokens.repeat(B, 1).to(device)
        middle_ids = self.middle_tokens.repeat(B, 1).to(device)
        end_ids    = self.end_tokens.repeat(B, 1).to(device)
        pad_idx    = torch.full(
            (B, 1), -100, dtype=text.dtype, device=device
        )
        labels = torch.cat([
            start_ids,
            text,
            middle_ids,
            pad_idx,
            codes_list,
            end_ids,
        ], dim=1)

        if model_inputs.size(1) > MAX_SEQ_LENGTH:
            print(f'model_inputs truncated by {model_inputs.size(1) - MAX_SEQ_LENGTH} tokens')
            model_inputs = model_inputs[:, :MAX_SEQ_LENGTH]
            labels = labels[:, :MAX_SEQ_LENGTH]

        # build attention mask
        attention_mask = torch.ones(
            model_inputs.size(0),
            model_inputs.size(1),
            dtype=torch.long,
            device=device,
        )

        # forward through the LM
        out = self.model(
            inputs_embeds=model_inputs,
            attention_mask=attention_mask,
            labels=labels,
            return_dict=True,
        )

        return out.loss, out.logits

SpeakerModelingLM.register_for_auto_class("AutoModelForCausalLM")
model = SpeakerModelingLM.from_pretrained(model_name)


training_args = TrainingArguments(
    output_dir="llama-voice-cloning",
    per_device_train_batch_size=1,
    num_train_epochs=1,
    bf16=True,
    logging_dir="logs",
    logging_steps=10,
    remove_unused_columns=False,
    report_to="wandb" if int(os.environ.get("LOCAL_RANK", -1)) in [-1, 0] else None,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    #data_collator=collate_fn,
)

print("training")
trainer.train()

print("saving")
trainer.push_to_hub("edwindn/llama-voice-cloning", safe_serialization=False)

