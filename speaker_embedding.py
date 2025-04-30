from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer, default_data_collator, AutoConfig, PreTrainedModel, TrainerCallback
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
import gc

torch.backends.cudnn.benchmark = False

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

start_of_speaker = llama_token_end + 8
end_of_speaker = llama_token_end + 9

audio_token_start = llama_token_end + 10

start = [start_of_human]
middle1 = [end_of_text, end_of_human, start_of_speaker]
middle2 = [end_of_speaker, start_of_gpt, start_of_audio]
end = [end_of_audio, end_of_gpt]

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

dataset = dataset.select(range(len(dataset) // 10))
dataset = dataset.shuffle(seed=42)
print(f'len dataset: {len(dataset)}')

# hf_login(os.getenv("HF_TOKEN_EDWIN"))

tokenizer = AutoTokenizer.from_pretrained(model_name)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

def map_fn(batch):
    text = batch["text"]
    text_tokens = tokenizer(text).input_ids

    audio_tokens = batch["codes_list"]
    audio_tokens = torch.tensor(audio_tokens)
    c0 = audio_tokens[::7]
    indices = torch.where(c0[:-1] == c0[1:])[0]
    if len(indices) > 0:
        mask_indices = (indices.unsqueeze(1) * 7 + torch.arange(7, device=indices.device)).flatten()
        mask = torch.ones(len(audio_tokens), dtype=torch.bool, device=audio_tokens.device)
        mask[mask_indices] = False
        audio_tokens = audio_tokens[mask]

    batch["text"] = text_tokens
    batch["codes_list"] = audio_tokens.tolist()
    return batch

dataset = dataset.map(map_fn, num_proc=NUM_WORKERS, batched=False)

class ProjectionLayer(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.linear = nn.Linear(input_dim, output_dim, bias=False)

    def forward(self, x):
        return self.linear(x)
        x = self.fc1(x)
        x = torch.relu(x)
        x = self.fc2(x)
        return x 

class SpeakerModelingLM(PreTrainedModel):
    config_class = AutoConfig
    base_model_prefix = ""

    def __init__(self, config, model):
        super().__init__(config)

        self.model = model

        self.speaker_projection = ProjectionLayer(SPEAKER_EMBEDDING_DIM, 768, LLAMA_EMBEDDING_DIM)
        self.embedding_layer = self.model.get_input_embeddings()

        self.register_buffer("start_tokens", torch.tensor(start, dtype=torch.long).unsqueeze(0))
        self.register_buffer("middle_tokens_1", torch.tensor(middle1, dtype=torch.long).unsqueeze(0))
        self.register_buffer("middle_tokens_2", torch.tensor(middle2, dtype=torch.long).unsqueeze(0))
        self.register_buffer("end_tokens", torch.tensor(end, dtype=torch.long).unsqueeze(0))

        self.end_of_audio = end_of_audio
        self.max_new_tokens = 250 * 7

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, **kwargs):
        model = AutoModelForCausalLM.from_pretrained(pretrained_model_name_or_path, **kwargs)
        return cls(model.config, model)
    
    @torch.no_grad()
    def generate(
        self,
        text: str,
        speaker_embedding: torch.Tensor,
        **kwargs
    ):
        
        text_tokens = tokenizer(text).input_ids
        input_ids_1 = [start_of_human] + text_tokens + [end_of_text, end_of_human, start_of_speaker]
        input_ids_2 = [end_of_speaker, start_of_gpt, start_of_audio]
        input_ids_1 = torch.tensor(input_ids_1, dtype=torch.long).unsqueeze(0).to(self.device)
        input_ids_2 = torch.tensor(input_ids_2, dtype=torch.long).unsqueeze(0).to(self.device)
        embds_1 = self.embedding_layer(input_ids_1)
        embds_2 = self.embedding_layer(input_ids_2)

        inputs_embeds = torch.cat([embds_1, speaker_embedding, embds_2], dim=1)
        attention_mask = torch.ones(inputs_embeds.size()[:-1], dtype=torch.long, device=device)

        output_tokens =  self.model.generate(
            inputs_embeds=inputs_embeds, # make sure skip embed step
            attention_mask=attention_mask,
            **kwargs
        )

        start_audio_idx = (output_tokens[0] == start_of_audio).nonzero(as_tuple=True)[0]
        output_tokens = output_tokens[0][start_audio_idx + 1:].tolist()
        print('output_tokens length:', len(output_tokens))

        if output_tokens[-1] == end_of_audio:
            output_tokens = output_tokens[:-1]

        assert len(output_tokens) % 7 == 0, "Token length must be divisible by 7"
        return output_tokens

    def forward(
            self,
            codes_list: torch.Tensor, # audio
            speaker_embedding: torch.Tensor,
            text: torch.Tensor,
            **kwargs
        ):

        device = next(self.parameters()).device
        codes_list, speaker_embedding, text = codes_list.to(device), speaker_embedding.to(device), text.to(device)
    
        B, _ = codes_list.size()

        start_embedding  = self.embedding_layer(self.start_tokens.to(device))
        middle_embedding_1 = self.embedding_layer(self.middle_tokens_1.to(device))
        middle_embedding_2 = self.embedding_layer(self.middle_tokens_2.to(device))
        end_embedding    = self.embedding_layer(self.end_tokens.to(device))

        speaker_embedding = self.speaker_projection(speaker_embedding).unsqueeze(1)
        audio_embedding = self.embedding_layer(codes_list)
        text_embedding = self.embedding_layer(text)

        model_inputs = torch.cat([start_embedding, text_embedding, middle_embedding_1,
                                  speaker_embedding, middle_embedding_2, audio_embedding, end_embedding], dim=1)

        start_ids  = self.start_tokens.repeat(B, 1).to(device)
        middle_ids_1 = self.middle_tokens_1.repeat(B, 1).to(device)
        middle_ids_2 = self.middle_tokens_2.repeat(B, 1).to(device)
        end_ids    = self.end_tokens.repeat(B, 1).to(device)
        pad_idx    = torch.full((B, 1), -100, dtype=text.dtype, device=device)

        labels = torch.cat([start_ids, text, middle_ids_1, pad_idx, middle_ids_2, codes_list, end_ids], dim=1)

        if model_inputs.size(1) > MAX_SEQ_LENGTH:
            print(f'model_inputs truncated by {model_inputs.size(1) - MAX_SEQ_LENGTH} tokens')
            model_inputs = model_inputs[:, :MAX_SEQ_LENGTH]
            labels = labels[:, :MAX_SEQ_LENGTH]

        attention_mask = torch.ones(model_inputs.size(0), model_inputs.size(1), dtype=torch.long, device=device)

        out = self.model(inputs_embeds=model_inputs, attention_mask=attention_mask, labels=labels, return_dict=True)

        return out.loss, out.logits

SpeakerModelingLM.register_for_auto_class("AutoModelForCausalLM")
model = SpeakerModelingLM.from_pretrained(model_name)

# Print model layers and exit
print("\nModel Layers:")
for name, module in model.named_modules():
    if len(list(module.children())) == 0:  # Only print leaf modules
        print(f"{name}: {type(module).__name__}")
print("\nExiting after printing model layers...")
exit()

class ClearCacheCallback(TrainerCallback):
    def __init__(self, n_steps=250):
        self.n_steps = n_steps
        self.step = 0

    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step % self.n_steps == 0:
            gc.collect()
            torch.cuda.empty_cache()
            if "model" in kwargs and hasattr(kwargs["model"], "trainer"):
                kwargs["model"].trainer.accelerator.clear()

training_args = TrainingArguments(
    output_dir="llama-voice-cloning",
    per_device_train_batch_size=1,
    gradient_accumulation_steps=1,
    learning_rate=2e-5,
    num_train_epochs=1,
    save_steps=10000,
    bf16=True,
    logging_dir="logs",
    logging_steps=1,
    remove_unused_columns=False,
    report_to="wandb" if int(os.environ.get("LOCAL_RANK", -1)) in [-1, 0] else None,
    dataloader_num_workers=4,
    optim="adamw_torch_fused",
    save_safetensors=False,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    #callbacks=[ClearCacheCallback()],
    #data_collator=collate_fn,
)

print("training")
trainer.train()

print("saving")
trainer.push_to_hub("edwindn/llama-voice-cloning", safe_serialization=False)

