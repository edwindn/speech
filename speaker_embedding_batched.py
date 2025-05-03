from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer, default_data_collator, AutoConfig, PreTrainedModel, TrainerCallback
import torch
import torch.nn as nn
from pyannote.audio import Model, Inference
from dotenv import load_dotenv
import os
from huggingface_hub import login as hf_login, snapshot_download, hf_hub_download, list_repo_files
import librosa
from snac import SNAC
import multiprocessing
import wandb
import gc
import glob


"""
!!use fsdp for device splitting
"""

torch.backends.cudnn.benchmark = False

load_dotenv()

SPEAKER_EMBEDDING_DIM = 192
LLAMA_EMBEDDING_DIM = 3072
AUDIO_EMBEDDING_SR = 16000
NUM_WORKERS = os.cpu_count()
MAX_SEQ_LENGTH = 8192 # 3072

# DE-DUPLICATE CODES

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

tokenizer = AutoTokenizer.from_pretrained("canopylabs/orpheus-3b-0.1-pretrained")

class ProjectionLayer(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim, bias=False)

    def forward(self, x):
        return self.linear(x)
    

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

        self.batching = None

        for param in self.speaker_projection.parameters():
            param.requires_grad = False

        for param in self.model.parameters():
            param.requires_grad = False

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, load_mode, **kwargs):
        assert load_mode in ["local", "online", "train"]

        if load_mode == "train": # base pretrained model
            model = AutoModelForCausalLM.from_pretrained(pretrained_model_name_or_path, **kwargs)
            instance = cls(model.config, model)
            return instance

        if load_mode == "online":
            config = AutoConfig.from_pretrained(pretrained_model_name_or_path, **kwargs)
            base_model = AutoModelForCausalLM.from_config(config)
            
            repo_files = list_repo_files(pretrained_model_name_or_path)
            shard_files = [f for f in repo_files if f.startswith("pytorch_model") and f.endswith(".bin")]
            shard_files.sort()
                        
            shard_paths = []
            for shard_file in shard_files:
                shard_path = hf_hub_download(
                    repo_id=pretrained_model_name_or_path,
                    filename=shard_file,
                    local_dir="model_weights"
                )
                shard_paths.append(shard_path)
        
            raw_state = {}
            for shard in shard_paths:
                sd = torch.load(shard, map_location="cpu")
                raw_state.update(sd)

            instance = cls(config, base_model)
            
            missing, unexpected = instance.load_state_dict(raw_state, strict=False)
            print("\nLoad results:")
            print("  Missing:", missing)
            print("  Unexpected:", unexpected)
            
            return instance

        if load_mode == "local":
            config = AutoConfig.from_pretrained(pretrained_model_name_or_path, **kwargs)
            base_model = AutoModelForCausalLM.from_config(config)

            ckpt_dir    = pretrained_model_name_or_path
            pattern     = os.path.join(ckpt_dir, "pytorch_model-*.bin")
            shard_paths = sorted(glob.glob(pattern))

            if not shard_paths:
                shard_paths = [os.path.join(ckpt_dir, "pytorch_model.bin")]

            raw_state = {}
            for shard in shard_paths:
                sd = torch.load(shard, map_location="cpu")
                raw_state.update(sd)

            fixed_state = {}
            for k, v in raw_state.items():
                if k.startswith("model."):
                    new_k = k
                    # new_k = k.replace("model.", "model.model.", 1)
                else:
                    new_k = k
                fixed_state[new_k] = v

            # missing, unexpected = base_model.load_state_dict(fixed_state, strict=False)
            # print(">>> base_model loaded. missing:", missing)
            # print(">>>                unexpected:", unexpected)

            instance = cls(config, base_model)
            missing_wrap, unexpected_wrap = instance.load_state_dict(fixed_state, strict=False)
            print("wrapper load missing:   ", missing_wrap)
            print("wrapper load unexpected:", unexpected_wrap)

            for key in instance.state_dict().keys():
                if 'lm_head.weight' in key:
                    ckpt_lm = fixed_state["model.lm_head.weight"] 
                    live_lm = instance.state_dict()[key]
                    assert torch.equal(ckpt_lm, live_lm), "lm_head.weight was not loaded correctly"
                    break

            ckpt_w = fixed_state["speaker_projection.linear.weight"]
            live_w = instance.state_dict()["speaker_projection.linear.weight"]
            assert torch.equal(ckpt_w, live_w), "speaker_projection.linear.weight was not loaded correctly"

            return instance
    
    @torch.no_grad()
    def generate(
        self,
        text: str,
        speaker_embedding: torch.Tensor,
        **kwargs
    ):
        device = next(self.parameters()).device
        speaker_embedding = speaker_embedding.to(device)
        
        text_tokens = tokenizer(text).input_ids
        input_ids_1 = [start_of_human] + text_tokens + [end_of_text, end_of_human, start_of_speaker]
        input_ids_2 = [end_of_speaker, start_of_gpt, start_of_audio]
        input_ids_1 = torch.tensor(input_ids_1, dtype=torch.long).unsqueeze(0).to(device)
        input_ids_2 = torch.tensor(input_ids_2, dtype=torch.long).unsqueeze(0).to(device)
        embds_1 = self.embedding_layer(input_ids_1)
        embds_2 = self.embedding_layer(input_ids_2)

        speaker_projection = self.speaker_projection(speaker_embedding.squeeze(1))

        print('embds_1', embds_1.shape)
        print('speaker_projection', speaker_projection.shape)
        print('embds_2', embds_2.shape)
        inputs_embeds = torch.cat([embds_1.squeeze(), speaker_projection, embds_2.squeeze()], dim=0).unsqueeze(0)
        attention_mask = torch.ones(inputs_embeds.size()[:-1], dtype=torch.long, device=device)

        output_tokens =  self.model.generate(
            inputs_embeds=inputs_embeds, # !! make sure skip embed step
            attention_mask=attention_mask,
            temperature=0.6,
            top_p=0.95,
            top_k=50,
            repetition_penalty=1.1,
            do_sample=True,
            max_new_tokens=self.max_new_tokens,
            eos_token_id=self.end_of_audio,
        )

        print('output_tokens', output_tokens.shape)
        # print(output_tokens)

        start_audio_idx = (output_tokens[0] == start_of_audio).nonzero(as_tuple=True)[0]
        if len(start_audio_idx) == 0:
            output_tokens = output_tokens[0].tolist()
        else:
            print('start_audio_idx', start_audio_idx)
            output_tokens = output_tokens[0][start_audio_idx.item() + 1:].tolist()

        print('output_tokens length:', len(output_tokens))

        if output_tokens[-1] == end_of_audio:
            output_tokens = output_tokens[:-1]

        assert len(output_tokens) % 7 == 0, "Token length must be divisible by 7"
        return output_tokens

    def forward(
            self,
            input_ids: torch.Tensor,
            speaker_embeddings: torch.Tensor,
            **kwargs
        ):

        speaker_embeddings = speaker_embeddings.view(-1, SPEAKER_EMBEDDING_DIM)
        B, _ = speaker_embeddings.size()
        
        device = next(self.parameters()).device
        input_ids, speaker_embeddings = input_ids.to(device), speaker_embeddings.to(device)

        input_ids[input_ids == -100] = pad_token
        placeholder_ids = (input_ids == pad_token).nonzero(as_tuple=True)[1]
        assert len(placeholder_ids) == B, "Placeholder ids must be equal to batch size"

        with torch.no_grad():
            speaker_projections = self.speaker_projection(speaker_embeddings)
        
        labels = input_ids.clone()
        for id in placeholder_ids:
            labels[:, id] = -100

        model_inputs = self.embedding_layer(input_ids)

        for ix, id in enumerate(placeholder_ids):
            model_inputs[:, id, :] = speaker_projections[ix, :]

        if model_inputs.size(1) > MAX_SEQ_LENGTH:
            print(f'model_inputs truncated by {model_inputs.size(1) - MAX_SEQ_LENGTH} tokens')
            model_inputs = model_inputs[:, :MAX_SEQ_LENGTH]
            labels = labels[:, :MAX_SEQ_LENGTH]

        attention_mask = torch.ones(model_inputs.size(0), model_inputs.size(1), dtype=torch.long, device=device)

        out = self.model(inputs_embeds=model_inputs, attention_mask=attention_mask, labels=labels, return_dict=True)

        return out.loss, out.logits
    

class ClearCacheCallback(TrainerCallback):
    def __init__(self, n_steps=10):
        self.n_steps = n_steps
        self.step = 0

    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step % self.n_steps == 0:
            gc.collect()
            torch.cuda.empty_cache()
            if "model" in kwargs and hasattr(kwargs["model"], "trainer"):
                kwargs["model"].trainer.accelerator.clear()


if __name__ == "__main__":
    model_name = "canopylabs/orpheus-3b-0.1-pretrained"
    output_dir = "model-for-voice-cloning-0.2"

    repo_id = "edwindn/voice_cloning_dataset" # input_ids, speaker_embedding
    snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        revision="main",        
        max_workers=NUM_WORKERS,
    ) 
    dataset = load_dataset(repo_id, split="train")
    dataset = dataset.shuffle(seed=42)
    print(f'len dataset: {len(dataset)}')

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    print(f"Using device: {device}")

    if int(os.environ.get("LOCAL_RANK", -1)) in [-1, 0]:
        wandb.init(project="speaker-embedding")

    SpeakerModelingLM.register_for_auto_class("AutoModelForCausalLM")
    model = SpeakerModelingLM.from_pretrained(model_name, load_mode="train")
    # model = SpeakerModelingLM.from_pretrained("../checkpoints/checkpoint-10000", load_mode="local")

    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=1,
        learning_rate=2e-5,
        num_train_epochs=1,
        save_steps=10000,
        bf16=True,
        fsdp=["auto_wrap"],
        logging_dir="logs",
        logging_steps=1,
        remove_unused_columns=False,
        report_to="wandb" if int(os.environ.get("LOCAL_RANK", -1)) in [-1, 0] else None,
        save_safetensors=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        callbacks=[ClearCacheCallback()],
        #data_collator=collate_fn,
    )

    print("training")
    trainer.train()

    print("saving")
    trainer.push_to_hub(output_dir, safe_serialization=False)

