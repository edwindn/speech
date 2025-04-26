from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
from transformers.modeling_outputs import CausalLMOutputWithPast
import numpy as np
import torch
import torch.nn as nn
from datasets import load_dataset
import wandb
from accelerate import Accelerator
from huggingface_hub import HfApi, snapshot_download

dsn = "amuvarma/em-EN"
suffix = "v1"

# Initialize accelerator first
accelerator = Accelerator()

learning_rate = 2e-5

if accelerator.is_main_process:
    wandb.init(project=f"training_bifrost_{suffix}", name=f"r1-{learning_rate}")

# Step 1: initialize text LLM and speech LLM
text_model_name = "meta-llama/Llama-3.2-3B-Instruct"
# Don't move to cuda immediately - let accelerator handle device placement
text_model = AutoModelForCausalLM.from_pretrained(text_model_name)

speech_model_name = "canopylabs/orpheus-3b-0.1-pretrained"
speech_model = AutoModelForCausalLM.from_pretrained(speech_model_name)

class Bifrost(nn.Module):
    def _init_(self, text_model_hidden_size, speech_model_embedding_size):
        super(Bifrost, self)._init_()
        self.linear = nn.Linear(text_model_hidden_size, speech_model_embedding_size)
        self.fc_1 = nn.Linear(speech_model_embedding_size, 4*speech_model_embedding_size, bias=True)
        self.fc_2 = nn.Linear(speech_model_embedding_size, 4*speech_model_embedding_size, bias=True)
        self.proj = nn.Linear(4*speech_model_embedding_size, speech_model_embedding_size, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.linear(x)
        x_fc_1 = self.fc_1(x)
        x_fc_2 = self.fc_2(x)
        x = torch.nn.functional.silu(x_fc_1) * x_fc_2
        return self.proj(x)

class HydraForCausalLM(AutoModelForCausalLM):
    def _init_(self, config):
        super()._init_(config)
        pass

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, **kwargs):
        model = super().from_pretrained(pretrained_model_name_or_path, **kwargs)
        model.bifrost = Bifrost(model.config.hidden_size, model.config.hidden_size)
        model.forward = cls.forward_v2._get_(model, type(model))
        return model

    def forward_v2(
            self, 
            hidden_states, 
            speech_ids, 
            logits_to_keep = 0,
            return_dict = None,
            **kwargs):
        
        # projected_embeds = self.bifrost(hidden_states)
        projected_embeds = hidden_states
        start_of_human_embed = self.model.embed_tokens(torch.tensor([[128259]], dtype=torch.int64).to(hidden_states.device))
        end_of_human_embed = self.model.embed_tokens(torch.tensor([[128260]], dtype=torch.int64).to(hidden_states.device))
        start_of_ai_embed = self.model.embed_tokens(torch.tensor([[128261]], dtype=torch.int64).to(hidden_states.device))
        start_of_speech_embed = self.model.embed_tokens(torch.tensor([[128257]], dtype=torch.int64).to(hidden_states.device))

        # Concatenate the embeddings
        extended_embeddings = torch.cat(
            (start_of_human_embed, projected_embeds, end_of_human_embed, start_of_ai_embed, start_of_speech_embed), dim=1
        )

        speech_embeds = self.model.embed_tokens(speech_ids)
        inputs_embeds = torch.cat((extended_embeddings, speech_embeds), dim=1)

        batch_size = inputs_embeds.shape[0]
        seq_length = inputs_embeds.shape[1]
        speech_length = speech_ids.shape[1]

        # Initialize all labels as -100
        labels = torch.full((batch_size, seq_length), -100, dtype=torch.long, device=inputs_embeds.device)

        # Fill in the speech_ids values at the end of the sequence
        start_idx = seq_length - speech_length
        labels[:, start_idx:] = speech_ids


        # Call the underlying base_model
        outputs = self.model(
            inputs_embeds=inputs_embeds,
            **kwargs 
        )

        hidden_states = outputs[0]
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(hidden_states[:, slice_indices, :])

        loss = None
        if labels is not None:
            loss = self.loss_function(logits=logits, labels=labels, vocab_size=self.config.vocab_size, **kwargs)

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

# Create the Hydra model
hydra_model = HydraForCausalLM.from_pretrained(pretrained_model_name_or_path=speech_model_name)

# Resize the text model's token embeddings
text_model.resize_token_embeddings(128256 + 10)
# Move to proper device with accelerator (will happen later)

class HydraDataCollator:
    def _init_(self, text_model, accelerator):
        self.greeting = "Hello world."
        self.text_model = text_model
        self.accelerator = accelerator

        
    def _turn_text_to_embeds(self, text):
        input_ids = tokenizer(text, return_tensors="pt").input_ids.to(self.text_model.device)
        with torch.no_grad():
            # Instead of unwrapping (which can move the model off the GPU),
            # check if the model is wrapped and then use its underlying module.
            if hasattr(self.text_model, "module"):
                inputs_embeds = self.text_model.module.get_input_embeddings()(input_ids)
            else:
                inputs_embeds = self.text_model.get_input_embeddings()(input_ids)
        return inputs_embeds
    
    def _generate_next_message_text(self, messages):
        return "Hello world."

    def _call_(self, features):
        messages = features[0]["messages"]
        speech_ids = torch.tensor([features[0]["speech_ids"]], dtype=torch.int64)
        # Turn text into embeddings
        text_embeds = self._turn_text_to_embeds(text)
        batch = {
            "speech_ids": speech_ids,
            "hidden_states": text_embeds
        }
        return batch

# Step 2: initialize tokenizers
tokenizer = AutoTokenizer.from_pretrained(text_model_name)
speech_tokenizer = AutoTokenizer.from_pretrained(speech_model_name)

# Step 3: load dataset and rename column
dataset = load_dataset(dsn, split="train")
dataset = dataset.rename_column("codes_list", "speech_ids")
dataset = dataset.shuffle(seed=42)

# Freeze speech_model parameters except for bifrost
# for param in hydra_model.parameters():
#     param.requires_grad = False
# for param in hydra_model.bifrost.parameters():
#     param.requires_grad = True

for param in text_model.parameters():
    param.requires_grad = False

# Use accelerator to prepare the models
text_model = accelerator.prepare(text_model)



training_args = TrainingArguments(
    output_dir="output",
    per_device_train_batch_size=1,
    gradient_accumulation_steps=1,
    num_train_epochs=1,
    learning_rate=learning_rate,
    logging_steps=1,
    evaluation_strategy="no",
    report_to="wandb",
    push_to_hub=False,
    dataloader_pin_memory=False,
    remove_unused_columns=False,
    lr_scheduler_type="cosine",
    bf16=True,
    save_steps=1000, 
    # fsdp = "auto_wrap"
)

trainer = Trainer(
    model=hydra_model,
    args=training_args,
    train_dataset=dataset,
    data_collator=HydraDataCollator(text_model, accelerator),
)

trainer.train()