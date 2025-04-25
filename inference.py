import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch.nn as nn
from typing import Optional
from snac import SNAC
import numpy as np
from scipy.io.wavfile import write
import torch.nn.functional as F
from dotenv import load_dotenv
from huggingface_hub import login as hf_login
import os

# ---------------------- #

llama_token_end = 128256
snac_vocab_size = 4096
start_of_text = 128000
end_of_text = 128001

start_of_human = llama_token_end + 1
end_of_human = llama_token_end + 2

start_of_gpt = llama_token_end + 3
end_of_gpt = llama_token_end + 4

start_of_audio = llama_token_end + 5
end_of_audio = llama_token_end + 6

pad_token = llama_token_end + 7

audio_token_start = llama_token_end + 10

# ---------------------- #


class OrpheusInference(nn.Module):
    def __init__(self, device):
        super().__init__()
        self.model = model.to(device)
        self.tokenizer = tokenizer
        self.snac = snac.to(device)
        self.device = device

        self.end_of_audio = end_of_audio
        self.max_new_tokens = 250 * 7

    def tokens_to_snac(
            self,
            tokens: list[int],
    ):  
        print(len(tokens))
        assert len(tokens) % 7 == 0, "Token length must be divisible by 7"
        tokens = torch.tensor(tokens, device=self.device).reshape(-1, 7) - audio_token_start
        
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

    def forward(
            self,
            text: str,
    ):
                
        input_ids = self.tokenizer(text).input_ids
        input_ids = [start_of_human] + input_ids + [end_of_text, end_of_human, start_of_gpt, start_of_audio] # start_of_text is given by tokenizer
        input_ids = torch.tensor(input_ids, dtype=torch.long).unsqueeze(0).to(self.device)

        model_input = {
            "input_ids": input_ids,
            "attention_mask": torch.ones_like(input_ids),
            "temperature": 0.6,
            "top_p": 0.95,
            "top_k": 50,
            "repetition_penalty": 1.1,
            "do_sample": True,
            "max_new_tokens": self.max_new_tokens,
            "eos_token_id": self.end_of_audio,
        }

        with torch.no_grad():
            output_tokens = self.model.generate(**model_input)

        start_audio_idx = (output_tokens[0] == start_of_audio).nonzero(as_tuple=True)[0]
        output_tokens = output_tokens[0][start_audio_idx + 1:].tolist()
        print(len(output_tokens))

        if output_tokens[-1] == end_of_audio:
            output_tokens = output_tokens[:-1]
        
        codes = self.tokens_to_snac(output_tokens)

        with torch.inference_mode():
            reconstructed_audio = snac.decode(codes)

        return reconstructed_audio


if __name__ == "__main__":

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    repo_id = "edwindn/orpheus-3b-voiceFinetune-0.1"

    print(f"Loading model from {repo_id}")

    if device == "cuda":
        model = AutoModelForCausalLM.from_pretrained(
            repo_id,
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
            low_cpu_mem_usage=True,
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            repo_id,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
        )

    load_dotenv()
    hf_login(os.getenv("HF_TOKEN_EDWIN"))

    tokenizer = AutoTokenizer.from_pretrained("canopylabs/orpheus-3b-0.1-pretrained")

    snac = SNAC.from_pretrained("hubertsiuzdak/snac_24khz").eval()
    snac = snac.to('cpu')

    snac_sample_rate = 24e3

    # ---------------------- #

    orpheus = OrpheusInference(device)
    #sample_text = "I am tara, one of Orpheus's voices <giggle>. I can speak pretty well considering I only have 1 billion parameters."
    #sample_text = "Hey there guys. It's Tara here, and let me introduce you to Zac... who seems to be asleep. Zac, it's time to wakey-wakey!"
    sample_text = "I am Orpheus-3b, and I was finetuned on Maya's voice. Do I sound like her at all?"

    reconstructed_audio = orpheus(sample_text)
    
    audio_data = reconstructed_audio.squeeze().cpu().numpy()
    
    write("reconstructed_audio.wav", int(snac_sample_rate), audio_data)

    