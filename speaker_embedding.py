from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import torch.nn as nn
from pyannote.audio import Model, Inference
from dotenv import load_dotenv
import os
from huggingface_hub import login as hf_login

load_dotenv()
hf_login(os.getenv("HF_TOKEN"))

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
        
        self.projection = GatedMLP().to(device)
        self.llama = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.1-3B").to(device)
        self.tokenizer = tokenizer

    def forward(self, input_ids, attention_mask):
        return
    

embedding = embed_speaker("reconstructed_audio.wav")
print(embedding.shape)


