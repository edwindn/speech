import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch.nn as nn
from typing import Optional
from snac import SNAC
import numpy as np
from scipy.io.wavfile import write
import torch.nn.functional as F
from dotenv import load_dotenv
import os
from pyannote.audio import Model, Inference
from speaker_embedding import SpeakerModelingLM
import torchaudio

load_dotenv()


SPEAKER_EMBEDDING_DIM = 192
LLAMA_EMBEDDING_DIM = 3072
AUDIO_EMBEDDING_SR = 16000
NUM_WORKERS = os.cpu_count()
MAX_SEQ_LENGTH = 2048
SNAC_SAMPLE_RATE = 24000

# DE-DUPLICATE CODES

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# model_name = "edwindn/llama-voice-cloning"
model_name = "model-for-voice-cloning/checkpoint-70000"

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

model = SpeakerModelingLM.from_pretrained(model_name).eval().to(device)

model.push_to_hub("edwindn/model-for-voice-cloning", safe_serialization=False)
print("Model uploaded successfully!")
