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
import sys

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

if __name__ == "__main__":
    # model = SpeakerModelingLM.from_pretrained("model-for-voice-cloning/checkpoint-80000", load_mode="local").eval().to(device)
    model = SpeakerModelingLM.from_pretrained("edwindn/model-for-voice-cloning-0.2", load_mode="online").eval().to(device)

    tokenizer = AutoTokenizer.from_pretrained("canopylabs/orpheus-3b-0.1-pretrained")
    snac = SNAC.from_pretrained("hubertsiuzdak/snac_24khz").eval()
    snac = snac.to(device)

    assert len(sys.argv) >= 2, "Please provide a reference audio file"
    ref_audios = sys.argv[1:]

    sample_text = "Hey, this is a test of voice cloning. I wonder if I sound like the original? Ha, I bet you can't tell the difference."

    from speechbrain.pretrained import SpeakerRecognition
    embedding_model = SpeakerRecognition.from_hparams(
        source="speechbrain/spkrec-ecapa-voxceleb",
        savedir="pretrained_models/spkrec-ecapa-voxceleb"
    )
    
    for ref_audio in ref_audios:
        signal, fs = torchaudio.load(ref_audio)
        print('original sr ', fs)
        signal = torchaudio.transforms.Resample(fs, AUDIO_EMBEDDING_SR)(signal) # 16kHz
        speaker_embedding = embedding_model.encode_batch(signal)
        speaker_embedding = speaker_embedding.to(device)
        print('speaker embedding ', speaker_embedding.shape)

        output_tokens = model.generate(text=sample_text, speaker_embedding=speaker_embedding)
        codes = detokenize_codes(output_tokens)
        with torch.inference_mode():
                reconstructed_audio = snac.decode(codes)
        
        audio = reconstructed_audio.squeeze().cpu().numpy()
        write(f"cloned_{ref_audio}", SNAC_SAMPLE_RATE, audio)
