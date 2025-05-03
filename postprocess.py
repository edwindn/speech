import os
from tqdm import tqdm
from pydub import AudioSegment
from elevenlabs import ElevenLabs
import torch
from speechbrain.pretrained import SpeakerRecognition
import time
import torchaudio
from datasets import Dataset
from transformers import AutoTokenizer
from snac import SNAC
import numpy as np
import ast
import soundfile as sf

"""
pipeline: youtube.py -> pre_postprocess.py -> postprocess.py

this script: tokenize and embed audio chunks

!! convert to wav before opening
"""

FILE_DIR = "audio_chunks/"
HF_REPO = "edwindn/voice_cloning_finetune_0.1"

tokenizer = AutoTokenizer.from_pretrained("canopylabs/orpheus-3b-0.1-pretrained")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

snac = SNAC.from_pretrained("hubertsiuzdak/snac_24khz").eval().to(device)

embedding_model = SpeakerRecognition.from_hparams(
    source="speechbrain/spkrec-ecapa-voxceleb",
    savedir="pretrained_models/spkrec-ecapa-voxceleb"
).to(device)

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

def encode_audio(audio):
    """
    must be a tensor of shape B, 1, T
    returns audio tokens ready for orpheus
    """

    with torch.inference_mode():
        codes = snac.encode(audio)

    c0 = codes[0].flatten()
    N = c0.size(0)

    c1 = codes[1].flatten().view(N, 2)
    c2 = codes[2].flatten().view(N, 4)
    out = [
        c0,
        c1[:, 0] + snac_vocab_size,
        c2[:, 0] + snac_vocab_size * 2,
        c2[:, 1] + snac_vocab_size * 3,
        c1[:, 1] + snac_vocab_size * 4,
        c2[:, 2] + snac_vocab_size * 5,
        c2[:, 3] + snac_vocab_size * 6
    ]
    out = torch.stack(out, dim=1).flatten()
    #print('out tokens (should be in increasing batches of 7): ', out[:70])

    # remove repeated tokens
    indices = torch.where(c0[:-1] == c0[1:])[0]
    if len(indices) > 0:
        mask_indices = (indices.unsqueeze(1) * 7 + torch.arange(7, device=indices.device)).flatten()
        mask = torch.ones(len(out), dtype=torch.bool, device=out.device)
        mask[mask_indices] = False
        out = out[mask]

    out = out + audio_token_start
    return out


def get_tokens(file_path):
    audio_input, sample_rate = sf.read(file_path)
    
    # Convert to mono if stereo by averaging channels
    if len(audio_input.shape) > 1 and audio_input.shape[1] > 1:
        print(f'found stereo file, shape {audio_input.shape}')
        audio_input = audio_input.mean(axis=1)
        
    audio = torch.tensor(audio_input, dtype=torch.float32).view(1, 1, -1)
    if sample_rate != 24000:
        audio = torch.nn.functional.interpolate(
            audio,
            scale_factor=24000/sample_rate,
            mode='linear',
            align_corners=False
        )

    audio_tokens = encode_audio(audio.to(device)).cpu().tolist()

    text = open(file_path.replace('.wav', '.txt')).read().strip()
    if not len(text):
        return []
    
    text_tokens = tokenizer(text).input_ids
    tokens = start + text_tokens + middle1 + [-100] + middle2 + audio_tokens + end
    return tokens

def embed_speaker(audio):
    signal, fs = torchaudio.load(audio)
    # Convert to mono by averaging channels if stereo
    if signal.size(0) > 1:
        signal = signal.mean(dim=0, keepdim=True)
    if fs != 16000:
        signal = torchaudio.transforms.Resample(fs, 16000)(signal)
    return embedding_model.encode_batch(signal).to(device)


if __name__ == "__main__":
    files = [f[:-4] for f in os.listdir(FILE_DIR) if f.endswith('.txt')]

    dataset = []

    for file in files:
        tokens = get_tokens(FILE_DIR + file + '.wav')
        if not tokens:
            print(f"Skipping {file} because no tokens found")
            continue

        speaker_embedding = embed_speaker(FILE_DIR + file + '.wav')

        dataset.append({
            "input_ids": tokens, # -100 placeholder for speaker embedding
            "speaker_embedding": speaker_embedding
        })

    dataset = Dataset.from_list(dataset)
    dataset.push_to_hub(HF_REPO)






