import os
from tqdm import tqdm
from pydub import AudioSegment
from elevenlabs import ElevenLabs
import torch
from speechbrain.pretrained import SpeakerRecognition
import time
import torchaudio
from datasets import Dataset
from snac import SNAC
import numpy as np
import ast
from pyannote.audio import Pipeline as PyannotePipeline

"""
chunk and transcribe audios & save as files
"""

# ----------------------
# Configuration
# ----------------------
AUDIO_DIR = 'chrisw/'
MAX_AUDIO_DURATION = 60  # seconds
SAVE_DIR = 'audio_chunks/'

# ElevenLabs API keys (cycled)
ELEVENLABS_API_KEYS = [
    "sk_a6af254a6712f67b1925b7fcc37b47ad24685e624a0e532c",
    "sk_dfe129dd45fade2811d07894b25be15d62c2487812358511",
    "sk_f4401523d9a6397222850ad22cc0b3f06ad1b370dead24e3",
]
current_key_index = 0

# Hugging Face repo to push partial/full dataset
HF_REPO = "edwindn/voice_cloning_finetune_0.2"
HF_SPLIT = "train"

# ----------------------
# Helper: ElevenLabs client cycling
# ----------------------
def get_elevenlabs_client():
    """Returns an ElevenLabs client with the next available API key, or None if exhausted."""
    global current_key_index
    if current_key_index >= len(ELEVENLABS_API_KEYS):
        return None
    api_key = ELEVENLABS_API_KEYS[current_key_index]
    current_key_index += 1
    return ElevenLabs(api_key=api_key)

# initialize first client
client = get_elevenlabs_client()

# ----------------------
# Models & pipelines
# ----------------------
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

pipeline = PyannotePipeline.from_pretrained(
            "pyannote/speaker-diarization",
            use_auth_token=True
)

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

#@title function definitions

def transcribe_with_scribe(path: str, model: str="scribe_v1", max_retries=1):
    audio_bytes = open(path, "rb").read()
    global client
    for attempt in range(max_retries + 1):
        if client is None:
            break
        try:
            resp = client.speech_to_text.convert(
                model_id=model,
                file=audio_bytes,
                timestamps_granularity="word",
            )
            return resp.words
        except Exception as e:
            print(f"Error with API key {attempt}, rotating key: {e}")
            client = get_elevenlabs_client()
            time.sleep(1)
    # no keys left or all retries failed
    raise RuntimeError("All ElevenLabs API keys exhausted during transcription")

def diarize_audio(path: str):
    print(f"Diarizing {path}")
    audio_bytes = open(AUDIO_DIR + path, "rb").read()

    diarization = pipeline(path)
    return [{"start": turn.start, "end": turn.end, "speaker": label}
            for turn, _, label in diarization.itertracks(yield_label=True)]

def get_main_speaker(segments):
    """
    Given diarization segments, pick the speaker label with longest total duration.
    """
    durations = {}
    for seg in segments:
        durations.setdefault(seg["speaker"], 0.0)
        durations[seg["speaker"]] += seg["end"] - seg["start"]
    # pick the speaker with max total time
    main = max(durations, key=durations.get)
    return main

def extract_and_concat(path: str, segments, speaker_label: str):
    """
    Extract all segments for speaker_label from `path`, concatenate, and return an AudioSegment.
    """
    audio = AudioSegment.from_file(path)
    parts = []
    # sort segments by start
    for seg in sorted(segments, key=lambda s: s["start"]):
        if seg["speaker"] == speaker_label:
            start_ms = int(seg["start"] * 1000)
            end_ms   = int(seg["end"]   * 1000)
            parts.append(audio[start_ms:end_ms])
    if not parts:
        raise RuntimeError(f"No segments found for speaker '{speaker_label}'")
    return sum(parts[1:], parts[0])  # concatenate

def transcribe_to_txt(audio_path: str, **scribe_kwargs):
    """
    Transcribe with Scribe and write plain text (no timestamps) to out_txt_path.
    """
    try:
        words = transcribe_with_scribe(audio_path, **scribe_kwargs)
        text = " ".join(w.text for w in words if getattr(w, "type", None) == "word")
        return text
    except Exception as e:
        print(f"Error transcribing {audio_path}: {e}")
        return ""

def diarize_and_transcribe(
    audio_path: str,
):
    """
    1) Diarize audio_path
    2) Find main speaker
    3) Extract & concatenate that speaker's audio
    4) Save new audio + its plain-text transcript
    """

    segments = diarize_audio(audio_path)

    main = get_main_speaker(segments)
    print(f"Main speaker = {main!r}")

    main_audio = extract_and_concat(audio_path, segments, main)
    text = transcribe_to_txt(audio_path)

    output_path = SAVE_DIR + audio_path.rsplit('.', 1)[0] + '_main.mp3'
    main_audio.export(output_path, format='mp3')

    output_txt = SAVE_DIR + audio_path.rsplit('.', 1)[0] + '_main.txt'
    with open(output_txt, 'w') as f:
        f.write(text)
    return main_audio, text


# def get_embedding(ref_audio):
#     signal, fs = torchaudio.load(ref_audio)
#     print('original sr ', fs)
#     signal = torchaudio.transforms.Resample(fs, 16000)(signal)
#     # convert to mono
#     if signal.shape[0] > 1:
#         signal = torch.mean(signal, dim=0, keepdim=True)
#     speaker_embedding = embedding_model.encode_batch(signal)
#     speaker_embedding = speaker_embedding.to(device)
#     return speaker_embedding

# def embed_speaker(audio):
#     samples = np.array(audio.get_array_of_samples(), dtype=np.float32)
#     if audio.channels > 1:
#         samples = samples.reshape(-1, audio.channels).mean(axis=1)
#     signal = torch.from_numpy(samples).unsqueeze(0).unsqueeze(0)
#     if audio.frame_rate != 16000:
#         signal = torchaudio.transforms.Resample(audio.frame_rate, 16000)(signal)

#     print('signal shape: ', signal.shape)
#     signal = signal.view(1, -1)
#     with torch.inference_mode():
#         emb = embedding_model.encode_batch(signal.to(device))
#     return emb
    
if __name__ == '__main__':
    files = [f for f in os.listdir(AUDIO_DIR) if f.endswith('.mp3')]
    for file in files:
        diarize_and_transcribe(file)