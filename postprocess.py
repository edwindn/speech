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
import soundfile as sf

"""
transcribe and tokenize mp3 audios
"""

# ----------------------
# Configuration
# ----------------------
AUDIO_DIR = 'chrisw/'
MAX_AUDIO_DURATION = 60  # seconds

# Hugging Face repo to push partial/full dataset
HF_REPO = "edwindn/voice_cloning_finetune_0.2"
HF_SPLIT = "train"

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

    audio = torch.tensor(audio_input, dtype=torch.float32).view(1, 1, -1)
    if sample_rate != 24000:
        audio = torch.nn.functional.interpolate(
            audio,
            scale_factor=24000/sample_rate,
            mode='linear',
            align_corners=False
        )

    audio_tokens = encode_audio(audio.to(device)).cpu().tolist()

    text = open(FILE + '.txt').read().strip()
    if not len(text):
        return []
    text_tokens = tokenizer(text).input_ids

    tokens = start + text_tokens + middle + audio_tokens + end
    return tokens

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

try:
    from pyannote.audio import Pipeline as PyannotePipeline
except ImportError:
    PyannotePipeline = None

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
    audio_bytes = open(path, "rb").read()
    global client
    # Try rotating keys for diarization
    for round in range(len(ELEVENLABS_API_KEYS)):
        if client is None:
            break
        try:
            resp = client.speech_to_text.diarize(
                file=audio_bytes,
                model_id="diarization_v1",
            )
            return [{"start": seg.start, "end": seg.end, "speaker": seg.speaker_label}
                    for seg in resp.segments]
        except Exception as e:
            print(f"Diarization error with key, rotating: {e}")
            client = get_elevenlabs_client()
    # fallback to pyannote if available
    if PyannotePipeline:
        pipeline = PyannotePipeline.from_pretrained(
            "pyannote/speaker-diarization",
            use_auth_token=True
        )
        diarization = pipeline(path)
        return [{"start": turn.start, "end": turn.end, "speaker": label}
                for turn, _, label in diarization.itertracks(yield_label=True)]
    raise RuntimeError("Diarization failed and no fallback available.")

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
    return main_audio, text


def get_embedding(ref_audio):
    signal, fs = torchaudio.load(ref_audio)
    print('original sr ', fs)
    signal = torchaudio.transforms.Resample(fs, 16000)(signal)
    # convert to mono
    if signal.shape[0] > 1:
        signal = torch.mean(signal, dim=0, keepdim=True)
    speaker_embedding = embedding_model.encode_batch(signal)
    speaker_embedding = speaker_embedding.to(device)
    return speaker_embedding

def embed_speaker(audio):
    samples = np.array(audio.get_array_of_samples(), dtype=np.float32)
    if audio.channels > 1:
        samples = samples.reshape(-1, audio.channels).mean(axis=1)
    signal = torch.from_numpy(samples).unsqueeze(0).unsqueeze(0)
    if audio.frame_rate != 16000:
        signal = torchaudio.transforms.Resample(audio.frame_rate, 16000)(signal)

    print('signal shape: ', signal.shape)
    signal = signal.view(1, -1)
    with torch.inference_mode():
        emb = embedding_model.encode_batch(signal.to(device))
    return emb

def encode_audio(audio):
    """
    must be a tensor of shape B, 1, T
    returns audio tokens ready for orpheus
    """

    samples = np.array(audio.get_array_of_samples(), dtype=np.float32)
    if audio.channels > 1:
        samples = samples.reshape(-1, audio.channels).mean(axis=1)
    signal = torch.from_numpy(samples).unsqueeze(0).unsqueeze(0)
    if audio.frame_rate != 24000:
        signal = torchaudio.transforms.Resample(audio.frame_rate, 24000)(signal)
    
    signal = signal.view(1, 1, -1)

    with torch.inference_mode():
        codes = snac.encode(signal)

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
    return out.tolist()



def process_and_build_dataset(files):
    dataset = []
    for idx, file in enumerate(tqdm(files)):
        try:
            path = os.path.join(AUDIO_DIR, file)
            embedding = get_embedding(path)
            signal, fs = torchaudio.load(path)

            if signal.shape[1] > MAX_AUDIO_DURATION * fs:
                signal = signal[:, :MAX_AUDIO_DURATION * fs]

            audio, text = diarize_and_transcribe(path)

            speaker_embedding = embed_speaker(audio)
            codes = encode_audio(audio)
            assert len(codes) % 7 == 0

            dataset.append({
                "text": text,
                "codes_list": codes,
                "speaker_embedding": speaker_embedding
            })

        except RuntimeError as e:
            print(f"Keys exhausted at index {idx}, saving partial dataset...")
            ds = Dataset.from_list(dataset)
            ds.push_to_hub(HF_REPO, split=HF_SPLIT, private=True)
            print(f"Saved slice [0:{idx}] to HuggingFace")
            return

    ds = Dataset.from_list(dataset)
    ds.push_to_hub(HF_REPO, split=HF_SPLIT, private=True)
    print("All done, full dataset saved.")
    
if __name__ == '__main__':
    files = [f for f in os.listdir(AUDIO_DIR) if f.endswith('.mp3')]
    process_and_build_dataset(files)