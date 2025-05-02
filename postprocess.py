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


"""
load in downloaded mp3 videos and transcribe
"""

files = [f for f in os.listdir('mp3_downloads') if f.endswith('.mp3')]
files

AUDIO_DIR = 'mp3_downloads/'
MAX_AUDIO_DURATION = 60

ELEVENLABS_API_KEY="sk_a6af254a6712f67b1925b7fcc37b47ad24685e624a0e532c"
client = ElevenLabs(api_key=ELEVENLABS_API_KEY)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

try:
    from pyannote.audio import Pipeline as PyannotePipeline
except ImportError:
    PyannotePipeline = None

snac = SNAC.from_pretrained("hubertsiuzdak/snac_24khz").eval()
snac = snac.to(device)

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
    """
    Read the entire file into memory once, then call Scribe exactly one time.
    Retry once on network error.
    Returns list of word‐timestamp objects.
    """
    audio_bytes = open(path, "rb").read()
    for attempt in range(max_retries+1):
        try:
            resp = client.speech_to_text.convert(
                model_id=model,
                file=audio_bytes,
                timestamps_granularity="word",
            )
            return resp.words
        except ElevenLabsException as e:
            if attempt < max_retries:
                time.sleep(1)
                continue
            else:
                raise

def diarize_audio(path: str):
    """
    Return list of {"start": sec, "end": sec, "speaker": str} by diarizing
    via ElevenLabs API, falling back to Pyannote if that fails.
    """
    audio_bytes = open(path, "rb").read()
    # try ElevenLabs diarization
    try:
        resp = client.speech_to_text.diarize(
            file=audio_bytes,
            model_id="diarization_v1",  # adjust if needed
        )
        segments = [
            {"start": seg.start, "end": seg.end, "speaker": seg.speaker_label}
            for seg in resp.segments
        ]
        return segments
    except Exception:
        if PyannotePipeline is None:
            raise RuntimeError("Pyannote not installed for fallback diarization")
        pipeline = PyannotePipeline.from_pretrained("pyannote/speaker-diarization", use_auth_token="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")
        diarization = pipeline(path)
        segments = [
            {"start": turn.start, "end": turn.end, "speaker": label}
            for turn, _, label in diarization.itertracks(yield_label=True)
        ]
        return segments

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
    words = transcribe_with_scribe(audio_path, **scribe_kwargs)
    text = " ".join(w.text for w in words if getattr(w, "type", None) == "word")
    return text

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


embedding_model = SpeakerRecognition.from_hparams(
    source="speechbrain/spkrec-ecapa-voxceleb",
    savedir="pretrained_models/spkrec-ecapa-voxceleb"
)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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


from speechbrain.pretrained import SpeakerRecognition
embedding_model = SpeakerRecognition.from_hparams(
    source="speechbrain/spkrec-ecapa-voxceleb",
    savedir="pretrained_models/spkrec-ecapa-voxceleb"
)

def embed_speaker(audio):
    signal, fs = torchaudio.load(audio)
    signal = torchaudio.transforms.Resample(fs, 16000)(signal)
    return embedding_model.encode_batch(signal)



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
    return out.tolist()



dataset = []

for file in tqdm(files):
    embedding = get_embedding(AUDIO_DIR + file)
    signal, fs = torchaudio.load(AUDIO_DIR + file)

    if signal.shape[1] > MAX_AUDIO_DURATION * fs:
        signal = signal[:, :MAX_AUDIO_DURATION * fs]

    audio, text = diarize_and_transcribe(AUDIO_DIR + file)


    speaker_embedding = embed_speaker(audio)
    codes = encode_audio(audio)
    assert len(codes) % 7 == 0
    
    dataset.append({
        "text": text,
        "codes_list": codes,
        "speaker_embedding": speaker_embedding
    })
    
dataset = Dataset.from_list(dataset)
dataset.push_to_hub("edwindn/voice_cloning_finetune_0.1", split="train", private=True)