import os
from tqdm import tqdm
from pydub import AudioSegment
from elevenlabs import ElevenLabs
import torch
from speechbrain.pretrained import SpeakerRecognition
import time
import torchaudio


"""
load in downloaded mp3 videos and transcribe
"""

files = [f for f in os.listdir('mp3_downloads') if f.endswith('.mp3')]
files

AUDIO_DIR = 'mp3_downloads/'
MAX_AUDIO_DURATION = 60

ELEVENLABS_API_KEY="sk_a6af254a6712f67b1925b7fcc37b47ad24685e624a0e532c"
client = ElevenLabs(api_key=ELEVENLABS_API_KEY)

try:
    from pyannote.audio import Pipeline as PyannotePipeline
except ImportError:
    PyannotePipeline = None

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
    out_dir: str = "audio_dataset",
    out_audio_name: str = "test.mp3",
    out_txt_name: str = "test.txt"
):
    """
    1) Diarize audio_path
    2) Find main speaker
    3) Extract & concatenate that speaker's audio
    4) Save new audio + its plain-text transcript
    """
    os.makedirs(out_dir, exist_ok=True)

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


for file in tqdm(files):
    embedding = get_embedding(AUDIO_DIR + file)
    signal, fs = torchaudio.load(AUDIO_DIR + file)

    if signal.shape[1] > MAX_AUDIO_DURATION * fs:
        signal = signal[:, :MAX_AUDIO_DURATION * fs]

    audio, text = diarize_and_transcribe(AUDIO_DIR + file)
    
