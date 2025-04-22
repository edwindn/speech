from datasets import load_dataset
from huggingface_hub import snapshot_download
import os

"""
finetuning possible:
https://huggingface.co/datasets/keithito/lj_speech -> single speaker
https://huggingface.co/datasets/badayvedat/VCTK

load_dataset("mozilla-foundation/common_voice_13_0", "en", split="train") -> common voice english

"""

CPU_COUNT = os.cpu_count()

voice_effects_path = "lmms-lab/vocalsound" # cough, sigh, laughter, sniff, sneeze, throat clearing
voice_effects = snapshot_download(
    repo_id=voice_effects_path,
    repo_type="dataset",
    revision="main",
    max_workers=CPU_COUNT,
)
voice_effects = load_dataset(voice_effects_path, split="train")
print(voice_effects)
print(len(voice_effects))


speaking_path = "mozilla-foundation/common_voice_13_0"
speaking = snapshot_download(
    repo_id=speaking_path,
    repo_type="dataset",
    revision="main",
    max_workers=CPU_COUNT,
)
speaking = load_dataset(speaking_path, split="train")
print(speaking)
print(len(speaking))





