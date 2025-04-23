from datasets import load_dataset, concatenate_datasets
from huggingface_hub import snapshot_download, login as hf_login
import os
from dotenv import load_dotenv

load_dotenv()
hf_login(os.getenv("HF_TOKEN_EDWIN"))
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
voice_effects_test = load_dataset(voice_effects_path, split="test")
voice_effects_val = load_dataset(voice_effects_path, split="val")
voice_effects = concatenate_datasets([voice_effects_test, voice_effects_val])
print(voice_effects)
print(len(voice_effects))
print(voice_effects[0])

speaking_path = "badayvedat/VCTK"
speaking = snapshot_download(
    repo_id=speaking_path,
    repo_type="dataset",
    revision="main",
    max_workers=CPU_COUNT,
)
speaking = load_dataset(speaking_path, split="train")
print(speaking)
print(len(speaking))
print(speaking[0])
# speaking_all = load_dataset("mozilla-foundation/common_voice_17_0", "en", split="train", streaming=True)
# speaking = [next(iter(speaking_all)) for _ in range(100)]






