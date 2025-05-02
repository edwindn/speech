from datasets import load_dataset, concatenate_datasets
from huggingface_hub import snapshot_download
import os

all_ds = []

for i in range(10):
    snapshot_download(repo_id=f"edwindn/voice_cloning_dataset_{i}", repo_type="dataset", num_workers=os.cpu_count())
    ds = load_dataset(f"edwindn/voice_cloning_dataset_{i}", split=f"train")
    all_ds.append(ds)

all_ds = concatenate_datasets(all_ds)

all_ds.push_to_hub("edwindn/voice_cloning_dataset", split="train", private=True)



