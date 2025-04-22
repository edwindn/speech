from datasets import load_dataset, snapshot_download
import os

CPU_COUNT = os.cpu_count()

ds1 = "lmms-lab/vocalsound" # cough, sigh, laughter, sniff, sneeze, throat clearing

dataset_path = snapshot_download(
    repo_id=ds1,
    repo_type="dataset",
    revision="main",
    max_workers=CPU_COUNT,
)

dataset = load_dataset(dataset_path, split="train")
print(dataset)
print(len(dataset))


