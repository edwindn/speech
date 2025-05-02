from datasets import load_dataset, concatenate_datasets


all_ds = []

for i in range(10):
    ds = load_dataset("edwindn/voice_cloning_dataset_0", split=f"train_{i}")
    all_ds.append(ds)

all_ds = concatenate_datasets(all_ds)

all_ds.push_to_hub("edwindn/voice_cloning_dataset", split="train", private=True)



