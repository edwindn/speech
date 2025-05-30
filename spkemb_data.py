from datasets import load_dataset, Dataset
from huggingface_hub import snapshot_download, login
import torch
from transformers import AutoTokenizer
import os
from tqdm import tqdm
import random
import multiprocessing as mp
from functools import partial

NUM_WORKERS = min(os.cpu_count(), 50)
MAX_SEQ_LENGTH = 8192
NUM_DS_CHUNKS = 200

tokenizer = AutoTokenizer.from_pretrained("canopylabs/orpheus-3b-0.1-pretrained")

PAD_TOKEN = 128256 + 7

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

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


def map_fn(batch):
    text = batch["text"]
    text_tokens = tokenizer(text).input_ids

    audio_tokens = batch["codes_list"]
    audio_tokens = torch.tensor(audio_tokens)
    c0 = audio_tokens[::7]
    indices = torch.where(c0[:-1] == c0[1:])[0]
    if len(indices) > 0:
        mask_indices = (indices.unsqueeze(1) * 7 + torch.arange(7, device=indices.device)).flatten()
        mask = torch.ones(len(audio_tokens), dtype=torch.bool, device=audio_tokens.device)
        mask[mask_indices] = False
        audio_tokens = audio_tokens[mask]

    input_ids = start + text_tokens + middle1 + [PAD_TOKEN] + middle2 + audio_tokens.tolist() + end
    assert input_ids.count(PAD_TOKEN) == 1, f"Expected exactly one placeholder token"

    return {
        "input_ids": input_ids,
        "speaker_embedding": batch["speaker_embedding"],
    }


repo_id = "amuvarma/snac_and_embs" # codes_list, speaker_embedding, text
snapshot_download(
    repo_id=repo_id,
    repo_type="dataset",
    revision="main",        
    max_workers=NUM_WORKERS,
) 
dataset = load_dataset(repo_id, split="train")
dataset = dataset.shuffle(seed=42)

print(f'len dataset: {len(dataset)}')

dataset = dataset.map(map_fn, num_proc=50, batched=False, remove_columns=dataset.column_names)

print('finished mapping')
avg_length = sum(len(item['input_ids']) for item in dataset.select(range(10000))) / 10000
print(f"Average length of input_ids: {avg_length}")

def process_chunk(dataset_chunk, dcix=0):
    print(f"Starting process_chunk {dcix} with {len(dataset_chunk)} items", flush=True)
    # length of chunk = len(text) + len(codes_list) + 1 + 9 = len(input_ids)

    train_dataset_chunk = []
    current_len = 0
    last_chunk = []
    last_embs = []

    for row in tqdm(dataset_chunk, desc=f"Processing chunk {dcix}"):
        input_ids = row['input_ids']
        speaker_embedding = row['speaker_embedding']

        row_len = len(input_ids)

        if current_len + row_len <= MAX_SEQ_LENGTH:
            last_chunk.extend(input_ids)
            last_embs.extend(speaker_embedding)
            current_len += row_len

        else:
            train_dataset_chunk.append({
                "input_ids": last_chunk,
                "speaker_embeddings": last_embs
            })

            if random.random() < 0.0001:
                print(f"Chunk {len(train_dataset_chunk)} for dataset chunk {dcix}: {len(last_chunk)}", flush=True)

            last_chunk = input_ids
            last_embs = speaker_embedding
            current_len = row_len

    print(f"finished process {dcix}", flush=True)
    return train_dataset_chunk

# train_dataset = process_chunk(dataset)
# train_dataset = Dataset.from_list(train_dataset)


NUM_DS_SHARDS = 10
NUM_CHUNKS = 50

login()

# Process the dataset in 10 sequential chunks
chunk_size = len(dataset) // NUM_DS_SHARDS

dataset_chunks = [dataset.shard(num_shards=NUM_DS_SHARDS, index=i) for i in range(NUM_DS_SHARDS)]

for idx, ds in enumerate(dataset_chunks):
    ds_chunks = [ds.shard(num_shards=NUM_CHUNKS, index=i) for i in range(NUM_CHUNKS)]

    # Process chunks in parallel
    pool = mp.Pool(processes=NUM_CHUNKS)
    try:
        process_chunk_with_index = partial(process_chunk)
        results = pool.starmap(process_chunk_with_index, [(chunk, i) for i, chunk in enumerate(ds_chunks)])
    finally:
        pool.close()
        pool.join()
    print(f"Finished processing {NUM_CHUNKS} chunks")

    # Combine results
    train_dataset = []
    for result in results:
        train_dataset.extend(result)

    print("Combined chunks")

    train_dataset = Dataset.from_list(train_dataset)
    #train_dataset = train_dataset.shuffle(seed=42)

    train_dataset.push_to_hub(
        f"edwindn/voice_cloning_dataset_{idx}",
        split="train",
        private=True
    )

    print(f"----- Pushed dataset {idx} to hub -----")

    # delete the item to free memory
    del ds
    del ds_chunks
    del train_dataset
    del results
    del process_chunk_with_index
    del pool



