from datasets import load_dataset, concatenate_datasets
from huggingface_hub import login as hf_login
from dotenv import load_dotenv
import os
import torch
from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq, Trainer, TrainingArguments
from scipy.signal import resample
import random
from torch.nn.utils.rnn import pad_sequence
import wandb
import numpy as np

## required packages
import librosa
import soundfile
import accelerate


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
processor = AutoProcessor.from_pretrained("openai/whisper-large-v3")
model = AutoModelForSpeechSeq2Seq.from_pretrained("openai/whisper-large-v3")
model.to(device)
model.train()

EPOCHS = 1

wandb.login(key=os.getenv("WANDB_API_KEY"))
wandb.init(
    project="whisper-finetuning",
    name="training-run",
    config={
        "model_name": "openai/whisper-large-v3",
        "learning_rate": 1e-5,
        "num_train_epochs": EPOCHS,
        "batch_size": 8,
    }
)

load_dotenv()
hf_login(os.getenv("HF_TOKEN_AMUVARMA"))

ds = load_dataset("amuvarma/vas-qa-13", split="train")

sound_idxs = []

for i in range(len(ds)):
    # check if the text entry contains < or >
    if '<' in ds[i]['text'] or '>' in ds[i]['text']:
        sound_idxs.append(i)

ds_sounds = ds.select(sound_idxs)
not_sound_idxs = list(set(range(len(ds))) - set(sound_idxs))
not_sound_idxs = random.sample(not_sound_idxs, len(sound_idxs))
ds_not_sounds = ds.select(not_sound_idxs)
ds_combined = concatenate_datasets([ds_sounds, ds_not_sounds])

# Split the dataset into train and validation
ds_combined = ds_combined.shuffle(seed=42)
split = ds_combined.train_test_split(test_size=0.1)
ds_train = split["train"]
ds_val = split["test"]

def map_fn(entry):
    audio = entry['audio']['array']
    text = entry['text']
    audio = resample(audio, int(len(audio) * 16000 / 48000))
    preproc_audio = processor.feature_extractor(audio, sampling_rate=16000, return_tensors='pt')

    input_features = preproc_audio['input_features'][0] # C should be 80
    labels = processor.tokenizer(text).input_ids
    return {
        'input_features': input_features,
        'labels': labels
    }

ds_train = ds_train.map(map_fn, batched=False, remove_columns=ds_sounds.column_names)
ds_val = ds_val.map(map_fn, batched=False, remove_columns=ds_sounds.column_names)

training_args = TrainingArguments(
    output_dir="./whisper-tags-finetuned",
    save_steps=500,
    save_total_limit=2,
    learning_rate=1e-5,
    num_train_epochs=EPOCHS,
    bf16=True,
    remove_unused_columns=False,
    report_to="wandb",
    logging_steps=10,
    eval_steps=20,
)

def data_collator(batch):
    input_features = torch.stack([torch.tensor(ex["input_features"]) for ex in batch])
    labels = [torch.tensor(ex["labels"], dtype=torch.long) for ex in batch]
    labels_padded = pad_sequence(labels, batch_first=True, padding_value=processor.tokenizer.pad_token_id)
    labels_padded[labels_padded == processor.tokenizer.pad_token_id] = -100
    return {"input_features": input_features, "labels": labels_padded}

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return {"accuracy": (predictions == labels).mean()}

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=ds_train,
    eval_dataset=ds_val,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

hf_login(os.getenv("HF_TOKEN_EDWIN"))

print('training')
trainer.train()

print('pushing to hub')
trainer.push_to_hub("edwindn/whisper-tags-finetuned")
