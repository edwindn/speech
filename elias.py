import torch
import random
import firebase_admin
import requests
import time
import os
import time
from snac import SNAC
import torchaudio.transforms as T
from firebase_admin import firestore, credentials
from datasets import load_dataset, concatenate_datasets
from speechbrain.pretrained import SpeakerRecognition
import pandas as pd

repo_id = "amphion/Emilia-Dataset"

model = SNAC.from_pretrained("hubertsiuzdak/snac_24khz")
model = model.to("cuda")

embedding_model = SpeakerRecognition.from_hparams(
    source="speechbrain/spkrec-ecapa-voxceleb",
    savedir="pretrained_models/spkrec-ecapa-voxceleb",  # any directory name you choose
    run_opts={"device": "cuda:0"}
)

sr__ = 24000

print("STARTTTING SCRIPT")
cred = credentials.Certificate("./serviceAccount.json")
firebase_admin.initialize_app(cred)
db = firestore.client()

selected_username = os.environ.get("selected_username")
print("Selected token: ", os.environ.get("selected_token"))
print(f"Selected username: {selected_username}")

# selected_username = "CanopyLabs"
# selected_username = "eliasfiz"

def update_vm_status(status):
    try:
        doc_ref = db.collection("vm_status").document("summary")
        doc = doc_ref.get()
        if doc.exists:
            data = doc.to_dict()
            data[status] = data.get(status, 0) + 1
            doc_ref.set(data)
        else:
            doc_ref.set({status: 1})
    except Exception as e:
        print("Failed to update Firebase")

def update_atomic_counter(status, error=None):
    print(f"Updating atomic counter to {status}")
    if status == "crashed":
        response = requests.get(f"http://34.55.2.250:8080/{status}/next?error={error}")
    else:
        response = requests.get(f"http://34.55.2.250:8080/{status}/next")


# Update VM creation count
update_atomic_counter("created")



def make_request():
    try:
        response = requests.get("http://34.55.2.250:8080/next")

        if response.status_code == 200:
            number = response.json()['number']
            print(f"Got number: {number}")
            return number
        else:
            print(f"Request failed with status {response.status_code}")
            return None
    except Exception as e:
        print(f"Error: {str(e)}")
        return None


base_model="large"
offset = 49162
if base_model == "small":
  offset = 49162 
else:
  offset = 128266


def tokenise_audio(waveform, og_sample_rate, target_sr=24000):
    #resample waveform from og_sample_rate to 24000

    #convert to float32
    waveform = torch.from_numpy(waveform).unsqueeze(0).float().to("cuda")
    
    resampler24 = T.Resample(og_sample_rate, target_sr).to("cuda")
    waveform = resampler24(waveform)

    waveform = waveform.to(dtype=torch.float32)

    #generate the codes from snac

    waveform = waveform.unsqueeze(0)
    # Move waveform to CUDA before processing
    
    with torch.inference_mode():
        codes = model.encode(waveform)


    all_codes = []
    for i in range(codes[0].shape[1]):
        all_codes.append(codes[0][0][i].item()+offset)
        all_codes.append(codes[1][0][2*i].item()+offset+4096)
        all_codes.append(codes[2][0][4*i].item()+offset+(2*4096))
        all_codes.append(codes[2][0][(4*i)+1].item()+offset+(3*4096))
        all_codes.append(codes[1][0][(2*i)+1].item()+offset+(4*4096))
        all_codes.append(codes[2][0][(4*i)+2].item()+offset+(5*4096))
        all_codes.append(codes[2][0][(4*i)+3].item()+offset+(6*4096))

    # Move waveform back to CPU for resampling
    resampler16 = T.Resample(og_sample_rate, 16000).to("cuda")
    waveform_16 = resampler16(waveform)
    waveform_16 = waveform_16.squeeze(0)
    # Move waveform_16 to CUDA for embedding model
    embedding = embedding_model.encode_batch(waveform_16)
    # embedding = embedding.squeeze(0).squeeze(0).cpu().numpy()

    return all_codes, embedding


def add_codes(example):
    # Always initialize codes_list to None
    codes_list = None

    try:
        answer_audio = example.get("mp3")
        # If there's a valid audio array, tokenise it
        if answer_audio and "array" in answer_audio:
            audio_array = answer_audio["array"]
            og_sample_rate = answer_audio["sampling_rate"]

            codes_list, embedding = tokenise_audio(audio_array, og_sample_rate)
    except Exception as e:
        print(f"Skipping row due to error: {e}")
        # Keep codes_list as None if we fail


    embedding = embedding.squeeze(0).squeeze(0).cpu().numpy()
    example["codes_list"] = codes_list
    example["speaker_embedding"] = embedding
    example["text"] = example["json"]["text"]

    return example



def process_tar_file(tar_index, batch_size=1000):

    start_time = time.time()
    
    # Determine the correct file path based on tar_index
    if tar_index >= 1000:
        path = f"Emilia-YODAS/EN/EN-B00{tar_index}.tar"
    elif tar_index >= 100:
        path = f"Emilia-YODAS/EN/EN-B000{tar_index}.tar"
    elif tar_index >= 10:
        path = f"Emilia-YODAS/EN/EN-B0000{tar_index}.tar"
    elif tar_index >= 1:
        path = f"Emilia-YODAS/EN/EN-B00000{tar_index}.tar"
    else:
        path = f"Emilia-YODAS/EN/EN-B000000.tar"

    # Load the dataset from the tar file
    ds = load_dataset("amphion/Emilia-Dataset", data_files={"en": path}, split="en")
    
    # ds = ds.select(range(200))
    
    total_rows = len(ds)
    
    print(f"Total rows in dataset: {total_rows}")

    processed_batches = []

    # Process the dataset in smaller batches
    for start in range(0, total_rows, batch_size):
        end = min(start + batch_size, total_rows)
        print(f"Processing batch rows {start} to {end}...")
        
        # Select a batch from the dataset
        ds_batch = ds.select(range(start, end))
        
        # Process the batch using the map function
        ds_batch = ds_batch.map(add_codes, remove_columns=ds_batch.column_names, num_proc=1)
        processed_batches.append(ds_batch)
        print(f"Batch processed in {time.time() - start_time:.2f} seconds")
    
    # Combine all processed batches into a single dataset
    combined_ds = concatenate_datasets(processed_batches)
    print(f"Combined dataset with {len(combined_ds)} rows. Pushing to hub...")
    
    # Push the final, combined dataset to the Hugging Face Hub
    combined_ds.push_to_hub(f"{selected_username}/emilia-snac-with-spk-emb", split=f"part_{tar_index}")


vm_has_crashed = False
while not vm_has_crashed:
    try:
        current_index = make_request()

        if current_index > 1139:
            print("ALL PROCESSES DONE")
            update_vm_status("done")
            vm_has_crashed = True
            break
        
        start_time = time.time()
        process_tar_file(current_index)
        print(f"Time taken to process tar file: {time.time() - start_time:.2f} seconds")

    except Exception as e:
        vm_has_crashed = True
        print("This VM is crashing")
        print(f"Failed to process tar file: {e}")
        update_atomic_counter("crashed", str(e))
        break