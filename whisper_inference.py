import torch
from transformers import WhisperProcessor, WhisperForConditionalGeneration
import librosa
import numpy as np

def transcribe_audio(audio_path, model_name="edwindn/whisper-tags-finetuned"):
    # Load processor and model
    processor = WhisperProcessor.from_pretrained(model_name)
    model = WhisperForConditionalGeneration.from_pretrained(model_name)
    
    # Load and preprocess audio
    audio, sr = librosa.load(audio_path, sr=16000)
    
    # Process audio
    input_features = processor(audio, sampling_rate=sr, return_tensors="pt").input_features
    
    # Generate transcription
    predicted_ids = model.generate(input_features)
    transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)
    
    return transcription[0]

if __name__ == "__main__":
    # Example usage
    audio_path = "path/to/your/audio/file.wav"  # Replace with your audio file path
    transcription = transcribe_audio(audio_path)
    print("Transcription:", transcription)
