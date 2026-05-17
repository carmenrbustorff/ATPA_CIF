#import os
import torch
import pandas as pd
import soundfile as sf
import torch.nn as nn
import numpy as np
import torchaudio.transforms as T
import timm
from pathlib import Path


# --- CONFIGURATION ---
# Point this to your actual checkpoint from the agent's run
MODEL_PATH = "/kaggle/input/datasets/carmenbustorff/model3/model.pt"

# Kaggle typically hides the real test audio until submission. 
# You will need to map this to wherever your local dummy test files are.
TEST_AUDIO_DIR = Path("/kaggle/input/competitions/birdclef-2026/test_soundscapes")
SAMPLE_SUB_PATH = "/kaggle/input/competitions/birdclef-2026/sample_submission.csv"
OUTPUT_PATH = "submission.csv"

# These MUST match the values in your data_loader.py perfectly
SAMPLE_RATE = 32000
CHUNK_SAMPLES = SAMPLE_RATE * 5
N_MELS = 128      # Check your data_loader.py
N_FFT = 1024      # Check your data_loader.py
HOP_LENGTH = 512  # Check your data_loader.py
F_MIN = 0.0
F_MAX = 16000.0
TOP_DB = 80.0

# 3. Model Architecture (Transfer Learning)
class BirdCLEFModel(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.base_model = timm.create_model("efficientnet_b0", pretrained=False, in_chans=1, num_classes=0)
        
        # Unfreeze the base model weights
        for param in self.base_model.parameters():
            param.requires_grad = True 
            
        # Bypass Pylance type checking dynamically
        in_features = getattr(self.base_model, "num_features")
        
        self.classifier = nn.Sequential(
            nn.Linear(in_features, 128),  # type: ignore
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )
        
    def forward(self, x):
        features = self.base_model(x)
        logits = self.classifier(features)
        return logits




def main():
    OUTPUT_PATH = "/kaggle/working/submission.csv"
    SAMPLE_SUB_PATH = "/kaggle/input/competitions/birdclef-2026/sample_submission.csv"
    TRAIN_CSV_PATH = "/kaggle/input/competitions/birdclef-2026/train.csv" 
    
    try:
        # --- 1. SETUP ---
        sample_df = pd.read_csv(SAMPLE_SUB_PATH)
        kaggle_species_cols = sample_df.columns[1:].tolist()

        # If Kaggle hides the train data during scoring, this will fail.
        # But now, our except block will catch it!
        train_df = pd.read_csv(TRAIN_CSV_PATH)
        trained_species = sorted(train_df["primary_label"].unique().tolist())
        num_trained_classes = len(trained_species) 

        # --- 2. INITIALIZE MODEL ---
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = BirdCLEFModel(num_classes=num_trained_classes)
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
        model.to(device)
        model.eval()

        mel_transform = T.MelSpectrogram(
            sample_rate=SAMPLE_RATE, n_fft=N_FFT, hop_length=HOP_LENGTH, 
            n_mels=N_MELS, f_min=F_MIN, f_max=F_MAX
        ).to(device)
        amplitude_to_db = T.AmplitudeToDB(top_db=TOP_DB).to(device)

        def process_audio(waveform):
            mel = mel_transform(waveform)
            log_mel = amplitude_to_db(mel)
            log_mel = (log_mel + TOP_DB) / TOP_DB
            return log_mel

        # Check subdirectories too with rglob
        if not TEST_AUDIO_DIR.exists() or len(list(TEST_AUDIO_DIR.rglob('*.ogg'))) == 0:
            sample_df.to_csv(OUTPUT_PATH, index=False)
            print("No test audio found. Exported default.")
            return

        results = []

        # --- 3. INFERENCE LOOP ---
        for audio_path in TEST_AUDIO_DIR.rglob('*.ogg'):
            filename = audio_path.stem
            
            waveform_np, orig_sr = sf.read(str(audio_path))
            waveform = torch.from_numpy(waveform_np).float()
            
            if waveform.dim() == 2:
                waveform = waveform.transpose(0, 1).mean(dim=0, keepdim=True)
            else:
                waveform = waveform.unsqueeze(0)
                
            if orig_sr != SAMPLE_RATE:
                resampler = T.Resample(orig_freq=orig_sr, new_freq=SAMPLE_RATE)
                waveform = resampler(waveform)
            
            total_samples = waveform.shape[-1]
            
            for start_idx in range(0, total_samples, CHUNK_SAMPLES):
                end_idx = start_idx + CHUNK_SAMPLES
                chunk = waveform[:, start_idx:end_idx]
                
                if chunk.shape[-1] < CHUNK_SAMPLES:
                    pad = CHUNK_SAMPLES - chunk.shape[-1]
                    chunk = torch.nn.functional.pad(chunk, (0, pad))
                    
                end_time = (start_idx + CHUNK_SAMPLES) // SAMPLE_RATE
                row_id = f"{filename}_{end_time}"
                
                with torch.no_grad():
                    chunk = chunk.to(device)
                    spec = process_audio(chunk)
                    spec = spec.unsqueeze(0) 
                    logits = model(spec)
                    probs = torch.sigmoid(logits).squeeze(0).cpu().numpy()
                    
                row_dict = {'row_id': row_id}
                for sp in kaggle_species_cols:
                    row_dict[sp] = 0.0
                    
                for i, sp in enumerate(trained_species):
                    if sp in kaggle_species_cols: 
                        row_dict[sp] = probs[i]
                        
                results.append(row_dict)

        # --- 4. FORMAT OUTPUT ---
        submission_df = pd.DataFrame(results)
        
        if submission_df.empty:
            sample_df.to_csv(OUTPUT_PATH, index=False)
            return

        final_submission = pd.merge(sample_df[['row_id']], submission_df, on='row_id', how='left')
        final_submission = final_submission.fillna(0.0)
        
        for col in kaggle_species_cols:
            if col not in final_submission.columns:
                final_submission[col] = 0.0
                
        final_submission = final_submission[['row_id'] + kaggle_species_cols]
        final_submission.to_csv(OUTPUT_PATH, index=False)
        print(f"Submission successfully generated: {OUTPUT_PATH}")

    # --- 5. THE ULTIMATE SAFETY NET ---
    except Exception as e:
        print(f"CRITICAL ERROR CAUGHT: {e}")
        # If the script crashes anywhere, we bypass everything, read the 
        # sample submission directly from the disk, and save it as our output.
        try:
            fallback_df = pd.read_csv(SAMPLE_SUB_PATH)
            fallback_df.to_csv(OUTPUT_PATH, index=False)
            print("Exported fallback sample_submission.csv due to exception.")
        except Exception as fallback_e:
            print(f"Even the fallback failed: {fallback_e}")
if __name__ == "__main__":
    main()