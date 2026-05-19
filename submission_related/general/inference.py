#!/usr/bin/env python3
"""
Bulletproof Inference Script for BirdCLEF 2026 Kaggle Competition
Designed specifically to survive Kaggle's hidden scoring environment.

Key Constraints:
1. Uses torchaudio.load() ONLY (no soundfile/libsndfile)
2. Slices audio into 5.0-second chunks with zero-padding
3. Pure log transformation: torch.log(mel_spec + 1e-9) — NO AmplitudeToDB
4. Pure FP32 inference (no mixed precision)
5. Nuclear failsafe: writes sample_submission.csv immediately
6. Dynamic dataframe merge with Kaggle's hidden template
"""

import sys
import os
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import torchaudio
import torchaudio.transforms as T
from pathlib import Path
from tqdm import tqdm

# ============================================================================
# GLOBAL CLASS LIST (206 species — extracted from train.csv, matches checkpoint)
# ============================================================================
LOCAL_CLASSES = [
    '1161364', '116570', '1176823', '1595929', '209233', '22930', '22956', '22961',
    '22967', '22973', '22983', '22985', '23150', '23154', '23158', '23176', '23724',
    '24279', '24285', '24287', '24321', '244024', '25092', '25214', '326272',
    '41970', '43435', '47144', '476521', '516975', '555123', '555145', '555146',
    '64898', '65377', '65380', '66971', '67107', '67252', '70711', '738183',
    '74113', '74580', '760266', 'ashgre1', 'astcra1', 'bafcur1', 'baffal1', 'banana',
    'barant1', 'batbel1', 'baymac', 'bbwduc', 'bcwfin2', 'bkcdon', 'bkhpar', 'blchaw1',
    'blheag1', 'blttit1', 'bncfly', 'bobfly1', 'brcmar1', 'brnowl', 'bucmot4', 'bucpar',
    'bufpar', 'bunibi1', 'burowl', 'camfli1', 'chacha1', 'chbmoc1', 'chobla1', 'chvcon1',
    'cibspi1', 'coffal1', 'compau', 'compot1', 'crbthr1', 'crebec1', 'dwatin1', 'epaori4',
    'eulfly1', 'fabwre1', 'fepowl', 'ficman1', 'flawar1', 'fotfly', 'fusfly1', 'gilhum1',
    'giwrai1', 'glteme1', 'grasal3', 'greani1', 'greant1', 'greela', 'grekis', 'grepot1',
    'gretho2', 'greyel', 'grfdov1', 'grhtan1', 'gycwor1', 'horscr1', 'houspa', 'hyamac1',
    'larela1', 'lesela1', 'lesgrf1', 'limpki', 'linwoo1', 'litcuc2', 'litnig1', 'mabpar',
    'magant1', 'magtan2', 'masgna1', 'nacnig1', 'ocecra1', 'oliwoo1', 'orbtro3', 'orwpar',
    'osprey', 'pabspi1', 'palhor3', 'paltan1', 'phecuc1', 'picpig2', 'pirfly1', 'plasla1',
    'platyr1', 'plcjay1', 'pluibi1', 'purjay1', 'pvttyr1', 'ragmac1', 'rebscy1', 'recfin1',
    'redjun', 'relser1', 'rinkin1', 'rivwar1', 'roahaw', 'rubthr1', 'rufcac2', 'rufcas2',
    'rufgna3', 'rufhor2', 'rufnig1', 'ruftho1', 'ruftof1', 'rumfly1', 'ruther1', 'rutjac1',
    'sabspa1', 'saffin', 'saytan1', 'scadov1', 'schpar1', 'scther1', 'shcfly1', 'shshaw',
    'shtnig1', 'sibtan2', 'smbani', 'smbtin1', 'sobcac1', 'sobtyr1', 'socfly1', 'sofspi1',
    'souant1', 'soulap1', 'souscr1', 'spbant3', 'spispi1', 'sptnig1', 'squcuc1', 'stbwoo2',
    'strcuc1', 'strher2', 'strowl1', 'swthum1', 'swtman1', 'tattin1', 'thlwre1', 'toctou1',
    'trokin', 'trsowl', 'undtin1', 'varant1', 'watjac1', 'wesfie1', 'wfwduc1', 'whbant2',
    'whbwar2', 'whiwoo1', 'whlspi1', 'whnjay1', 'whtdov', 'whwpic1', 'y00678', 'yebcar',
    'yebela1', 'yecmac', 'yecpar', 'yehcar1', 'yeofly1'
]

NUM_CLASSES = len(LOCAL_CLASSES)
CHUNK_DURATION = 5.0
SR = 32000
N_MELS = 128
N_FFT = 1024
HOP_LENGTH = 512


# ============================================================================
# MODEL ARCHITECTURE (matching train.py exactly)
# ============================================================================
class BirdCLEFModel(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        import timm
        self.base_model = timm.create_model(
            "efficientnet_b1", pretrained=True, in_chans=1, num_classes=0
        )
        for param in self.base_model.parameters():
            param.requires_grad = True
        in_features = getattr(self.base_model, "num_features")

        self.attention = nn.Sequential(
            nn.Linear(in_features, in_features // 2),
            nn.ReLU(),
            nn.Linear(in_features // 2, in_features),
            nn.Sigmoid()
        )

        self.classifier = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        features = self.base_model(x)
        attention_weights = self.attention(features)
        attended_features = features * attention_weights
        logits = self.classifier(attended_features)
        return logits


# ============================================================================
# AUDIO PROCESSING
# ============================================================================
def load_and_slice_audio(audio_path):
    """
    Load audio using torchaudio.load() and slice into 5-second chunks.
    Zero-pad final chunk if shorter than 5 seconds.
    """
    try:
        waveform, sample_rate = torchaudio.load(audio_path)
    except Exception as e:
        print(f"ERROR loading {audio_path}: {e}")
        return []

    if sample_rate != SR:
        resampler = T.Resample(sample_rate, SR)
        waveform = resampler(waveform)

    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    waveform = waveform.squeeze(0).numpy()

    chunk_samples = int(CHUNK_DURATION * SR)
    chunks = []

    for start_idx in range(0, len(waveform), chunk_samples):
        chunk = waveform[start_idx:start_idx + chunk_samples]

        if len(chunk) < chunk_samples:
            chunk = np.pad(chunk, (0, chunk_samples - len(chunk)), mode='constant', constant_values=0.0)

        chunks.append(torch.from_numpy(chunk).float())

    return chunks if chunks else [torch.zeros(chunk_samples, dtype=torch.float32)]


def extract_mel_spectrogram(audio_chunk, device):
    """
    Extract mel spectrogram from audio chunk.
    Apply pure log transformation: torch.log(mel_spec + 1e-9)
    """
    mel_transform = T.MelSpectrogram(
        sample_rate=SR,
        n_mels=N_MELS,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        power=2.0
    ).to(device)

    mel_spec = mel_transform(audio_chunk)

    mel_spec = torch.log(mel_spec + 1e-9)

    return mel_spec


# ============================================================================
# INFERENCE
# ============================================================================
def generate_submission_csv(model_path, test_audio_dir, output_csv, device):
    """
    Main inference pipeline.
    CRITICAL: Write sample_submission.csv first as nuclear failsafe.
    """

    # === NUCLEAR FAILSAFE: Write empty sample_submission.csv immediately ===
    if os.path.exists("sample_submission.csv"):
        try:
            sample_df = pd.read_csv("sample_submission.csv")
            sample_df.to_csv(output_csv, index=False)
            print(f"[FAILSAFE] Wrote sample_submission.csv to {output_csv}")
        except Exception as e:
            print(f"[FAILSAFE] Warning: Could not copy sample_submission.csv: {e}")

    # Load model
    print(f"Loading model from {model_path}...")
    model = BirdCLEFModel(num_classes=NUM_CLASSES)

    try:
        state_dict = torch.load(model_path, map_location=device)
        model.load_state_dict(state_dict)
    except Exception as e:
        print(f"ERROR loading model state dict: {e}")
        return

    model.to(device)
    model.eval()
    print("Model loaded successfully.")

    # Find all audio files
    audio_dir = Path(test_audio_dir)
    audio_files = sorted(
        list(audio_dir.glob("*.mp3")) +
        list(audio_dir.glob("*.wav")) +
        list(audio_dir.glob("*.ogg")) +
        list(audio_dir.glob("*.flac"))
    )

    if not audio_files:
        print(f"WARNING: No audio files found in {test_audio_dir}")
        print("Checking alternative paths...")
        alt_audio_paths = [
            Path("/kaggle/input") / "test_soundscapes",
            Path("/kaggle/input") / "test",
            Path("/kaggle/input/birdclef-2026") / "test",
            Path("/kaggle/input/birdclef-2026") / "test_soundscapes",
        ]
        for alt_path in alt_audio_paths:
            if alt_path.exists():
                test_alt_files = sorted(
                    list(alt_path.glob("*.mp3")) +
                    list(alt_path.glob("*.wav")) +
                    list(alt_path.glob("*.ogg")) +
                    list(alt_path.glob("*.flac"))
                )
                if test_alt_files:
                    print(f"Found {len(test_alt_files)} audio files at {alt_path}")
                    audio_files = test_alt_files
                    break

    if not audio_files:
        print("WARNING: No audio files found in any location!")
        audio_files = []

    print(f"Found {len(audio_files)} audio files.")

    # Process each audio file
    predictions_dict = {}

    with torch.no_grad():
        for audio_path in tqdm(audio_files, desc="Processing audio files"):
            row_id = audio_path.stem

            chunks = load_and_slice_audio(str(audio_path))

            if not chunks:
                predictions_dict[row_id] = np.zeros(NUM_CLASSES, dtype=np.float32)
                continue

            chunk_preds = []
            for chunk in chunks:
                chunk = chunk.to(device)
                mel_spec = extract_mel_spectrogram(chunk, device)
                mel_spec = mel_spec.unsqueeze(0).unsqueeze(0).to(device)  # (1, 1, n_mels, time)

                logits = model(mel_spec)
                probs = torch.sigmoid(logits).cpu().numpy()
                probs = probs.reshape(-1)
                chunk_preds.append(probs)

            agg_pred = np.mean(chunk_preds, axis=0)
            predictions_dict[row_id] = agg_pred

    # Build prediction dataframe
    rows = []
    for row_id, pred_vec in predictions_dict.items():
        row = {"row_id": row_id}
        for class_idx, class_name in enumerate(LOCAL_CLASSES):
            row[class_name] = float(pred_vec[class_idx])
        rows.append(row)

    pred_df = pd.DataFrame(rows)
    print(f"Built prediction dataframe with {len(pred_df)} rows.")

    # === DYNAMIC MERGE with Kaggle's hidden sample_submission.csv ===
    try:
        if os.path.exists("sample_submission.csv"):
            sample_df = pd.read_csv("sample_submission.csv")
            print(f"Loaded sample_submission.csv with columns: {list(sample_df.columns)}")
            print(f"Sample shape: {sample_df.shape}, Predictions shape: {pred_df.shape}")

            template_species = [col for col in sample_df.columns if col != "row_id"]
            print(f"Template species: {len(template_species)}")

            merged_df = sample_df[["row_id"]].copy()

            for species in template_species:
                if species in pred_df.columns:
                    merged_df[species] = pred_df.set_index("row_id").loc[merged_df["row_id"], species].values
                else:
                    merged_df[species] = 0.0

            merged_df = merged_df.fillna(0.0)
            output_df = merged_df[["row_id"] + template_species]
        else:
            output_df = pred_df
    except Exception as e:
        print(f"WARNING: Merge failed: {e}. Using raw predictions.")
        output_df = pred_df

    output_df.to_csv(output_csv, index=False)
    print(f"Predictions written to {output_csv}")
    print(f"Output shape: {output_df.shape}")


# ============================================================================
# MAIN
# ============================================================================
if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python inference.py <model_path> <test_audio_dir> <output_csv>")
        sys.exit(1)

    model_path = sys.argv[1]
    test_audio_dir = sys.argv[2]
    output_csv = sys.argv[3]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    generate_submission_csv(model_path, test_audio_dir, output_csv, device)
    print("Inference complete.")