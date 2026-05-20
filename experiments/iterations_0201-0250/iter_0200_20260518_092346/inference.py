import os
import gc
from pathlib import Path
from typing import List, Dict

import numpy as np
import pandas as pd
import soundfile as sf
import torch
import torch.nn as nn
import torchaudio.transforms as T
import timm

# =============================================================================
# Configuration (must match training)
# =============================================================================
SAMPLE_RATE = 32_000
CHUNK_DURATION = 5.0
CHUNK_SAMPLES = int(SAMPLE_RATE * CHUNK_DURATION)
N_MELS = 128
N_FFT = 1024
HOP_LENGTH = 512
F_MIN = 50.0
F_MAX = 14_000.0
TOP_DB = 80.0
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Paths (Kaggle defaults; override via env if needed)
MODEL_PATH = os.environ.get("MODEL_PATH", "/kaggle/input/datasets/carmenbustorff/model200-7/model200.pt")
TEST_AUDIO_DIR = Path(os.environ.get("TEST_AUDIO_DIR", "/kaggle/input/competitions/birdclef-2026/test_soundscapes"))
SAMPLE_SUB_PATH = os.environ.get("SAMPLE_SUB_PATH", "/kaggle/input/competitions/birdclef-2026/sample_submission.csv")
TRAIN_META_PATH = os.environ.get("TRAIN_META_PATH", "/kaggle/input/competitions/birdclef-2026/train_metadata.csv")
CLASSES_PATH = os.environ.get("CLASSES_PATH", "")  # optional path to classes.txt
OUTPUT_PATH = os.environ.get("OUTPUT_PATH", "/kaggle/working/submission.csv")

# =============================================================================
# Model (must match training)
# =============================================================================
class BirdCLEFModel(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()
        self.base_model = timm.create_model(
            "efficientnet_b1", pretrained=False, in_chans=1, num_classes=0
        )
        in_features = getattr(self.base_model, "num_features")

        self.attention = nn.Sequential(
            nn.Linear(in_features, in_features // 2),
            nn.ReLU(),
            nn.Linear(in_features // 2, in_features),
            nn.Sigmoid(),
        )

        self.classifier = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes),
        )

    def forward(self, x):
        features = self.base_model(x)
        attention_weights = self.attention(features)
        attended_features = features * attention_weights
        logits = self.classifier(attended_features)
        return logits


# =============================================================================
# Class list (must match training order)
# =============================================================================
# Fallback list — replace if you have classes.txt from training
LOCAL_CLASSES_FALLBACK = [
    '1161364', '116570', '1176823', '1595929', '209233', '22930', '22956', '22961', '22967',
    '22973', '22983', '22985', '23150', '23154', '23158', '23176', '23724', '24279', '24285',
    '24287', '24321', '244024', '25073', '25092', '25214', '326272', '41970', '43435', '47144',
    '47158son01', '47158son02', '47158son03', '47158son04', '47158son05', '47158son06',
    '47158son07', '47158son08', '47158son09', '47158son10', '47158son11', '47158son12',
    '47158son13', '47158son14', '47158son15', '47158son16', '47158son17', '47158son18',
    '47158son19', '47158son20', '47158son21', '47158son22', '47158son23', '47158son24',
    '47158son25', '476521', '516975', '517063', '555123', '555145', '555146', '64898', '65377',
    '65380', '66971', '67107', '67252', '70711', '738183', '74113', '74580', '760266', 'ashgre1',
    'astcra1', 'bafcur1', 'baffal1', 'banana', 'barant1', 'batbel1', 'baymac', 'bbwduc', 'bcwfin2',
    'bkcdon', 'bkhpar', 'blchaw1', 'blheag1', 'blttit1', 'bncfly', 'bobfly1', 'brcmar1', 'brnowl',
    'bucmot4', 'bucpar', 'bufpar', 'bunibi1', 'burowl', 'camfli1', 'chacha1', 'chbmoc1', 'chobla1',
    'chvcon1', 'cibspi1', 'coffal1', 'compau', 'compot1', 'crbthr1', 'crebec1', 'dwatin1', 'epaori4',
    'eulfly1', 'fabwre1', 'fepowl', 'ficman1', 'flawar1', 'fotfly', 'fusfly1', 'gilhum1', 'giwrai1',
    'glteme1', 'grasal3', 'greani1', 'greant1', 'greela', 'grekis', 'grepot1', 'gretho2', 'greyel',
    'grfdov1', 'grhtan1', 'gycwor1', 'horscr1', 'houspa', 'hyamac1', 'larela1', 'lesela1', 'lesgrf1',
    'limpki', 'linwoo1', 'litcuc2', 'litnig1', 'mabpar', 'magant1', 'magtan2', 'masgna1', 'nacnig1',
    'ocecra1', 'oliwoo1', 'orbtro3', 'orwpar', 'osprey', 'pabspi1', 'palhor3', 'paltan1', 'phecuc1',
    'picpig2', 'pirfly1', 'plasla1', 'platyr1', 'plcjay1', 'pluibi1', 'purjay1', 'pvttyr1', 'ragmac1',
    'rebscy1', 'recfin1', 'redjun', 'relser1', 'rinkin1', 'rivwar1', 'roahaw', 'rubthr1', 'rufcac2',
    'rufcas2', 'rufgna3', 'rufhor2', 'rufnig1', 'ruftho1', 'ruftof1', 'rumfly1', 'ruther1', 'rutjac1',
    'sabspa1', 'saffin', 'saytan1', 'scadov1', 'schpar1', 'scther1', 'shcfly1', 'shshaw', 'shtnig1',
    'sibtan2', 'smbani', 'smbtin1', 'sobcac1', 'sobtyr1', 'socfly1', 'sofspi1', 'souant1', 'soulap1',
    'souscr1', 'spbant3', 'spispi1', 'sptnig1', 'squcuc1', 'stbwoo2', 'strcuc1', 'strher2', 'strowl1',
    'swthum1', 'swtman1', 'tattin1', 'thlwre1', 'toctou1', 'trokin', 'trsowl', 'undtin1', 'varant1',
    'watjac1', 'wesfie1', 'wfwduc1', 'whbant2', 'whbwar2', 'whiwoo1', 'whlspi1', 'whnjay1', 'whtdov',
    'whwpic1', 'y00678', 'yebcar', 'yebela1', 'yecmac', 'yecpar', 'yehcar1', 'yeofly1'
]


def load_training_classes() -> List[str]:
    # 1) classes.txt (recommended)
    if CLASSES_PATH:
        path = Path(CLASSES_PATH)
        if path.exists():
            classes = [line.strip() for line in path.read_text().splitlines() if line.strip()]
            if classes:
                return classes

    # 2) train_metadata.csv (matches data_loader.py sorting)
    try:
        df = pd.read_csv(TRAIN_META_PATH)
        species = sorted(df["primary_label"].unique().tolist())
        if species:
            return species
    except Exception:
        pass

    # 3) fallback
    return LOCAL_CLASSES_FALLBACK


# =============================================================================
# Audio preprocessing (matches training)
# =============================================================================
mel_transform = T.MelSpectrogram(
    sample_rate=SAMPLE_RATE,
    n_mels=N_MELS,
    n_fft=N_FFT,
    hop_length=HOP_LENGTH,
    f_min=F_MIN,
    f_max=F_MAX,
).to(DEVICE)

amplitude_to_db = T.AmplitudeToDB(top_db=TOP_DB).to(DEVICE)


def process_audio(waveform: torch.Tensor) -> torch.Tensor:
    mel_spec = mel_transform(waveform)
    log_mel = amplitude_to_db(mel_spec)
    log_mel = (log_mel + TOP_DB) / TOP_DB
    return log_mel


# =============================================================================
# Inference
# =============================================================================

def infer_file(audio_path: Path, model: nn.Module, classes: List[str]) -> Dict[str, np.ndarray]:
    """Return dict of row_id -> probs array"""
    results: Dict[str, np.ndarray] = {}

    try:
        waveform_np, orig_sr = sf.read(str(audio_path))
    except Exception:
        return results

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

        end_time = int((start_idx + CHUNK_SAMPLES) / SAMPLE_RATE)
        row_id = f"{audio_path.stem}_{end_time}"

        with torch.no_grad():
            chunk = chunk.to(DEVICE)
            spec = process_audio(chunk)
            if spec.dim() == 2:
                spec = spec.unsqueeze(0).unsqueeze(0)
            elif spec.dim() == 3:
                spec = spec.unsqueeze(0)

            logits = model(spec)
            probs = torch.sigmoid(logits).squeeze(0).cpu().numpy().astype(np.float32)

        results[row_id] = probs

    return results


def generate_submission_csv():
    sample_df = pd.read_csv(SAMPLE_SUB_PATH)
    kaggle_species_cols = sample_df.columns[1:].tolist()

    classes = load_training_classes()
    num_classes = len(classes)

    device = DEVICE
    model = BirdCLEFModel(num_classes=num_classes)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.to(device)
    model.eval()

    if not TEST_AUDIO_DIR.exists():
        sample_df.to_csv(OUTPUT_PATH, index=False)
        return

    test_files = list(TEST_AUDIO_DIR.rglob("*.ogg")) + list(TEST_AUDIO_DIR.rglob("*.mp3")) + list(TEST_AUDIO_DIR.rglob("*.flac"))
    if not test_files:
        sample_df.to_csv(OUTPUT_PATH, index=False)
        return

    # Collect predictions into a mapping
    pred_map: Dict[str, np.ndarray] = {}
    for audio_path in test_files:
        pred_map.update(infer_file(audio_path, model, classes))

    # Build submission aligned to sample_submission
    final_data = {"row_id": sample_df["row_id"].values}
    for col in kaggle_species_cols:
        final_data[col] = np.zeros(len(sample_df), dtype=np.float32)

    # Fill from predictions by class-name mapping
    class_to_idx = {c: i for i, c in enumerate(classes)}

    for i, row_id in enumerate(sample_df["row_id"].values):
        probs = pred_map.get(row_id)
        if probs is None:
            continue
        for col in kaggle_species_cols:
            idx = class_to_idx.get(col)
            if idx is not None and idx < probs.shape[0]:
                final_data[col][i] = float(probs[idx])

    final_df = pd.DataFrame(final_data)
    for col in kaggle_species_cols:
        final_df[col] = final_df[col].astype(np.float32)

    final_df.to_csv(OUTPUT_PATH, index=False)


if __name__ == "__main__":
    generate_submission_csv()
