#!/usr/bin/env python3
"""
BULLETPROOF KAGGLE SUBMISSION — BirdCLEF 2026
Critical: This must produce valid predictions or Kaggle gets a 0.5 score.
Model AUC: 96% — must reflect in Kaggle score.
"""

import os
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import torchaudio
import torchaudio.transforms as T
from pathlib import Path
from tqdm import tqdm
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s')
logger = logging.getLogger(__name__)

# ============================================================================
# KAGGLE PATHS (hardcoded for Kaggle environment)
# ============================================================================
INPUT_DIR = Path("/kaggle/input/competitions/birdclef-2026")
WORKING_DIR = Path("/kaggle/working")
OUTPUT_CSV = WORKING_DIR / "submission.csv"
SAMPLE_CSV = INPUT_DIR / "sample_submission.csv"
MODEL_PATH = INPUT_DIR / "model" / "model.pt"
TEST_AUDIO_DIR = INPUT_DIR / "test_soundscapes"

# ============================================================================
# AUDIO CONFIG (must match training exactly)
# ============================================================================
SR = 32000
N_MELS = 128
N_FFT = 1024
HOP_LENGTH = 512
CHUNK_DURATION = 5.0
CHUNK_SAMPLES = int(SR * CHUNK_DURATION)

# ============================================================================
# CLASS LIST (will be populated from sample_submission.csv template)
# ============================================================================
LOCAL_CLASSES = []
NUM_CLASSES = 0


# ============================================================================
# MODEL ARCHITECTURE
# ============================================================================
class BirdCLEFModel(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        import timm
        self.base_model = timm.create_model(
            "efficientnet_b1", pretrained=False, in_chans=1, num_classes=0
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
    """Load audio and slice into 5-second chunks with zero-padding."""
    try:
        waveform, sample_rate = torchaudio.load(audio_path)
        logger.info(f"Loaded {audio_path}: sr={sample_rate}, shape={waveform.shape}")
    except Exception as e:
        logger.error(f"Failed to load {audio_path}: {e}")
        return []

    if sample_rate != SR:
        resampler = T.Resample(sample_rate, SR)
        waveform = resampler(waveform)
        logger.info(f"Resampled to {SR}")

    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    waveform = waveform.squeeze(0).numpy()

    chunks = []
    for start_idx in range(0, len(waveform), CHUNK_SAMPLES):
        chunk = waveform[start_idx:start_idx + CHUNK_SAMPLES]
        if len(chunk) < CHUNK_SAMPLES:
            chunk = np.pad(chunk, (0, CHUNK_SAMPLES - len(chunk)), mode='constant')
        chunks.append(torch.from_numpy(chunk).float())

    logger.info(f"Created {len(chunks)} chunks")
    return chunks


def extract_mel_spectrogram(audio_chunk, device):
    """Extract mel spectrogram with pure log transform."""
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
# MAIN INFERENCE PIPELINE
# ============================================================================
def main():
    logger.info("="*80)
    logger.info("KAGGLE SUBMISSION START")
    logger.info("="*80)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    # ========== STEP 1: Verify sample_submission.csv exists ==========
    sample_csv = Path(SAMPLE_CSV)
    if not sample_csv.exists():
        logger.error(f"sample_submission.csv not found at {sample_csv}")
        logger.error(f"Checking alternative paths...")
        alt_paths = [
            Path("/kaggle/input") / "sample_submission.csv",
            Path("/kaggle/input/birdclef-2026") / "sample_submission.csv",
        ]
        for alt in alt_paths:
            if alt.exists():
                logger.info(f"Found at {alt}")
                sample_csv = Path(alt)
                break
        else:
            logger.error("sample_submission.csv not found anywhere!")
            return False

    sample_df = pd.read_csv(sample_csv)
    logger.info(f"Loaded sample_submission.csv: shape={sample_df.shape}")
    logger.info(f"Columns: {list(sample_df.columns)}")
    logger.info(f"Sample rows:\n{sample_df.head()}")

    # Extract class list from template
    global LOCAL_CLASSES, NUM_CLASSES
    LOCAL_CLASSES = [col for col in sample_df.columns if col != "row_id"]
    NUM_CLASSES = len(LOCAL_CLASSES)
    logger.info(f"Loaded {NUM_CLASSES} classes from template")

    # Validate class list against template
    template_species = LOCAL_CLASSES
    logger.info(f"Template has {len(template_species)} species columns")

    # ========== STEP 2: Load model ==========
    model_path = Path(MODEL_PATH)
    logger.info(f"Loading model from {model_path}...")
    if not model_path.exists():
        logger.error(f"Model not found at {model_path}")
        logger.error(f"Checking alternative paths...")
        alt_paths = [
            Path("/kaggle/input") / "model.pt",
            Path("/kaggle/input/model") / "model.pt",
            Path("/kaggle/input/datasets") / "carmenbustorff" / "model200-9" / "model200.pt",
        ]
        for alt in alt_paths:
            if alt.exists():
                logger.info(f"Found at {alt}")
                model_path = Path(alt)
                break
        else:
            logger.error("model.pt not found anywhere!")
            return False

    model = BirdCLEFModel(num_classes=NUM_CLASSES)
    try:
        state_dict = torch.load(model_path, map_location=device)

        # Handle class mismatch: checkpoint may have different num_classes
        classifier_weight_key = "classifier.3.weight"
        classifier_bias_key = "classifier.3.bias"

        if classifier_weight_key in state_dict:
            checkpoint_num_classes = state_dict[classifier_weight_key].shape[0]
            if checkpoint_num_classes != NUM_CLASSES:
                logger.warning(f"Class mismatch: checkpoint has {checkpoint_num_classes} classes, template has {NUM_CLASSES}")
                logger.info(f"Extending model to handle {NUM_CLASSES} classes...")

                # Extract original weights/biases
                orig_weight = state_dict[classifier_weight_key]  # [206, 512]
                orig_bias = state_dict[classifier_bias_key]      # [206]

                # Create new weights/biases for extended model [234, 512]
                new_weight = torch.zeros(NUM_CLASSES, orig_weight.shape[1], device=device)
                new_bias = torch.zeros(NUM_CLASSES, device=device)

                # Copy original weights to first 206 positions
                new_weight[:checkpoint_num_classes] = orig_weight.to(device)
                new_bias[:checkpoint_num_classes] = orig_bias.to(device)

                # Initialize new species with small random values
                torch.manual_seed(42)
                new_weight[checkpoint_num_classes:] = torch.randn(
                    NUM_CLASSES - checkpoint_num_classes, orig_weight.shape[1], device=device
                ) * 0.01
                new_bias[checkpoint_num_classes:] = torch.randn(
                    NUM_CLASSES - checkpoint_num_classes, device=device
                ) * 0.01

                # Update state dict
                state_dict[classifier_weight_key] = new_weight.to('cpu')
                state_dict[classifier_bias_key] = new_bias.to('cpu')

                logger.info(f"✓ Extended classifier from {checkpoint_num_classes} to {NUM_CLASSES} classes")

        model.load_state_dict(state_dict)
        logger.info("Model loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        return False

    model.to(device)
    model.eval()

    # ========== STEP 3: Find test audio files ==========
    test_audio_dir = TEST_AUDIO_DIR
    logger.info(f"Searching for audio files in {test_audio_dir}...")
    audio_files = sorted(
        list(test_audio_dir.glob("*.mp3")) +
        list(test_audio_dir.glob("*.wav")) +
        list(test_audio_dir.glob("*.ogg")) +
        list(test_audio_dir.glob("*.flac"))
    )
    logger.info(f"Found {len(audio_files)} audio files")

    if not audio_files:
        logger.warning(f"No audio files in {test_audio_dir}")
        logger.info("Checking alternative paths...")
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
                    logger.info(f"Found {len(test_alt_files)} audio files at {alt_path}")
                    audio_files = test_alt_files
                    break

    if not audio_files:
        logger.warning("No audio files found in any location!")
        logger.info("Using all zeros for missing predictions")

    # ========== STEP 4: Process each audio file ==========
    predictions_dict = {}

    with torch.no_grad():
        for audio_path in tqdm(audio_files, desc="Processing audio"):
            row_id = audio_path.stem
            logger.info(f"\nProcessing {row_id}...")

            chunks = load_and_slice_audio(str(audio_path))

            if not chunks:
                logger.warning(f"No chunks for {row_id}, using zeros")
                predictions_dict[row_id] = np.zeros(NUM_CLASSES, dtype=np.float32)
                continue

            chunk_preds = []
            for i, chunk in enumerate(chunks):
                chunk = chunk.to(device)
                mel_spec = extract_mel_spectrogram(chunk, device)
                mel_spec = mel_spec.unsqueeze(0).unsqueeze(0).to(device)

                logits = model(mel_spec)
                probs = torch.sigmoid(logits).cpu().numpy()
                probs = probs.reshape(-1)

                # Verify predictions are valid
                if np.isnan(probs).any() or np.isinf(probs).any():
                    logger.error(f"NaN/Inf detected in chunk {i} of {row_id}")
                    probs = np.zeros(NUM_CLASSES, dtype=np.float32)

                chunk_preds.append(probs)

            agg_pred = np.mean(chunk_preds, axis=0)
            predictions_dict[row_id] = agg_pred
            pred_min = float(np.min(agg_pred))
            pred_max = float(np.max(agg_pred))
            pred_mean = float(np.mean(agg_pred))
            logger.info(f"{row_id}: pred_min={pred_min:.4f}, pred_max={pred_max:.4f}, pred_mean={pred_mean:.4f}")

    logger.info(f"\nProcessed {len(predictions_dict)} files with predictions")

    # ========== STEP 5: Build prediction dataframe ==========
    logger.info("Building prediction dataframe...")
    try:
        rows = []
        for row_id, pred_vec in predictions_dict.items():
            row = {"row_id": row_id}
            for class_idx, class_name in enumerate(LOCAL_CLASSES):
                row[class_name] = float(pred_vec[class_idx])
            rows.append(row)

        pred_df = pd.DataFrame(rows)
        logger.info(f"Prediction dataframe: shape={pred_df.shape}")
        logger.info(f"Columns: {list(pred_df.columns)[:5]}... (showing first 5)")

        # DEBUG: Write intermediate predictions
        debug_csv = WORKING_DIR / "debug_predictions.csv"
        pred_df.to_csv(debug_csv, index=False)
        logger.info(f"✓ Debug file written: {debug_csv}")

    except Exception as e:
        logger.error(f"Failed to build prediction dataframe: {e}", exc_info=True)
        return False

    # ========== STEP 6: Merge with sample_submission.csv ==========
    logger.info("Merging predictions with sample_submission template...")
    try:
        merged_df = sample_df[["row_id"]].copy()

        for species in template_species:
            if species in pred_df.columns:
                merged_df[species] = pred_df.set_index("row_id").loc[merged_df["row_id"], species].values
            else:
                merged_df[species] = 0.0
                logger.warning(f"Species {species} not in predictions, filling with 0.0")

        merged_df = merged_df.fillna(0.0)
        merged_df = merged_df[["row_id"] + template_species]
        logger.info(f"Final submission shape: {merged_df.shape}")

    except Exception as e:
        logger.error(f"Merge failed: {e}", exc_info=True)
        return False

    # ========== STEP 7: Validation before writing ==========
    logger.info("Validating submission...")

    # Check for NaN/Inf
    if merged_df.isnull().any().any():
        logger.error("NaN values detected!")
        merged_df = merged_df.fillna(0.0)
        logger.warning("Filled NaN with 0.0")

    # Check prediction ranges
    pred_cols_to_check = [col for col in merged_df.columns if col != "row_id"]
    for col in pred_cols_to_check[:5]:  # Check first 5
        vals = np.asarray(merged_df[col].values, dtype=np.float32)
        col_min = float(np.min(vals))
        col_max = float(np.max(vals))
        col_mean = float(np.mean(vals))
        logger.info(f"{col}: min={col_min:.4f}, max={col_max:.4f}, mean={col_mean:.4f}")

    # Check row_id matches
    template_ids = set(sample_df["row_id"])
    submitted_ids = set(merged_df["row_id"])
    logger.info(f"Template row_ids: {len(template_ids)}")
    logger.info(f"Submitted row_ids: {len(submitted_ids)}")
    if template_ids != submitted_ids:
        missing = template_ids - submitted_ids
        logger.warning(f"Missing {len(missing)} row_ids")
        if len(missing) <= 10:
            logger.warning(f"Missing: {missing}")

    # ========== STEP 8: Write submission.csv ==========
    logger.info(f"Writing submission to {OUTPUT_CSV}...")
    try:
        # Ensure output directory exists
        OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)

        # Write CSV
        merged_df.to_csv(OUTPUT_CSV, index=False)
        logger.info("✓ Submission written successfully")

        # Force disk sync
        import os
        os.sync()
        logger.info("✓ Disk sync completed")

        # Verify file was written
        if not OUTPUT_CSV.exists():
            logger.error("File was not written (does not exist)!")
            return False

        file_size = OUTPUT_CSV.stat().st_size
        if file_size == 0:
            logger.error("File was written but is empty!")
            return False

        logger.info(f"✓ File size: {file_size} bytes")

        # Quick sanity check
        check_df = pd.read_csv(OUTPUT_CSV)
        logger.info(f"✓ Verified: shape={check_df.shape}, columns={len(check_df.columns)}")
        logger.info(f"✓ Sample:\n{check_df.head()}")

        # Double-check file is readable
        with open(OUTPUT_CSV, 'r') as f:
            first_line = f.readline()
            logger.info(f"✓ File is readable: {first_line[:80]}")

        return True

    except Exception as e:
        logger.error(f"Failed to write submission: {e}", exc_info=True)
        return False


# ============================================================================
# ULTIMATE FALLBACK HANDLER
# ============================================================================
def fallback_submission():
    """Last resort: just copy sample_submission.csv if everything fails."""
    logger.info("="*80)
    logger.info("EMERGENCY FALLBACK: Attempting to copy sample_submission.csv")
    logger.info("="*80)

    fallback_paths = [
        Path("/kaggle/input/competitions/birdclef-2026/sample_submission.csv"),
        Path("/kaggle/input/birdclef-2026/sample_submission.csv"),
        Path("/kaggle/input/sample_submission.csv"),
    ]

    for sample_path in fallback_paths:
        if sample_path.exists():
            try:
                logger.info(f"Found sample_submission.csv at {sample_path}")
                sample_df = pd.read_csv(sample_path)
                logger.info(f"Copied to {OUTPUT_CSV}: shape={sample_df.shape}")
                sample_df.to_csv(OUTPUT_CSV, index=False)

                if OUTPUT_CSV.exists():
                    logger.info(f"✓ Fallback file written: {OUTPUT_CSV.stat().st_size} bytes")
                    return True
            except Exception as e:
                logger.error(f"Fallback copy failed: {e}")
                continue

    logger.error("Could not find sample_submission.csv for fallback!")
    return False


# ============================================================================
# ENTRY POINT
# ============================================================================
if __name__ == "__main__":
    try:
        success = main()
        if success:
            logger.info("="*80)
            logger.info("SUBMISSION COMPLETE ✓")
            logger.info("="*80)
        else:
            logger.error("SUBMISSION FAILED, trying fallback...")
            fallback_submission()
    except Exception as e:
        logger.error(f"CRITICAL ERROR: {e}", exc_info=True)
        logger.error("Attempting fallback...")
        fallback_submission()
