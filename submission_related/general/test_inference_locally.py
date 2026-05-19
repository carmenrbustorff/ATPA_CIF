#!/usr/bin/env python3
"""
Local test script: Use training audio as mock test data to verify inference pipeline.
"""

import os
import sys
import shutil
import tempfile
from pathlib import Path

sys.path.insert(0, '/home/carme/ATPA_CIF')

# Create temporary mock test directory
mock_test_dir = Path(tempfile.gettempdir()) / "mock_test_soundscapes"
mock_test_dir.mkdir(exist_ok=True)

# Use training audio as mock test data (pick first 3 files from different species)
train_audio_base = Path("/home/carme/ATPA_CIF/data/train_audio") if Path("/home/carme/ATPA_CIF/data/train_audio").exists() else None

if not train_audio_base or not train_audio_base.exists():
    print("Training audio not found locally. Checking if this is Kaggle environment...")
    sys.exit(1)

# Find some audio files
audio_files = list(train_audio_base.glob("*/*.ogg"))[:3]
if not audio_files:
    print(f"No audio files found in {train_audio_base}")
    sys.exit(1)

print(f"Found {len(audio_files)} training audio files to use as mock test data")
print(f"Copying to {mock_test_dir}...")

for audio_file in audio_files:
    dest = mock_test_dir / audio_file.stem
    # Rename to look like test files
    dest = mock_test_dir / f"test_{audio_file.stem}.ogg"
    shutil.copy(audio_file, dest)
    print(f"  Copied: {dest.name}")

# Now run inference on mock test data
print("\n" + "="*80)
print("Running inference on mock test data...")
print("="*80 + "\n")

output_csv = Path(tempfile.gettempdir()) / "test_submission.csv"

# Import and run the inference
from submission_related.general.inference import generate_submission_csv

device_str = "cuda" if os.environ.get("CUDA_AVAILABLE") else "cpu"
print(f"Using device: {device_str}")

# You'll need to have your model.pt in the local directory
model_path = Path("/home/carme/ATPA_CIF/model.pt")
if not model_path.exists():
    print(f"Model not found at {model_path}")
    print("Looking for model in common locations...")
    alt_paths = [
        Path.home() / "ATPA_CIF" / "model.pt",
        Path.cwd() / "model.pt",
    ]
    for alt in alt_paths:
        if alt.exists():
            model_path = alt
            print(f"Found at {model_path}")
            break
    else:
        print("Model not found. Please provide the model path.")
        sys.exit(1)

print(f"\nRunning inference:")
print(f"  Model: {model_path}")
print(f"  Test audio: {mock_test_dir}")
print(f"  Output: {output_csv}\n")

generate_submission_csv(
    model_path=str(model_path),
    test_audio_dir=str(mock_test_dir),
    output_csv=str(output_csv),
    device="cpu"  # Use CPU for local testing
)

# Verify output
print("\n" + "="*80)
print("Verification")
print("="*80)

import pandas as pd

if output_csv.exists():
    df = pd.read_csv(output_csv)
    print(f"✓ CSV created: {output_csv}")
    print(f"  Shape: {df.shape}")
    print(f"  Columns: {list(df.columns)[:10]}... ({len(df.columns)} total)")
    print(f"\n  Sample predictions:\n{df.head()}")

    # Check for valid predictions
    pred_cols = [col for col in df.columns if col != "row_id"]
    pred_values = df[pred_cols].values.flatten()

    print(f"\n  Prediction stats:")
    print(f"    Min: {pred_values.min():.4f}")
    print(f"    Max: {pred_values.max():.4f}")
    print(f"    Mean: {pred_values.mean():.4f}")
    print(f"    Non-zero predictions: {(pred_values > 0).sum()} / {len(pred_values)}")

    if (pred_values > 0).sum() > 0:
        print(f"\n  ✓ SUCCESS: Model produced non-zero predictions!")
    else:
        print(f"\n  ⚠ WARNING: All predictions are zero")
else:
    print(f"✗ CSV not created at {output_csv}")

print("\n" + "="*80)
print("Cleanup: Remove mock test directory")
print("="*80)
shutil.rmtree(mock_test_dir)
print(f"Removed {mock_test_dir}")
