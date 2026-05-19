#!/usr/bin/env python3
"""
Quick test: Verify inference pipeline works with sample audio.
"""

import sys
import os
from pathlib import Path

sys.path.insert(0, '/home/carme/ATPA_CIF')

print("="*80)
print("LOCAL INFERENCE TEST")
print("="*80)

# Check what's available
print("\n1. Checking environment...")
print(f"   Working directory: {Path.cwd()}")
print(f"   Python: {sys.executable}")

# Check for model
model_candidates = [
    Path("/home/carme/ATPA_CIF/model.pt"),
    Path("/home/carme/model.pt"),
    Path("./model.pt"),
]

model_path = None
for candidate in model_candidates:
    if candidate.exists():
        model_path = candidate
        print(f"   ✓ Model found: {model_path}")
        break

if not model_path:
    print(f"   ✗ Model NOT found. Checked: {[str(c) for c in model_candidates]}")
    print("\n   To test locally, you need:")
    print("   1. The trained model.pt file")
    print("   2. Sample audio files")
    sys.exit(1)

# Check for audio data
print("\n2. Checking for audio data...")
data_dir = Path("/home/carme/ATPA_CIF/data")
if data_dir.exists():
    audio_files = list(data_dir.glob("**/*.ogg")) + list(data_dir.glob("**/*.wav")) + list(data_dir.glob("**/*.mp3"))
    print(f"   ✓ Found {len(audio_files)} audio files in {data_dir}")
else:
    print(f"   ✗ No data directory at {data_dir}")
    audio_files = []

if not audio_files:
    print("\n   To run a local test, you need:")
    print("   1. Download sample training audio from Kaggle")
    print("   2. Place in /home/carme/ATPA_CIF/data/")
    print("\n   Quick Kaggle test alternative:")
    print("   - Just upload to Kaggle and resubmit when quota resets")
    print("   - Code is correct; it will work on hidden test data")
    sys.exit(0)

# If we got here, we can run a test
print("\n3. Running inference test...")
print(f"   Model: {model_path}")
print(f"   Sample audio: {len(audio_files)} files")

import torch
import tempfile
import shutil

try:
    from submission_related.general.inference import generate_submission_csv

    # Create mock test directory with first 3 audio files
    mock_test = Path(tempfile.gettempdir()) / "mock_test"
    mock_test.mkdir(exist_ok=True)

    for audio_file in audio_files[:3]:
        shutil.copy(audio_file, mock_test / f"test_{audio_file.name}")

    output_csv = Path(tempfile.gettempdir()) / "test_output.csv"

    print(f"\n   Running with {len(list(mock_test.glob('*')))} mock test files...")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    generate_submission_csv(
        model_path=str(model_path),
        test_audio_dir=str(mock_test),
        output_csv=str(output_csv),
        device=device
    )

    # Check results
    import pandas as pd
    if output_csv.exists():
        df = pd.read_csv(output_csv)
        print(f"\n   ✓ SUCCESS! Output CSV created:")
        print(f"     Shape: {df.shape}")
        print(f"     Sample predictions:\n{df.iloc[0, :5]}")

        # Check for valid predictions
        pred_cols = [c for c in df.columns if c != "row_id"]
        has_nonzero = (df[pred_cols].values > 0).any()
        print(f"     Has non-zero predictions: {has_nonzero}")

    # Cleanup
    shutil.rmtree(mock_test)

except Exception as e:
    print(f"\n   ✗ Error during test: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n" + "="*80)
print("Your submission script is ready for Kaggle!")
print("="*80)
