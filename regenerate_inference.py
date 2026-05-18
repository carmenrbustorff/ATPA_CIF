"""
Regenerate inference.py from existing iterations.

This script allows you to generate inference.py for iterations that were
completed before the adaptive inference pipeline was implemented.

Usage:
    python regenerate_inference.py \\
        --iteration-dir ./experiments/iterations_0001-0050/iter_0050_20260515_100000
"""

import argparse
import ast
import logging
from pathlib import Path
from typing import Optional

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

NUM_SPECIES = 206

KAGGLE_INFERENCE_TEMPLATE = '''\
"""
Auto-generated Kaggle inference script.
Loads a multi-hour audio file, processes it in 5-second windows,
and generates a submission CSV compatible with the BirdCLEF competition.
"""

import os
import gc
import json
import torch
import torch.nn as nn
import torchaudio
import torchaudio.transforms as T
import numpy as np
import soundfile as sf
import pandas as pd
from pathlib import Path

# ============================================================================
# INJECTED MODEL CLASSES (AUTO-GENERATED FROM TRAIN.PY)
# ============================================================================

{INJECTED_MODEL_CLASSES}

# ============================================================================
# Inference Configuration
# ============================================================================

SAMPLE_RATE = 32000
WINDOW_DURATION = 5.0  # seconds
MEL_BINS = 128
N_FFT = 2048
HOP_LENGTH = 512
BATCH_SIZE_INFERENCE = 16
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ============================================================================
# CPU-based Mel-Spectrogram Extraction
# ============================================================================

def extract_mel_spectrogram_cpu(audio_chunk: np.ndarray, sr: int = SAMPLE_RATE) -> np.ndarray:
    """
    Extract mel-spectrogram on CPU to avoid GPU memory overhead.
    Audio is expected as float32 numpy array.
    Returns log-mel spectrogram of shape (MEL_BINS, time_steps).
    """
    if audio_chunk.shape[0] == 0:
        return np.zeros((MEL_BINS, 1), dtype=np.float32)

    # Convert to torch tensor (CPU)
    audio_tensor = torch.from_numpy(audio_chunk).float()

    # Create mel-spectrogram transform on CPU
    mel_transform = T.MelSpectrogram(
        sample_rate=sr,
        n_mels=MEL_BINS,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH
    )

    # Compute mel-spectrogram
    mel_spec = mel_transform(audio_tensor)

    # Convert to log scale
    log_mel_spec = torch.log(mel_spec + 1e-9)

    return log_mel_spec.numpy().astype(np.float32)

# ============================================================================
# Inference Pipeline with GPU Memory Efficiency
# ============================================================================

def infer_on_audio_file(
    audio_path: str,
    model: nn.Module,
    device: torch.device,
    local_num_classes: int = 206,
) -> dict:
    """
    Load audio file, split into windows, extract spectrograms (CPU),
    and run inference in batches (GPU). Return predictions per chunk.
    """
    try:
        # Load audio file on CPU
        audio_data, sr = sf.read(audio_path, dtype=np.float32)

        if len(audio_data.shape) > 1:
            audio_data = audio_data.mean(axis=1)

        # Resample if necessary
        if sr != SAMPLE_RATE:
            resampler = T.Resample(orig_freq=sr, new_freq=SAMPLE_RATE)
            audio_tensor = torch.from_numpy(audio_data).float()
            audio_data = resampler(audio_tensor).numpy()

    except Exception as e:
        print(f"Error loading {{audio_path}}: {{e}}")
        return {{
            "file": Path(audio_path).stem,
            "predictions": np.zeros((1, local_num_classes), dtype=np.float32),
            "error": str(e)
        }}

    # Split audio into 5-second windows
    num_samples_per_window = int(WINDOW_DURATION * SAMPLE_RATE)
    num_windows = int(np.ceil(len(audio_data) / num_samples_per_window))

    all_predictions = []

    # Process windows in CPU batches, then move to GPU
    for window_idx in range(num_windows):
        start_sample = window_idx * num_samples_per_window
        end_sample = min((window_idx + 1) * num_samples_per_window, len(audio_data))

        audio_chunk = audio_data[start_sample:end_sample]

        # Extract mel-spectrogram on CPU
        mel_spec = extract_mel_spectrogram_cpu(audio_chunk, sr=SAMPLE_RATE)

        # Ensure correct shape (1, MEL_BINS, time_steps)
        if len(mel_spec.shape) == 2:
            mel_spec = np.expand_dims(mel_spec, axis=0)

        # Convert to tensor and move to GPU
        mel_tensor = torch.from_numpy(mel_spec).float().to(device)

        # Inference with autocast for memory efficiency
        with torch.no_grad(), torch.autocast(device_type="cuda" if device.type == "cuda" else "cpu"):
            logits = model(mel_tensor.unsqueeze(0))  # Add batch dimension
            probs = torch.sigmoid(logits).cpu().numpy()

        all_predictions.append(probs)

        # Clean up GPU memory
        del mel_tensor, logits
        if device.type == "cuda":
            torch.cuda.empty_cache()
        gc.collect()

    # Stack all window predictions
    if all_predictions:
        all_preds = np.vstack(all_predictions)
        # Take max probability across all windows per class
        max_probs = np.max(all_preds, axis=0)
    else:
        max_probs = np.zeros((1, local_num_classes), dtype=np.float32)

    return {{
        "file": Path(audio_path).stem,
        "predictions": max_probs,
        "num_windows": num_windows,
        "error": None
    }}

# ============================================================================
# Main Inference and Submission Generation
# ============================================================================

def generate_submission_csv(
    test_audio_dir: str,
    model_path: str,
    output_csv: str,
    submission_template: str = "sample_submission.csv",
):
    """
    Main inference pipeline:
    1. Load best_model.pth
    2. Infer on all audio files in test_audio_dir
    3. Merge with sample_submission.csv (234 Kaggle classes)
    4. Export submission CSV
    """

    # Load model – find the first nn.Module class from injected classes
    print("[Inference] Loading model from:", model_path)

    # Find the model class (should be BirdCLEFModel if enforced by agent constraints)
    import inspect
    model_class = None
    for name, obj in globals().items():
        if (inspect.isclass(obj) and
            issubclass(obj, nn.Module) and
            name != 'nn' and
            not name.startswith('_')):
            model_class = obj
            break

    if model_class is None:
        raise RuntimeError("No nn.Module subclass found in generated code. Check injected model classes.")

    model = model_class(num_classes={NUM_LOCAL_CLASSES})
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model = model.to(DEVICE)
    model.eval()

    print("[Inference] Model loaded successfully")

    # Get list of test audio files
    audio_dir = Path(test_audio_dir)
    audio_exts = {{".ogg", ".wav", ".flac", ".mp3"}}
    audio_files = [f for f in audio_dir.rglob("*") if f.suffix.lower() in audio_exts]

    if not audio_files:
        print(f"[Warning] No audio files found in {{test_audio_dir}}")
        audio_files = []

    print(f"[Inference] Found {{len(audio_files)}} audio files to process")

    # Run inference on all files
    results = []
    for i, audio_file in enumerate(audio_files):
        print(f"[Inference] Processing {{i+1}}/{{len(audio_files)}}: {{audio_file.name}}")

        result = infer_on_audio_file(
            str(audio_file),
            model,
            DEVICE,
            local_num_classes={NUM_LOCAL_CLASSES}
        )
        results.append(result)

        if result["error"]:
            print(f"  Warning: {{result['error']}}")

    # Aggregate predictions by file_id (group by max probability across all audio)
    file_predictions = {{}}
    for result in results:
        file_id = result["file"]
        probs = result["predictions"]

        if file_id not in file_predictions:
            file_predictions[file_id] = probs.flatten()
        else:
            # Take max if multiple entries for same file
            file_predictions[file_id] = np.maximum(file_predictions[file_id], probs.flatten())

    print(f"[Inference] Aggregated {{len(file_predictions)}} unique files")

    # Create local submission with 206 classes
    local_submission = pd.DataFrame({{
        "row_id": list(file_predictions.keys()),
    }})
    for class_idx in range({NUM_LOCAL_CLASSES}):
        local_submission[f"class_{{class_idx}}"] = [
            file_predictions[fid][class_idx] if class_idx < len(file_predictions[fid]) else 0.0
            for fid in local_submission["row_id"]
        ]

    # Load Kaggle submission template (234 classes)
    try:
        submission_template_df = pd.read_csv(submission_template)
    except FileNotFoundError:
        print(f"[Warning] Submission template {{submission_template}} not found. Creating with all zeros.")
        # Create a dummy 234-column template
        submission_template_df = pd.DataFrame()
        submission_template_df["row_id"] = local_submission["row_id"]
        for i in range(234):
            submission_template_df[f"species_{{i:03d}}"] = 0.0

    # Left-join: local predictions with 206 classes, fill missing Kaggle classes with 0.0
    # Match on row_id
    final_submission = submission_template_df[["row_id"]].copy()

    # Merge local predictions
    for col in submission_template_df.columns:
        if col == "row_id":
            continue
        # Try to match local class index from column name
        if col.startswith("species_"):
            try:
                # Kaggle uses 0-based indexing: species_000, species_001, ..., species_233
                kaggle_idx = int(col.split("_")[1])
                if kaggle_idx < {NUM_LOCAL_CLASSES}:
                    # Try to find matching local class
                    final_submission[col] = 0.0
                    for idx, row in local_submission.iterrows():
                        matching_row = submission_template_df[submission_template_df["row_id"] == row["row_id"]]
                        if not matching_row.empty:
                            local_idx = kaggle_idx
                            if local_idx < {NUM_LOCAL_CLASSES}:
                                final_submission.loc[final_submission["row_id"] == row["row_id"], col] = row[f"class_{{local_idx}}"]
                else:
                    final_submission[col] = 0.0
            except Exception:
                final_submission[col] = 0.0
        else:
            final_submission[col] = submission_template_df[col]

    # Fill any NaNs with 0.0
    final_submission = final_submission.fillna(0.0)

    # Ensure exactly 234 columns + row_id
    kaggle_cols = [c for c in final_submission.columns if c.startswith("species_")]
    if len(kaggle_cols) < 234:
        for i in range(234):
            col_name = f"species_{{i:03d}}"
            if col_name not in final_submission.columns:
                final_submission[col_name] = 0.0

    # Reorder: row_id first, then all species columns
    final_submission = final_submission[["row_id"] + sorted([c for c in final_submission.columns if c.startswith("species_")])]

    # Save submission
    output_path = Path(output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    final_submission.to_csv(output_path, index=False)

    print(f"[Inference] Submission saved to: {{output_path}}")
    print(f"[Inference] Shape: {{final_submission.shape}} (rows, cols)")

    return final_submission

# ============================================================================
# Entry Point
# ============================================================================

if __name__ == "__main__":
    import sys

    # Expect arguments: model_path, test_audio_dir, output_csv
    if len(sys.argv) < 4:
        print("Usage: inference.py <model_path> <test_audio_dir> <output_csv>")
        sys.exit(1)

    model_path = sys.argv[1]
    test_audio_dir = sys.argv[2]
    output_csv = sys.argv[3]

    print(f"[Inference] Starting inference pipeline")
    print(f"  Model: {{model_path}}")
    print(f"  Test dir: {{test_audio_dir}}")
    print(f"  Output: {{output_csv}}")

    submission_df = generate_submission_csv(
        test_audio_dir=test_audio_dir,
        model_path=model_path,
        output_csv=output_csv,
    )

    print("[Inference] Complete!")
'''

# ---------------------------------------------------------------------------
# Helper: extract model classes from existing train.py
# ---------------------------------------------------------------------------

def extract_model_classes(source_code: str) -> str:
    """
    Parse Python source code using AST and extract all class definitions
    that inherit from nn.Module.

    Returns the raw source code of these classes as a single string.
    """
    try:
        tree = ast.parse(source_code)
    except SyntaxError as e:
        logger.warning("Could not parse source code for class extraction: %s", e)
        return ""

    extracted_code = []
    lines = source_code.split('\n')

    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            # Check if it inherits from nn.Module or has nn.Module in bases
            is_model_class = any(
                (isinstance(base, ast.Name) and base.id in ("nn.Module", "Module")) or
                (isinstance(base, ast.Attribute) and base.attr == "Module")
                for base in node.bases
            )

            # Also include any custom classes (to be safe), but prioritize nn.Module subclasses
            if is_model_class or "Model" in node.name or "model" in node.name.lower():
                # Extract source code for this class
                start_line = node.lineno - 1
                end_line = node.end_lineno if node.end_lineno else start_line + 10

                class_source = '\n'.join(lines[start_line:end_line])
                extracted_code.append(class_source)

    return '\n\n'.join(extracted_code)


# ---------------------------------------------------------------------------
# Main regeneration logic
# ---------------------------------------------------------------------------

def regenerate_inference(iteration_dir: Path) -> bool:
    """
    Regenerate inference.py from an existing iteration's train.py.

    Returns True if successful, False otherwise.
    """
    iteration_dir = iteration_dir.resolve()

    logger.info("=" * 70)
    logger.info("REGENERATING INFERENCE.PY")
    logger.info("=" * 70)
    logger.info("Iteration directory: %s", iteration_dir)

    # Check required files
    train_py = iteration_dir / "train.py"
    model_pt = iteration_dir / "best_model.pth"

    if not train_py.exists():
        logger.error("train.py not found in %s", iteration_dir)
        return False

    if not model_pt.exists():
        # Try model.pt as fallback
        model_pt = iteration_dir / "model.pt"
        if not model_pt.exists():
            logger.warning("No model checkpoint found (best_model.pth or model.pt)")
            logger.info("inference.py will be created, but needs a model checkpoint to run")

    # Read train.py and extract model classes
    logger.info("Reading train.py…")
    try:
        train_code = train_py.read_text(encoding="utf-8")
    except Exception as e:
        logger.error("Failed to read train.py: %s", e)
        return False

    # Extract model classes
    logger.info("Extracting model classes from train.py…")
    model_classes = extract_model_classes(train_code)

    if not model_classes.strip():
        logger.warning("No model classes found in train.py")
        logger.warning("Make sure train.py defines at least one class inheriting from nn.Module")
        return False

    logger.info("Found model classes:")
    for line in model_classes.split('\n')[:5]:
        logger.info("  %s", line)
    if model_classes.count('\n') > 5:
        logger.info("  … (and more)")

    # Generate inference.py
    logger.info("Generating inference.py…")
    inference_script = KAGGLE_INFERENCE_TEMPLATE.format(
        INJECTED_MODEL_CLASSES=model_classes,
        NUM_LOCAL_CLASSES=NUM_SPECIES,
    )

    inference_path = iteration_dir / "inference.py"
    try:
        inference_path.write_text(inference_script, encoding="utf-8")
    except Exception as e:
        logger.error("Failed to write inference.py: %s", e)
        return False

    logger.info("=" * 70)
    logger.info("✓ Successfully generated: %s", inference_path)
    logger.info("=" * 70)
    logger.info("\nNext steps:")
    logger.info("  1. Test locally:")
    logger.info("     python generate_local_submission.py \\")
    logger.info("         --iteration-dir %s \\", iteration_dir)
    logger.info("         --test-audio-dir ./test_audio \\")
    logger.info("         --output-dir ./submissions")
    logger.info("\n  2. If validation passes, submit to Kaggle")

    return True


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Regenerate inference.py from existing iterations",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--iteration-dir",
        type=Path,
        required=True,
        help="Path to iteration folder containing train.py (e.g., ./experiments/iterations_0001-0050/iter_0050_20260515_100000)",
    )

    args = parser.parse_args()

    success = regenerate_inference(args.iteration_dir)

    if not success:
        logger.error("Failed to regenerate inference.py")
        return 1

    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
