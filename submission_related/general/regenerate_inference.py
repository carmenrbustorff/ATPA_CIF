"""Regenerate inference.py from existing iterations (fixed format: per-chunk, float32)."""

import argparse
import ast
import logging
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

NUM_SPECIES = 206

# Template with proper chunk-based row generation
KAGGLE_INFERENCE_TEMPLATE = '''\
import os
import gc
import torch
import torch.nn as nn
import torchaudio.transforms as T
import numpy as np
import soundfile as sf
import pandas as pd
import timm
from pathlib import Path

SAMPLE_RATE = 32000
CHUNK_DURATION = 5.0
MEL_BINS = 128
N_FFT = 2048
HOP_LENGTH = 512
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

{INJECTED_MODEL_CLASSES}

def extract_mel_spectrogram_cpu(audio_chunk, sr=SAMPLE_RATE):
    if audio_chunk.shape[0] == 0:
        return np.zeros((1, MEL_BINS, 1), dtype=np.float32)
    audio_tensor = torch.from_numpy(audio_chunk.astype(np.float32))
    mel_transform = T.MelSpectrogram(sample_rate=sr, n_mels=MEL_BINS, n_fft=N_FFT, hop_length=HOP_LENGTH)
    mel_spec = mel_transform(audio_tensor)
    log_mel = torch.log(mel_spec + 1e-9)
    return log_mel.unsqueeze(0).numpy().astype(np.float32)

def infer_on_audio_file(audio_path, model, device):
    results = []
    try:
        audio_data, sr = sf.read(audio_path, dtype=np.float32)
        if len(audio_data.shape) > 1:
            audio_data = audio_data.mean(axis=1)
        if sr != SAMPLE_RATE:
            resampler = T.Resample(orig_freq=sr, new_freq=SAMPLE_RATE)
            audio_tensor = torch.from_numpy(audio_data).float()
            audio_data = resampler(audio_tensor).numpy()
    except Exception as e:
        print(f"Error loading {{audio_path}}: {{e}}")
        return results

    file_stem = Path(audio_path).stem
    num_samples_per_chunk = int(CHUNK_DURATION * SAMPLE_RATE)
    num_chunks = int(np.ceil(len(audio_data) / num_samples_per_chunk))

    for chunk_idx in range(num_chunks):
        start_sample = chunk_idx * num_samples_per_chunk
        end_sample = min((chunk_idx + 1) * num_samples_per_chunk, len(audio_data))
        audio_chunk = audio_data[start_sample:end_sample]
        mel_spec = extract_mel_spectrogram_cpu(audio_chunk)
        mel_tensor = torch.from_numpy(mel_spec).to(device)

        with torch.no_grad(), torch.autocast(device_type="cuda" if device.type == "cuda" else "cpu"):
            logits = model(mel_tensor.unsqueeze(0))  # Add batch dimension
            probs = torch.sigmoid(logits).squeeze(0).cpu().numpy().astype(np.float32)

        end_time_seconds = int((chunk_idx + 1) * CHUNK_DURATION)
        row_id = f"{{file_stem}}_{{end_time_seconds}}"
        results.append({{"row_id": row_id, "predictions": probs}})

        del mel_tensor, logits
        if device.type == "cuda":
            torch.cuda.empty_cache()
        gc.collect()

    return results

def generate_submission_csv(test_audio_dir, model_path, output_csv, sample_submission_path="sample_submission.csv"):
    print("[Inference] Loading model…")

    import inspect
    model_class = None
    for name, obj in globals().items():
        if inspect.isclass(obj) and issubclass(obj, nn.Module) and name not in ('nn', 'Module') and not name.startswith('_'):
            model_class = obj
            break

    if model_class is None:
        raise RuntimeError("No nn.Module subclass found")

    model = model_class(num_classes={NUM_CLASSES})
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model = model.to(DEVICE)
    model.eval()

    print(f"[Inference] Model loaded")

    audio_dir = Path(test_audio_dir)
    audio_exts = {{".ogg", ".wav", ".flac", ".mp3"}}
    audio_files = sorted([f for f in audio_dir.rglob("*") if f.suffix.lower() in audio_exts])

    if not audio_files:
        print(f"[Warning] No audio files in {{test_audio_dir}}")
        audio_files = []

    print(f"[Inference] Found {{len(audio_files)}} audio files")

    all_results = []
    for i, audio_file in enumerate(audio_files):
        if (i + 1) % 100 == 0 or i == 0:
            print(f"[Inference] {{i+1}}/{{len(audio_files)}}…")
        chunk_results = infer_on_audio_file(str(audio_file), model, DEVICE)
        all_results.extend(chunk_results)

    print(f"[Inference] Total chunks: {{len(all_results)}}")

    if not all_results:
        print("[Warning] No predictions generated.")
        submission_df = pd.DataFrame({{"row_id": []}})
        submission_df.to_csv(output_csv, index=False)
        return submission_df

    local_data = {{"row_id": [r["row_id"] for r in all_results]}}
    for local_idx in range({NUM_CLASSES}):
        local_data[f"c{{local_idx}}"] = np.array([r["predictions"][local_idx] for r in all_results], dtype=np.float32)

    local_df = pd.DataFrame(local_data)

    try:
        kaggle_df = pd.read_csv(sample_submission_path)
        kaggle_cols = [c for c in kaggle_df.columns if c != "row_id"]
    except FileNotFoundError:
        kaggle_cols = [f"species_{{i:03d}}" for i in range(234)]

    print(f"[Inference] Mapping {{len([c for c in local_data.keys()])}} local classes to {{len(kaggle_cols)}} Kaggle classes")

    final_data = {{"row_id": local_df["row_id"].values}}
    for kaggle_col_idx in range(len(kaggle_cols)):
        if kaggle_col_idx < {NUM_CLASSES}:
            final_data[kaggle_cols[kaggle_col_idx]] = local_df[f"c{{kaggle_col_idx}}"].values.astype(np.float32)
        else:
            final_data[kaggle_cols[kaggle_col_idx]] = np.zeros(len(local_df), dtype=np.float32)

    final_df = pd.DataFrame(final_data)
    final_df = final_df[["row_id"] + kaggle_cols]

    for col in kaggle_cols:
        final_df[col] = final_df[col].astype(np.float32)

    output_path = Path(output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    final_df.to_csv(output_path, index=False)

    print(f"[Inference] Saved: {{output_path}}")
    print(f"[Inference] Shape: {{final_df.shape}} ({{len(final_df)}} rows = 1 per chunk)")

    return final_df

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 4:
        print("Usage: inference.py <model_path> <test_audio_dir> <output_csv>")
        sys.exit(1)
    generate_submission_csv(sys.argv[2], sys.argv[1], sys.argv[3])
    print("[Inference] Complete!")
'''


def extract_model_classes(source_code: str) -> str:
    """Extract model classes from train.py."""
    try:
        tree = ast.parse(source_code)
    except SyntaxError as e:
        logger.warning("Could not parse source code: %s", e)
        return ""

    extracted_code = []
    lines = source_code.split('\n')
    excluded_names = {"FocalLoss", "MixupLoss", "Loss", "Metric", "Scheduler"}

    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            if node.name in excluded_names or any(exc in node.name for exc in excluded_names):
                continue

            is_model_class = any(
                (isinstance(base, ast.Name) and base.id in ("nn.Module", "Module")) or
                (isinstance(base, ast.Attribute) and base.attr == "Module")
                for base in node.bases
            )

            if is_model_class or "Model" in node.name:
                start_line = node.lineno - 1
                end_line = node.end_lineno if node.end_lineno else start_line + 10
                class_source = '\n'.join(lines[start_line:end_line])
                extracted_code.append(class_source)

    return '\n\n'.join(extracted_code)


def regenerate_inference(iteration_dir: Path) -> bool:
    """Regenerate inference.py from existing iteration."""
    iteration_dir = iteration_dir.resolve()

    logger.info("=" * 70)
    logger.info("REGENERATING INFERENCE.PY (Per-Chunk Format)")
    logger.info("=" * 70)

    train_py = iteration_dir / "train.py"
    if not train_py.exists():
        logger.error("train.py not found")
        return False

    try:
        train_code = train_py.read_text(encoding="utf-8")
    except Exception as e:
        logger.error("Failed to read train.py: %s", e)
        return False

    model_classes = extract_model_classes(train_code)

    if not model_classes.strip():
        logger.warning("No model classes found")
        return False

    inference_script = KAGGLE_INFERENCE_TEMPLATE.format(
        INJECTED_MODEL_CLASSES=model_classes,
        NUM_CLASSES=NUM_SPECIES,
    )

    inference_path = iteration_dir / "inference.py"
    try:
        inference_path.write_text(inference_script, encoding="utf-8")
    except Exception as e:
        logger.error("Failed to write inference.py: %s", e)
        return False

    logger.info("✓ Generated: %s", inference_path)
    return True


def main():
    parser = argparse.ArgumentParser(description="Regenerate inference.py from existing iterations")
    parser.add_argument("--iteration-dir", type=Path, required=True, help="Path to iteration folder")
    args = parser.parse_args()
    success = regenerate_inference(args.iteration_dir)
    return 0 if success else 1


if __name__ == "__main__":
    import sys
    sys.exit(main())
