#!/usr/bin/env python3
"""
Generate a clean Kaggle submission notebook for BirdCLEF+ 2026.

The generated notebook expects the trained model to be attached as a Kaggle
data source and loads the state dict from /kaggle/input/... at runtime.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from textwrap import dedent

DEFAULT_WEIGHTS_DATASET = "birdclef-pytorch-weights"
DEFAULT_WEIGHTS_FILENAME = "model.pt"


def _cell(source: str) -> str:
    return dedent(source).strip("\n")


def _cell_source(source: str) -> list[str]:
    lines = source.splitlines()
    return [f"{line}\n" for line in lines[:-1]] + ([lines[-1]] if lines else [])


def _build_cells(weights_dataset: str, weights_filename: str) -> list[tuple[str, str]]:
    weights_input_dir = f"/kaggle/input/{weights_dataset}"

    return [
        (
            "markdown",
            _cell(
                """
                # BirdCLEF+ 2026 Kaggle Submission

                Clean submission notebook that loads model weights from a Kaggle
                data source, runs inference on the competition test audio, and
                writes `submission.csv`.
                """
            ),
        ),
        (
            "markdown",
            _cell(
                """
                ## Kaggle inputs

                Attach the competition dataset and one model dataset. The notebook
                looks for the model weights at `/kaggle/input/<weights-dataset>/<weights-filename>`.

                If you regenerate the notebook for a different checkpoint, pass new
                values for `--weights-dataset` and `--weights-filename`.
                """
            ),
        ),
        (
            "code",
            _cell(
                """
                import os
                import time
                import warnings
                from pathlib import Path

                warnings.filterwarnings("ignore")

                import librosa
                import numpy as np
                import pandas as pd
                import torch
                import torch.nn as nn

                SUBMISSION_START = time.time()
                MAX_RUNTIME_S = 90 * 60
                KAGGLE_INPUT_DIR = Path("/kaggle/input")
                MODEL_INPUT_DIR = Path("__WEIGHTS_INPUT_DIR__")
                MODEL_FILENAME = "__WEIGHTS_FILENAME__"

                def time_remaining() -> float:
                    return MAX_RUNTIME_S - (time.time() - SUBMISSION_START)

                print(f"Submission started. Runtime budget: {MAX_RUNTIME_S / 60:.0f} min")
                print(f"PyTorch: {torch.__version__} | CUDA available: {torch.cuda.is_available()}")
                """
            )
            .replace("__WEIGHTS_INPUT_DIR__", weights_input_dir)
            .replace("__WEIGHTS_FILENAME__", weights_filename),
        ),
        (
            "code",
            _cell(
                """
                def find_competition_root() -> tuple[Path, Path, Path]:
                    for candidate in sorted(KAGGLE_INPUT_DIR.iterdir()):
                        if not candidate.is_dir():
                            continue
                        sample_path = candidate / "sample_submission.csv"
                        test_path = candidate / "test_soundscapes"
                        if sample_path.exists() and test_path.exists():
                            return candidate, sample_path, test_path
                    raise FileNotFoundError(
                        "Could not find a Kaggle input directory with sample_submission.csv and test_soundscapes."
                    )

                def find_model_path() -> Path:
                    explicit = MODEL_INPUT_DIR / MODEL_FILENAME
                    if explicit.exists():
                        return explicit

                    if MODEL_INPUT_DIR.exists():
                        for pattern in ("*.pt", "*.pth", "*.bin"):
                            matches = sorted(MODEL_INPUT_DIR.rglob(pattern))
                            if matches:
                                return matches[0]

                    raise FileNotFoundError(
                        f"Could not find model weights under {MODEL_INPUT_DIR}."
                    )

                COMPETITION_ROOT, SAMPLE_SUB_PATH, TEST_DIR = find_competition_root()
                MODEL_PATH = find_model_path()

                sample_sub = pd.read_csv(SAMPLE_SUB_PATH)
                SPECIES_COLS = [column for column in sample_sub.columns if column != "row_id"]
                NUM_SPECIES = len(SPECIES_COLS)

                print(f"Competition root: {COMPETITION_ROOT}")
                print(f"Test dir: {TEST_DIR}")
                print(f"Model path: {MODEL_PATH}")
                print(f"Species columns: {NUM_SPECIES}")
                """
            ),
        ),
        (
            "code",
            _cell(
                """
                SR = 32000
                DURATION = 5.0
                CHUNK_SAMPLES = int(SR * DURATION)
                N_MELS = 128
                N_FFT = 1024
                HOP_LENGTH = 512
                FMIN = 50
                FMAX = 14000
                TOP_DB = 80.0
                TARGET_FRAMES = 216

                def audio_to_melspec(waveform, sr=SR):
                    mel = librosa.feature.melspectrogram(
                        y=waveform,
                        sr=sr,
                        n_fft=N_FFT,
                        hop_length=HOP_LENGTH,
                        n_mels=N_MELS,
                        fmin=FMIN,
                        fmax=FMAX,
                    )
                    log_mel = librosa.power_to_db(mel, ref=np.max, top_db=TOP_DB)
                    return ((log_mel + TOP_DB) / TOP_DB).astype(np.float32)

                def pad_or_trim(spec, target_frames=TARGET_FRAMES):
                    if spec.shape[1] < target_frames:
                        spec = np.pad(spec, ((0, 0), (0, target_frames - spec.shape[1])))
                    else:
                        spec = spec[:, :target_frames]
                    return spec

                def load_chunks(filepath):
                    try:
                        waveform, _ = librosa.load(filepath, sr=SR, mono=True)
                    except Exception as exc:
                        print(f"Warning: failed to load {filepath}: {exc}")
                        return

                    for start in range(0, len(waveform), CHUNK_SAMPLES):
                        chunk = waveform[start : start + CHUNK_SAMPLES]
                        if len(chunk) < CHUNK_SAMPLES // 4:
                            break
                        if len(chunk) < CHUNK_SAMPLES:
                            chunk = np.pad(chunk, (0, CHUNK_SAMPLES - len(chunk)))

                        spec = audio_to_melspec(chunk)
                        spec = pad_or_trim(spec)
                        yield spec[np.newaxis, ...]

                print("Audio preprocessing ready.")
                """
            ),
        ),
        (
            "code",
            _cell(
                """
                class BirdCLEFModel(nn.Module):
                    def __init__(self, num_species=NUM_SPECIES, dropout_rate=0.5):
                        super().__init__()
                        self.conv_layers = nn.Sequential(
                            nn.Conv2d(1, 32, kernel_size=3, padding=1),
                            nn.BatchNorm2d(32),
                            nn.ReLU(),
                            nn.Conv2d(32, 64, kernel_size=3, padding=1),
                            nn.BatchNorm2d(64),
                            nn.ReLU(),
                        )
                        self.pool = nn.AdaptiveAvgPool2d((1, 1))
                        self.classifier = nn.Sequential(
                            nn.Linear(64, 128),
                            nn.ReLU(),
                            nn.Dropout(dropout_rate),
                            nn.Linear(128, num_species),
                        )

                    def forward(self, x):
                        x = self.conv_layers(x)
                        x = self.pool(x).view(x.size(0), -1)
                        return torch.sigmoid(self.classifier(x))
                """
            ),
        ),
        (
            "code",
            _cell(
                """
                device = torch.device("cpu")
                model = BirdCLEFModel(num_species=NUM_SPECIES).to(device)
                state_dict = torch.load(MODEL_PATH, map_location=device)
                model.load_state_dict(state_dict)
                model.eval()

                print(f"Loaded model with {sum(p.numel() for p in model.parameters()):,} parameters")
                """
            ),
        ),
        (
            "code",
            _cell(
                """
                BATCH_SIZE = 8
                results = []
                test_files = sorted(
                    file_path for file_path in TEST_DIR.iterdir()
                    if file_path.suffix.lower() in {".ogg", ".wav", ".mp3", ".flac"}
                )

                print(f"Test files found: {len(test_files)}")

                for file_idx, file_path in enumerate(test_files, start=1):
                    if time_remaining() < 300:
                        print("Approaching the Kaggle time limit; stopping early.")
                        break

                    if file_idx % 10 == 0:
                        print(f"  [{file_idx}/{len(test_files)}] {time_remaining() / 60:.1f} min remaining")

                    chunks = list(load_chunks(file_path))
                    if not chunks:
                        continue

                    for batch_start in range(0, len(chunks), BATCH_SIZE):
                        batch_specs = np.stack(chunks[batch_start : batch_start + BATCH_SIZE], axis=0)
                        batch_tensor = torch.from_numpy(batch_specs).float().to(device)

                        with torch.inference_mode():
                            batch_preds = model(batch_tensor).cpu().numpy()

                        for offset, pred in enumerate(batch_preds):
                            end_sec = int((batch_start + offset + 1) * DURATION)
                            row = {"row_id": f"{file_path.stem}_{end_sec}"}
                            row.update(dict(zip(SPECIES_COLS, pred.tolist())))
                            results.append(row)

                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

                print(f"Inference complete. Total rows: {len(results)}")
                print(f"Time remaining: {time_remaining() / 60:.1f} min")
                """
            ),
        ),
        (
            "code",
            _cell(
                """
                if results:
                    prediction_frame = pd.DataFrame(results)
                    submission = sample_sub[["row_id"]].merge(prediction_frame, on="row_id", how="left")
                    submission[SPECIES_COLS] = submission[SPECIES_COLS].fillna(0.0)
                else:
                    submission = sample_sub.copy()
                    submission[SPECIES_COLS] = 0.0

                submission = submission[["row_id"] + SPECIES_COLS]
                submission.to_csv("submission.csv", index=False)
                print(f"Saved submission.csv with shape {submission.shape}")
                """
            ),
        ),
    ]


def build_notebook(weights_dataset: str, weights_filename: str) -> dict:
    cells = []
    for cell_type, source in _build_cells(weights_dataset, weights_filename):
        cell = {
            "cell_type": cell_type,
            "metadata": {"language": cell_type},
            "source": _cell_source(source),
        }
        if cell_type == "code":
            cell["outputs"] = []
            cell["execution_count"] = None
        cells.append(cell)

    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            }
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a Kaggle submission notebook for BirdCLEF+ 2026.")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("submission_pytorch.ipynb"),
        help="Notebook path to write.",
    )
    parser.add_argument(
        "--weights-dataset",
        default=DEFAULT_WEIGHTS_DATASET,
        help="Kaggle dataset name that contains the model weights.",
    )
    parser.add_argument(
        "--weights-filename",
        default=DEFAULT_WEIGHTS_FILENAME,
        help="Filename to load from the Kaggle dataset.",
    )
    args = parser.parse_args()

    notebook = build_notebook(args.weights_dataset, args.weights_filename)
    args.output.write_text(json.dumps(notebook, indent=2) + "\n", encoding="utf-8")

    print(f"Wrote {args.output}")
    print(f"Cells: {len(notebook['cells'])}")
    print(f"Weights dataset: /kaggle/input/{args.weights_dataset}/{args.weights_filename}")


if __name__ == "__main__":
    main()
