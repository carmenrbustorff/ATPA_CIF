"""
Precompute and cache mel-spectrograms to disk for fast training.

Usage:
    python cache_spectrograms.py --output-dir /tmp/birdclef-specs --num-workers 4

This generates .npz files on disk that can be loaded ~100x faster than
recomputing spectrograms from OGG files during training.
"""

import argparse
import sys
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import pandas as pd
import torch
import librosa
from tqdm import tqdm

from config import SR, N_MELS, N_FFT, HOP_LENGTH, CLIP_DURATION
from data_loader import METADATA_CSV, AUDIO_DIR, F_MIN, F_MAX, TOP_DB, SAMPLE_RATE, CLIP_SAMPLES


def compute_spectrogram(audio_path, sample_rate=SAMPLE_RATE, clip_samples=CLIP_SAMPLES,
                       n_mels=N_MELS, n_fft=N_FFT, hop_length=HOP_LENGTH,
                       f_min=F_MIN, f_max=F_MAX, top_db=TOP_DB):
    """Load OGG and compute normalized mel-spectrogram."""
    try:
        # Load audio
        audio, orig_sr = librosa.load(str(audio_path), sr=None, mono=False)
        if audio.ndim == 1:
            audio = audio[None, :]
        
        # Resample if needed
        if orig_sr != sample_rate:
            resampled = []
            for ch in audio:
                resampled_ch = librosa.resample(ch.astype(np.float32), orig_sr, sample_rate)
                resampled.append(resampled_ch)
            audio = np.stack(resampled, axis=0)
        
        # Extract center crop
        total_samples = audio.shape[-1]
        if total_samples >= clip_samples:
            start = (total_samples - clip_samples) // 2
            audio_window = audio[:, start : start + clip_samples]
        else:
            pad = clip_samples - total_samples
            audio_window = np.pad(audio, ((0, 0), (0, pad)), mode="constant")
        
        # Mix to mono
        if audio_window.shape[0] > 1:
            audio_window = audio_window.mean(axis=0, keepdims=True)
        
        # Compute mel-spectrogram
        y = audio_window.squeeze(0).astype(np.float32)
        mel = librosa.feature.melspectrogram(
            y=y, sr=sample_rate, n_fft=n_fft, hop_length=hop_length,
            n_mels=n_mels, fmin=f_min, fmax=f_max, power=2.0,
        )
        log_mel = librosa.power_to_db(mel, top_db=top_db)  # range [-top_db, 0]
        log_mel = (log_mel + top_db) / top_db  # normalise to [0, 1]
        spec = log_mel.astype(np.float32)[None, :, :]  # add channel dim
        
        return spec
    except Exception as e:
        print(f"  ERROR loading {audio_path}: {e}", file=sys.stderr)
        return None


def cache_spectrograms(metadata_csv, audio_dir, output_dir, num_workers=4):
    """Precompute and cache all spectrograms."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    df = pd.read_csv(metadata_csv)
    print(f"Caching {len(df)} spectrograms to {output_dir}...")
    
    def process_row(idx, row):
        audio_path = Path(audio_dir) / row["filename"]
        spec = compute_spectrogram(audio_path)
        if spec is not None:
            # Use a safe filename (replace path separators)
            safe_name = str(row["filename"]).replace("/", "_").replace(".", "_")
            output_path = output_dir / f"{safe_name}.npz"
            np.savez_compressed(output_path, spectrogram=spec)
            return (idx, True)
        return (idx, False)
    
    successful = 0
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = [
            executor.submit(process_row, idx, row)
            for idx, row in df.iterrows()
        ]
        for future in tqdm(futures, total=len(df), desc="Caching"):
            try:
                idx, success = future.result(timeout=60)
                if success:
                    successful += 1
            except Exception as e:
                print(f"  Failed: {e}", file=sys.stderr)
    
    print(f"Cached {successful}/{len(df)} spectrograms successfully")
    return successful


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="/tmp/birdclef-specs",
                       help="Directory to cache spectrograms (default: /tmp/birdclef-specs)")
    parser.add_argument("--num-workers", type=int, default=4,
                       help="Number of worker threads (default: 4)")
    args = parser.parse_args()
    
    cache_spectrograms(METADATA_CSV, AUDIO_DIR, args.output_dir, args.num_workers)
