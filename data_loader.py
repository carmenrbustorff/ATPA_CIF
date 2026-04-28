"""
Memory-efficient PyTorch Dataset and DataLoader for BirdCLEF 2026 (Phase 1).

Reads train_metadata.csv, loads .ogg audio from the shared disk at
/mnt/disks/data/birdclef/train_audio/, and extracts random 5-second
mel-spectrogram windows on the fly during training.

Audio spec:
  - Native sample rate  : 32 kHz
  - Clip length         : 5 s  → 160,000 samples
  - Mel bins (n_mels)   : 128
  - FFT window          : 1024 samples
  - Hop length          : 512 samples
  - Time frames per clip: ⌈160 000 / 512⌉ = 313
  - Output tensor shape : (1, 128, 313)  [channel, n_mels, time]

DataLoader tuning (4-vCPU, shared NVIDIA L4):
  - num_workers = 3  (leaves 1 CPU for the main process / training loop)
  - pin_memory  = True  (zero-copy transfer to GPU)
  - persistent_workers = True  (avoids worker re-spawn overhead per epoch)
"""

from __future__ import annotations

import logging
import os
import random
import sys
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torchaudio
import torchaudio.transforms as T
from torch.utils.data import DataLoader, Dataset

# ---------------------------------------------------------------------------
# Module-level logger
# ---------------------------------------------------------------------------
logging.basicConfig(
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    level=logging.INFO,
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DATA_ROOT = Path("/mnt/disks/data/birdclef")
METADATA_CSV = DATA_ROOT / "train_metadata.csv"
AUDIO_DIR = DATA_ROOT / "train_audio"

SAMPLE_RATE = 32_000          # native rate of BirdCLEF OGG files
CLIP_DURATION = 5.0           # seconds
CLIP_SAMPLES = int(SAMPLE_RATE * CLIP_DURATION)  # 160,000

N_MELS = 128
N_FFT = 1024
HOP_LENGTH = 512
F_MIN = 50.0
F_MAX = 14_000.0
TOP_DB = 80.0

# Derived time dimension: ceil(CLIP_SAMPLES / HOP_LENGTH) = 313
TIME_FRAMES = (CLIP_SAMPLES // HOP_LENGTH) + 1  # 313

# DataLoader workers: leave 1 CPU free for the training loop
MAX_WORKERS = 3
NUM_WORKERS = min(MAX_WORKERS, max(1, os.cpu_count() - 1))  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class BirdCLEFDataset(Dataset):
    """
    PyTorch Dataset for BirdCLEF 2026 training audio.

    Each call to ``__getitem__`` loads an OGG file, extracts a random
    5-second window, and returns a log-scaled, normalised mel-spectrogram
    tensor of shape ``(1, N_MELS, TIME_FRAMES)`` together with an integer
    class label.

    Parameters
    ----------
    metadata_csv:
        Path to ``train_metadata.csv``.
    audio_dir:
        Root directory that contains per-species sub-directories of OGG files.
    sample_rate:
        Target sample rate in Hz (default 32 000).
    clip_samples:
        Number of waveform samples per clip (default 160 000 = 5 s × 32 kHz).
    n_mels:
        Number of mel filter-bank bins.
    n_fft:
        FFT window size in samples.
    hop_length:
        Hop length between STFT frames in samples.
    f_min / f_max:
        Frequency bounds of the mel filter bank.
    top_db:
        Dynamic range for AmplitudeToDB.
    augment:
        If True, apply basic time-shift augmentation (reserved for training).
    """

    def __init__(
        self,
        metadata_csv: Path = METADATA_CSV,
        audio_dir: Path = AUDIO_DIR,
        sample_rate: int = SAMPLE_RATE,
        clip_samples: int = CLIP_SAMPLES,
        n_mels: int = N_MELS,
        n_fft: int = N_FFT,
        hop_length: int = HOP_LENGTH,
        f_min: float = F_MIN,
        f_max: float = F_MAX,
        top_db: float = TOP_DB,
        augment: bool = False,
    ) -> None:
        super().__init__()

        self.audio_dir = Path(audio_dir)
        self.sample_rate = sample_rate
        self.clip_samples = clip_samples
        self.augment = augment

        # ------------------------------------------------------------------
        # Load metadata and build a flat list of (filepath, label_idx) pairs
        # ------------------------------------------------------------------
        df = pd.read_csv(metadata_csv)

        # Build sorted label encoder so indices are deterministic across runs
        species = sorted(df["primary_label"].unique().tolist())
        self.label_to_idx: dict[str, int] = {sp: i for i, sp in enumerate(species)}
        self.idx_to_label: list[str] = species
        self.num_classes: int = len(species)

        # Each row in the CSV has a "filename" column with a relative path
        # like "XC12345/XC12345.ogg" (relative to train_audio/).
        self._samples: list[Tuple[Path, int]] = []
        for _, row in df.iterrows():
            rel_path = row["filename"]
            full_path = self.audio_dir / rel_path
            label_idx = self.label_to_idx[row["primary_label"]]
            self._samples.append((full_path, label_idx))

        logger.info(
            "Dataset initialised: %d samples, %d species",
            len(self._samples),
            self.num_classes,
        )

        # ------------------------------------------------------------------
        # Mel-spectrogram transform (constructed once; shared across workers
        # because torchaudio transforms are stateless / thread-safe)
        # ------------------------------------------------------------------
        self._mel_transform = T.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels,
            f_min=f_min,
            f_max=f_max,
        )
        self._amplitude_to_db = T.AmplitudeToDB(top_db=top_db)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load_waveform(self, path: Path) -> torch.Tensor:
        """
        Load an OGG file and return a mono waveform of exactly
        ``self.clip_samples`` frames.

        A random ``clip_samples``-length window is extracted when the file
        is longer; the waveform is zero-padded when it is shorter.
        """
        try:
            waveform, orig_sr = torchaudio.load(str(path))
        except Exception as exc:
            logger.warning("Failed to load %s: %s — returning silence", path, exc)
            return torch.zeros(1, self.clip_samples)

        # Mix to mono
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        # Resample if the file is not already at the target rate
        if orig_sr != self.sample_rate:
            waveform = T.Resample(orig_freq=orig_sr, new_freq=self.sample_rate)(waveform)

        total_samples = waveform.shape[-1]

        if total_samples >= self.clip_samples:
            # Random window extraction
            if self.augment:
                max_start = total_samples - self.clip_samples
                start = random.randint(0, max_start)
            else:
                # Deterministic centre crop for validation / inference
                start = (total_samples - self.clip_samples) // 2
            waveform = waveform[:, start : start + self.clip_samples]
        else:
            # Zero-pad short files on the right
            pad = self.clip_samples - total_samples
            waveform = torch.nn.functional.pad(waveform, (0, pad))

        return waveform  # shape: (1, clip_samples)

    def _waveform_to_melspec(self, waveform: torch.Tensor) -> torch.Tensor:
        """
        Convert a (1, clip_samples) waveform to a normalised log-mel
        spectrogram of shape (1, n_mels, time_frames).

        Values are normalised to approximately [0, 1].
        """
        mel = self._mel_transform(waveform)          # (1, n_mels, time)
        log_mel = self._amplitude_to_db(mel)          # (1, n_mels, time), range [-top_db, 0]
        log_mel = (log_mel + TOP_DB) / TOP_DB         # normalise to [0, 1]
        return log_mel.float()

    # ------------------------------------------------------------------
    # Dataset interface
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        path, label = self._samples[idx]
        waveform = self._load_waveform(path)
        spectrogram = self._waveform_to_melspec(waveform)
        return spectrogram, label


# ---------------------------------------------------------------------------
# DataLoader factory
# ---------------------------------------------------------------------------

def build_dataloader(
    metadata_csv: Path = METADATA_CSV,
    audio_dir: Path = AUDIO_DIR,
    batch_size: int = 32,
    shuffle: bool = True,
    num_workers: int = NUM_WORKERS,
    pin_memory: bool = True,
    augment: bool = False,
    prefetch_factor: Optional[int] = 2,
) -> DataLoader:
    """
    Build a memory-efficient DataLoader for BirdCLEF training data.

    Parameters
    ----------
    metadata_csv:
        Path to ``train_metadata.csv``.
    audio_dir:
        Root directory of the OGG training audio files.
    batch_size:
        Samples per batch. Adjust downward if GPU OOM occurs.
    shuffle:
        Shuffle the dataset each epoch (set False for validation).
    num_workers:
        Sub-processes for data loading. Defaults to ``min(3, cpu_count-1)``.
    pin_memory:
        Enables faster CPU→GPU transfers via pinned memory. Set True when
        running on a CUDA-capable host.
    augment:
        Pass True during training to enable random window extraction.
    prefetch_factor:
        Batches to pre-load per worker. None disables prefetching.

    Returns
    -------
    torch.utils.data.DataLoader
    """
    dataset = BirdCLEFDataset(
        metadata_csv=metadata_csv,
        audio_dir=audio_dir,
        augment=augment,
    )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=(num_workers > 0),
        prefetch_factor=prefetch_factor if num_workers > 0 else None,
        drop_last=False,
    )

    logger.info(
        "DataLoader ready: batch_size=%d, num_workers=%d, pin_memory=%s",
        batch_size,
        num_workers,
        pin_memory,
    )
    return loader


# ---------------------------------------------------------------------------
# Validation entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Allow overriding paths via env vars for testing in alternate environments
    metadata_csv = Path(os.environ.get("BIRDCLEF_METADATA", str(METADATA_CSV)))
    audio_dir = Path(os.environ.get("BIRDCLEF_AUDIO_DIR", str(AUDIO_DIR)))

    if not metadata_csv.exists():
        logger.error("Metadata CSV not found: %s", metadata_csv)
        sys.exit(1)

    BATCH_SIZE = 16

    logger.info("Initialising DataLoader (batch_size=%d)...", BATCH_SIZE)
    loader = build_dataloader(
        metadata_csv=metadata_csv,
        audio_dir=audio_dir,
        batch_size=BATCH_SIZE,
        shuffle=True,
        augment=True,
    )

    logger.info("Fetching first batch...")
    spectrograms, labels = next(iter(loader))

    # Report tensor shape
    print(f"\n{'='*60}")
    print(f"  Batch tensor shape : {tuple(spectrograms.shape)}")
    print(f"  Expected           : ({BATCH_SIZE}, 1, {N_MELS}, {TIME_FRAMES})")
    print(f"  dtype              : {spectrograms.dtype}")
    print(f"  Labels shape       : {tuple(labels.shape)}")
    print(f"  Label range        : [{labels.min().item()}, {labels.max().item()}]")

    # Memory footprint
    bytes_per_batch = spectrograms.element_size() * spectrograms.nelement()
    print(f"  Batch memory (CPU) : {bytes_per_batch / 1024**2:.2f} MB")

    # GPU readiness check
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Target device      : {device}")
    if device.type == "cuda":
        spectrograms_gpu = spectrograms.to(device, non_blocking=True)
        torch.cuda.synchronize()
        gpu_bytes = spectrograms_gpu.element_size() * spectrograms_gpu.nelement()
        print(f"  Batch memory (GPU) : {gpu_bytes / 1024**2:.2f} MB")
        print(f"  GPU VRAM allocated : {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
        torch.cuda.empty_cache()
    print(f"{'='*60}\n")

    logger.info("Validation complete — DataLoader is working correctly.")
