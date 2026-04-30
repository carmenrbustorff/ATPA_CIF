"""
Memory-efficient PyTorch Dataset and DataLoader for BirdCLEF+ 2026 (Track B).

Reads train.csv, loads .ogg audio from ~/birdclef-data/train_audio/, and
extracts random 5-second mel-spectrogram windows on the fly during training.

Audio spec (from config.py):
  - Sample rate     : 32 kHz
  - Clip length     : 5 s  → 160 000 samples
  - Mel bins        : 128
  - FFT window      : 1024 samples
  - Hop length      : 512 samples
  - Time frames     : (160 000 // 512) + 1 = 313
  - Output shape    : (1, 128, 313)  [channel, n_mels, time]

Augmentations (augment=True):
  - Random 5-second window extraction
  - SpecAugment: TimeMasking(param=40), FrequencyMasking(param=15)
  - Additive Gaussian noise (std=0.005)

DataLoader tuning (4-vCPU, shared NVIDIA L4):
  - num_workers = 3  (leaves 1 CPU for the main process / training loop)
  - pin_memory  = True  (zero-copy transfer to GPU)
  - persistent_workers = True  (avoids worker re-spawn overhead per epoch)
"""

from __future__ import annotations

import ast
import logging
import os
import random
import sys
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torchaudio
import torchaudio.transforms as T
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset, Subset

from config import (
    AUDIO_DIR,
    CLIP_DURATION,
    CLIP_SAMPLES,
    F_MAX,
    F_MIN,
    HOP_LENGTH,
    N_FFT,
    N_MELS,
    NUM_SPECIES,
    SR,
    TOP_DB,
    TRAIN_CSV,
)

# ---------------------------------------------------------------------------
# Module-level logger
# ---------------------------------------------------------------------------
logging.basicConfig(
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    level=logging.INFO,
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)

# DataLoader workers: leave 1 CPU free for the training loop
_MAX_WORKERS = 3
NUM_WORKERS = min(_MAX_WORKERS, max(1, (os.cpu_count() or 1) - 1))


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class BirdCLEFDataset(Dataset):
    """
    PyTorch Dataset for BirdCLEF 2026 training audio.

    Each ``__getitem__`` call loads an OGG file, extracts a 5-second window,
    and returns a log-scaled, normalised mel-spectrogram tensor of shape
    ``(1, N_MELS, TIME_FRAMES)`` together with a multi-hot float label vector
    of shape ``(num_classes,)``.

    Parameters
    ----------
    train_csv:
        Path to ``train.csv``.
    audio_dir:
        Root directory containing per-species sub-directories of OGG files.
    sample_rate:
        Target sample rate in Hz (default from config).
    clip_samples:
        Number of waveform samples per clip (default from config).
    n_mels, n_fft, hop_length, f_min, f_max, top_db:
        Mel-spectrogram parameters (defaults from config).
    augment:
        If True, apply random window extraction and SpecAugment.
    species_list:
        Optional fixed list of species codes (sorted). When provided the
        label dimension is ``len(species_list)``; species not in the list
        are ignored. Defaults to all unique primary labels in the CSV.
    """

    def __init__(
        self,
        train_csv: Path = TRAIN_CSV,
        audio_dir: Path = AUDIO_DIR,
        sample_rate: int = SR,
        clip_samples: int = CLIP_SAMPLES,
        n_mels: int = N_MELS,
        n_fft: int = N_FFT,
        hop_length: int = HOP_LENGTH,
        f_min: float = F_MIN,
        f_max: float = F_MAX,
        top_db: float = TOP_DB,
        augment: bool = False,
        species_list: Optional[List[str]] = None,
    ) -> None:
        super().__init__()

        self.audio_dir = Path(audio_dir)
        self.sample_rate = sample_rate
        self.clip_samples = clip_samples
        self.top_db = top_db
        self.augment = augment

        # ------------------------------------------------------------------
        # Load metadata
        # ------------------------------------------------------------------
        df = pd.read_csv(train_csv)

        # Build sorted label encoder
        if species_list is not None:
            self.idx_to_label: List[str] = sorted(species_list)
        else:
            self.idx_to_label = sorted(df["primary_label"].unique().tolist())
        self.label_to_idx: dict = {sp: i for i, sp in enumerate(self.idx_to_label)}
        self.num_classes: int = len(self.idx_to_label)

        # Build flat sample list: (filepath, multi_hot_label_tensor)
        self._samples: List[Tuple[Path, torch.Tensor]] = []
        has_secondary = "secondary_labels" in df.columns

        for _, row in df.iterrows():
            full_path = self.audio_dir / row["filename"]
            label_vec = torch.zeros(self.num_classes, dtype=torch.float32)

            primary = row["primary_label"]
            if primary in self.label_to_idx:
                label_vec[self.label_to_idx[primary]] = 1.0

            if has_secondary:
                for sp in self._parse_secondary_labels(row["secondary_labels"]):
                    if sp in self.label_to_idx:
                        label_vec[self.label_to_idx[sp]] = 1.0

            self._samples.append((full_path, label_vec))

        logger.info(
            "Dataset initialised: %d samples, %d classes",
            len(self._samples),
            self.num_classes,
        )

        # ------------------------------------------------------------------
        # Mel-spectrogram transform (stateless; shared across workers)
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

        # SpecAugment transforms (only used when augment=True)
        self._time_masking = T.TimeMasking(time_mask_param=40)
        self._freq_masking = T.FrequencyMasking(freq_mask_param=15)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_secondary_labels(value) -> List[str]:
        """Parse a secondary_labels cell into a list of species strings."""
        if pd.isna(value):
            return []
        s = str(value).strip()
        if not s or s in ("[]", "''", '""'):
            return []
        try:
            parsed = ast.literal_eval(s)
            if isinstance(parsed, (list, tuple)):
                return [str(x) for x in parsed]
        except (ValueError, SyntaxError):
            pass
        # Fallback: treat as whitespace/comma-separated string
        return [x.strip() for x in s.replace(",", " ").split() if x.strip()]

    def _load_waveform(self, path: Path) -> torch.Tensor:
        """
        Load an OGG file and return a mono waveform of exactly
        ``self.clip_samples`` frames.
        """
        try:
            waveform, orig_sr = torchaudio.load(str(path))
        except Exception as exc:
            logger.warning("Failed to load %s: %s — returning silence", path, exc)
            return torch.zeros(1, self.clip_samples)

        # Mix to mono
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        # Resample if needed
        if orig_sr != self.sample_rate:
            waveform = T.Resample(orig_freq=orig_sr, new_freq=self.sample_rate)(waveform)

        total_samples = waveform.shape[-1]

        if total_samples >= self.clip_samples:
            if self.augment:
                max_start = total_samples - self.clip_samples
                start = random.randint(0, max_start)
            else:
                start = (total_samples - self.clip_samples) // 2
            waveform = waveform[:, start : start + self.clip_samples]
        else:
            pad = self.clip_samples - total_samples
            waveform = torch.nn.functional.pad(waveform, (0, pad))

        return waveform  # shape: (1, clip_samples)

    def _waveform_to_melspec(self, waveform: torch.Tensor) -> torch.Tensor:
        """
        Convert a (1, clip_samples) waveform to a normalised log-mel
        spectrogram of shape (1, n_mels, time_frames).
        """
        mel = self._mel_transform(waveform)
        log_mel = self._amplitude_to_db(mel)
        log_mel = (log_mel + self.top_db) / self.top_db   # normalise to [0, 1]
        return log_mel.float()

    def _apply_augmentation(self, spec: torch.Tensor) -> torch.Tensor:
        """Apply SpecAugment and Gaussian noise to a spectrogram tensor."""
        spec = self._time_masking(spec)
        spec = self._freq_masking(spec)
        spec = spec + 0.005 * torch.randn_like(spec)
        return torch.clamp(spec, 0.0, 1.0)

    # ------------------------------------------------------------------
    # Dataset interface
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        path, label_vec = self._samples[idx]
        waveform = self._load_waveform(path)
        spec = self._waveform_to_melspec(waveform)
        if self.augment:
            spec = self._apply_augmentation(spec)
        return spec, label_vec


# ---------------------------------------------------------------------------
# DataLoader factories
# ---------------------------------------------------------------------------

def build_dataloader(
    train_csv: Path = TRAIN_CSV,
    audio_dir: Path = AUDIO_DIR,
    batch_size: int = 32,
    shuffle: bool = True,
    num_workers: int = NUM_WORKERS,
    pin_memory: bool = True,
    augment: bool = False,
    prefetch_factor: Optional[int] = 2,
    species_list: Optional[List[str]] = None,
) -> DataLoader:
    """
    Build a DataLoader over the full training set.

    Returns
    -------
    torch.utils.data.DataLoader
    """
    dataset = BirdCLEFDataset(
        train_csv=train_csv,
        audio_dir=audio_dir,
        augment=augment,
        species_list=species_list,
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
        "DataLoader ready: batch_size=%d, num_workers=%d, pin_memory=%s, augment=%s",
        batch_size, num_workers, pin_memory, augment,
    )
    return loader


def build_train_val_dataloaders(
    train_csv: Path = TRAIN_CSV,
    audio_dir: Path = AUDIO_DIR,
    batch_size: int = 32,
    val_batch_size: int = 32,
    num_workers: int = NUM_WORKERS,
    pin_memory: bool = True,
    val_split: float = 0.2,
    random_state: int = 42,
    prefetch_factor: Optional[int] = 2,
) -> Tuple[DataLoader, DataLoader, List[str]]:
    """
    Build stratified 80/20 train and validation DataLoaders.

    Stratification is by primary label so rare species appear in both splits.

    Returns
    -------
    (train_loader, val_loader, species_list)
        ``species_list`` is the sorted list of species codes used for label
        encoding — pass ``len(species_list)`` to the model as ``num_classes``.
    """
    df = pd.read_csv(train_csv)
    species_list: List[str] = sorted(df["primary_label"].unique().tolist())

    indices = list(range(len(df)))
    train_idx, val_idx = train_test_split(
        indices,
        test_size=val_split,
        stratify=df["primary_label"].values,
        random_state=random_state,
    )

    train_dataset = BirdCLEFDataset(
        train_csv=train_csv,
        audio_dir=audio_dir,
        augment=True,
        species_list=species_list,
    )
    val_dataset = BirdCLEFDataset(
        train_csv=train_csv,
        audio_dir=audio_dir,
        augment=False,
        species_list=species_list,
    )

    train_subset = Subset(train_dataset, train_idx)
    val_subset = Subset(val_dataset, val_idx)

    train_loader = DataLoader(
        train_subset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=(num_workers > 0),
        prefetch_factor=prefetch_factor if num_workers > 0 else None,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_subset,
        batch_size=val_batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=(num_workers > 0),
        prefetch_factor=prefetch_factor if num_workers > 0 else None,
        drop_last=False,
    )

    logger.info(
        "Train/Val split: %d / %d samples (%d classes), batch_size=%d",
        len(train_subset), len(val_subset), len(species_list), batch_size,
    )
    return train_loader, val_loader, species_list


# ---------------------------------------------------------------------------
# Validation entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from config import N_MELS, TIME_FRAMES

    _train_csv = Path(os.environ.get("BIRDCLEF_TRAIN_CSV", str(TRAIN_CSV)))
    _audio_dir = Path(os.environ.get("BIRDCLEF_AUDIO_DIR", str(AUDIO_DIR)))

    if not _train_csv.exists():
        logger.error("train.csv not found: %s", _train_csv)
        sys.exit(1)

    BATCH_SIZE = 16
    logger.info("Building train/val DataLoaders (batch_size=%d)…", BATCH_SIZE)
    train_loader, val_loader, species = build_train_val_dataloaders(
        train_csv=_train_csv,
        audio_dir=_audio_dir,
        batch_size=BATCH_SIZE,
    )

    logger.info("Fetching first training batch…")
    specs, labels = next(iter(train_loader))

    print(f"\n{'='*60}")
    print(f"  Batch tensor shape : {tuple(specs.shape)}")
    print(f"  Expected           : ({BATCH_SIZE}, 1, {N_MELS}, {TIME_FRAMES})")
    print(f"  dtype              : {specs.dtype}")
    print(f"  Labels shape       : {tuple(labels.shape)}")
    print(f"  Species classes    : {len(species)}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Target device      : {device}")
    print(f"{'='*60}\n")
    logger.info("Validation complete — DataLoader is working correctly.")
