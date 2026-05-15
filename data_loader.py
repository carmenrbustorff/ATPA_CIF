"""
Memory-efficient PyTorch Dataset and DataLoader for BirdCLEF 2026 (Phase 1).

Reads train.csv, loads .ogg audio from the shared disk at
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
from collections import Counter
from functools import partial
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torchaudio
import torchaudio.transforms as T
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset, Subset

from config import SR, N_MELS, N_FFT, HOP_LENGTH, CLIP_DURATION


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
METADATA_CSV = DATA_ROOT / "train.csv"
AUDIO_DIR = DATA_ROOT / "train_audio"

SAMPLE_RATE = SR
CLIP_DURATION = CLIP_DURATION
CLIP_SAMPLES = int(SAMPLE_RATE * CLIP_DURATION)  # 160,000

N_MELS = N_MELS
N_FFT = N_FFT
HOP_LENGTH = HOP_LENGTH
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

    Each call to __getitem__ loads an OGG file, extracts a random
    5-second window, and returns a log-scaled, normalised mel-spectrogram
    tensor of shape (1, N_MELS, TIME_FRAMES) together with an integer
    class label.

    Parameters
    ----------
    metadata_csv:
        Path to ``train.csv``.
    audio_dir:
        Root directory that contains per-species sub-directories of OGG files.
    sample_rate:
        Target sample rate in Hz (default 32 000).
    clip_samples:
        Number of waveform samples per clip (default 160 000 = 5 s x 32 kHz).
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
        If True, apply random time-shift augmentation (training only).
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

        self._samples: list[Tuple[Path, int]] = []
        for _, row in df.iterrows():
            full_path = self.audio_dir / row["filename"]
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

        # SpecAugment transforms — only applied when augment=True
        self._time_masking = T.TimeMasking(time_mask_param=40)
        self._freq_masking = T.FrequencyMasking(freq_mask_param=15)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load_waveform(self, path: Path) -> torch.Tensor:
        """
        Load an OGG file and return a mono waveform of exactly
        self.clip_samples frames.

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

        # Random Gaussian noise (training only)
        if self.augment:
            waveform = waveform + torch.randn_like(waveform) * 0.005

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

        # SpecAugment — time and frequency masking (training only)
        if self.augment:
            log_mel = self._time_masking(log_mel)
            log_mel = self._freq_masking(log_mel)

        return log_mel.float()

    # ------------------------------------------------------------------
    # Dataset interface
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        try:
            # Your existing logic
            path, label = self._samples[idx]
            waveform = self._load_waveform(path)
            spectrogram = self._waveform_to_melspec(waveform)
            
            return spectrogram, label
            
        except Exception as e:
            # Catch the corrupted file error
            logger.warning("Skipping corrupted file at index %d: %s", idx, e)

            # Pick a random new index and recursively try again
            new_idx = random.randint(0, len(self._samples) - 1)
            return self.__getitem__(new_idx)

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
        Path to ``train.csv``.
    audio_dir:
        Root directory of the OGG training audio files.
    batch_size:
        Samples per batch. Adjust downward if GPU OOM occurs.
    shuffle:
        Shuffle the dataset each epoch (set False for validation).
    num_workers:
        Sub-processes for data loading. Defaults to min(3, cpu_count-1).
    pin_memory:
        Enables faster CPU->GPU transfers via pinned memory.
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
# Mixup collate function
# ---------------------------------------------------------------------------

def mixup_collate_fn(
    batch: list,
    alpha: float = 0.4,
    num_classes: int = 206,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Mixup augmentation applied at the batch level.

    Each spectrogram is linearly interpolated with a randomly chosen partner
    from the same batch. Labels are mixed with the same lambda so the model
    learns a soft target that reflects both species.

    Parameters
    ----------
    batch:
        List of (spectrogram, label) tuples from __getitem__.
    alpha:
        Beta distribution parameter controlling mix strength.
    num_classes:
        Total number of species — passed in from dataset.num_classes.

    Returns
    -------
    (mixed_specs, mixed_labels) : Tuple[torch.Tensor, torch.Tensor]
        mixed_specs  : (batch_size, 1, n_mels, time_frames)
        mixed_labels : (batch_size, num_classes) soft one-hot float targets
    """
    specs, labels = zip(*batch)
    specs  = torch.stack(specs)
    labels = torch.tensor(labels)

    one_hot = torch.zeros(len(labels), num_classes)
    one_hot.scatter_(1, labels.unsqueeze(1), 1.0)

    lam = float(np.random.beta(alpha, alpha))
    idx = torch.randperm(len(specs))

    mixed_specs  = lam * specs   + (1 - lam) * specs[idx]
    mixed_labels = lam * one_hot + (1 - lam) * one_hot[idx]

    return mixed_specs, mixed_labels


# ---------------------------------------------------------------------------
# Stratified train / val split factory
# ---------------------------------------------------------------------------

def build_train_val_dataloaders(
    metadata_csv: Path = METADATA_CSV,
    audio_dir: Path = AUDIO_DIR,
    val_split: float = 0.2,
    batch_size: int = 32,
    num_workers: int = NUM_WORKERS,
    pin_memory: bool = True,
    augment: bool = False,
    prefetch_factor: Optional[int] = 2,
    random_state: int = 42,
    use_cache: bool = False,
    cache_dir: Path = Path("/tmp/birdclef-specs"),
) -> Tuple[DataLoader, DataLoader]:
    """
    Build stratified 80/20 train and validation DataLoaders.

    Stratification is on primary_label. Species with fewer than 2 samples
    are dropped with a warning. The train loader uses random-crop augmentation
    if augment=True; the val loader always uses a deterministic centre crop.

    Parameters
    ----------
    metadata_csv:
        Path to train.csv.
    audio_dir:
        Root directory of the OGG training audio files.
    val_split:
        Fraction of the dataset reserved for validation (default 0.20).
    batch_size:
        Samples per batch for both loaders.
    num_workers:
        Sub-processes for data loading.
    pin_memory:
        Enables faster CPU->GPU transfers via pinned memory.
    augment:
        If True, the train loader uses random window extraction plus
        SpecAugment and mixup. The val loader always uses a deterministic centre crop.
    prefetch_factor:
        Batches to pre-load per worker. None disables prefetching.
    random_state:
        Seed passed to train_test_split for reproducibility.
    use_cache:
        If True, load precomputed spectrograms from cache_dir instead of
        processing OGG files on the fly. Requires cache_spectrograms.py
        to have been run first. Note: augment and mixup are disabled when
        using the cache since BirdCLEFCachedDataset does not support them.
    cache_dir:
        Directory containing cached .npz spectrogram files.

    Returns
    -------
    (train_loader, val_loader) : Tuple[DataLoader, DataLoader]
    """
    if not 0.0 < val_split < 1.0:
        raise ValueError(f"val_split must be in (0, 1), got {val_split}")

    if use_cache:
        from data_loader_cached import BirdCLEFCachedDataset
        logger.info("Using cached spectrograms from %s", cache_dir)
        if augment:
            logger.warning(
                "augment=True is ignored when use_cache=True — "
                "BirdCLEFCachedDataset does not support augmentation."
            )
        # Cached dataset: single object, no augment support
        train_dataset = BirdCLEFCachedDataset(metadata_csv=metadata_csv, cache_dir=cache_dir)
        val_dataset   = BirdCLEFCachedDataset(metadata_csv=metadata_csv, cache_dir=cache_dir)
        all_labels    = [lbl for _, lbl in train_dataset._samples]
    else:
        # Two separate dataset objects so augment never leaks into val
        train_dataset = BirdCLEFDataset(metadata_csv=metadata_csv, audio_dir=audio_dir, augment=augment)
        val_dataset   = BirdCLEFDataset(metadata_csv=metadata_csv, audio_dir=audio_dir, augment=False)
        all_labels    = [lbl for _, lbl in train_dataset._samples]
    label_counts = Counter(all_labels)
    eligible_idx = [i for i, (_, lbl) in enumerate(train_dataset._samples)
                    if label_counts[lbl] >= 2]
    eligible_labels = [all_labels[i] for i in eligible_idx]

    dropped = len(train_dataset) - len(eligible_idx)
    if dropped:
        logger.warning("Dropped %d samples from singleton classes.", dropped)

    train_local, val_local = train_test_split(
        range(len(eligible_idx)),
        test_size=val_split,
        stratify=eligible_labels,
        random_state=random_state,
    )
    train_idx = [eligible_idx[i] for i in train_local]
    val_idx   = [eligible_idx[i] for i in val_local]

    logger.info("Stratified split — train: %d  val: %d  dropped: %d",
                len(train_idx), len(val_idx), dropped)

    loader_kwargs = dict(
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=(num_workers > 0),
        prefetch_factor=prefetch_factor if num_workers > 0 else None,
        drop_last=False,
    )
    train_loader = DataLoader(
        Subset(train_dataset, train_idx),
        shuffle=True,
        collate_fn=partial(mixup_collate_fn, num_classes=train_dataset.num_classes) if (augment and not use_cache) else None,
        **loader_kwargs,
    )
    val_loader = DataLoader(Subset(val_dataset, val_idx), shuffle=False, **loader_kwargs)

    logger.info("Train: %d batches  |  Val: %d batches  (batch_size=%d)",
                len(train_loader), len(val_loader), batch_size)
    return train_loader, val_loader


# ---------------------------------------------------------------------------
# Validation entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    metadata_csv = Path(os.environ.get("BIRDCLEF_METADATA", str(METADATA_CSV)))
    audio_dir = Path(os.environ.get("BIRDCLEF_AUDIO_DIR", str(AUDIO_DIR)))

    if not metadata_csv.exists():
        logger.error("Metadata CSV not found: %s", metadata_csv)
        sys.exit(1)

    BATCH_SIZE = 16

    # Smoke-test build_dataloader
    logger.info("Initialising DataLoader (batch_size=%d)...", BATCH_SIZE)
    loader = build_dataloader(
        metadata_csv=metadata_csv,
        audio_dir=audio_dir,
        batch_size=BATCH_SIZE,
        shuffle=True,
        augment=True,
    )
    spectrograms, labels = next(iter(loader))
    print(f"\n{'='*60}")
    print(f"  Batch tensor shape : {tuple(spectrograms.shape)}")
    print(f"  Expected           : ({BATCH_SIZE}, 1, {N_MELS}, {TIME_FRAMES})")
    print(f"  dtype              : {spectrograms.dtype}")
    print(f"  Labels shape       : {tuple(labels.shape)}")
    print(f"  Label range        : [{labels.min().item()}, {labels.max().item()}]")
    bytes_per_batch = spectrograms.element_size() * spectrograms.nelement()
    print(f"  Batch memory (CPU) : {bytes_per_batch / 1024**2:.2f} MB")
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
    logger.info("DataLoader smoke-test passed.")

    # Smoke-test build_train_val_dataloaders
    logger.info("Testing build_train_val_dataloaders ...")
    train_loader, val_loader = build_train_val_dataloaders(
        metadata_csv=metadata_csv,
        audio_dir=audio_dir,
        val_split=0.2,
        batch_size=BATCH_SIZE,
        augment=True,
    )
    print(f"\n{'='*60}")
    print(f"  Stratified split smoke-test")
    print(f"  Train batches : {len(train_loader)}")
    print(f"  Val   batches : {len(val_loader)}")
    tr_specs, tr_labels = next(iter(train_loader))
    va_specs, va_labels = next(iter(val_loader))
    print(f"  Train batch   : specs={tuple(tr_specs.shape)}  labels={tuple(tr_labels.shape)}")
    print(f"  Val   batch   : specs={tuple(va_specs.shape)}  labels={tuple(va_labels.shape)}")
    overlap = set(train_loader.dataset.indices) & set(val_loader.dataset.indices)
    print(f"  Index overlap : {len(overlap)} (must be 0)")
    assert len(overlap) == 0, "BUG: train/val index lists overlap!"
    print(f"{'='*60}\n")
    logger.info("Stratified split smoke-test passed.")
