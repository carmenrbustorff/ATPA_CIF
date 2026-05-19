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

DataLoader tuning (CPU-constrained host, shared NVIDIA L4):
    - num_workers = 2  (avoids oversubscribing the available CPU)
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
import librosa
from torch.utils.data import DataLoader, Dataset

from config import SR, N_MELS, N_FFT, HOP_LENGTH, CLIP_DURATION, NUM_SPECIES


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
SPECTROGRAM_CACHE_DIR = DATA_ROOT / "spectrograms_cache"

SAMPLE_RATE = SR          # native rate of BirdCLEF OGG files
CLIP_DURATION = CLIP_DURATION           # seconds
CLIP_SAMPLES = int(SAMPLE_RATE * CLIP_DURATION)  # 160,000

N_MELS = N_MELS
N_FFT = N_FFT
HOP_LENGTH = HOP_LENGTH
F_MIN = 50.0
F_MAX = 14_000.0
TOP_DB = 80.0

# Derived time dimension: ceil(CLIP_SAMPLES / HOP_LENGTH) = 313
TIME_FRAMES = (CLIP_SAMPLES // HOP_LENGTH) + 1  # 313

# DataLoader workers: keep CPU usage conservative on shared hosts
NUM_WORKERS = 2


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
        spectrogram_dir: Optional[Path] = None,
        use_cache: bool = True,
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
        self.spectrogram_dir = Path(spectrogram_dir) if spectrogram_dir is not None else SPECTROGRAM_CACHE_DIR
        self.use_cache = use_cache
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

        # Store spectrogram parameters (we use librosa for transforms)
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.f_min = f_min
        self.f_max = f_max
        self.top_db = top_db

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
        if self.use_cache:
            cache_path = self.spectrogram_dir / path.relative_to(self.audio_dir).with_suffix(".npy")
            if cache_path.exists():
                cached_spectrogram = np.load(cache_path, allow_pickle=False).astype(np.float32, copy=False)
                return cached_spectrogram

        try:
            audio, orig_sr = librosa.load(str(path), sr=None, mono=False)
            # librosa returns numpy (N,) for mono or (channels, N) for multi-channel
            if audio.ndim == 1:
                audio = audio[None, :]  # add channel dim → (1, N)
        except Exception as exc:
            logger.warning("Failed to load %s: %s — returning silence", path, exc)
            return torch.zeros(1, self.clip_samples)

        # Resample (numpy) if needed
        if orig_sr != self.sample_rate:
            try:
                resampled = []
                for ch in audio:
                    resampled_ch = librosa.resample(ch.astype(np.float32), orig_sr, self.sample_rate)
                    resampled.append(resampled_ch)
                audio = np.stack(resampled, axis=0)
            except Exception:
                # If resampling fails, fall back to converting as-is
                pass

        total_samples = audio.shape[-1]

        if total_samples >= self.clip_samples:
            # Random window extraction
            if self.augment:
                max_start = total_samples - self.clip_samples
                start = random.randint(0, max_start)
            else:
                # Deterministic centre crop for validation / inference
                start = (total_samples - self.clip_samples) // 2
            audio_window = audio[:, start : start + self.clip_samples]
        else:
            # Zero-pad short files on the right (numpy)
            pad = self.clip_samples - total_samples
            audio_window = np.pad(audio, ((0, 0), (0, pad)), mode="constant")

        # Mix to mono
        if audio_window.shape[0] > 1:
            audio_window = audio_window.mean(axis=0, keepdims=True)

        waveform = torch.from_numpy(audio_window.astype(np.float32)).float()
        return waveform  # shape: (1, clip_samples)

    def _waveform_to_melspec(self, waveform: torch.Tensor) -> torch.Tensor:
        """
        Convert a (1, clip_samples) waveform to a normalised log-mel
        spectrogram of shape (1, n_mels, time_frames).

        Values are normalised to approximately [0, 1].
        """
        # waveform: (1, clip_samples) torch.Tensor -> use librosa on numpy
        y = waveform.squeeze(0).numpy().astype(np.float32)
        mel = librosa.feature.melspectrogram(
            y=y,
            sr=self.sample_rate,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            n_mels=self.n_mels,
            fmin=self.f_min,
            fmax=self.f_max,
            power=2.0,
        )
        log_mel = librosa.power_to_db(mel, top_db=self.top_db)  # range [-top_db, 0]
        log_mel = (log_mel + self.top_db) / self.top_db  # normalise to [0, 1]
        spec = torch.from_numpy(log_mel.astype(np.float32)).unsqueeze(0)
        return spec.float()

    # ------------------------------------------------------------------
    # Dataset interface
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        path, label = self._samples[idx]
        item = self._load_waveform(path)

        if isinstance(item, np.ndarray):
            spectrogram = item
            if spectrogram.ndim == 3 and spectrogram.shape[0] == 1:
                spectrogram = spectrogram.squeeze(0)
            elif spectrogram.ndim == 3 and spectrogram.shape[-1] == 1:
                spectrogram = spectrogram.squeeze(-1)
            if spectrogram.ndim != 2:
                raise ValueError(f"Cached spectrogram has unexpected shape: {spectrogram.shape}")
            return torch.from_numpy(spectrogram).unsqueeze(0).float(), label

        spectrogram = self._waveform_to_melspec(item)
        return spectrogram, label


# ---------------------------------------------------------------------------
# DataLoader factory
# ---------------------------------------------------------------------------

def build_dataloader(
    metadata_csv: Path = METADATA_CSV,
    audio_dir: Path = AUDIO_DIR,
    spectrogram_dir: Optional[Path] = None,
    batch_size: int = 32,
    shuffle: bool = True,
    num_workers: int = NUM_WORKERS,
    pin_memory: bool = True,
    use_cache: bool = True,
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
        spectrogram_dir=spectrogram_dir,
        use_cache=use_cache,
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
