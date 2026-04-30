"""
Canonical project-wide configuration for BirdCLEF+ 2026 (Track B).

Import constants from this module to keep preprocessing, data loading,
and training consistently aligned.
"""
from __future__ import annotations

import os
from pathlib import Path

# ---------------------------------------------------------------------------
# Data paths
# ---------------------------------------------------------------------------

DATA_DIR: Path = Path(os.path.expanduser("~/birdclef-data"))
AUDIO_DIR: Path = DATA_DIR / "train_audio"
TRAIN_CSV: Path = DATA_DIR / "train.csv"
TAXONOMY_CSV: Path = DATA_DIR / "taxonomy.csv"
TEST_SOUNDSCAPES_DIR: Path = DATA_DIR / "test_soundscapes"
TRAIN_SOUNDSCAPES_DIR: Path = DATA_DIR / "train_soundscapes"
TRAIN_SOUNDSCAPES_LABELS_CSV: Path = DATA_DIR / "train_soundscapes_labels.csv"
SAMPLE_SUBMISSION_CSV: Path = DATA_DIR / "sample_submission.csv"

# ---------------------------------------------------------------------------
# Audio / mel-spectrogram parameters
# ---------------------------------------------------------------------------

SR: int = 32_000                        # native sample rate of BirdCLEF OGG files
CLIP_DURATION: float = 5.0              # seconds per training clip
CLIP_SAMPLES: int = int(SR * CLIP_DURATION)   # 160 000

N_MELS: int = 128
N_FFT: int = 1024
HOP_LENGTH: int = 512
F_MIN: float = 50.0
F_MAX: float = 14_000.0
TOP_DB: float = 80.0

# Derived: (CLIP_SAMPLES // HOP_LENGTH) + 1 = 313
TIME_FRAMES: int = (CLIP_SAMPLES // HOP_LENGTH) + 1

# ---------------------------------------------------------------------------
# Species count — derived from taxonomy.csv when available
# ---------------------------------------------------------------------------

def _count_species() -> int:
    """Count species rows in taxonomy.csv (header line is subtracted)."""
    if TAXONOMY_CSV.exists():
        try:
            with open(TAXONOMY_CSV, encoding="utf-8") as fh:
                return sum(1 for _ in fh) - 1  # subtract header
        except OSError:
            pass
    return 206   # BirdCLEF+ 2026 Track B fallback


NUM_SPECIES: int = _count_species()
