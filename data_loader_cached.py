"""
Fast cached dataset using precomputed spectrograms from disk.

Usage:
    # First, precompute spectrograms:
    python cache_spectrograms.py --output-dir /tmp/birdclef-specs --num-workers 4
    
    # Then use in training:
    from data_loader_cached import BirdCLEFCachedDataset
    dataset = BirdCLEFCachedDataset(
        metadata_csv=METADATA_CSV,
        cache_dir="/tmp/birdclef-specs"
    )
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from config import NUM_SPECIES

logging.basicConfig(
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    level=logging.INFO,
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)


class BirdCLEFCachedDataset(Dataset):
    """
    Fast dataset using precomputed cached spectrograms.
    
    Each sample is loaded from a .npz file on disk (~0.01s vs 0.4s from OGG).
    """
    
    def __init__(self, metadata_csv: Path, cache_dir: Path) -> None:
        self.cache_dir = Path(cache_dir)
        
        # Load metadata
        df = pd.read_csv(metadata_csv)
        
        # Build label encoder
        species = sorted(df["primary_label"].unique().tolist())
        self.label_to_idx: dict[str, int] = {sp: i for i, sp in enumerate(species)}
        self.idx_to_label: list[str] = species
        self.num_classes: int = len(species)
        
        # Build sample list with cache paths
        self._samples: list[Tuple[Path, int]] = []
        missing_count = 0
        for _, row in df.iterrows():
            safe_name = str(row["filename"]).replace("/", "_").replace(".", "_")
            cache_path = self.cache_dir / f"{safe_name}.npz"
            
            if cache_path.exists():
                label_idx = self.label_to_idx[row["primary_label"]]
                self._samples.append((cache_path, label_idx))
            else:
                missing_count += 1
        
        if missing_count > 0:
            logger.warning(
                "Missing %d cached spectrograms. Run cache_spectrograms.py first.",
                missing_count
            )
        
        logger.info(
            "Cached dataset initialised: %d samples, %d species",
            len(self._samples),
            self.num_classes,
        )
    
    def __len__(self) -> int:
        return len(self._samples)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        cache_path, label = self._samples[idx]
        
        try:
            data = np.load(cache_path)
            spec = data["spectrogram"]  # shape: (1, n_mels, time_frames)
            spec_tensor = torch.from_numpy(spec).float()
            return spec_tensor, label
        except Exception as e:
            logger.error("Failed to load %s: %s", cache_path, e)
            # Return silence as fallback
            return torch.zeros(1, 128, 313), label
