"""Build a spectrogram cache for BirdCLEF training audio.

This script reuses BirdCLEFDataset's waveform-to-spectrogram path so the
cached .npy files match the training-time preprocessing exactly.
"""
from __future__ import annotations

import logging
import time
from functools import partial
from multiprocessing import Pool
from pathlib import Path
from typing import Iterable, Optional, Tuple

import numpy as np
from tqdm import tqdm

from data_loader import AUDIO_DIR, DATA_ROOT, BirdCLEFDataset, METADATA_CSV, SPECTROGRAM_CACHE_DIR

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

NUM_WORKERS = 2
FAILURES_LOG = Path("cache_failures.txt")
AUDIO_SUFFIXES = {".ogg"}

_worker_dataset: Optional[BirdCLEFDataset] = None


def _init_worker() -> None:
    global _worker_dataset
    _worker_dataset = BirdCLEFDataset(
        metadata_csv=METADATA_CSV,
        audio_dir=AUDIO_DIR,
        spectrogram_dir=SPECTROGRAM_CACHE_DIR,
        use_cache=False,
        augment=False,
    )


def _cache_path_for(audio_path: Path) -> Path:
    return SPECTROGRAM_CACHE_DIR / audio_path.relative_to(AUDIO_DIR).with_suffix(".npy")


def _process_one(audio_path_str: str) -> Tuple[str, bool, str]:
    if _worker_dataset is None:
        raise RuntimeError("Worker dataset not initialised")

    audio_path = Path(audio_path_str)
    cache_path = _cache_path_for(audio_path)
    cache_path.parent.mkdir(parents=True, exist_ok=True)

    if cache_path.exists():
        return (str(audio_path), True, "skipped")

    try:
        waveform = _worker_dataset._load_waveform(audio_path)
        if not isinstance(waveform, np.ndarray):
            spectrogram_tensor = _worker_dataset._waveform_to_melspec(waveform)
            spectrogram = spectrogram_tensor.squeeze(0).cpu().numpy().astype(np.float32, copy=False)
        else:
            spectrogram = waveform.astype(np.float32, copy=False)

        np.save(cache_path, spectrogram)
        return (str(audio_path), True, "saved")
    except Exception as exc:  # noqa: BLE001
        return (str(audio_path), False, f"{type(exc).__name__}: {exc}")


def _iter_audio_files() -> Iterable[Path]:
    for path in sorted(AUDIO_DIR.rglob("*")):
        if path.is_file() and path.suffix.lower() in AUDIO_SUFFIXES:
            yield path


def main() -> None:
    start = time.time()
    audio_files = list(_iter_audio_files())
    logger.info("Found %d audio files under %s", len(audio_files), AUDIO_DIR)
    logger.info("Writing cache to %s", SPECTROGRAM_CACHE_DIR)

    failures: list[str] = []
    processed = 0

    with Pool(processes=NUM_WORKERS, initializer=_init_worker) as pool:
        for _, ok, message in tqdm(pool.imap_unordered(_process_one, map(str, audio_files)), total=len(audio_files)):
            processed += 1
            if not ok and message != "skipped":
                failures.append(message)

    if failures:
        FAILURES_LOG.write_text("\n".join(failures) + "\n")
        logger.info("Logged %d failures to %s", len(failures), FAILURES_LOG)
    elif FAILURES_LOG.exists():
        FAILURES_LOG.unlink()

    elapsed = time.time() - start
    logger.info("Completed %d files in %.1f seconds", processed, elapsed)
    print(f"Completed {processed} files in {elapsed:.1f} seconds")


if __name__ == "__main__":
    main()
