"""
Audio Preprocessing Templates for BirdCLEF+ 2026 (Track B).

Provides utilities to convert raw audio files (.ogg, .wav, .flac, etc.)
into mel-spectrograms suitable for 2-D CNN input.

Two backends are supported:
  - ``librosa``  (default, CPU-only, broad format support)
  - ``torchaudio``  (optional, faster on GPU, PyTorch-native)

All constants default to the project-wide values defined in ``config.py``.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Literal, Optional, Tuple

import numpy as np

from config import (
    SR as DEFAULT_SR,
    CLIP_DURATION as DEFAULT_DURATION,
    N_MELS as DEFAULT_N_MELS,
    N_FFT as DEFAULT_N_FFT,
    HOP_LENGTH as DEFAULT_HOP_LENGTH,
    F_MIN as DEFAULT_FMIN,
    F_MAX as DEFAULT_FMAX,
    TOP_DB,
    NUM_SPECIES,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Librosa backend
# ---------------------------------------------------------------------------

def load_audio_librosa(
    path: str | Path,
    sr: int = DEFAULT_SR,
    duration: float = DEFAULT_DURATION,
    offset: float = 0.0,
) -> Tuple[np.ndarray, int]:
    """
    Load an audio file with librosa.

    Returns
    -------
    (waveform, sample_rate)
        waveform: float32 numpy array of shape (num_samples,)
        sample_rate: int
    """
    import librosa  # type: ignore

    waveform, _ = librosa.load(
        path,
        sr=sr,
        duration=duration,
        offset=offset,
        mono=True,
    )
    return waveform.astype(np.float32), sr


def audio_to_melspec_librosa(
    waveform: np.ndarray,
    sr: int = DEFAULT_SR,
    n_mels: int = DEFAULT_N_MELS,
    n_fft: int = DEFAULT_N_FFT,
    hop_length: int = DEFAULT_HOP_LENGTH,
    fmin: float = DEFAULT_FMIN,
    fmax: float = DEFAULT_FMAX,
    top_db: float = 80.0,
) -> np.ndarray:
    """
    Convert a waveform to a log-scaled mel-spectrogram using librosa.

    Returns
    -------
    np.ndarray
        Shape (n_mels, time_frames), dtype float32, values in [0, 1].
    """
    import librosa  # type: ignore

    mel = librosa.feature.melspectrogram(
        y=waveform,
        sr=sr,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels,
        fmin=fmin,
        fmax=fmax,
    )
    log_mel = librosa.power_to_db(mel, ref=np.max, top_db=top_db)
    # Normalise to [0, 1]
    log_mel = (log_mel + top_db) / top_db
    return log_mel.astype(np.float32)


def file_to_melspec_librosa(
    path: str | Path,
    sr: int = DEFAULT_SR,
    duration: float = DEFAULT_DURATION,
    offset: float = 0.0,
    n_mels: int = DEFAULT_N_MELS,
    n_fft: int = DEFAULT_N_FFT,
    hop_length: int = DEFAULT_HOP_LENGTH,
    fmin: float = DEFAULT_FMIN,
    fmax: float = DEFAULT_FMAX,
) -> np.ndarray:
    """
    Load an audio file and return a normalised mel-spectrogram (librosa backend).
    """
    waveform, sr = load_audio_librosa(path, sr=sr, duration=duration, offset=offset)
    return audio_to_melspec_librosa(
        waveform, sr=sr, n_mels=n_mels, n_fft=n_fft, hop_length=hop_length,
        fmin=fmin, fmax=fmax,
    )


# ---------------------------------------------------------------------------
# Torchaudio backend
# ---------------------------------------------------------------------------

def load_audio_torchaudio(
    path: str | Path,
    sr: int = DEFAULT_SR,
    duration: float = DEFAULT_DURATION,
    offset: float = 0.0,
):
    """
    Load an audio file with torchaudio.

    Returns
    -------
    (waveform_tensor, sample_rate)
        waveform_tensor: torch.Tensor of shape (1, num_samples)
    """
    import torch  # type: ignore
    import torchaudio  # type: ignore
    import torchaudio.transforms as T  # type: ignore

    num_frames = int(duration * sr) if duration > 0 else -1
    frame_offset = int(offset * sr)

    waveform, orig_sr = torchaudio.load(
        str(path),
        num_frames=num_frames,
        frame_offset=frame_offset,
    )
    # Mix down to mono
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    # Resample if needed
    if orig_sr != sr:
        waveform = T.Resample(orig_freq=orig_sr, new_freq=sr)(waveform)
    # Pad / trim to exact length
    target_len = int(duration * sr)
    if waveform.shape[-1] < target_len:
        pad = target_len - waveform.shape[-1]
        waveform = torch.nn.functional.pad(waveform, (0, pad))
    else:
        waveform = waveform[..., :target_len]
    return waveform, sr


def audio_to_melspec_torchaudio(
    waveform,
    sr: int = DEFAULT_SR,
    n_mels: int = DEFAULT_N_MELS,
    n_fft: int = DEFAULT_N_FFT,
    hop_length: int = DEFAULT_HOP_LENGTH,
    fmin: float = DEFAULT_FMIN,
    fmax: float = DEFAULT_FMAX,
    top_db: float = 80.0,
) -> np.ndarray:
    """
    Convert a torchaudio waveform tensor to a normalised mel-spectrogram.

    Returns
    -------
    np.ndarray  shape (n_mels, time_frames), dtype float32, values in [0, 1].
    """
    import torch  # type: ignore
    import torchaudio.transforms as T  # type: ignore

    mel_transform = T.MelSpectrogram(
        sample_rate=sr,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels,
        f_min=fmin,
        f_max=fmax,
    )
    amplitude_to_db = T.AmplitudeToDB(top_db=top_db)

    mel = mel_transform(waveform)           # (1, n_mels, time)
    log_mel = amplitude_to_db(mel)          # (1, n_mels, time)
    log_mel = log_mel.squeeze(0)            # (n_mels, time)
    # Normalise to [0, 1]
    log_mel = (log_mel + top_db) / top_db
    return log_mel.numpy().astype(np.float32)


def file_to_melspec_torchaudio(
    path: str | Path,
    sr: int = DEFAULT_SR,
    duration: float = DEFAULT_DURATION,
    offset: float = 0.0,
    n_mels: int = DEFAULT_N_MELS,
    n_fft: int = DEFAULT_N_FFT,
    hop_length: int = DEFAULT_HOP_LENGTH,
    fmin: float = DEFAULT_FMIN,
    fmax: float = DEFAULT_FMAX,
) -> np.ndarray:
    """
    Load an audio file and return a normalised mel-spectrogram (torchaudio backend).
    """
    waveform, sr = load_audio_torchaudio(path, sr=sr, duration=duration, offset=offset)
    return audio_to_melspec_torchaudio(
        waveform, sr=sr, n_mels=n_mels, n_fft=n_fft, hop_length=hop_length,
        fmin=fmin, fmax=fmax,
    )


# ---------------------------------------------------------------------------
# Unified entry point
# ---------------------------------------------------------------------------

def file_to_melspec(
    path: str | Path,
    backend: Literal["librosa", "torchaudio"] = "librosa",
    sr: int = DEFAULT_SR,
    duration: float = DEFAULT_DURATION,
    offset: float = 0.0,
    n_mels: int = DEFAULT_N_MELS,
    n_fft: int = DEFAULT_N_FFT,
    hop_length: int = DEFAULT_HOP_LENGTH,
    fmin: float = DEFAULT_FMIN,
    fmax: float = DEFAULT_FMAX,
) -> np.ndarray:
    """
    Convert an audio file to a mel-spectrogram using the chosen backend.

    Parameters
    ----------
    path:
        Path to the audio file (.ogg, .wav, .flac, etc.).
    backend:
        ``"librosa"`` (default, broader format support) or
        ``"torchaudio"`` (PyTorch-native, GPU-acceleratable).
    sr:
        Target sample rate in Hz.
    duration:
        Clip duration in seconds. Default 5 s for fast iteration.
    offset:
        Start offset in seconds.
    n_mels:
        Number of mel filter-bank bins. Default 64 (low-res).
    n_fft:
        FFT window size in samples.
    hop_length:
        Hop length between STFT frames in samples.
    fmin / fmax:
        Frequency range of the mel filter bank (Hz).

    Returns
    -------
    np.ndarray
        Shape ``(n_mels, time_frames)``, dtype float32, values in ``[0, 1]``.
    """
    if backend == "librosa":
        return file_to_melspec_librosa(
            path, sr=sr, duration=duration, offset=offset,
            n_mels=n_mels, n_fft=n_fft, hop_length=hop_length,
            fmin=fmin, fmax=fmax,
        )
    if backend == "torchaudio":
        return file_to_melspec_torchaudio(
            path, sr=sr, duration=duration, offset=offset,
            n_mels=n_mels, n_fft=n_fft, hop_length=hop_length,
            fmin=fmin, fmax=fmax,
        )
    raise ValueError(f"Unknown backend: {backend!r}. Choose 'librosa' or 'torchaudio'.")


# ---------------------------------------------------------------------------
# Dataset helpers
# ---------------------------------------------------------------------------

def pad_or_trim(spec: np.ndarray, target_frames: int) -> np.ndarray:
    """
    Pad (with zeros) or trim a spectrogram to a fixed number of time frames.

    Parameters
    ----------
    spec:
        Shape (n_mels, time_frames).
    target_frames:
        Desired number of time frames.

    Returns
    -------
    np.ndarray  shape (n_mels, target_frames).
    """
    current = spec.shape[1]
    if current == target_frames:
        return spec
    if current < target_frames:
        pad_width = target_frames - current
        return np.pad(spec, ((0, 0), (0, pad_width)), mode="constant")
    return spec[:, :target_frames]


def spec_to_cnn_input(
    spec: np.ndarray,
    target_frames: Optional[int] = None,
    add_channel_dim: bool = True,
) -> np.ndarray:
    """
    Prepare a spectrogram for CNN input.

    Optionally pads/trims to ``target_frames`` and adds a channel dimension
    so the shape becomes ``(n_mels, target_frames, 1)``.

    Parameters
    ----------
    spec:
        Shape (n_mels, time_frames).
    target_frames:
        If provided, pad or trim to this many time frames.
    add_channel_dim:
        If True (default), append a trailing channel dimension.

    Returns
    -------
    np.ndarray
        Shape (n_mels, target_frames, 1) if add_channel_dim else
        (n_mels, target_frames).
    """
    if target_frames is not None:
        spec = pad_or_trim(spec, target_frames)
    if add_channel_dim:
        spec = spec[..., np.newaxis]
    return spec


def batch_process_directory(
    audio_dir: str | Path,
    output_dir: str | Path,
    backend: Literal["librosa", "torchaudio"] = "librosa",
    extensions: Tuple[str, ...] = (".ogg", ".wav", ".flac", ".mp3"),
    **melspec_kwargs,
) -> int:
    """
    Process all audio files in a directory and save mel-spectrograms as .npy files.

    Parameters
    ----------
    audio_dir:
        Directory containing raw audio files.
    output_dir:
        Directory where ``.npy`` spectrogram files will be saved.
    backend:
        ``"librosa"`` or ``"torchaudio"``.
    extensions:
        Tuple of file extensions to process.
    **melspec_kwargs:
        Additional keyword arguments forwarded to :func:`file_to_melspec`.

    Returns
    -------
    int
        Number of files processed successfully.
    """
    audio_dir = Path(audio_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    processed = 0
    for fpath in sorted(audio_dir.rglob("*")):
        if fpath.suffix.lower() not in extensions:
            continue
        try:
            spec = file_to_melspec(fpath, backend=backend, **melspec_kwargs)
            out_path = output_dir / (fpath.stem + ".npy")
            np.save(out_path, spec)
            processed += 1
            logger.debug("Saved spectrogram: %s", out_path)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Failed to process %s: %s", fpath, exc)

    logger.info("Processed %d audio files -> %s", processed, output_dir)
    return processed
