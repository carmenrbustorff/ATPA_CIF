# CRITICAL FIXES FOR agent.py - KAGGLE SUBMISSION PIPELINE
# ============================================================
# Replace the old KAGGLE_INFERENCE_TEMPLATE with this corrected version.
# This template generates submission.csv with ONE ROW PER 5-SECOND CHUNK,
# properly maps 206 local classes to 234 Kaggle classes, and ensures float32 types.

KAGGLE_INFERENCE_TEMPLATE = '''\
"""
Auto-generated Kaggle inference script – BirdCLEF 2026 Track B.

CRITICAL: One row per 5-second chunk (not aggregated by file).
Maps 206 local classes to 234 Kaggle species columns.
All predictions cast to float32 before saving.
"""

import os
import gc
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
import torchaudio.transforms as T
import numpy as np
import soundfile as sf
import pandas as pd
import timm
from pathlib import Path
from typing import List, Dict

# ============================================================================
# CONFIGURATION & INJECTED DATA
# ============================================================================

SAMPLE_RATE = 32000
CHUNK_DURATION = 5.0  # seconds
MEL_BINS = 128
N_FFT = 2048
HOP_LENGTH = 512
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Injected at generation time: list of 206 local species names
LOCAL_CLASSES: List[str] = {INJECTED_LOCAL_CLASSES}

# Injected at generation time: model class definition(s)
{INJECTED_MODEL_CLASSES}

# ============================================================================
# SPECTROGRAM EXTRACTION
# ============================================================================

def extract_mel_spectrogram_cpu(audio_chunk: np.ndarray, sr: int = SAMPLE_RATE) -> np.ndarray:
    """Extract log-mel spectrogram on CPU. Returns shape (1, MEL_BINS, time_steps)."""
    if audio_chunk.shape[0] == 0:
        return np.zeros((1, MEL_BINS, 1), dtype=np.float32)

    audio_tensor = torch.from_numpy(audio_chunk.astype(np.float32))
    mel_transform = T.MelSpectrogram(
        sample_rate=sr,
        n_mels=MEL_BINS,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH
    )
    mel_spec = mel_transform(audio_tensor)
    log_mel = torch.log(mel_spec + 1e-9)
    return log_mel.unsqueeze(0).numpy().astype(np.float32)


# ============================================================================
# INFERENCE ON SINGLE FILE
# ============================================================================

def infer_on_audio_file(
    audio_path: str,
    model: nn.Module,
    device: torch.device,
) -> List[Dict]:
    """
    Process audio file in 5-second chunks.

    Returns list of dicts, one per chunk:
        {{
            "row_id": "BC2026_Train_0001_S08_20250606_030007_5",  # {filename}_{end_seconds}
            "chunk_idx": 0,
            "predictions": np.array of shape (206,)  float32 probabilities
        }}
    """
    results = []

    try:
        audio_data, sr = sf.read(audio_path, dtype=np.float32)
        if len(audio_data.shape) > 1:
            audio_data = audio_data.mean(axis=1)

        if sr != SAMPLE_RATE:
            resampler = T.Resample(orig_freq=sr, new_freq=SAMPLE_RATE)
            audio_tensor = torch.from_numpy(audio_data).float()
            audio_data = resampler(audio_tensor).numpy()
    except Exception as e:
        print(f"[Error] Failed to load {{audio_path}}: {{e}}")
        return results

    file_stem = Path(audio_path).stem
    num_samples_per_chunk = int(CHUNK_DURATION * SAMPLE_RATE)
    num_chunks = int(np.ceil(len(audio_data) / num_samples_per_chunk))

    print(f"[Inference] {{file_stem}}: {{num_chunks}} chunks")

    for chunk_idx in range(num_chunks):
        start_sample = chunk_idx * num_samples_per_chunk
        end_sample = min((chunk_idx + 1) * num_samples_per_chunk, len(audio_data))

        audio_chunk = audio_data[start_sample:end_sample]
        mel_spec = extract_mel_spectrogram_cpu(audio_chunk, sr=SAMPLE_RATE)
        mel_tensor = torch.from_numpy(mel_spec).to(device)

        # Inference
        with torch.no_grad(), torch.autocast(device_type="cuda" if device.type == "cuda" else "cpu"):
            logits = model(mel_tensor)
            probs = torch.sigmoid(logits).squeeze(0).cpu().numpy().astype(np.float32)

        # Row ID: filename_end_time_in_seconds
        end_time_seconds = int((chunk_idx + 1) * CHUNK_DURATION)
        row_id = f"{{file_stem}}_{{end_time_seconds}}"

        results.append({{
            "row_id": row_id,
            "chunk_idx": chunk_idx,
            "predictions": probs  # shape: (206,) dtype: float32
        }})

        del mel_tensor, logits
        if device.type == "cuda":
            torch.cuda.empty_cache()
        gc.collect()

    return results


# ============================================================================
# SUBMISSION GENERATION WITH CLASS MAPPING
# ============================================================================

def generate_submission_csv(
    test_audio_dir: str,
    model_path: str,
    output_csv: str,
    sample_submission_path: str = "sample_submission.csv",
):
    """
    1. Load model
    2. Infer on all test audio (one row per 5-sec chunk)
    3. Map 206 local -> 234 Kaggle classes
    4. Ensure float32 types
    5. Save submission.csv
    """
    print("[Inference] Loading model…")

    # Find model class dynamically
    import inspect
    model_class = None
    for name, obj in globals().items():
        if (inspect.isclass(obj) and
            issubclass(obj, nn.Module) and
            name not in ('nn', 'Module') and
            not name.startswith('_')):
            model_class = obj
            break

    if model_class is None:
        raise RuntimeError("No nn.Module subclass found in generated code")

    model = model_class(num_classes=len(LOCAL_CLASSES))
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model = model.to(DEVICE)
    model.eval()

    print(f"[Inference] Model loaded. {{len(LOCAL_CLASSES)}} local classes, DEVICE={{DEVICE}}")

    # Find audio files
    audio_dir = Path(test_audio_dir)
    audio_exts = {{".ogg", ".wav", ".flac", ".mp3"}}
    audio_files = sorted([f for f in audio_dir.rglob("*") if f.suffix.lower() in audio_exts])

    if not audio_files:
        print(f"[Warning] No audio files in {{test_audio_dir}}")
        audio_files = []

    print(f"[Inference] Found {{len(audio_files)}} audio files")

    # Infer on all files - collect ALL chunk predictions (no aggregation!)
    all_chunk_results = []
    for i, audio_file in enumerate(audio_files):
        if (i + 1) % 100 == 0 or i == 0:
            print(f"[Inference] Processing {{i+1}}/{{len(audio_files)}}…")

        chunk_results = infer_on_audio_file(str(audio_file), model, DEVICE)
        all_chunk_results.extend(chunk_results)

    print(f"[Inference] Total chunks: {{len(all_chunk_results)}}")

    if not all_chunk_results:
        print("[Warning] No predictions generated. Creating empty submission.")
        submission_df = pd.DataFrame({{"row_id": []}})
        submission_df.to_csv(output_csv, index=False)
        return submission_df

    # ===== CRITICAL: Build LOCAL submission with one row per chunk =====
    # Each row has row_id + 206 local species probabilities
    local_data = {{"row_id": [r["row_id"] for r in all_chunk_results]}}
    for local_idx, species_name in enumerate(LOCAL_CLASSES):
        local_data[species_name] = np.array(
            [r["predictions"][local_idx] for r in all_chunk_results],
            dtype=np.float32
        )

    local_df = pd.DataFrame(local_data)
    print(f"[Inference] Local DataFrame: {{local_df.shape}} - {{local_df['row_id'].dtype}}, {{local_df[LOCAL_CLASSES[0]].dtype}}")

    # Load Kaggle sample_submission.csv to get the 234 species columns
    try:
        kaggle_df = pd.read_csv(sample_submission_path)
        kaggle_cols = [c for c in kaggle_df.columns if c != "row_id"]
    except FileNotFoundError:
        print(f"[Warning] {{sample_submission_path}} not found. Generating dummy Kaggle format.")
        kaggle_cols = [f"species_{{i:03d}}" for i in range(234)]

    print(f"[Inference] Kaggle submission expects {{len(kaggle_cols)}} species columns")

    # ===== BUILD FINAL SUBMISSION =====
    # Start with row_id
    final_data = {{"row_id": local_df["row_id"].values}}

    # For each Kaggle species column, find the corresponding local class
    for kaggle_col in kaggle_cols:
        if kaggle_col in LOCAL_CLASSES:
            # Direct match: species name in both local and Kaggle
            final_data[kaggle_col] = local_df[kaggle_col].values.astype(np.float32)
        else:
            # Species not in local training set - fill with 0.0
            final_data[kaggle_col] = np.zeros(len(local_df), dtype=np.float32)

    # Convert to DataFrame with explicit column order
    final_df = pd.DataFrame(final_data)
    final_df = final_df[["row_id"] + kaggle_cols]

    # CRITICAL: Ensure all columns (except row_id) are float32
    for col in kaggle_cols:
        final_df[col] = final_df[col].astype(np.float32)

    # Validate before saving
    print(f"[Inference] Final DataFrame dtypes:")
    print(f"  row_id: {{final_df['row_id'].dtype}}")
    print(f"  {{kaggle_cols[0]}}: {{final_df[kaggle_cols[0]].dtype}}")
    print(f"  Shape: {{final_df.shape}}")
    print(f"  NaNs: {{final_df.isna().sum().sum()}}")

    # Save
    output_path = Path(output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    final_df.to_csv(output_path, index=False)

    print(f"[Inference] Submission saved: {{output_path}}")
    print(f"[Inference] ✓ Ready for Kaggle submission")

    return final_df


# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    import sys

    if len(sys.argv) < 4:
        print("Usage: inference.py <model_path> <test_audio_dir> <output_csv>")
        sys.exit(1)

    model_path = sys.argv[1]
    test_audio_dir = sys.argv[2]
    output_csv = sys.argv[3]

    print(f"[Inference] LOCAL_CLASSES: {{len(LOCAL_CLASSES)}} species")
    print(f"[Inference] Model: {{model_path}}")
    print(f"[Inference] Test dir: {{test_audio_dir}}")
    print(f"[Inference] Output: {{output_csv}}")

    submission_df = generate_submission_csv(
        test_audio_dir=test_audio_dir,
        model_path=model_path,
        output_csv=output_csv,
    )

    print("[Inference] Complete!")
'''

# ============================================================================
# UPDATED propose_and_generate_code() FUNCTION
# ============================================================================
# This is the KEY function change. Replace your propose_and_generate_code()
# with this version. The critical changes:
# 1. Call extract_local_classes(data_dir) to get 206 species names
# 2. Format template with BOTH {INJECTED_MODEL_CLASSES} and {INJECTED_LOCAL_CLASSES}

def propose_and_generate_code_UPDATED(
    llm,  # LLMClient
    task_context: str,
    previous_results: Optional[str],
    iteration_dir: Path,
    data_dir: Optional[Path] = None,  # ADD THIS PARAMETER
) -> Path:
    """
    Ask the LLM to propose an architecture and write the generated code to a file.
    NOW ALSO GENERATES inference.py with proper class mapping.

    Returns Path to the generated train.py script.
    """
    logger.info("Step 2: Requesting architecture proposal from LLM (%s)…", llm.model)
    response = ""
    try:
        response = llm.propose_architecture(task_context, previous_results)
    except Exception as exc:
        logger.warning("LLM proposal failed, using fallback script: %s", exc)
        (iteration_dir / "llm_error.txt").write_text(str(exc), encoding="utf-8")

    (iteration_dir / "llm_proposal.txt").write_text(response, encoding="utf-8")
    logger.info("LLM proposal saved.")

    # Step 3: extract code
    logger.info("Step 3: Extracting code from LLM response…")
    code_blocks = extract_code_blocks(response)

    if not code_blocks:
        logger.warning("No code blocks found in LLM response. Using fallback script.")
        code = _fallback_training_script()
    else:
        code = "\n\n".join(code_blocks)

    # [Previous header + filtering logic remains unchanged...]
    project_root = EXPERIMENTS_DIR.parent.resolve()
    header = (
        "# AUTO-GENERATED by the Autonomous Research Agent\n"
        f"# Iteration: {iteration_dir.name}\n"
        f"# Generated: {datetime.now(timezone.utc).isoformat()}\n\n"
        "import sys\n"
        f"sys.path.insert(0, r'{project_root}')\n\n"
        "import os\n"
        "import torch\n"
        "import torch.nn as nn\n"
        "import torch.nn.functional as F\n"
        "import torchaudio.transforms as T\n"
        "import torch.optim as optim\n"
        "from data_loader import build_train_val_dataloaders\n"
        "import json\n"
        "from tqdm import tqdm\n"
        "import numpy as np\n"
        "from sklearn.metrics import roc_auc_score\n"
    )

    # Remove duplicate imports
    lines = code.split('\n')
    filtered_lines = []
    skip_imports = {'import os', 'import torch', 'import json', 'from tqdm import tqdm',
                    'import numpy as np', 'from sklearn.metrics import roc_auc_score',
                    'import torch.nn as nn', 'import torch.nn.functional as F',
                    'import torch.optim as optim', 'from data_loader import build_train_val_dataloaders'}
    for line in lines:
        stripped = line.strip()
        if any(stripped.startswith(skip) for skip in skip_imports):
            continue
        filtered_lines.append(line)
    code = '\n'.join(filtered_lines)

    # Validate
    checks = validate_generated_code(code)
    (iteration_dir / "generated_code_checks.json").write_text(
        json.dumps(checks, indent=2),
        encoding="utf-8",
    )
    for warning in checks["warnings"]:
        logger.warning("Code generation warning: %s", warning)
    if checks["blockers"]:
        for blocker in checks["blockers"]:
            logger.warning("Code generation blocker: %s", blocker)
        logger.warning("Falling back to safe template due to generated code blockers.")
        code = _fallback_training_script()

    # Math injection
    MATH_INJECTION = """
# OVERRIDE: Mathematically Sound Multi-Label Continuous Focal Loss
class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        probs = torch.sigmoid(inputs)
        p_t = targets * probs + (1 - targets) * (1 - probs)
        focal_weight = (1 - p_t) ** self.gamma
        alpha_t = targets * self.alpha + (1 - targets) * (1 - self.alpha)
        loss = alpha_t * focal_weight * bce_loss
        return loss.sum(dim=1).mean()

# OVERRIDE: Multi-Label Mixup (Direct Label Blending)
def mixup_data(x, y, alpha=0.2):
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1
    batch_size = x.size()[0]
    index = torch.randperm(batch_size).to(x.device)

    mixed_x = lam * x + (1 - lam) * x[index, :]

    return mixed_x, y, y[index, :], lam
"""

    script_path = iteration_dir / "train.py"
    script_path.write_text(header + "\n" + MATH_INJECTION + "\n" + code, encoding="utf-8")
    logger.info("Generated training script: %s", script_path)

    # ===================================================================
    # GENERATE inference.py WITH PROPER CLASS MAPPING
    # ===================================================================
    logger.info("Generating inference.py from model classes…")
    model_classes = extract_model_classes(code)

    if not model_classes.strip():
        logger.warning("No model classes found. Inference generation skipped.")
        return script_path

    # CRITICAL: Extract local 206 species names
    local_classes = extract_local_classes(data_dir)
    logger.info("Extracted %d local species", len(local_classes))

    # Format template with BOTH model classes AND local class names
    inference_script = KAGGLE_INFERENCE_TEMPLATE.format(
        INJECTED_MODEL_CLASSES=model_classes,
        INJECTED_LOCAL_CLASSES=repr(local_classes),  # repr() converts list to Python literal
    )

    inference_path = iteration_dir / "inference.py"
    inference_path.write_text(inference_script, encoding="utf-8")
    logger.info("Generated inference script: %s", inference_path)

    return script_path
