"""
Autonomous Research Agent – Core Loop.

Implements the 7-step iterative cycle:

  Step 1 – Data Exploration    : scan the dataset, compute statistics, and
                                  produce a context summary for the LLM.
  Step 2 – Architecture Proposal: ask the LLM to propose a model architecture
                                  (or improvements) based on the context.
  Step 3 – Code Generation     : extract runnable Python code from the LLM
                                  response and write it to the experiments dir.
  Step 4 – Sandboxed Execution : run the generated code in a subprocess with a
                                  configurable timeout.
  Step 5 – Results Capture     : parse stdout/stderr and any metrics files left
                                  by the generated code.
  Step 6 – LLM-Driven Analysis : send results back to the LLM for structured
                                  analysis and improvement suggestions.
  Step 7 – Iteration           : decide whether to keep iterating or stop.

Usage
-----
    python agent.py [--iterations N] [--model <ollama-model>] [--data-dir /path/to/data]
like:
   python3 agent.py --iterations 1 --model qwen2.5-coder:14b --data-dir /mnt/disks/data/birdclef

The agent stores all artefacts under ``experiments/<iteration_id>/``.
"""

from __future__ import annotations

import argparse
import ast
import json
import logging
import os
import re
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Optional

from llm_client import LLMClient
import warnings
from sklearn.exceptions import UndefinedMetricWarning
warnings.filterwarnings("ignore", category=UndefinedMetricWarning)
import torch

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def resolve_device(allow_cpu_fallback: bool = True) -> torch.device:
    if torch.cuda.is_available():
        device = torch.device("cuda")
        torch.backends.cudnn.benchmark = True
        logger.info("Using device: %s (%s)", device, torch.cuda.get_device_name(0))
        return device

    if allow_cpu_fallback:
        logger.warning("CUDA is not available; falling back to CPU.")
        return torch.device("cpu")

    raise RuntimeError(
        "CUDA is not available. This agent is configured to run on the VM's GPU only."
    )

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

EXPERIMENTS_DIR = Path(__file__).parent / "experiments"
NUM_SPECIES = 206  # Update this if dataset changes
MAX_EXEC_TIMEOUT = 10800   # seconds
MODEL_RELOAD_DELAY_S = 3


TASK_CONTEXT_TEMPLATE = """\
Task: BirdCLEF+ 2026 – Track B (audio classification)
Goal: Multi-label classification of {num_species} bird species from mel-spectrograms.
### CRITICAL EXECUTION CONSTRAINTS ###
You are an execution agent. You must follow these architectural rules exactly. Any deviation will result in a fatal script crash.

Current iteration: {iteration_num}

CRITICAL RULES - READ CAREFULLY:

0. MANDATORY IMPORTS:  Your code MUST start with these exact imports. Do NOT skip any:
You must include the following imports at the top of your script. Missing these will cause a NameError for SpecAugment operations.
CRITICAL: You MUST include this exact import block to the top of your script. Use the exact aliases as shown (e.g., 'import torch.nn as nn' is required for 'nn.Module' to work). Missing any of these imports will cause your script to fail with a NameError when it tries to use the missing module.
NEVER IMPORT MODULES WITHOUT ALIASES
   import os
   import torch
   import torch.nn as nn
   import torch.nn.functional as F
   import torch.optim as optim
   from data_loader import build_train_val_dataloaders
   import json
   from tqdm import tqdm
   import soundfile as sf
   import numpy as np
   from sklearn.metrics import roc_auc_score
   import timm
   import torchaudio.transforms as T

1. STRICTLY PYTORCH: You are absolutely forbidden from using TensorFlow, Keras, or `model.fit()`.

2. DATA INGESTION & AUGMENTATION: You must use our pre-built PyTorch DataLoader. Do not write your own data loaders. Use the build_train_val_dataloaders function from data_loader.py.
   - ALWAYS use augment=True for training.
   - You MUST set 'num_workers=4' and 'pin_memory=True' in your DataLoader to prevent CPU bottlenecking.
   
3. INPUT DIMENSIONS: Each batch item has shape (1, 128, 216):
   - Channels: 1 (mono mel-spectrogram)
   - When batched: (batch_size, 1, 128, 216)

4. OUTPUT DIMENSIONS: Multi-label binary classification with {num_species} labels.
   - Output shape: (batch_size, {num_species})
   - You MUST use Focal Loss for training to handle heavy class imbalance. DO NOT define it yourself. It is injected globally into your runtime environment. Simply instantiate it with 'criterion = FocalLoss(alpha=0.25, gamma=2.0)' and call it in your training loop.

5. ARCHITECTURE & TRANSFER LEARNING:
   - CRITICAL: You MUST use 'efficientnet_b1' via the timm library for transfer learning. Instantiate it exactly like this: 'base_model = timm.create_model("efficientnet_b1", pretrained=True, in_chans=1, num_classes=0)'.
   - You MUST unfreeze the base_model by setting 'requires_grad = True' (we are fine-tuning).
   - Use 'in_features = getattr(self.base_model, "num_features")' to size your custom head.
   - Classifier head MUST be: Linear(in_features, 512) -> ReLU -> Dropout(0.3) -> Linear(512, NUM_SPECIES).
    STRICT BACKBONE ENFORCEMENT:
        You are strictly limited to the EfficientNet-B1 architecture to comply with downstream VRAM and inference limits.
        DO NOT use efficientnet_b3, efficientnet_b0, or any other variant.
        If you change the model string, the run will be considered an immediate failure.

6. TRAINING CONSTRAINTS (IMPORTANT FOR ITERATION SPEED)
    - Set max_epochs=40 and initial batch_size=64.
   - You MUST implement Mixup data augmentation in your training loop using the provided mixup_data function. Do NOT define it yourself. It is injected globally into your runtime environment. Simply call 'mixed_x, mixed_y = mixup_data(x, y, alpha=0.2)' directly in your training loop.
   - You MUST apply SpecAugment (T.FrequencyMasking and T.TimeMasking) to the training batches.
   - Wrap the batch training step in a try/except block for OOM safety.
   - Include torch.cuda.empty_cache() AFTER EACH BATCH.
   - You MUST implement Automatic Mixed Precision (AMP) using 'scaler = torch.amp.GradScaler("cuda")' and 'with torch.autocast(device_type="cuda"):'.
    - Implement Early Stopping: automatically stop training if the validation AUC does not improve for 5 consecutive epochs.
    Do not define FocalLoss or mixup_data. These have already been defined and injected globally into your runtime environment. Simply instantiate criterion = FocalLoss(alpha=0.25, gamma=2.0) and call mixed_x, mixed_y = mixup_data(x, y, alpha=0.2) directly in your training loop.

7A. ANTI-OVERFITTING RULES (MANDATORY):
    - Reduce default training budget: prefer `max_epochs=20` and `batch_size=32` unless explicit reasons exist.
    - Use stronger regularisation: set `weight_decay >= 1e-3`, increase dropout in classifier head to >=0.5.
    - Use conservative learning rates (e.g., lr <= 5e-5) when fine-tuning pretrained backbones.
    - Use a LR scheduler (e.g. `ReduceLROnPlateau`) and lower LR on plateau to avoid overfitting.
    - Reduce early-stopping patience to 4 and checkpoint the best model only.
    - Increase SpecAugment strength (larger time/frequency masks) and ensure `augment=True` is used.
    - Log/print training and validation losses and AUC each epoch (required for analysis and rollback).

7. METRICS CAPTURE & MODEL SAVING:
   - Validation = same val_loader but with model.eval() and torch.no_grad().
   - Use roc_auc_score(all_labels, all_preds, average='macro').
   - Save metrics dict to 'metrics.json'.
   - Save ONLY the best model weights: torch.save(model.state_dict(), 'model.pt').

8. MANDATORY SCRIPT TEMPLATE (CRITICAL):
   You MUST use the exact code structure below as the foundation for your script. Do NOT skip any imports. You must fill in the missing logic where indicated.
   Every import that uses an alias must be explicitly imported with that alias (e.g., 'import torch.nn as nn' is required for 'nn.Module' to work).

   ```python
   import os
   import json
   import torch
   import torch.nn as nn
   import torch.nn.functional as F
   import torch.optim as optim
   from tqdm import tqdm
   import numpy as np
   from sklearn.metrics import roc_auc_score
   from data_loader import build_train_val_dataloaders
   import timm
   import torchaudio.transforms as T 
   

   # 1. Setup
   NUM_SPECIES = 206
   
   # 2. Data Loaders
   batch_size = 64
   train_loader, val_loader = build_train_val_dataloaders(
       batch_size=batch_size,
       augment=True,
       val_split=0.2,
       num_workers=4,
       pin_memory=True
   )
    device = resolve_device()
   
   # NOTE: FocalLoss and mixup_data are already injected into your runtime environment.
   # Do NOT define them. Simply use them as shown below.
    
   # 3. Model Architecture
    class BirdCLEFModel(nn.Module):
       def __init__(self, num_classes):
           super().__init__()
           self.base_model = timm.create_model("efficientnet_b1", pretrained=True, in_chans=1, num_classes=0)
           # Start training with the backbone frozen; unfreeze after a small number of epochs for fine-tuning
           for param in self.base_model.parameters():
               param.requires_grad = False
           in_features = getattr(self.base_model, "num_features")
           self.classifier = nn.Sequential(
               nn.Linear(in_features, 512),
               nn.ReLU(),
               nn.Dropout(0.5),
               nn.Linear(512, num_classes)
           )
       def forward(self, x):
           features = self.base_model(x)
           logits = self.classifier(features)
           return logits
   
    model = BirdCLEFModel(num_classes=NUM_SPECIES).to(device)
   
    # 4. Optimizer, Loss, AMP Scaler
    criterion = FocalLoss(alpha=0.25, gamma=2.0)
    # Conservative LR and stronger weight decay to reduce overfitting
    optimizer = optim.Adam(model.parameters(), lr=5e-5, weight_decay=1e-3)
    scaler = torch.amp.GradScaler("cuda")  # type: ignore
    max_epochs = 20

    # Augmentations (stronger SpecAugment for improved generalisation)
    freq_mask = T.FrequencyMasking(freq_mask_param=48).to(device)
    time_mask = T.TimeMasking(time_mask_param=96).to(device)

    # LR scheduler to reduce LR on plateau (helps prevent overfitting)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=2, min_lr=1e-7)

   # --- TRAINING AND VALIDATION LOOPS ---
    all_train_losses = []
    best_auc = 0.0
    patience = 4
    early_stop_counter = 0

   for epoch in range(max_epochs):
       model.train()
       epoch_loss = 0.0
         print(f"Starting epoch {{epoch + 1}}/{{max_epochs}}")
       
       for batch_idx, (inputs, labels) in enumerate(train_loader):
           try:
               inputs = inputs.to(device)
               labels = labels.float().to(device)
               
               # Apply Mixup
               mixed_x, mixed_y = mixup_data(inputs, labels, alpha=0.2)
               
               # Apply SpecAugment
               mixed_x = freq_mask(mixed_x)
               mixed_x = time_mask(mixed_x)
               
               optimizer.zero_grad()
               
               with torch.autocast(device_type="cuda"):
                   logits = model(mixed_x)
                   loss = criterion(logits, mixed_y)
               
               scaler.scale(loss).backward()
               scaler.unscale_(optimizer)
               torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
               scaler.step(optimizer)
               scaler.update()
               
               epoch_loss += loss.item()
               print(f"Epoch {{epoch + 1}}/{{max_epochs}} - batch {{batch_idx + 1}}/{{len(train_loader)}}")
               torch.cuda.empty_cache()
           except RuntimeError as e:
               if "out of memory" in str(e).lower():
                   print("OOM caught during training step. Skipping batch.")
                   torch.cuda.empty_cache()
                   continue
               else:
                   raise e
       
       avg_epoch_loss = epoch_loss / len(train_loader)
       all_train_losses.append(avg_epoch_loss)
             print(f"Epoch {{epoch + 1}}/{{max_epochs}}, Train Loss: {{avg_epoch_loss:.4f}}")

       # Validation Pass
       model.eval()
       all_preds = []
       all_labels = []

       with torch.no_grad():
           for inputs, labels in val_loader:
               inputs = inputs.to(device)
               labels = labels.float().to(device)
               
               with torch.autocast(device_type="cuda"):
                   logits = model(inputs)
                   preds = torch.sigmoid(logits).cpu().numpy()
               
               all_preds.extend(preds)
               all_labels.extend(labels.cpu().numpy())

       all_preds = np.array(all_preds)
       y_true_multi = np.array(all_labels) 

       col_sums = np.sum(y_true_multi, axis=0)
       valid_classes = (col_sums > 0) & (col_sums < len(y_true_multi))

       if np.any(valid_classes):
           filtered_true = y_true_multi[:, valid_classes]
           filtered_preds = all_preds[:, valid_classes]
           try:
               current_auc = float(roc_auc_score(filtered_true, filtered_preds, average='macro'))
           except Exception as e:
               current_auc = 0.5000
       else:
           current_auc = 0.5000

         print(f"Epoch {{epoch + 1}}/{{max_epochs}}, Validation AUC: {{current_auc:.4f}}")

       # Early Stopping & Best Weights Tracking
       if current_auc > best_auc:
           best_auc = current_auc
           early_stop_counter = 0
           torch.save(model.state_dict(), 'model.pt')
           print(f"--> Saved new best checkpoint with AUC: {{best_auc:.4f}}")
       else:
           early_stop_counter += 1
           print(f"Early stopping counter: {{early_stop_counter}}/{{patience}}")
           if early_stop_counter >= patience:
               print("Early stopping triggered.")
               break

   # Save Metrics
   metrics = {{
       "final_train_loss": all_train_losses[-1] if all_train_losses else 0.0,
       "final_auc": best_auc,
       "num_params": sum(p.numel() for p in model.parameters()),
       "epochs_trained": epoch + 1,
       "batch_size": batch_size, 
       "training_samples": len(train_loader.dataset), 
       "eval_samples": len(val_loader.dataset) 
   }}

   with open('metrics.json', 'w') as f:
       json.dump(metrics, f)

   print("METRICS: ", json.dumps(metrics))

Dataset summary:
{dataset_summary}

Previous best result: {best_result}
"""# ---------------------------------------------------------------------------
# Helper: extract local class names from data
# ---------------------------------------------------------------------------

def extract_local_classes(data_dir: Optional[Path] = None) -> list[str]:
    """
    Extract the list of 206 local training classes from the dataset.

    Reads train.csv and returns sorted list of unique primary_label values.
    If data_dir is None, returns a placeholder list.
    """
    if data_dir is None or not data_dir.exists():
        logger.warning("Data directory not provided. Using placeholder class list.")
        return [f"species_{i:03d}" for i in range(206)]

    train_csv = data_dir / "train.csv"
    if not train_csv.exists():
        logger.warning("train.csv not found in %s. Using placeholder class list.", data_dir)
        return [f"species_{i:03d}" for i in range(206)]

    try:
        import pandas as pd
        df = pd.read_csv(train_csv)
        species = sorted(df["primary_label"].unique().tolist())
        logger.info("Extracted %d local species from dataset", len(species))
        return species
    except Exception as e:
        logger.warning("Failed to extract species from train.csv: %s. Using placeholder.", e)
        return [f"species_{i:03d}" for i in range(206)]


# ---------------------------------------------------------------------------
# Helper: extract model classes from LLM-generated code
# ---------------------------------------------------------------------------

def extract_model_classes(source_code: str) -> str:
    """
    Parse Python source code using AST and extract all class definitions
    that inherit from nn.Module, excluding loss functions and utility classes.

    Returns the raw source code of these classes as a single string.
    """
    try:
        tree = ast.parse(source_code)
    except SyntaxError as e:
        logger.warning("Could not parse source code for class extraction: %s", e)
        return ""

    extracted_code = []
    lines = source_code.split('\n')

    # Classes to exclude (loss functions, metrics, utilities)
    excluded_names = {"FocalLoss", "MixupLoss", "Loss", "Metric", "Scheduler"}

    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            # Skip excluded classes
            if node.name in excluded_names or any(exc in node.name for exc in excluded_names):
                continue

            # Check if it inherits from nn.Module or has nn.Module in bases
            is_model_class = any(
                (isinstance(base, ast.Name) and base.id in ("nn.Module", "Module")) or
                (isinstance(base, ast.Attribute) and base.attr == "Module")
                for base in node.bases
            )

            # Include model classes (those with "Model" in name or inheriting from nn.Module)
            if is_model_class or "Model" in node.name or "model" in node.name.lower():
                # Extract source code for this class
                start_line = node.lineno - 1
                end_line = node.end_lineno if node.end_lineno else start_line + 10

                class_source = '\n'.join(lines[start_line:end_line])
                extracted_code.append(class_source)

    return '\n\n'.join(extracted_code)


# ---------------------------------------------------------------------------
# Helper: extract Python code blocks from LLM output
# ---------------------------------------------------------------------------

def extract_code_blocks(text: str) -> list[str]:
    """
    Extract all ```python ... ``` fenced code blocks from a string.
    Falls back to the entire text if no fenced blocks are found.
    """
    pattern = r"```(?:python)?\s*(.*?)```"
    blocks = re.findall(pattern, text, re.DOTALL | re.IGNORECASE)
    if blocks:
        return [b.strip() for b in blocks]
    # No fenced blocks – treat the whole text as code (last-resort)
    stripped = text.strip()
    return [stripped] if stripped else []


def validate_generated_code(code: str) -> Dict[str, list[str]]:
    """
    Validate generated training code against known project API constraints.

    Returns a dict with:
      - blockers: patterns that should trigger fallback script usage
      - warnings: non-fatal issues worth logging
    """
    blocked_patterns = [
        (r"\bset_mode\s*\(", "Uses set_mode(), but BirdCLEFDataset has no set_mode() API."),
        (
            r"\b\w+\s*,\s*\w+\s*=\s*get_dataloader\s*\(",
            "Unpacks get_dataloader() into multiple loaders, but it returns one loader.",
        ),
        (r"\blimit_samples\s*\(", "Uses limit_samples(), but dataset has no limit_samples() API."),
        (r"\bmodel\.fit\s*\(", "Uses model.fit(), but only manual PyTorch loops are supported."),
        (r"\bimport\s+tensorflow\b", "Imports TensorFlow, which is forbidden in this pipeline."),
        (r"\bfrom\s+tensorflow\b", "Imports TensorFlow, which is forbidden in this pipeline."),
        (r"\bimport\s+keras\b", "Imports Keras, which is forbidden in this pipeline."),
        (r"\bfrom\s+keras\b", "Imports Keras, which is forbidden in this pipeline."),
    ]
    warnings = []
    blockers = []

    for pattern, message in blocked_patterns:
        if re.search(pattern, code, re.IGNORECASE):
            blockers.append(message)

    if "metrics.json" not in code:
        warnings.append("Generated code does not explicitly mention metrics.json writing.")
    if "METRICS:" not in code:
        warnings.append("Generated code does not explicitly print METRICS: line for log parsing.")
    if "model.pt" not in code:
        warnings.append("Generated code does not explicitly mention saving model.pt checkpoint.")

    return {"blockers": blockers, "warnings": warnings}


# ---------------------------------------------------------------------------
# Step 1 – Data Exploration
# ---------------------------------------------------------------------------

def explore_data(data_dir: Optional[Path]) -> Dict:
    """
    Scan the dataset directory and collect basic statistics.

    Returns a dict suitable for JSON serialisation.
    """
    summary: Dict = {
        "data_dir": str(data_dir) if data_dir else "not provided",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "audio_files": 0,
        "unique_species": 0,
        "total_duration_estimate_s": 0,
        "notes": [],
    }

    if data_dir is None or not data_dir.exists():
        summary["notes"].append(
            "No data directory found. Using synthetic/demo data for architecture search."
        )
        return summary

    audio_exts = {".ogg", ".wav", ".flac", ".mp3"}
    species_dirs: set = set()
    count = 0

    for fpath in data_dir.rglob("*"):
        if fpath.suffix.lower() in audio_exts:
            count += 1
            # Assume data/<species>/<file>.ogg structure
            if fpath.parent != data_dir:
                species_dirs.add(fpath.parent.name)

    summary["audio_files"] = count
    summary["unique_species"] = len(species_dirs)
    # Rough estimate: assume 5 s average clip length
    summary["total_duration_estimate_s"] = count * 5

    if count == 0:
        summary["notes"].append("No audio files found in data directory.")
    else:
        summary["notes"].append(
            f"Found {count} audio files across {len(species_dirs)} species directories."
        )

    return summary


# ---------------------------------------------------------------------------
# Steps 2 & 3 – Architecture Proposal + Code Generation
# ---------------------------------------------------------------------------

def propose_and_generate_code(
    llm: LLMClient,
    task_context: str,
    previous_results: Optional[str],
    iteration_dir: Path,
) -> Path:
    """
    Ask the LLM to propose an architecture and write the generated code to a file.

    Returns
    -------
    Path to the generated script.
    """
    logger.info("Step 2: Requesting architecture proposal from LLM (%s)…", llm.model)
    response = ""
    try:
        response = llm.propose_architecture(task_context, previous_results)
    except Exception as exc:
        logger.warning("LLM proposal failed, using fallback script: %s", exc)
        (iteration_dir / "llm_error.txt").write_text(str(exc), encoding="utf-8")

    # Save the raw LLM response (if any)
    (iteration_dir / "llm_proposal.txt").write_text(response, encoding="utf-8")
    logger.info("LLM proposal saved.")

    # Step 3: extract code
    logger.info("Step 3: Extracting code from LLM response…")
    code_blocks = extract_code_blocks(response)

    if not code_blocks:
        logger.warning("No code blocks found in LLM response. Using fallback script.")
        code = _fallback_training_script()
    else:
        # Concatenate all code blocks
        code = "\n\n".join(code_blocks)

    # Prepend a safety header
    # Calculate the project root relative to the experiments directory
    project_root = EXPERIMENTS_DIR.parent.resolve()
    header = (
        "# AUTO-GENERATED by the Autonomous Research Agent\n"
        f"# Iteration: {iteration_dir.name}\n"
        f"# Generated: {datetime.now(timezone.utc).isoformat()}\n\n"
        "import sys\n"
        f"# Ensure project root is on sys.path before importing any project modules\n"
        f"sys.path.insert(0, r'{project_root}')\n\n"
        "# Essential imports (guaranteed to be available)\n"
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

    # Resolve device helper injected after imports to ensure torch is available
    header += (
        "\n# Device resolver helper\n"
        "def resolve_device(allow_cpu_fallback: bool = True):\n"
        "    if torch.cuda.is_available():\n"
        "        device = torch.device('cuda')\n"
        "        torch.backends.cudnn.benchmark = True\n"
        "        return device\n"
        "    if allow_cpu_fallback:\n"
        "        print('CUDA unavailable; falling back to CPU.')\n"
        "        return torch.device('cpu')\n"
        "    raise RuntimeError('CUDA is not available.')\n\n"
    )
    
    # Remove duplicate imports from LLM code to avoid "import redefinition" issues
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

    # Validate generated code and fallback early on known API-incompatible patterns.
    checks = validate_generated_code(code)
    (iteration_dir / "generated_code_checks.json").write_text(
        json.dumps(checks, indent=2),
        encoding="utf-8",
    )
    for warning in checks["warnings"]:
        logger.warning("Code generation warning: %s", warning)
    # ... [existing validation checks] ...
    if checks["blockers"]:
        for blocker in checks["blockers"]:
            logger.warning("Code generation blocker: %s", blocker)
        logger.warning("Falling back to safe template due to generated code blockers.")
        code = _fallback_training_script()

    if not torch.cuda.is_available():
        logger.warning("CUDA is unavailable; using the built-in CPU fallback training script.")
        code = _fallback_training_script()

    # Generated scripts in the wild use two common mixup styles:
    # 1) mixed_x, mixed_y = mixup_data(...)
    # 2) inputs, targets_a, targets_b, lam = mixup_data(...)
    # Match the injected helper signature to the generated usage for this iteration.
    mixup_expects_four = re.search(
        r"\b\w+\s*,\s*\w+\s*,\s*\w+\s*,\s*\w+\s*=\s*mixup_data\s*\(",
        code,
    ) is not None

    if mixup_expects_four:
        logger.info("Detected 4-value mixup_data unpack pattern in generated code.")
        mixup_body = """
    mixed_x = lam * x + (1 - lam) * x[index, :]

    return mixed_x, y, y[index, :], lam
"""
    else:
        mixup_body = """
    mixed_x = lam * x + (1 - lam) * x[index, :]
    mixed_y = lam * y + (1 - lam) * y[index, :]

    return mixed_x, mixed_y
"""
    
    MATH_INJECTION = """
# OVERRIDE: Mathematically Sound Multi-Label Continuous Focal Loss
class _InjectedFocalLoss(nn.Module):
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
def _injected_mixup_data(x, y, alpha=0.2):
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1
    batch_size = x.size()[0]
    index = torch.randperm(batch_size).to(x.device)
    
{mixup_body}

FocalLoss = _InjectedFocalLoss
mixup_data = _injected_mixup_data
""".format(mixup_body=mixup_body)

    DEVICE_COMPAT_INJECTION = """
# DEVICE COMPATIBILITY SHIM: make common GPU idioms safe on CPU
from contextlib import contextmanager
device = resolve_device(allow_cpu_fallback=True)
# Autocast shim: no-op on CPU, real autocast on CUDA
print(f"Resolved device: {device}, torch.cuda.is_available={torch.cuda.is_available()}")

# Autocast shim: no-op on CPU, real autocast on CUDA
# Autocast shim: no-op on CPU, real autocast on CUDA
def _autocast(device_type='cuda'):
    if device.type == 'cuda':
        # Use torch.amp.autocast when CUDA is available
        return torch.amp.autocast(device_type='cuda')
    @contextmanager
    def _noop(*args, **kwargs):
        yield
    return _noop()

# GradScaler shim: enables scaler only when CUDA is available
# Prefer torch.amp.GradScaler (new API), then torch.cuda.amp.GradScaler, else no-op scaler
_amp_gradscaler = getattr(torch.amp, 'GradScaler', None)
_cuda_gradscaler = getattr(getattr(torch, 'cuda', None), 'amp', None)
_cuda_gradscaler = getattr(_cuda_gradscaler, 'GradScaler', None)

if _amp_gradscaler is None and _cuda_gradscaler is None:
    class _NoopGradScaler:
        def __init__(self, enabled=True, *args, **kwargs):
            self.enabled = enabled
        def scale(self, loss):
            return loss
        def unscale_(self, optimizer):
            return
        def step(self, optimizer):
            return optimizer.step()
        def update(self):
            return
    _amp_gradscaler = _NoopGradScaler

class _GradScalerWrapper:
    def __init__(self, *args, **kwargs):
        enabled = kwargs.pop('enabled', device.type == 'cuda')

        # New API first: torch.amp.GradScaler(device_type, ...)
        if _amp_gradscaler is not None:
            try:
                if args:
                    self._scaler = _amp_gradscaler(*args, enabled=enabled, **kwargs)
                else:
                    device_type = 'cuda' if device.type == 'cuda' else 'cpu'
                    self._scaler = _amp_gradscaler(device_type, enabled=enabled, **kwargs)
                return
            except TypeError:
                try:
                    self._scaler = _amp_gradscaler(*args, **kwargs)
                    return
                except TypeError:
                    pass

        # Compatibility fallback for older torch versions
        if _cuda_gradscaler is not None:
            try:
                self._scaler = _cuda_gradscaler(enabled=enabled)
            except TypeError:
                self._scaler = _cuda_gradscaler()
            return

        # Final safe fallback (only if a no-op scaler was installed above)
        self._scaler = _amp_gradscaler(enabled=enabled)

    def __getattr__(self, name):
        return getattr(self._scaler, name)

torch.amp.GradScaler = _GradScalerWrapper
"""

    script_path = iteration_dir / "train.py"
    # Inject device shim and math helpers before the generated code, then restore
    # the aliases at the end so any later redeclarations in generated code do not
    # overwrite the injected implementations.
    script_path.write_text(
        header + "\n" + DEVICE_COMPAT_INJECTION + "\n" + MATH_INJECTION + "\n" + code + "\n"
        + "\n# Restore injected math helpers after generated code\n"
        + "FocalLoss = _InjectedFocalLoss\n"
        + "mixup_data = _injected_mixup_data\n",
        encoding="utf-8",
    )

    logger.info("Generated training script: %s", script_path)
    return script_path

def _fallback_training_script() -> str:
    """
    Return a minimal PyTorch training script for when LLM produces incomplete code.
    Guarantees: metrics capture, validation AUC, and model checkpoint.
    """
    return '''\
    import json
    import torch
    import torch.nn as nn
    import pathlib
    from data_loader import build_train_val_dataloaders

    def resolve_device(allow_cpu_fallback=True):
        if torch.cuda.is_available():
            device = torch.device("cuda")
            torch.backends.cudnn.benchmark = True
            return device
        if allow_cpu_fallback:
            print("CUDA unavailable; falling back to CPU.")
            return torch.device("cpu")
        raise RuntimeError("CUDA is not available.")

    DATA_DIR = "/mnt/disks/data/birdclef"
    METADATA_FILE = DATA_DIR + "/train.csv"
    train_loader, val_loader = build_train_val_dataloaders(
        metadata_csv=METADATA_FILE,
        audio_dir=DATA_DIR,
        batch_size=128,
        augment=True,
        val_split=0.2,
    )

    device = resolve_device()

    class SimpleModel(nn.Module):
        def __init__(self, num_classes):
            super().__init__()
            self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
            self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
            self.pool = nn.AdaptiveAvgPool2d((1, 1))
            self.fc = nn.Linear(64, num_classes)  # Use num_classes parameter for output dimension
        def forward(self, x):
            x = torch.relu(self.conv1(x))
            x = torch.relu(self.conv2(x))
            x = self.pool(x)
            x = x.view(x.size(0), -1)
            return torch.sigmoid(self.fc(x))

    model = SimpleModel(num_classes=206).to(device)
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    # ===== Training Loop =====
    best_auc = 0.0
    patience_counter = 0
    all_train_losses = []

    for epoch in range(max_epochs):
        model.train()
        epoch_loss = 0.0
        batch_count = 0
        model.train()
        for inputs, labels in train_loader:
            if batch_count >= (15000 // 128):
                break  # This safely breaks the BATCH loop, but keeps the EPOCH loop alive
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels.float())
            loss.backward()
            optimizer.step()
            loss_sum += loss.item()
            batch_count += 1
            torch.cuda.empty_cache()
        print(f"Epoch {epoch+1}, Loss: {loss_sum/max(1, batch_count):.4f}")
        # DO NOT put a break statement here!

    # Evaluation loop: compute validation AUC
    model.eval()
    all_preds, all_labels = [], []
    eval_batch_count = 0
    with torch.no_grad():
        for inputs, labels in val_loader:
            if eval_batch_count >= 10:
                break
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            all_preds.append(outputs.cpu().numpy())
            all_labels.append(labels.cpu().numpy())
            eval_batch_count += 1
            torch.cuda.empty_cache()

    if all_preds:
        all_preds = np.concatenate(all_preds)
        all_labels = np.concatenate(all_labels)
        try:
            final_auc = float(roc_auc_score(all_labels, all_preds, average="weighted"))
        except Exception:
            final_auc = 0.5
    else:
        final_auc = 0.5

    # Build metrics dict with all required fields
    metrics = {
        "final_train_loss": float(loss_sum / max(1, batch_count)) if batch_count > 0 else 0.0,
        "final_auc": final_auc,
        "num_params": sum(p.numel() for p in model.parameters()),
        "epochs_trained": 20,
        "batch_size": batch_size,
        "training_samples": batch_count * batch_size, # type: ignore
        "eval_samples": eval_batch_count * batch_size, # type: ignore
    }

    # Save metrics.json (Priority 1: CAPTURE METRICS)
    metrics_path = pathlib.Path(__file__).parent / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2))
    print("METRICS:", json.dumps(metrics))

    # Save model checkpoint (Priority 3: MODEL CHECKPOINT)
    model_path = pathlib.Path(__file__).parent / "model.pt"
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")
    '''

# ---------------------------------------------------------------------------
# Step 4 – Sandboxed Execution
# ---------------------------------------------------------------------------

def execute_script(script_path: Path, timeout: int = MAX_EXEC_TIMEOUT) -> Dict:
    """
    Run the generated training script in a subprocess.

    Returns a dict with keys: ``returncode``, ``stdout``, ``stderr``,
    ``timed_out``, ``duration_s``.
    """
    logger.info("Step 4: Executing generated script (timeout=%ds)…", timeout)
    t0 = time.time()
    stdout_output = ""
    
    try:
        # Use .venv Python to ensure PyTorch and dependencies are available
        venv_python = Path(__file__).parent / ".venv" / "bin" / "python3"
        
        # Use Popen with the "-u" flag to stream output live
        proc = subprocess.Popen(
            [str(venv_python), "-u", str(script_path)],
            cwd=str(script_path.parent),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT, # Merge stderr into stdout
            text=True,
            bufsize=1
        )
        
        # Read output line-by-line and print it to the live terminal
        if proc.stdout:
            for line in iter(proc.stdout.readline, ''):
                print(line, end="") 
                stdout_output += line
            proc.stdout.close()
            
        returncode = proc.wait(timeout=timeout)
        duration = time.time() - t0
        
        return {
            "returncode": returncode,
            "stdout": stdout_output,
            "stderr": "", 
            "timed_out": False,
            "duration_s": round(duration, 2),
        }
        
    except subprocess.TimeoutExpired:
        proc.kill()
        duration = time.time() - t0
        logger.warning("Script execution timed out after %.0f s.", duration)
        return {
            "returncode": -1,
            "stdout": stdout_output,
            "stderr": "TimeoutExpired",
            "timed_out": True,
            "duration_s": round(duration, 2),
        }

# ---------------------------------------------------------------------------
# Step 5 – Results Capture
# ---------------------------------------------------------------------------

def capture_results(execution_result: Dict, iteration_dir: Path) -> Dict:
    """
    Parse stdout/stderr for metrics and check for a metrics.json file.

    Returns a combined results dict.
    """
    logger.info("Step 5: Capturing results…")
    results = dict(execution_result)

    # Try to load metrics.json written by the generated script
    metrics_file = iteration_dir / "metrics.json"
    if metrics_file.exists():
        try:
            results["metrics"] = json.loads(metrics_file.read_text(encoding="utf-8"))
            logger.info("Loaded metrics.json: %s", results["metrics"])
        except json.JSONDecodeError as exc:
            logger.warning("Could not parse metrics.json: %s", exc)

    # Also scan stdout for "METRICS: {...}" lines
    if "metrics" not in results:
        for line in results.get("stdout", "").splitlines():
            if line.startswith("METRICS:"):
                try:
                    results["metrics"] = json.loads(line[len("METRICS:"):].strip())
                    logger.info("Extracted inline metrics: %s", results["metrics"])
                    break
                except json.JSONDecodeError:
                    pass

    # Save execution log
    (iteration_dir / "execution.json").write_text(
        json.dumps(results, indent=2, default=str), encoding="utf-8"
    )

    if results["returncode"] != 0 and not results["timed_out"]:
        logger.error(
            "Script exited with code %d.\nSTDERR:\n%s",
            results["returncode"],
            results.get("stderr", "")[-2000:],
        )
    else:
        logger.info("Script completed in %.1f s.", results.get("duration_s", 0))

    return results


# ---------------------------------------------------------------------------
# Step 6 – LLM-Driven Analysis
# ---------------------------------------------------------------------------

def analyse_results(
    llm: LLMClient,
    results: Dict,
    task_context: str,
    iteration_dir: Path,
) -> str:
    """
    Send results to the LLM for analysis.

    Returns the LLM's textual analysis.
    """
    logger.info("Step 6: Sending results to LLM for analysis…")
    results_text = json.dumps(results, indent=2, default=str)
    analysis = llm.analyse_results(results_text, task_context)
    (iteration_dir / "llm_analysis.txt").write_text(analysis, encoding="utf-8")
    logger.info("LLM analysis saved.")
    return analysis


# ---------------------------------------------------------------------------
# Iteration state helpers
# ---------------------------------------------------------------------------

def get_iteration_bucket_dir(experiments_dir: Path, global_iter: int, bucket_size: int = 50) -> Path:
    """
    Return the bucket directory for a given iteration number.
    Organizes iterations into folders like iterations_0001-0050/, iterations_0051-0100/, etc.
    """
    bucket_num = global_iter // bucket_size
    start = bucket_num * bucket_size + 1
    end = (bucket_num + 1) * bucket_size
    bucket_dir = experiments_dir / f"iterations_{start:04d}-{end:04d}"
    return bucket_dir


# def manage_model_checkpoints(
#     state: Dict,
#     iteration_id: str,
#     iteration_dir: Path,
#     auc: float,
#     keep_top_n: int = 1,
# ) -> Dict:
#     """
#     Manage model checkpoints to keep only the top N best models.
#     Deletes older model files when a better one is found.
    
#     Parameters
#     ----------
#     state: Current agent state dict
#     iteration_id: Current iteration identifier
#     iteration_dir: Path to current iteration directory
#     auc: AUC score for current iteration
#     keep_top_n: Number of top models to keep (default: 1, options: 1 or 3)
    
#     Returns
#     -------
#     Updated state dict with top_models list
#     """
#     model_path = iteration_dir / "model.pt"
    
#     # Initialize top_models list if not present
#     if "top_models" not in state:
#         state["top_models"] = []  # List of dicts: {"auc": X, "path": Y, "iteration": Z}
    
#     # Add current model to tracking (if model.pt exists)
#     if model_path.exists():
#         state["top_models"].append({
#             "auc": auc,
#             "path": str(model_path),
#             "iteration": iteration_id,
#         })
        
#         # Sort by AUC (descending) and keep only top N
#         state["top_models"].sort(key=lambda x: x["auc"], reverse=True)
        
#         # Delete models outside top N
#         for model_info in state["top_models"][keep_top_n:]:
#             old_path = Path(model_info["path"])
#             if old_path.exists():
#                 old_path.unlink()
#                 logger.info("Deleted old model: %s (AUC: %.4f)", old_path, model_info["auc"])
        
#         # Keep only top N in the list
#         state["top_models"] = state["top_models"][:keep_top_n]
        
#         logger.info(
#             "Top %d models: %s",
#             keep_top_n,
#             ", ".join([f"{m['iteration']}(AUC={m['auc']:.4f})" for m in state["top_models"]]),
#         )
    
#     return state


def load_state(experiments_dir: Path) -> Dict:
    """Load persistent agent state (best results, iteration counter)."""
    state_file = experiments_dir / "agent_state.json"
    if state_file.exists():
        return json.loads(state_file.read_text(encoding="utf-8"))
    return {"iteration": 0, "best_auc": 0.0, "best_iteration": None, "history": []}


def save_state(experiments_dir: Path, state: Dict) -> None:
    """Persist agent state to disk."""
    state_file = experiments_dir / "agent_state.json"
    state_file.write_text(json.dumps(state, indent=2, default=str), encoding="utf-8")


def update_state(state: Dict, iteration_id: str, results: Dict, iteration_dir: Path) -> Dict:
    """Update state with the latest iteration results and manage model checkpoints."""
    metrics = results.get("metrics", {})
    auc = metrics.get("final_auc", 0.0)  # Changed from final_val_auc to final_auc
    if auc > state["best_auc"]:
        state["best_auc"] = auc
        state["best_iteration"] = iteration_id
        logger.info("New best AUC: %.4f (iteration %s)", auc, iteration_id)
    state["history"].append(
        {"iteration": iteration_id, "auc": auc, "metrics": metrics}
    )
    
   
    state["iteration"] += 1
    return state


# ---------------------------------------------------------------------------
# Main agent loop
# ---------------------------------------------------------------------------

def run_agent(
    num_iterations: int = 5,
    model_name: str = "gemma3",
    data_dir: Optional[Path] = None,
    exec_timeout: int = MAX_EXEC_TIMEOUT,
) -> None:
    """
    Run the autonomous research agent loop.

    Parameters
    ----------
    num_iterations:
        Number of architecture search iterations to perform.
    model_name:
        Name of the Ollama model to use.
    data_dir:
        Path to the BirdCLEF audio dataset. May be None (uses synthetic data).
    exec_timeout:
        Maximum seconds to allow a generated training script to run.
    """
    EXPERIMENTS_DIR.mkdir(parents=True, exist_ok=True)

    # Initialise LLM client
    llm = LLMClient(model=model_name)
  #  if not llm.is_available():
  #      logger.error(
  #          "Ollama server is not reachable at %s. "
  #          "Start it with: ollama serve",
  #          llm.base_url,
  #      )
  #      sys.exit(1)
    logger.info("LLM client ready. Model: %s", llm.model)

    # Load persistent state
    state = load_state(EXPERIMENTS_DIR)
    logger.info(
        "Resuming from iteration %d (best AUC so far: %.4f).",
        state["iteration"],
        state["best_auc"],
    )

    for i in range(num_iterations):
        global_iter = state["iteration"]
        iteration_id = datetime.now(timezone.utc).strftime(f"iter_{global_iter:04d}_%Y%m%d_%H%M%S")
        # Organize iterations into buckets (iterations_0001-0050, iterations_0051-0100, etc.)
        bucket_dir = get_iteration_bucket_dir(EXPERIMENTS_DIR, global_iter, bucket_size=50)
        iteration_dir = bucket_dir / iteration_id
        iteration_dir.mkdir(parents=True, exist_ok=True)

        logger.info("=" * 60)
        logger.info("ITERATION %d / %d  [%s]", i + 1, num_iterations, iteration_id)
        logger.info("=" * 60)

        # --- Step 1: Data Exploration ---
        logger.info("Step 1: Exploring data…")
        dataset_summary = explore_data(data_dir)
        (iteration_dir / "dataset_summary.json").write_text(
            json.dumps(dataset_summary, indent=2), encoding="utf-8"
        )

        best_result_str = (
            f"AUC={state['best_auc']:.4f} (iteration {state['best_iteration']})"
            if state["best_iteration"]
            else "No previous results"
        )
        previous_results_text = None
        if state["history"]:
            previous_results_text = json.dumps(state["history"][-1], indent=2)

        task_context = TASK_CONTEXT_TEMPLATE.format(
            iteration_num=i + 1,
            num_species=NUM_SPECIES,
            dataset_summary=json.dumps(dataset_summary, indent=2),
            best_result=best_result_str,
        )

        # --- Steps 2 & 3: Propose + Generate ---
        script_path = propose_and_generate_code(
            llm, task_context, previous_results_text, iteration_dir
        )

        # --- Step 4: Execute ---
        exec_result = execute_script(script_path, timeout=exec_timeout)

        # --- Step 5: Capture Results ---
        results = capture_results(exec_result, iteration_dir)

      
        # --- Step 6: Analyse ---
        import time
        logger.info("Giving Ollama %d seconds to load the model back into VRAM...")
      
        try:
            analysis = analyse_results(llm, results, task_context, iteration_dir)
            logger.info("Analysis snippet: %s", analysis[:400])
        except Exception as exc:
            logger.warning("LLM analysis failed for this iteration: %s", exc)
        # --- Step 7: Iterate ---
        state = update_state(state, iteration_id, results, iteration_dir)
        save_state(EXPERIMENTS_DIR, state)

        logger.info(
            "Iteration %d complete. Best AUC so far: %.4f", i + 1, state["best_auc"]
        )

    logger.info("Agent loop finished. Best iteration: %s (AUC=%.4f)",
                state["best_iteration"], state["best_auc"])


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="BirdCLEF+ 2026 Autonomous Research Agent",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--iterations", type=int, default=5,
        help="Number of architecture-search iterations to run.",
    )
    parser.add_argument(
        "--model", type=str, default="gemma3",
        help="Ollama model name (e.g. gemma3, qwen2.5-coder, llama3).",
    )
    parser.add_argument(
        "--data-dir", type=Path, default=None,
        help="Path to the BirdCLEF audio dataset root directory.",
    )
    parser.add_argument(
        "--timeout", type=int, default=MAX_EXEC_TIMEOUT,
        help="Maximum seconds per generated training script.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    run_agent(
        num_iterations=args.iterations,
        model_name=args.model,
        data_dir=args.data_dir,
        exec_timeout=args.timeout,
    )
