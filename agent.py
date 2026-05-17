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

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

EXPERIMENTS_DIR = Path(__file__).parent / "experiments"
NUM_SPECIES = 206  # Update this if dataset changes
MAX_EXEC_TIMEOUT = 10800   # seconds 


TASK_CONTEXT_TEMPLATE = """\
Task: BirdCLEF+ 2026 – Track B (audio classification)
Goal: Multi-label classification of {num_species} bird species from mel-spectrograms.

Current iteration: {iteration_num}

CRITICAL RULES - READ CAREFULLY:
0. IMPORTS (MANDATORY): Your code MUST start with these exact imports. Do not skip any:
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
   - You MUST use Focal Loss for training to handle heavy class imbalance.

5. ARCHITECTURE & TRANSFER LEARNING:
   - CRITICAL: You MUST use 'efficientnet_b1' via the timm library for transfer learning. Instantiate it exactly like this: 'base_model = timm.create_model("efficientnet_b1", pretrained=True, in_chans=1, num_classes=0)'.
   - You MUST unfreeze the base_model by setting 'requires_grad = True' (we are fine-tuning).
   - Use 'in_features = getattr(self.base_model, "num_features")' to size your custom head.
   - Classifier head MUST be: Linear(in_features, 512) -> ReLU -> Dropout(0.3) -> Linear(512, NUM_SPECIES).

6. TRAINING CONSTRAINTS (IMPORTANT FOR ITERATION SPEED)
   - Set max_epochs=40 and initial batch_size=64.
   - You MUST implement Mixup (alpha=0.2) in the training loop to simulate overlapping bird calls.
   - You MUST apply SpecAugment (T.FrequencyMasking and T.TimeMasking) to the training batches.
   - Wrap the batch training step in a try/except block for OOM safety.
   - Include torch.cuda.empty_cache() AFTER EACH BATCH.
   - You MUST implement Automatic Mixed Precision (AMP) using 'scaler = torch.amp.GradScaler("cuda")' and 'with torch.autocast(device_type="cuda"):'.
   - Implement Early Stopping: automatically stop training if the validation AUC does not improve for 5 consecutive epochs.

7. METRICS CAPTURE & MODEL SAVING:
   - Validation = same val_loader but with model.eval() and torch.no_grad().
   - Use roc_auc_score(all_labels, all_preds, average='macro').
   - Save metrics dict to 'metrics.json'.
   - Save ONLY the best model weights: torch.save(model.state_dict(), 'model.pt').

8. MANDATORY SCRIPT TEMPLATE (CRITICAL):
   You MUST use the exact code structure below as the foundation for your script. Do NOT skip any imports. You must fill in the missing logic where indicated.
   
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
   device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
   
   # Custom Focal Loss
   class FocalLoss(nn.Module):
       def __init__(self, alpha=0.25, gamma=2.0):
           super().__init__()
           self.alpha = alpha
           self.gamma = gamma
       def forward(self, inputs, targets):
           bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
           pt = torch.exp(-bce_loss)
           return (self.alpha * (1 - pt) ** self.gamma * bce_loss).mean()

   # Mixup Function
   def mixup_data(x, y, alpha=0.2):
       if alpha > 0:
           lam = np.random.beta(alpha, alpha)
       else:
           lam = 1
       batch_size = x.size()[0]
       index = torch.randperm(batch_size).to(x.device)
       mixed_x = lam * x + (1 - lam) * x[index, :]
       y_a, y_b = y, y[index]
       return mixed_x, y_a, y_b, lam

   # 3. Model Architecture
   class BirdCLEFModel(nn.Module):
       def __init__(self, num_classes):
           super().__init__()
           self.base_model = timm.create_model("efficientnet_b1", pretrained=True, in_chans=1, num_classes=0)
           for param in self.base_model.parameters():
               param.requires_grad = True 
           in_features = getattr(self.base_model, "num_features")
           self.classifier = nn.Sequential(
               nn.Linear(in_features, 512), 
               nn.ReLU(),
               nn.Dropout(0.3),
               nn.Linear(512, num_classes)
           )
       def forward(self, x):
           features = self.base_model(x)
           logits = self.classifier(features)
           return logits
   
   model = BirdCLEFModel(num_classes=NUM_SPECIES).to(device)
   
   # 4. Optimizer, Loss, AMP Scaler
   criterion = FocalLoss(alpha=0.25, gamma=2.0)
   optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-4) 
   scaler = torch.amp.GradScaler("cuda")  # type: ignore
   max_epochs = 40

   # Augmentations
   freq_mask = T.FrequencyMasking(freq_mask_param=24).to(device)
   time_mask = T.TimeMasking(time_mask_param=64).to(device)

   # --- TRAINING AND VALIDATION LOOPS ---
   all_train_losses = []
   best_auc = 0.0
   patience = 5
   early_stop_counter = 0

   for epoch in range(max_epochs):
       model.train()
       epoch_loss = 0.0
       
       for batch_idx, (inputs, labels) in enumerate(train_loader):
           try:
               inputs = inputs.to(device)
               labels = labels.float().to(device)
               
               # Apply Mixup
               inputs, targets_a, targets_b, lam = mixup_data(inputs, labels, alpha=0.2)
               
               # Apply SpecAugment
               inputs = freq_mask(inputs)
               inputs = time_mask(inputs)
               
               optimizer.zero_grad()
               
               with torch.autocast(device_type="cuda"):
                   logits = model(inputs)
                   loss = lam * criterion(logits, targets_a) + (1 - lam) * criterion(logits, targets_b)
               
               scaler.scale(loss).backward()
               scaler.unscale_(optimizer)
               torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
               scaler.step(optimizer)
               scaler.update()
               
               epoch_loss += loss.item()
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
        "import torch.optim as optim\n"
        "from data_loader import build_train_val_dataloaders\n"
        "import json\n"
        "from tqdm import tqdm\n"
        "import numpy as np\n"
        "from sklearn.metrics import roc_auc_score\n"
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
    if checks["blockers"]:
        for blocker in checks["blockers"]:
            logger.warning("Code generation blocker: %s", blocker)
        logger.warning("Falling back to safe template due to generated code blockers.")
        code = _fallback_training_script()
    
    script_path = iteration_dir / "train.py"
    script_path.write_text(header + "\n" + code, encoding="utf-8")
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

    DATA_DIR = "/mnt/disks/data/birdclef"
    METADATA_FILE = DATA_DIR + "/train.csv"
    train_loader, val_loader = build_train_val_dataloaders(
        metadata_csv=METADATA_FILE,
        audio_dir=DATA_DIR,
        batch_size=128,
        augment=True,
        val_split=0.2,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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


def manage_model_checkpoints(
    state: Dict,
    iteration_id: str,
    iteration_dir: Path,
    auc: float,
    keep_top_n: int = 1,
) -> Dict:
    """
    Manage model checkpoints to keep only the top N best models.
    Deletes older model files when a better one is found.
    
    Parameters
    ----------
    state: Current agent state dict
    iteration_id: Current iteration identifier
    iteration_dir: Path to current iteration directory
    auc: AUC score for current iteration
    keep_top_n: Number of top models to keep (default: 1, options: 1 or 3)
    
    Returns
    -------
    Updated state dict with top_models list
    """
    model_path = iteration_dir / "model.pt"
    
    # Initialize top_models list if not present
    if "top_models" not in state:
        state["top_models"] = []  # List of dicts: {"auc": X, "path": Y, "iteration": Z}
    
    # Add current model to tracking (if model.pt exists)
    if model_path.exists():
        state["top_models"].append({
            "auc": auc,
            "path": str(model_path),
            "iteration": iteration_id,
        })
        
        # Sort by AUC (descending) and keep only top N
        state["top_models"].sort(key=lambda x: x["auc"], reverse=True)
        
        # Delete models outside top N
        for model_info in state["top_models"][keep_top_n:]:
            old_path = Path(model_info["path"])
            if old_path.exists():
                old_path.unlink()
                logger.info("Deleted old model: %s (AUC: %.4f)", old_path, model_info["auc"])
        
        # Keep only top N in the list
        state["top_models"] = state["top_models"][:keep_top_n]
        
        logger.info(
            "Top %d models: %s",
            keep_top_n,
            ", ".join([f"{m['iteration']}(AUC={m['auc']:.4f})" for m in state["top_models"]]),
        )
    
    return state


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
    
    # Manage model checkpoints: keep only the best model (or top 3 if you prefer)
    # Change keep_top_n to 3 if you want to keep 3 models instead of 1
    state = manage_model_checkpoints(state, iteration_id, iteration_dir, auc, keep_top_n=1)
    
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
        logger.info("Giving Ollama 15 seconds to load the model back into VRAM...")
        time.sleep(15)
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
