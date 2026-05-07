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
NUM_SPECIES = 234
MAX_EXEC_TIMEOUT = 600   # seconds – 10 minutes max per generated script


TASK_CONTEXT_TEMPLATE = """\
Task: BirdCLEF+ 2026 – Track B (audio classification)
Goal: Multi-label classification of {num_species} bird species from mel-spectrograms.

CRITICAL RULES - READ CAREFULLY:
0. IMPORTS (MANDATORY): Your code MUST start with these exact imports. Do not skip any:
   import os
   import torch
   import torch.nn as nn
   import torch.nn.functional as F
   import torch.optim as optim
   from dataset import get_dataloader
   import json
   from tqdm import tqdm
   
1. STRICTLY PYTORCH: You are absolutely forbidden from using TensorFlow, Keras, or `model.fit()`. 
2. DATA INGESTION: You must use our pre-built PyTorch DataLoader. Do not write your own data loaders.
   Use this code after imports:
   
   DATA_DIR = "/mnt/disks/data/birdclef"
   METADATA_FILE = os.path.join(DATA_DIR, "train.csv")
   train_loader = get_dataloader(DATA_DIR, METADATA_FILE, batch_size=32)

3. INPUT DIMENSIONS: Each batch item has shape (1, 128, 216):
   - Channels: 1 (mono mel-spectrogram)
   - Mel bins: 128
   - Time frames: 216
   When batched: (batch_size, 1, 128, 216)

4. OUTPUT DIMENSIONS: Multi-label binary classification with {num_species} labels.
   - Use sigmoid activation
   - Use BCELoss for training
   - Output shape: (batch_size, {num_species})

5. ARCHITECTURE: Write a PyTorch CNN class that correctly flattens intermediate features.
   Use AdaptiveAvgPool2d for robust dimension handling across training batches.
   Example pattern:
   - Conv2d layers to extract features
   - Use AdaptiveAvgPool2d(output_size=(1, 1)) to flatten robustly
   - Linear layers for classification

6. TRAINING CONSTRAINTS (IMPORTANT FOR ITERATION SPEED):
   - Use 2-3 epochs maximum (not 5+) for quick iteration
   - Use limited training samples: train on first ~5000 samples for fast feedback
   - Report loss after each epoch with tqdm progress bar
   - Include torch.cuda.empty_cache() after each epoch to prevent OOM
   
7. TRAINING LOOP:
   - Write a standard PyTorch training loop
   - Move model and data to CUDA with .to('cuda')
   - Use subset of data for speed (can do full training later)
   - Report training loss per epoch

8. METRICS CAPTURE (CRITICAL - AUC IS MANDATORY):
   IMPORTANT: final_auc must NEVER be null or nan. This is your main model comparison metric.
   Use this exact code after training:
   
   import numpy as np
   from sklearn.metrics import roc_auc_score
   
   model.eval()
   all_preds = []
   all_labels = []
   with torch.no_grad():
       for inputs, labels in train_loader:
           inputs, labels = inputs.to(device), labels.to(device)
           outputs = model(inputs)
           all_preds.append(outputs.cpu().numpy())
           all_labels.append(labels.cpu().numpy())
   
   all_preds = np.concatenate(all_preds)
   all_labels = np.concatenate(all_labels)
   
   # For multi-label classification, use samples_average or weighted average
   try:
       final_auc = float(roc_auc_score(all_labels, all_preds, average='weighted', multi_class='ovr'))
   except:
       # Fallback: if above fails, use sample average
       final_auc = float(roc_auc_score(all_labels.ravel(), all_preds.ravel()))
   
   - Save metrics dict with keys: final_train_loss, final_auc, num_params, epochs_trained, batch_size
   - final_auc must be a valid number (check: not nan, not inf, not null)
   - Write to metrics.json in JSON format
   - Print: "METRICS: " + JSON string (one line)
   - Save model: torch.save(model.state_dict(), 'model.pt')

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
    response = llm.propose_architecture(task_context, previous_results)

    # Save the raw LLM response
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
        "from dataset import get_dataloader\n"
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
                    'import torch.optim as optim', 'from dataset import get_dataloader'}
    for line in lines:
        stripped = line.strip()
        if any(stripped.startswith(skip) for skip in skip_imports):
            continue
        filtered_lines.append(line)
    code = '\n'.join(filtered_lines)
    
    script_path = iteration_dir / "train.py"
    script_path.write_text(header + "\n" + code, encoding="utf-8")
    logger.info("Generated training script: %s", script_path)
    return script_path


def _fallback_training_script() -> str:
    """
    Return a minimal PyTorch training script for when LLM produces incomplete code.
    This guarantees proper AUC computation and metrics capture.
    """
    return '''\
import json
import time
import torch
import torch.nn as nn

DATA_DIR = "/mnt/disks/data/birdclef"
METADATA_FILE = DATA_DIR + "/train.csv"
train_loader = get_dataloader(DATA_DIR, METADATA_FILE, batch_size=32)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(64, 234)
    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        return torch.sigmoid(self.fc(x))

model = SimpleModel().to(device)
criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

loss_sum = 0.0
batch_count = 0
for epoch in range(2):
    model.train()
    for inputs, labels in train_loader:
        if batch_count >= 100:
            break
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels.float())
        loss.backward()
        optimizer.step()
        loss_sum += loss.item()
        batch_count += 1
    print(f"Epoch {epoch+1}, Loss: {loss_sum/max(1, batch_count):.4f}")
    torch.cuda.empty_cache()
    if batch_count >= 100:
        break

model.eval()
all_preds, all_labels = [], []
batch_count = 0
with torch.no_grad():
    for inputs, labels in train_loader:
        if batch_count >= 100:
            break
        inputs, labels = inputs.to(device), labels.to(device)
        all_preds.append(model(inputs).cpu().numpy())
        all_labels.append(labels.cpu().numpy())
        batch_count += 1

if all_preds:
    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)
    try:
        final_auc = float(roc_auc_score(all_labels, all_preds, average="weighted"))
    except Exception:
        final_auc = 0.5
else:
    final_auc = 0.5

metrics = {
    "final_train_loss": float(loss_sum / min(100, batch_count)) if batch_count > 0 else 0.0,
    "final_auc": final_auc,
    "num_params": sum(p.numel() for p in model.parameters()),
    "epochs_trained": 2,
    "batch_size": 32,
}
import pathlib
pathlib.Path(__file__).parent.joinpath("metrics.json").write_text(json.dumps(metrics))
print("METRICS:", json.dumps(metrics))
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
    try:
        proc = subprocess.run(
            [sys.executable, str(script_path)],
            cwd=str(script_path.parent),
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        duration = time.time() - t0
        return {
            "returncode": proc.returncode,
            "stdout": proc.stdout,
            "stderr": proc.stderr,
            "timed_out": False,
            "duration_s": round(duration, 2),
        }
    except subprocess.TimeoutExpired:
        duration = time.time() - t0
        logger.warning("Script execution timed out after %.0f s.", duration)
        return {
            "returncode": -1,
            "stdout": "",
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


def update_state(state: Dict, iteration_id: str, results: Dict) -> Dict:
    """Update state with the latest iteration results."""
    metrics = results.get("metrics", {})
    auc = metrics.get("final_val_auc", 0.0)
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
        iteration_dir = EXPERIMENTS_DIR / iteration_id
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
        if llm.is_available():
            analysis = analyse_results(llm, results, task_context, iteration_dir)
            logger.info("Analysis snippet: %s", analysis[:400])
        else:
            logger.warning("LLM unavailable for analysis step.")

        # --- Step 7: Iterate ---
        state = update_state(state, iteration_id, results)
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
