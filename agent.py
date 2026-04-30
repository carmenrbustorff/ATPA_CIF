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
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

from llm_client import LLMClient
from config import NUM_SPECIES

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
MAX_EXEC_TIMEOUT = 600   # seconds – 10 minutes max per generated script

TASK_CONTEXT_TEMPLATE = """\
Task: BirdCLEF+ 2026 – Track B (audio classification)
Goal: Multi-label classification of {num_species} bird species from mel-spectrograms.
Constraints:
  - PyTorch only (no TensorFlow/Keras)
  - Prioritise small-scale, fast iterations (short clips, low-res spectrograms)
  - Final Kaggle submission must run on CPU in ≤ 90 minutes
  - Audio → mel-spectrogram preprocessing is in preprocessing.py (SR=32000, N_MELS=128)
  - Model scaffolds are in models.py
  - Evaluation metric: macro-averaged ROC-AUC skipping classes with no true positives

Dataset summary:
{dataset_summary}

Previous best result: {best_result}
"""


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
        "timestamp": datetime.utcnow().isoformat(),
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
    header = (
        "# AUTO-GENERATED by the Autonomous Research Agent\n"
        f"# Iteration: {iteration_dir.name}\n"
        f"# Generated: {datetime.utcnow().isoformat()}\n\n"
        "import sys, os\n"
        "# Ensure project root is on sys.path\n"
        "sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))\n\n"
    )
    script_path = iteration_dir / "train.py"
    script_path.write_text(header + code, encoding="utf-8")
    logger.info("Generated training script: %s", script_path)
    return script_path


def _fallback_training_script() -> str:
    """
    Return a minimal PyTorch baseline training script for when the LLM
    produces no code.
    """
    return '''\
import json, time, pathlib, numpy as np
import torch
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler
from sklearn.metrics import roc_auc_score

from models import build_simple_cnn_torch
from config import NUM_SPECIES, N_MELS, TIME_FRAMES

BATCH_SIZE = 16
EPOCHS = 3
LR = 1e-3
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Synthetic data (replace with real dataset loader) ---
np.random.seed(42)
n_train, n_val = 200, 40
X_train = torch.rand(n_train, 1, N_MELS, TIME_FRAMES)
y_train = (torch.rand(n_train, NUM_SPECIES) > 0.95).float()
X_val   = torch.rand(n_val, 1, N_MELS, TIME_FRAMES)
y_val   = (torch.rand(n_val, NUM_SPECIES) > 0.95).float()

model = build_simple_cnn_torch(num_classes=NUM_SPECIES).to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=LR)
criterion = nn.BCELoss()
scaler = GradScaler()

t0 = time.time()
for epoch in range(EPOCHS):
    model.train()
    for i in range(0, n_train, BATCH_SIZE):
        xb = X_train[i:i+BATCH_SIZE].to(DEVICE)
        yb = y_train[i:i+BATCH_SIZE].to(DEVICE)
        optimizer.zero_grad()
        with autocast():
            loss = criterion(model(xb), yb)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

    # Validation
    model.eval()
    with torch.no_grad():
        preds = model(X_val.to(DEVICE)).cpu().numpy()
    y_np = y_val.numpy()
    valid_cols = [c for c in range(NUM_SPECIES) if y_np[:, c].sum() > 0]
    val_auc = roc_auc_score(y_np[:, valid_cols], preds[:, valid_cols],
                            average="macro") if valid_cols else 0.0

    if DEVICE.type == "cuda":
        torch.cuda.empty_cache()

elapsed = time.time() - t0

metrics = {
    "model": "SimpleCNN_baseline_torch",
    "epochs": EPOCHS,
    "final_val_auc": round(float(val_auc), 4),
    "training_time_s": round(elapsed, 2),
}
out_path = pathlib.Path(__file__).parent / "metrics.json"
out_path.write_text(json.dumps(metrics, indent=2))
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
    model_name: str = "deepseek-r1:8b",
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
    if not llm.is_available():
        logger.error(
            "Ollama server is not reachable at %s. "
            "Start it with: ollama serve",
            llm.base_url,
        )
        sys.exit(1)
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
        iteration_id = datetime.utcnow().strftime(f"iter_{global_iter:04d}_%Y%m%d_%H%M%S")
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
        "--model", type=str, default="deepseek-r1:8b",
        help="Ollama model name (e.g. deepseek-r1:8b, qwen2.5-coder, llama3).",
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
