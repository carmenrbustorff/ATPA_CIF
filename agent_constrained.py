"""agent_constrained.py — Constrained autonomous research agent for BirdCLEF.

Design: LLM picks hyperparameters within a fixed search space; train.py is the
unchanging scaffold. Contrast with Carmen's unconstrained agent (LLM writes
full training code each iteration).

Usage:
    python agent_constrained.py --iterations 1 --llm qwen2.5-coder:14b
"""
import argparse
import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import requests

EXPERIMENTS = Path("experiments")
AGENT_LOG = EXPERIMENTS / "constrained_agent_state.json"
OLLAMA_URL = "http://localhost:11434/api/generate"


PROMPT_TEMPLATE = """You are an ML research assistant. Task: BirdCLEF 2026 bird species classification (multi-label, 206 classes, audio→mel-spectrogram→CNN).

A training script `train.py` exists with these CLI arguments:
- --model: "simple_cnn_torch" or "efficientnet_torch"
- --epochs: int 1-5
- --batch-size: 8, 16, or 32
- --lr: float 1e-5 to 1e-2
- --weight-decay: float 0.0 to 1e-3
- --augment: boolean (true enables random-crop augmentation)

Manual baselines: SimpleCNN from scratch achieves ~0.52 AUC. EfficientNet (pretrained, transfer learning) is expected to do much better.

Previous experiment results (last 5):
{history}

Best so far: {best}

Propose the NEXT experiment based on what you've learned. Respond with EXACTLY this JSON and nothing else:

{{"reasoning": "<one short sentence>", "model": "...", "epochs": ..., "batch_size": ..., "lr": ..., "weight_decay": ..., "augment": true_or_false}}
"""


def call_llm(model, prompt, timeout=120):
    response = requests.post(
        OLLAMA_URL,
        json={"model": model, "prompt": prompt, "stream": False, "format": "json"},
        timeout=timeout,
    )
    response.raise_for_status()
    return response.json()["response"]


def parse_proposal(raw_response):
    raw = raw_response.strip()
    if "```" in raw:
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
        raw = raw.strip()
    return json.loads(raw)


def validate_proposal(p):
    """Clamp values into valid ranges. Returns None if proposal is unsalvageable."""
    try:
        if p["model"] not in ("simple_cnn_torch", "efficientnet_torch"):
            return None
        p["epochs"] = max(1, min(15, int(p["epochs"])))
        bs = int(p["batch_size"])
        p["batch_size"] = bs if bs in (8, 16, 32) else 16
        p["lr"] = max(1e-5, min(1e-2, float(p["lr"])))
        p["weight_decay"] = max(0.0, min(1e-3, float(p.get("weight_decay", 0.0))))
        p["augment"] = bool(p.get("augment", False))
        return p
    except (KeyError, ValueError, TypeError):
        return None


def build_history_summary(state):
    if not state.get("history"):
        return "(no previous experiments — propose a reasonable starting point)"
    lines = []
    for i, e in enumerate(state["history"][-5:], 1):
        cfg = e.get("config", {})
        lines.append(
            f"  {i}. model={cfg.get('model')} lr={cfg.get('lr')} "
            f"bs={cfg.get('batch_size')} ep={cfg.get('epochs')} aug={cfg.get('augment')} "
            f"→ AUC={e.get('best_val_auc')} status={e.get('status')}"
        )
    return "\n".join(lines)


def best_summary(state):
    if state.get("best") is None:
        return "none"
    return f"AUC={state['best']['auc']:.4f} from {state['best']['run_id']}"


def load_state():
    if AGENT_LOG.exists():
        return json.loads(AGENT_LOG.read_text())
    return {"history": [], "best": None}


def save_state(state):
    AGENT_LOG.write_text(json.dumps(state, indent=2))


def run_train_script(proposal, llm_name):
    run_id = datetime.now().strftime(f"agent_{llm_name}_%Y%m%d_%H%M%S")
    cmd = [
        sys.executable, "train.py",
        "--model", proposal["model"],
        "--epochs", str(proposal["epochs"]),
        "--batch-size", str(proposal["batch_size"]),
        "--lr", str(proposal["lr"]),
        "--weight-decay", str(proposal["weight_decay"]),
        "--run-id", run_id,
        "--llm-name", llm_name,
    ]
    if proposal["augment"]:
        cmd.append("--augment")
    print(f"  → {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)

    metrics_path = EXPERIMENTS / run_id / "metrics.json"
    if metrics_path.exists():
        m = json.loads(metrics_path.read_text())
        m["run_id"] = run_id
        return m
    return {"run_id": run_id, "status": "failed_no_metrics", "best_val_auc": None,
            "config": proposal, "stderr": result.stderr[-500:]}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--iterations", type=int, default=10)
    p.add_argument("--llm", default="qwen3-coder",
                   help="Ollama model name (e.g. qwen3-coder, nemotron, gemma3)")
    args = p.parse_args()

    EXPERIMENTS.mkdir(exist_ok=True)
    state = load_state()
    print(f"Loaded state: {len(state['history'])} previous runs, best={best_summary(state)}")

    for i in range(1, args.iterations + 1):
        print(f"\n{'='*60}\nITERATION {i}/{args.iterations}\n{'='*60}")
        prompt = PROMPT_TEMPLATE.format(
            history=build_history_summary(state),
            best=best_summary(state),
        )
        try:
            t0 = time.time()
            raw = call_llm(args.llm, prompt)
            print(f"LLM responded in {time.time()-t0:.1f}s")
            proposal = parse_proposal(raw)
            proposal = validate_proposal(proposal)
            if proposal is None:
                raise ValueError("invalid proposal after validation")
            print(f"  Proposal: {proposal}")
        except Exception as e:
            print(f"  ❌ LLM failure: {e}")
            state["history"].append({"iteration": i, "status": "llm_failed", "error": str(e)})
            save_state(state)
            continue

        metrics = run_train_script(proposal, args.llm)
        print(f"  Result: status={metrics.get('status')} AUC={metrics.get('best_val_auc')}")

        state["history"].append(metrics)
        if metrics.get("best_val_auc") is not None:
            current = metrics["best_val_auc"]
            if state["best"] is None or current > state["best"]["auc"]:
                state["best"] = {"auc": current, "run_id": metrics["run_id"]}
                print(f"  ⭐ NEW BEST AUC: {current:.4f}")
        save_state(state)

    successful = sum(1 for h in state["history"] if h.get("status") == "success")
    failed = len(state["history"]) - successful
    print(f"\n{'='*60}\nFINAL\n{'='*60}")
    print(f"Total: {len(state['history'])}  Success: {successful}  Skipped: {failed}")
    print(f"Skip rate: {failed/max(1,len(state['history'])):.0%}")
    print(f"Best: {best_summary(state)}")


if __name__ == "__main__":
    main()