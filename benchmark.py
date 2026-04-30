"""
Benchmark Table Generator for BirdCLEF+ 2026 (Track B).

Aggregates results from all experiment runs into a single benchmark table
that compares:
  - Model architectures (SimpleCNN, EfficientNet, agent-generated, …)
  - LLM models used for code generation (deepseek-r1:8b, qwen2.5-coder, …)
  - Macro ROC-AUC scores (best val and final val)
  - Code-generation error rates (per LLM / per agent session)
  - Training time and GPU memory

Results are written to:
  experiments/benchmark.json   — machine-readable full record
  experiments/benchmark.md     — human-readable Markdown table

Sources read (in priority order):
  1. experiments/benchmark.json  — persistent registry updated by evaluate.py / benchmark.py
  2. experiments/agent_state.json — iteration history from agent.py
  3. experiments/iter_*/execution.json  — per-iteration runtime data (error codes, duration)
  4. experiments/iter_*/metrics.json    — per-iteration training metrics
  5. experiments/**/eval_results.json   — results written by evaluate.py

Usage
-----
    # Rebuild full benchmark table from all experiment artefacts:
    python benchmark.py

    # Register a manual run (e.g. after running train.py + evaluate.py):
    python benchmark.py register \\
        --run-id my_run_001 \\
        --model efficientnet_torch \\
        --llm deepseek-r1:8b \\
        --macro-auc 0.7842 \\
        --training-time-s 1234 \\
        --notes "EfficientNet, 5 epochs, lr=5e-4"
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

EXPERIMENTS_DIR = Path(__file__).parent / "experiments"
BENCHMARK_JSON = EXPERIMENTS_DIR / "benchmark.json"
BENCHMARK_MD = EXPERIMENTS_DIR / "benchmark.md"


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

def _empty_entry(run_id: str) -> Dict[str, Any]:
    return {
        "run_id": run_id,
        "timestamp": None,
        "source": "unknown",      # "agent", "manual", "evaluate"
        "model_arch": None,
        "llm_model": None,
        "macro_auc": None,
        "best_val_auc": None,
        "final_val_auc": None,
        "training_time_s": None,
        "exec_error_rate": None,  # fraction of iterations that failed (0–1)
        "timed_out": False,
        "epochs": None,
        "batch_size": None,
        "notes": "",
    }


# ---------------------------------------------------------------------------
# Registry helpers
# ---------------------------------------------------------------------------

def load_registry() -> List[Dict[str, Any]]:
    """Load the existing benchmark registry, or return an empty list."""
    if BENCHMARK_JSON.exists():
        try:
            data = json.loads(BENCHMARK_JSON.read_text(encoding="utf-8"))
            if isinstance(data, list):
                return data
        except json.JSONDecodeError as exc:
            logger.warning("Could not parse benchmark.json: %s", exc)
    return []


def save_registry(entries: List[Dict[str, Any]]) -> None:
    """Persist the benchmark registry."""
    EXPERIMENTS_DIR.mkdir(parents=True, exist_ok=True)
    BENCHMARK_JSON.write_text(
        json.dumps(entries, indent=2, default=str), encoding="utf-8"
    )
    logger.info("Saved benchmark registry: %d entries → %s", len(entries), BENCHMARK_JSON)


def _upsert(registry: List[Dict[str, Any]], entry: Dict[str, Any]) -> None:
    """Insert or update an entry by run_id."""
    for i, existing in enumerate(registry):
        if existing["run_id"] == entry["run_id"]:
            registry[i] = entry
            return
    registry.append(entry)


# ---------------------------------------------------------------------------
# Harvest agent_state.json
# ---------------------------------------------------------------------------

def _harvest_agent_state(experiments_dir: Path) -> List[Dict[str, Any]]:
    """
    Read experiments/agent_state.json and produce one entry per LLM session
    (grouped by consecutive runs with the same LLM model).

    Also scans per-iteration execution.json files to compute error rates.
    """
    state_file = experiments_dir / "agent_state.json"
    if not state_file.exists():
        return []

    try:
        state = json.loads(state_file.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        logger.warning("Could not parse agent_state.json: %s", exc)
        return []

    history = state.get("history", [])
    if not history:
        return []

    # Group consecutive iterations that share the same llm_model
    sessions: List[Dict] = []
    current_session: Optional[Dict] = None

    for h in history:
        iteration_id = h.get("iteration", "")
        llm = h.get("metrics", {}).get("llm_model") or h.get("llm_model")
        auc = h.get("auc", 0.0) or 0.0
        metrics = h.get("metrics", {})

        # Try to load per-iteration execution.json for error/timeout info
        iter_dir = experiments_dir / iteration_id
        exec_failed = False
        timed_out = False
        duration_s = metrics.get("training_time_s")

        exec_file = iter_dir / "execution.json"
        if exec_file.exists():
            try:
                exec_data = json.loads(exec_file.read_text(encoding="utf-8"))
                exec_failed = exec_data.get("returncode", 0) != 0
                timed_out = exec_data.get("timed_out", False)
                if duration_s is None:
                    duration_s = exec_data.get("duration_s")
            except json.JSONDecodeError:
                pass

        if current_session is None or current_session["llm_model"] != llm:
            current_session = {
                "run_id": f"agent_{iteration_id}",
                "llm_model": llm,
                "aucs": [],
                "failures": 0,
                "timeouts": 0,
                "iterations": 0,
                "total_time_s": 0.0,
                "model_arch": metrics.get("model"),
            }
            sessions.append(current_session)

        current_session["aucs"].append(auc)
        current_session["iterations"] += 1
        if exec_failed:
            current_session["failures"] += 1
        if timed_out:
            current_session["timeouts"] += 1
        if duration_s:
            current_session["total_time_s"] += float(duration_s)

    entries = []
    for s in sessions:
        n = s["iterations"]
        aucs = [a for a in s["aucs"] if a and a > 0]
        entry = _empty_entry(s["run_id"])
        entry.update({
            "timestamp": None,
            "source": "agent",
            "model_arch": s["model_arch"],
            "llm_model": s["llm_model"],
            "macro_auc": round(max(aucs), 6) if aucs else None,
            "best_val_auc": round(max(aucs), 6) if aucs else None,
            "final_val_auc": round(aucs[-1], 6) if aucs else None,
            "training_time_s": round(s["total_time_s"], 1) if s["total_time_s"] else None,
            "exec_error_rate": round(s["failures"] / n, 3) if n else None,
            "timed_out": s["timeouts"] > 0,
            "notes": f"{n} iteration(s), {s['failures']} error(s), {s['timeouts']} timeout(s)",
        })
        entries.append(entry)

    return entries


# ---------------------------------------------------------------------------
# Harvest evaluate.py outputs (eval_results.json)
# ---------------------------------------------------------------------------

def _harvest_eval_results(experiments_dir: Path) -> List[Dict[str, Any]]:
    """Scan all eval_results.json files and produce one entry each."""
    entries = []
    for eval_file in sorted(experiments_dir.rglob("eval_results.json")):
        try:
            data = json.loads(eval_file.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            logger.warning("Skipping %s: %s", eval_file, exc)
            continue

        run_id = f"eval_{eval_file.parent.name}"
        entry = _empty_entry(run_id)
        entry.update({
            "timestamp": datetime.fromtimestamp(eval_file.stat().st_mtime).isoformat(),
            "source": "evaluate",
            "model_arch": data.get("model"),
            "llm_model": data.get("llm_model"),
            "macro_auc": data.get("macro_auc"),
            "best_val_auc": data.get("macro_auc"),
            "notes": data.get("notes", f"checkpoint: {data.get('checkpoint', '')}"),
        })
        entries.append(entry)

    return entries


# ---------------------------------------------------------------------------
# Harvest per-iteration metrics.json files
# ---------------------------------------------------------------------------

def _harvest_iter_metrics(experiments_dir: Path) -> List[Dict[str, Any]]:
    """Scan iter_*/metrics.json and return one entry per iteration."""
    entries = []
    for metrics_file in sorted(experiments_dir.glob("iter_*/metrics.json")):
        try:
            data = json.loads(metrics_file.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            logger.warning("Skipping %s: %s", metrics_file, exc)
            continue

        iter_name = metrics_file.parent.name
        run_id = f"iter_{iter_name}"

        # Check execution result for error rate
        exec_failed = False
        timed_out = False
        exec_file = metrics_file.parent / "execution.json"
        if exec_file.exists():
            try:
                exec_data = json.loads(exec_file.read_text(encoding="utf-8"))
                exec_failed = exec_data.get("returncode", 0) != 0
                timed_out = exec_data.get("timed_out", False)
            except json.JSONDecodeError:
                pass

        entry = _empty_entry(run_id)
        entry.update({
            "timestamp": datetime.fromtimestamp(metrics_file.stat().st_mtime).isoformat(),
            "source": "agent_iter",
            "model_arch": data.get("model"),
            "llm_model": data.get("llm_model"),
            "macro_auc": data.get("final_val_auc"),
            "best_val_auc": data.get("best_val_auc") or data.get("final_val_auc"),
            "final_val_auc": data.get("final_val_auc"),
            "training_time_s": data.get("training_time_s"),
            "exec_error_rate": 1.0 if exec_failed else 0.0,
            "timed_out": timed_out,
            "epochs": data.get("epochs"),
            "batch_size": data.get("batch_size"),
            "notes": f"iter: {iter_name}",
        })
        entries.append(entry)

    return entries


# ---------------------------------------------------------------------------
# Build / refresh the full benchmark table
# ---------------------------------------------------------------------------

def build_benchmark(experiments_dir: Path = EXPERIMENTS_DIR) -> List[Dict[str, Any]]:
    """
    Scan all experiment artefacts and merge them into the benchmark registry.

    Existing manually-registered entries are preserved; auto-discovered entries
    are added or updated.
    """
    registry = load_registry()

    # Auto-harvest from artefacts
    harvested: List[Dict[str, Any]] = []
    harvested.extend(_harvest_agent_state(experiments_dir))
    harvested.extend(_harvest_eval_results(experiments_dir))
    harvested.extend(_harvest_iter_metrics(experiments_dir))

    for entry in harvested:
        _upsert(registry, entry)

    # Sort: highest AUC first, then by run_id
    def _sort_key(e: Dict) -> tuple:
        auc = e.get("macro_auc") or e.get("best_val_auc") or 0.0
        return (-auc, e.get("run_id", ""))

    registry.sort(key=_sort_key)
    return registry


# ---------------------------------------------------------------------------
# Render Markdown benchmark table
# ---------------------------------------------------------------------------

_MD_COLS = [
    ("Run ID",           "run_id",           "l"),
    ("Source",           "source",           "l"),
    ("Model Arch",       "model_arch",       "l"),
    ("LLM",              "llm_model",        "l"),
    ("Best AUC",         "best_val_auc",     "r"),
    ("Final AUC",        "final_val_auc",    "r"),
    ("Err Rate",         "exec_error_rate",  "r"),
    ("Train Time (s)",   "training_time_s",  "r"),
    ("Epochs",           "epochs",           "r"),
    ("Notes",            "notes",            "l"),
]


def _fmt(value: Any, col_id: str) -> str:
    if value is None:
        return "—"
    if col_id in ("best_val_auc", "final_val_auc", "macro_auc"):
        try:
            return f"{float(value):.4f}"
        except (TypeError, ValueError):
            return str(value)
    if col_id == "exec_error_rate":
        try:
            return f"{float(value):.1%}"
        except (TypeError, ValueError):
            return str(value)
    return str(value)


def render_markdown(registry: List[Dict[str, Any]]) -> str:
    """Render the benchmark registry as a Markdown table string."""
    if not registry:
        return "_No benchmark entries yet. Run experiments and then `python benchmark.py`._\n"

    headers = [c[0] for c in _MD_COLS]
    col_ids = [c[1] for c in _MD_COLS]
    aligns = [c[2] for c in _MD_COLS]

    rows = [[_fmt(e.get(cid), cid) for cid in col_ids] for e in registry]

    # Compute column widths
    widths = [max(len(h), max((len(r[i]) for r in rows), default=0))
              for i, h in enumerate(headers)]

    def _row(cells: list[str]) -> str:
        parts = []
        for i, cell in enumerate(cells):
            if aligns[i] == "r":
                parts.append(cell.rjust(widths[i]))
            else:
                parts.append(cell.ljust(widths[i]))
        return "| " + " | ".join(parts) + " |"

    sep_row = "|" + "|".join(
        ("-" * (w + 2) if a == "l" else ("-" * w + ":" + "-").replace("-:", ":-", 1)
         for w, a in zip(widths, aligns))
    ) + "|"
    # Simpler separator
    sep_parts = []
    for w, a in zip(widths, aligns):
        if a == "r":
            sep_parts.append("-" * w + ":")
        else:
            sep_parts.append("-" * (w + 1))
    sep_row = "| " + " | ".join(sep_parts) + " |"

    lines = [
        "# BirdCLEF+ 2026 — Benchmark Table",
        "",
        f"_Last updated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}_",
        "",
        _row(headers),
        sep_row,
    ]
    for row in rows:
        lines.append(_row(row))

    lines += [
        "",
        "## Column definitions",
        "| Column | Description |",
        "|--------|-------------|",
        "| Best AUC | Highest macro ROC-AUC achieved across all epochs/iterations |",
        "| Final AUC | Macro ROC-AUC at the last epoch/iteration |",
        "| Err Rate | Fraction of code-generation iterations that exited non-zero |",
        "| Train Time | Wall-clock training time in seconds |",
        "",
        "> Metric: macro-averaged ROC-AUC skipping classes with no true-positive labels.",
    ]
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# LLM error-rate summary helper
# ---------------------------------------------------------------------------

def llm_error_summary(registry: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """
    Aggregate error rates and AUC scores per LLM model.

    Returns a dict keyed by LLM name with:
        runs, avg_auc, best_auc, avg_error_rate
    """
    from collections import defaultdict
    agg: Dict[str, Dict] = defaultdict(lambda: {
        "runs": 0,
        "aucs": [],
        "error_rates": [],
    })

    for e in registry:
        llm = e.get("llm_model") or "unknown"
        agg[llm]["runs"] += 1
        auc = e.get("best_val_auc") or e.get("macro_auc")
        if auc is not None:
            agg[llm]["aucs"].append(float(auc))
        err = e.get("exec_error_rate")
        if err is not None:
            agg[llm]["error_rates"].append(float(err))

    summary = {}
    for llm, data in sorted(agg.items()):
        aucs = data["aucs"]
        errs = data["error_rates"]
        summary[llm] = {
            "runs": data["runs"],
            "avg_auc": round(sum(aucs) / len(aucs), 4) if aucs else None,
            "best_auc": round(max(aucs), 4) if aucs else None,
            "avg_error_rate": round(sum(errs) / len(errs), 3) if errs else None,
        }
    return summary


# ---------------------------------------------------------------------------
# register sub-command
# ---------------------------------------------------------------------------

def cmd_register(args: argparse.Namespace) -> None:
    """Register a single run result into the benchmark registry."""
    registry = load_registry()
    entry = _empty_entry(args.run_id)
    entry.update({
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "source": "manual",
        "model_arch": args.model,
        "llm_model": args.llm,
        "macro_auc": args.macro_auc,
        "best_val_auc": args.macro_auc,
        "final_val_auc": args.final_auc or args.macro_auc,
        "training_time_s": args.training_time_s,
        "exec_error_rate": args.error_rate,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "notes": args.notes or "",
    })
    _upsert(registry, entry)

    def _sort_key(e: Dict) -> tuple:
        auc = e.get("macro_auc") or e.get("best_val_auc") or 0.0
        return (-auc, e.get("run_id", ""))

    registry.sort(key=_sort_key)
    save_registry(registry)
    md = render_markdown(registry)
    BENCHMARK_MD.write_text(md, encoding="utf-8")
    logger.info("Registered run '%s'. Benchmark table: %s", args.run_id, BENCHMARK_MD)


# ---------------------------------------------------------------------------
# show sub-command
# ---------------------------------------------------------------------------

def cmd_show(args: argparse.Namespace) -> None:
    """Print the current benchmark table to stdout."""
    if args.refresh:
        registry = build_benchmark()
        save_registry(registry)
        md = render_markdown(registry)
        BENCHMARK_MD.write_text(md, encoding="utf-8")
    else:
        registry = load_registry()

    if args.format == "json":
        print(json.dumps(registry, indent=2, default=str))
    elif args.format == "llm-summary":
        print(json.dumps(llm_error_summary(registry), indent=2, default=str))
    else:
        print(render_markdown(registry))


# ---------------------------------------------------------------------------
# Default (no sub-command): rebuild + print
# ---------------------------------------------------------------------------

def cmd_build(args: argparse.Namespace) -> None:
    """Rebuild benchmark from all experiment artefacts and write files."""
    registry = build_benchmark()
    save_registry(registry)
    md = render_markdown(registry)
    BENCHMARK_MD.write_text(md, encoding="utf-8")

    # Print to stdout
    print(md)

    # Also print LLM error summary
    summary = llm_error_summary(registry)
    if summary:
        print("\n## LLM Error-Rate Summary\n")
        print(f"{'LLM':<30} {'Runs':>5} {'Avg AUC':>10} {'Best AUC':>10} {'Avg Err%':>10}")
        print("-" * 65)
        for llm, s in summary.items():
            avg_auc = f"{s['avg_auc']:.4f}" if s["avg_auc"] is not None else "    —"
            best_auc = f"{s['best_auc']:.4f}" if s["best_auc"] is not None else "    —"
            avg_err = f"{s['avg_error_rate']:.1%}" if s["avg_error_rate"] is not None else "    —"
            print(f"{llm:<30} {s['runs']:>5} {avg_auc:>10} {best_auc:>10} {avg_err:>10}")

    logger.info("Benchmark written to %s and %s", BENCHMARK_JSON, BENCHMARK_MD)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="BirdCLEF+ 2026 Benchmark Table Generator",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command")

    # --- register ---
    reg = sub.add_parser("register", help="Register a single run result.")
    reg.add_argument("--run-id", required=True, help="Unique identifier for this run.")
    reg.add_argument("--model", default=None, help="Model architecture name.")
    reg.add_argument("--llm", default=None, help="LLM model used for code generation.")
    reg.add_argument("--macro-auc", type=float, default=None, help="Best macro ROC-AUC.")
    reg.add_argument("--final-auc", type=float, default=None, help="Final epoch macro AUC.")
    reg.add_argument("--training-time-s", type=float, default=None, help="Training time (s).")
    reg.add_argument("--error-rate", type=float, default=None,
                     help="Fraction of code-gen iterations that failed (0–1).")
    reg.add_argument("--epochs", type=int, default=None)
    reg.add_argument("--batch-size", type=int, default=None)
    reg.add_argument("--notes", default="", help="Free-text notes.")

    # --- show ---
    show = sub.add_parser("show", help="Print the benchmark table.")
    show.add_argument("--refresh", action="store_true",
                      help="Re-scan experiment artefacts before printing.")
    show.add_argument("--format", choices=["markdown", "json", "llm-summary"],
                      default="markdown", help="Output format.")

    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    if args.command == "register":
        cmd_register(args)
    elif args.command == "show":
        cmd_show(args)
    else:
        # Default: rebuild everything
        cmd_build(args)


if __name__ == "__main__":
    main()
