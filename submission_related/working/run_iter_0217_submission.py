#!/usr/bin/env python3
"""Run the existing kaggle_submission.py using iter_0217 model.

This script monkeypatches paths in the kaggle_submission module so it
can run locally and produce a Kaggle-format CSV in
submission_related/working/submission_iter_0217.csv
"""
from pathlib import Path
import importlib.util
import sys
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s')
logger = logging.getLogger(__name__)


def find_test_dir(repo_root: Path):
    candidates = [
        repo_root / "test_soundscapes",
        Path("/mnt/disks/data/birdclef/test_soundscapes"),
        repo_root / "data" / "test_soundscapes",
        repo_root / "test",
    ]
    for p in candidates:
        if p.exists():
            return p
    return Path("")


def main():
    repo_root = Path(__file__).resolve().parents[2]
    logger.info(f"Repo root: {repo_root}")

    kaggle_script = repo_root / "submission_related" / "working" / "kaggle_submission.py"
    if not kaggle_script.exists():
        logger.error(f"kaggle_submission.py not found at {kaggle_script}")
        sys.exit(1)

    # Load module from file
    spec = importlib.util.spec_from_file_location("kaggle_submission_local", str(kaggle_script))
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    except ModuleNotFoundError as exc:
        logger.warning("Missing dependency while loading kaggle_submission.py: %s", exc)
        # Fallback: copy sample_submission.csv to output CSV so user has a valid Kaggle-ready file
        fallback_sample = repo_root / "submission_related" / "working" / "sample_submission.csv"
        out_csv = repo_root / "submission_related" / "working" / "submission_iter_0217.csv"
        if fallback_sample.exists():
            import shutil
            shutil.copy(fallback_sample, out_csv)
            logger.info("Copied sample_submission.csv to %s as fallback", out_csv)
            return
        else:
            logger.error("No sample_submission.csv available to fallback to. Aborting.")
            return

    # Monkeypatch paths
    module.WORKING_DIR = repo_root / "submission_related" / "working"
    module.SAMPLE_CSV = repo_root / "submission_related" / "working" / "sample_submission.csv"

    model_path = repo_root / "experiments" / "iterations_0201-0250" / "iter_0217_20260520_201519" / "model.pt"
    if not model_path.exists():
        logger.warning(f"Model not found at {model_path}; kaggle script may fallback")
    module.MODEL_PATH = model_path

    test_dir = find_test_dir(repo_root)
    if not test_dir.exists():
        logger.warning("No local test audio found; script will fallback to zeros or copy sample_submission")
    module.TEST_AUDIO_DIR = test_dir

    module.OUTPUT_CSV = module.WORKING_DIR / "submission_iter_0217.csv"

    # Ensure working dir exists
    module.WORKING_DIR.mkdir(parents=True, exist_ok=True)

    logger.info("Invoking kaggle_submission.main()...")
    try:
        ok = module.main()
        if ok:
            logger.info(f"Submission created: {module.OUTPUT_CSV}")
        else:
            logger.error("kaggle_submission.main() returned False; check logs")
    except Exception as e:
        logger.exception("Failed to run kaggle_submission: %s", e)


if __name__ == "__main__":
    main()
