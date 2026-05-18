"""
Local Submission Generator – Test dynamically generated inference.py scripts.

This utility validates inference.py before submitting to Kaggle by:
1. Loading a generated inference.py module
2. Running inference on a local test audio directory
3. Validating the output CSV format (234 columns, no NaNs)

Usage:
    python generate_local_submission.py \\
        --iteration-dir /path/to/iteration/folder \\
        --test-audio-dir /path/to/test/audio \\
        --output-dir ./submissions/
"""

import argparse
import importlib.util
import json
import logging
import sys
import traceback
from pathlib import Path

import pandas as pd
import numpy as np

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
# Configuration
# ---------------------------------------------------------------------------

NUM_KAGGLE_CLASSES = 234
REQUIRED_COLUMNS = 234  # Excluding row_id


# ---------------------------------------------------------------------------
# Helper: Dynamically load inference.py
# ---------------------------------------------------------------------------

def load_inference_module(inference_path: Path):
    """
    Dynamically import inference.py from a given path.

    Returns the loaded module or None if loading fails.
    """
    if not inference_path.exists():
        logger.error("Inference script not found: %s", inference_path)
        return None

    try:
        spec = importlib.util.spec_from_file_location("inference", inference_path)
        if spec is None or spec.loader is None:
            logger.error("Could not create import spec for: %s", inference_path)
            return None

        module = importlib.util.module_from_spec(spec)
        sys.modules["inference"] = module
        spec.loader.exec_module(module)

        logger.info("Successfully loaded inference module from: %s", inference_path)
        return module

    except Exception as e:
        logger.error("Failed to load inference module: %s", e)
        traceback.print_exc()
        return None


# ---------------------------------------------------------------------------
# Helper: Run inference and validate output
# ---------------------------------------------------------------------------

def run_inference(
    iteration_dir: Path,
    test_audio_dir: Path,
    output_dir: Path,
) -> tuple[bool, str]:
    """
    Run the dynamically loaded inference.py and validate the output.

    Returns: (success: bool, message: str)
    """
    # Verify required files exist
    inference_path = iteration_dir / "inference.py"
    model_path = iteration_dir / "best_model.pth"

    if not inference_path.exists():
        msg = f"inference.py not found in {iteration_dir}"
        logger.error(msg)
        return False, msg

    if not model_path.exists():
        logger.warning(
            "best_model.pth not found. Checking for model.pt instead…"
        )
        model_path = iteration_dir / "model.pt"
        if not model_path.exists():
            msg = f"No model checkpoint found in {iteration_dir}"
            logger.error(msg)
            return False, msg

    if not test_audio_dir.exists():
        msg = f"Test audio directory not found: {test_audio_dir}"
        logger.error(msg)
        return False, msg

    # Load inference module
    inference_module = load_inference_module(inference_path)
    if inference_module is None:
        msg = "Failed to load inference module"
        logger.error(msg)
        return False, msg

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Run inference
    logger.info("Running inference on test audio directory: %s", test_audio_dir)
    output_csv = output_dir / "submission.csv"

    try:
        # Call the inference module's main function
        if hasattr(inference_module, "generate_submission_csv"):
            logger.info("Calling generate_submission_csv()…")
            submission_df = inference_module.generate_submission_csv(
                test_audio_dir=str(test_audio_dir),
                model_path=str(model_path),
                output_csv=str(output_csv),
            )
        else:
            msg = "inference.py does not have generate_submission_csv() function"
            logger.error(msg)
            return False, msg

    except Exception as e:
        logger.error("Inference execution failed: %s", e)
        traceback.print_exc()
        return False, f"Inference execution failed: {str(e)}"

    logger.info("Inference completed. Output saved to: %s", output_csv)
    return True, "Inference completed successfully"


# ---------------------------------------------------------------------------
# Helper: Validate submission CSV
# ---------------------------------------------------------------------------

def validate_submission(csv_path: Path) -> tuple[bool, dict]:
    """
    Validate the generated submission CSV.

    Checks:
    - File exists
    - Exactly 235 columns (1 row_id + 234 species)
    - First column is 'row_id'
    - No NaN values
    - All numeric columns are float/int

    Returns: (is_valid: bool, report: dict)
    """
    report = {
        "file_exists": False,
        "shape": None,
        "columns": 0,
        "first_column": None,
        "has_nan": False,
        "nan_count": 0,
        "errors": [],
        "warnings": [],
    }

    if not csv_path.exists():
        report["errors"].append(f"CSV file not found: {csv_path}")
        return False, report

    report["file_exists"] = True

    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        report["errors"].append(f"Failed to parse CSV: {str(e)}")
        return False, report

    report["shape"] = tuple(df.shape)
    report["columns"] = df.shape[1]
    report["first_column"] = df.columns[0] if len(df.columns) > 0 else None

    # Validate column count
    if df.shape[1] != NUM_KAGGLE_CLASSES + 1:
        report["errors"].append(
            f"Expected {NUM_KAGGLE_CLASSES + 1} columns (1 row_id + {NUM_KAGGLE_CLASSES} species), "
            f"got {df.shape[1]}"
        )

    # Validate first column name
    if report["first_column"] != "row_id":
        report["errors"].append(
            f"First column must be 'row_id', got '{report['first_column']}'"
        )

    # Check for NaN values
    nan_count = df.isna().sum().sum()
    report["has_nan"] = nan_count > 0
    report["nan_count"] = int(nan_count)

    if report["has_nan"]:
        report["errors"].append(
            f"Found {nan_count} NaN values in CSV. All values must be numeric."
        )

    # Validate numeric columns (all except row_id should be numeric)
    for col in df.columns[1:]:
        try:
            if not pd.api.types.is_numeric_dtype(df[col]):
                report["warnings"].append(
                    f"Column '{col}' is not numeric: {df[col].dtype}"
                )
        except Exception as e:
            report["warnings"].append(f"Could not validate column '{col}': {str(e)}")

    is_valid = len(report["errors"]) == 0
    return is_valid, report


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Generate and validate local Kaggle submissions",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--iteration-dir",
        type=Path,
        required=True,
        help="Path to iteration folder containing inference.py and best_model.pth",
    )
    parser.add_argument(
        "--test-audio-dir",
        type=Path,
        required=True,
        help="Path to directory containing test audio files (.ogg, .wav, .flac, .mp3)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("./submissions"),
        help="Path to save submission.csv (defaults to ./submissions/)",
    )

    args = parser.parse_args()

    logger.info("=" * 70)
    logger.info("LOCAL SUBMISSION GENERATOR")
    logger.info("=" * 70)
    logger.info("Iteration directory: %s", args.iteration_dir.resolve())
    logger.info("Test audio directory: %s", args.test_audio_dir.resolve())
    logger.info("Output directory: %s", args.output_dir.resolve())
    logger.info("=" * 70)

    # Run inference
    success, msg = run_inference(args.iteration_dir, args.test_audio_dir, args.output_dir)

    if not success:
        logger.error("Inference failed: %s", msg)
        sys.exit(1)

    logger.info("Inference succeeded. Validating output…")

    # Validate output CSV
    csv_path = args.output_dir / "submission.csv"
    is_valid, report = validate_submission(csv_path)

    logger.info("=" * 70)
    logger.info("VALIDATION REPORT")
    logger.info("=" * 70)
    logger.info("File exists: %s", report["file_exists"])
    logger.info("Shape: %s", report["shape"])
    logger.info("Columns: %d (expected %d)", report["columns"], NUM_KAGGLE_CLASSES + 1)
    logger.info("First column: %s (expected 'row_id')", report["first_column"])
    logger.info("NaN values: %d (expected 0)", report["nan_count"])

    if report["warnings"]:
        logger.warning("Warnings:")
        for w in report["warnings"]:
            logger.warning("  - %s", w)

    if report["errors"]:
        logger.error("Errors:")
        for e in report["errors"]:
            logger.error("  - %s", e)

    # Save validation report
    report_path = args.output_dir / "validation_report.json"
    report["is_valid"] = is_valid
    report_path.write_text(json.dumps(report, indent=2, default=str))
    logger.info("Validation report saved to: %s", report_path)

    logger.info("=" * 70)

    if is_valid:
        logger.info("✓ SUBMISSION IS VALID AND READY FOR KAGGLE")
        logger.info("CSV location: %s", csv_path)
        sys.exit(0)
    else:
        logger.error("✗ SUBMISSION HAS VALIDATION ERRORS")
        logger.error("Please fix the issues above and try again.")
        sys.exit(1)


if __name__ == "__main__":
    main()
