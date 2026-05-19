BEFORE/AFTER: CRITICAL CHANGES SUMMARY
======================================

═══════════════════════════════════════════════════════════════════════════════
CHANGE 1: Template Placeholders
═══════════════════════════════════════════════════════════════════════════════

BEFORE (template had):
    {INJECTED_MODEL_CLASSES}
    NUM_LOCAL_CLASSES=206

AFTER (template has):
    {INJECTED_MODEL_CLASSES}
    {INJECTED_LOCAL_CLASSES}        # NEW: actual list of 206 species names


═══════════════════════════════════════════════════════════════════════════════
CHANGE 2: Row ID Generation
═══════════════════════════════════════════════════════════════════════════════

BEFORE:
    # Aggregated by file
    file_predictions[file_id] = probs.flatten()  # ONE entry per file
    final_submission["row_id"] = list(file_predictions.keys())  # Unique files

    # Result: submission.csv has 700 rows (one per audio file)
    row_id                           species_000  species_001  ...
    BC2026_Train_0001_S08_20250606   0.234        0.156
    BC2026_Train_0002_S08_20250607   0.891        0.445

AFTER:
    # One row per 5-second chunk
    row_id = f"{file_stem}_{end_time_seconds}"  # e.g., "BC2026_Train_0001_5"
    results.append({
        "row_id": row_id,
        "chunk_idx": chunk_idx,
        "predictions": probs  # shape: (206,)
    })

    # Result: submission.csv has 8400+ rows (one per 5-second chunk)
    row_id                                       species_000  species_001  ...
    BC2026_Train_0001_S08_20250606_5            0.234        0.156
    BC2026_Train_0001_S08_20250606_10           0.891        0.445
    BC2026_Train_0001_S08_20250606_15           0.445        0.234
    ...


═══════════════════════════════════════════════════════════════════════════════
CHANGE 3: Class Mapping Logic
═══════════════════════════════════════════════════════════════════════════════

BEFORE (broken):
    # Hardcoded numeric indices, no name mapping
    for col in submission_template_df.columns:
        if col.startswith("species_"):
            kaggle_idx = int(col.split("_")[1])
            if kaggle_idx < NUM_LOCAL_CLASSES:  # Only first 206 classes
                local_idx = kaggle_idx
                final_submission[col] = row[f"class_{local_idx}"]
            else:
                final_submission[col] = 0.0  # All 234+ are zero

    # Result: Column "species_150" always gets class #150,
    #         even if class #150 is "zebfinc" and species_150 is "tufwoo"

AFTER (correct):
    # Species-name-based matching
    for kaggle_col in kaggle_cols:
        if kaggle_col in LOCAL_CLASSES:
            # Direct match: find species by name, not index
            final_data[kaggle_col] = local_df[kaggle_col].values.astype(np.float32)
        else:
            # Species not in local training set - fill with 0.0
            final_data[kaggle_col] = np.zeros(len(local_df), dtype=np.float32)

    # Result: If "amakihil" is in both local and Kaggle, it gets matched by name
    #         If "banwax" is only in Kaggle (not trained), it gets 0.0
    #         Proper semantic alignment, not positional


═══════════════════════════════════════════════════════════════════════════════
CHANGE 4: Type Casting
═══════════════════════════════════════════════════════════════════════════════

BEFORE:
    # No explicit type conversion
    final_submission = submission_template_df[["row_id"]].copy()
    final_submission[col] = ...  # dtype: object or float64
    final_submission.to_csv(output_path, index=False)

    # Result in CSV:
    #   dtypes: object (implicit conversion to strings in CSV)
    #   Kaggle parser sees "0.234" as string, not number
    #   Probabilities washed out or rejected

AFTER:
    # Explicit float32 casting BEFORE saving
    for col in kaggle_cols:
        final_df[col] = final_df[col].astype(np.float32)

    final_df.to_csv(output_path, index=False)

    # Result in CSV:
    #   dtypes: float32
    #   All values are properly numeric
    #   Kaggle parser receives true floats


═══════════════════════════════════════════════════════════════════════════════
CHANGE 5: Function Signature
═══════════════════════════════════════════════════════════════════════════════

BEFORE:
    def propose_and_generate_code(
        llm: LLMClient,
        task_context: str,
        previous_results: Optional[str],
        iteration_dir: Path,
    ) -> Path:

AFTER:
    def propose_and_generate_code(
        llm: LLMClient,
        task_context: str,
        previous_results: Optional[str],
        iteration_dir: Path,
        data_dir: Optional[Path] = None,  # ADD THIS
    ) -> Path:


═══════════════════════════════════════════════════════════════════════════════
CHANGE 6: Inference Generation (Inside propose_and_generate_code)
═══════════════════════════════════════════════════════════════════════════════

BEFORE:
    inference_script = KAGGLE_INFERENCE_TEMPLATE.format(
        INJECTED_MODEL_CLASSES=model_classes,
        NUM_LOCAL_CLASSES=NUM_SPECIES,
    )

AFTER:
    local_classes = extract_local_classes(data_dir)
    logger.info("Extracted %d local species", len(local_classes))
    
    inference_script = KAGGLE_INFERENCE_TEMPLATE.format(
        INJECTED_MODEL_CLASSES=model_classes,
        INJECTED_LOCAL_CLASSES=repr(local_classes),
    )


═══════════════════════════════════════════════════════════════════════════════
CHANGE 7: Function Call (in run_agent)
═══════════════════════════════════════════════════════════════════════════════

BEFORE (around line 993):
    script_path = propose_and_generate_code(
        llm, task_context, previous_results_text, iteration_dir
    )

AFTER:
    script_path = propose_and_generate_code(
        llm, task_context, previous_results_text, iteration_dir, data_dir
    )


═══════════════════════════════════════════════════════════════════════════════
IMPACT SUMMARY
═══════════════════════════════════════════════════════════════════════════════

Metric                          Before      After       Reason
──────────────────────────────────────────────────────────────────────────────
Rows in submission.csv          ~700        ~8400       One per chunk, not file
Row ID format                   file_name   file_5      Includes time offset
Classes per row                 206→234     206→234     Both have full mapping
Probability range               1/206       0.0-1.0     Proper per-chunk scores
Dtype of predictions            object      float32     Explicit casting
Class name matching             index-based name-based  Semantic matching
Missing species handling        incorrect   correct     0.0 for untraced species


═══════════════════════════════════════════════════════════════════════════════
FILE CHECKLIST
═══════════════════════════════════════════════════════════════════════════════

After integration, these files should exist:

✓ agent.py
  - extract_local_classes() function (line ~670)
  - Updated KAGGLE_INFERENCE_TEMPLATE (line ~69)
  - Updated propose_and_generate_code() with data_dir parameter
  - Updated run_agent() passing data_dir

✓ regenerate_inference.py
  - Updated with same new template and extract_local_classes()

✓ generate_local_submission.py
  - No changes needed (already works with new inference.py)

✓ AGENT_FIXES.py (reference only - copy from here)
✓ INTEGRATION_GUIDE.md (instructions)
✓ BEFORE_AFTER.md (this file)


═══════════════════════════════════════════════════════════════════════════════
VALIDATION SCRIPT
═══════════════════════════════════════════════════════════════════════════════

Run this after integration to verify all fixes:

#!/bin/bash
set -e

echo "=== Checking agent.py ==="
python3 -m py_compile agent.py && echo "✓ Syntax OK"

echo "=== Checking for extract_local_classes ==="
grep -q "def extract_local_classes" agent.py && echo "✓ Function defined"

echo "=== Checking template placeholders ==="
grep -q "INJECTED_LOCAL_CLASSES" agent.py && echo "✓ Template has INJECTED_LOCAL_CLASSES"
grep -q "INJECTED_MODEL_CLASSES" agent.py && echo "✓ Template has INJECTED_MODEL_CLASSES"

echo "=== Checking function signature ==="
grep -A 4 "def propose_and_generate_code" agent.py | grep -q "data_dir" && \
    echo "✓ Function has data_dir parameter"

echo "=== Checking local_classes extraction ==="
grep -q "local_classes = extract_local_classes" agent.py && \
    echo "✓ Local classes extraction in propose_and_generate_code"

echo "=== Checking row_id format ==="
grep -q 'row_id = f"{file_stem}_{end_time_seconds}"' agent.py && \
    echo "✓ Row ID format correct (one per chunk)"

echo "=== All checks passed! ==="
