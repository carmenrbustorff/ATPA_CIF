CRITICAL BUGFIXES FOR agent.py - INTEGRATION GUIDE
====================================================

PROBLEMS FIXED:
===============
1. ✗ Row aggregation by file    → ✓ One row per 5-second chunk
2. ✗ Washed-out probabilities  → ✓ Proper 206→234 class mapping  
3. ✗ Object dtypes in CSV       → ✓ Explicit float32 casting
4. ✗ Missing local class list   → ✓ Extract from data_loader

INTEGRATION STEPS:
==================

## Step 1: Replace KAGGLE_INFERENCE_TEMPLATE (Lines 69-665 approx)

Locate the current KAGGLE_INFERENCE_TEMPLATE in agent.py (starts at line 69).
Delete everything from line 69 to the closing triple-quotes around line 661.

Then copy the ENTIRE template from AGENT_FIXES.py starting with:
    KAGGLE_INFERENCE_TEMPLATE = '''\
    """
    Auto-generated Kaggle inference script...
    
And paste it into agent.py at line 69.

**KEY CHANGES IN TEMPLATE:**
- Row ID format: `{filename_stem}_{end_time_seconds}` (e.g., "BC2026_Train_0001_S08_20250606_030007_5")
- NO aggregation by file - EVERY 5-second chunk is a separate row
- Two injected variables:
  * {INJECTED_LOCAL_CLASSES} → List of 206 species names like ["amakihil", "amtkif", ...]
  * {INJECTED_MODEL_CLASSES} → Model class definitions
- Proper mapping from 206 local classes to 234 Kaggle classes
- All float columns explicitly cast to np.float32 before saving


## Step 2: Ensure extract_local_classes() is in agent.py

The function extract_local_classes(data_dir: Optional[Path]) should be present.
It's already added if you ran the earlier command, starting around line 669-693.

If NOT present, add it before extract_model_classes():

    def extract_local_classes(data_dir: Optional[Path] = None) -> list[str]:
        """Extract 206 local species names from train.csv."""
        if data_dir is None or not data_dir.exists():
            logger.warning("Using placeholder class list")
            return [f"species_{i:03d}" for i in range(206)]
        
        train_csv = data_dir / "train.csv"
        if not train_csv.exists():
            return [f"species_{i:03d}" for i in range(206)]
        
        try:
            import pandas as pd
            df = pd.read_csv(train_csv)
            species = sorted(df["primary_label"].unique().tolist())
            logger.info("Extracted %d local species", len(species))
            return species
        except Exception as e:
            logger.warning("Failed to extract species: %s", e)
            return [f"species_{i:03d}" for i in range(206)]


## Step 3: Update propose_and_generate_code() Function

Locate the propose_and_generate_code() function (starts around line 855).

KEY CHANGES NEEDED:
a) Add data_dir parameter to function signature:

    def propose_and_generate_code(
        llm: LLMClient,
        task_context: str,
        previous_results: Optional[str],
        iteration_dir: Path,
        data_dir: Optional[Path] = None,  # ADD THIS
    ) -> Path:

b) At the END of the function (after writing train.py), replace the inference.py generation block:

    # OLD CODE (DELETE):
    inference_script = KAGGLE_INFERENCE_TEMPLATE.format(
        INJECTED_MODEL_CLASSES=model_classes,
        NUM_LOCAL_CLASSES=NUM_SPECIES,
    )

    # NEW CODE (REPLACE WITH):
    local_classes = extract_local_classes(data_dir)
    logger.info("Extracted %d local species", len(local_classes))
    
    inference_script = KAGGLE_INFERENCE_TEMPLATE.format(
        INJECTED_MODEL_CLASSES=model_classes,
        INJECTED_LOCAL_CLASSES=repr(local_classes),  # Critical: use repr()
    )

c) The rest of the function stays the same.


## Step 4: Update run_agent() Function

Locate run_agent() (around line 916).

In the main loop where propose_and_generate_code() is called (around line 993), pass data_dir:

    # OLD:
    script_path = propose_and_generate_code(
        llm, task_context, previous_results_text, iteration_dir
    )

    # NEW:
    script_path = propose_and_generate_code(
        llm, task_context, previous_results_text, iteration_dir, data_dir
    )


## VALIDATION CHECKLIST:
=======================

After making changes, verify:

1. agent.py compiles without syntax errors:
   python3 -m py_compile agent.py

2. Template has both placeholders:
   grep "INJECTED_LOCAL_CLASSES\|INJECTED_MODEL_CLASSES" agent.py

3. extract_local_classes() is defined:
   grep -n "def extract_local_classes" agent.py

4. propose_and_generate_code() calls extract_local_classes():
   grep -A 2 "local_classes = extract_local_classes" agent.py

5. Inference template generates one row per chunk (not aggregated):
   grep -c "row_id = f" agent.py  # Should be 1 (the format string in template)
   grep -c "end_time_seconds" agent.py  # Should be 3+


## WHAT THE FIX DOES:
======================

BEFORE (broken):
- File "BC2026_Train_0001.ogg" (10 minutes) → 1 row in submission.csv
- Aggregates all chunks with max pooling
- Probability per species becomes 1/234 (washed out)
- Loses temporal information about which 5-sec chunks are confident

AFTER (fixed):
- File "BC2026_Train_0001.ogg" (10 minutes, 120 chunks) → 120 rows in submission.csv
- Each row_id: "BC2026_Train_0001_5", "BC2026_Train_0001_10", ..., "BC2026_Train_0001_600"
- Row_id_N corresponds to the Nth 5-second chunk
- Each chunk has independent predictions (206 local → 234 Kaggle classes)
- Kaggle can assign per-chunk predictions, not just per-file aggregates
- Proper float32 dtype throughout


## TESTING:
===========

After integration, test with:

1. Regenerate inference for an existing iteration:
   .venv/bin/python regenerate_inference.py \
       --iteration-dir ./experiments/iterations_0201-0250/iter_0200_20260518_092346

2. Test inference locally:
   .venv/bin/python generate_local_submission.py \
       --iteration-dir ./experiments/iterations_0201-0250/iter_0200_20260518_092346 \
       --test-audio-dir /mnt/disks/data/birdclef/train_soundscapes \
       --output-dir ./submissions

3. Verify submission.csv format:
   head -5 submissions/submission.csv  # Check row_id format
   wc -l submissions/submission.csv    # Should be high (1 row per chunk)
   file submissions/submission.csv     # Should be text/CSV
   python3 -c "import pandas as pd; df = pd.read_csv('submissions/submission.csv'); \
       print(f'Shape: {df.shape}'); \
       print(f'Dtypes: {df.dtypes.value_counts()}'); \
       print(f'NaNs: {df.isna().sum().sum()}')"


## REGENERATE + APPLY FIXES TO OLD ITERATIONS:
===============================================

Once you've updated agent.py, regenerate inference.py for all previous iterations:

    for iter_dir in experiments/iterations_*/iter_*; do
        echo "Regenerating: $iter_dir"
        .venv/bin/python regenerate_inference.py --iteration-dir "$iter_dir"
    done

Then test the best iteration:

    BEST_ITER=$(jq -r '.best_iteration' experiments/agent_state.json)
    .venv/bin/python generate_local_submission.py \
        --iteration-dir "./experiments/iterations_*/$BEST_ITER" \
        --test-audio-dir /mnt/disks/data/birdclef/train_soundscapes \
        --output-dir ./best_submission
    
    cat best_submission/validation_report.json


QUESTIONS?
==========
- Row format: See line ~215 in new template: row_id = f"{file_stem}_{end_time_seconds}"
- Class mapping: See lines ~239-244 in new template: if kaggle_col in LOCAL_CLASSES
- Type casting: See lines ~251-254 in new template: final_df[col].astype(np.float32)
