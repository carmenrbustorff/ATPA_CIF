QUICKSTART: Apply Critical Fixes in 5 Minutes
==============================================

FILES CREATED:
- AGENT_FIXES.py        ← Copy the corrected code from here
- INTEGRATION_GUIDE.md  ← Detailed step-by-step instructions
- BEFORE_AFTER.md       ← Before/after comparison of all changes

TLDR SUMMARY:
=============

Problem: Kaggle submission.csv is broken because:
  1. Rows aggregated by file instead of per 5-sec chunk
  2. Classes incorrectly mapped (index-based instead of name-based)
  3. Probabilities washed out or object type instead of float32

Solution: 3 changes to agent.py:

  1. Replace KAGGLE_INFERENCE_TEMPLATE (lines 69-661)
     → Use new template from AGENT_FIXES.py
  
  2. Ensure extract_local_classes() exists
     → Should be lines 669-693 (already added in earlier step)
  
  3. Update propose_and_generate_code() function
     → Add data_dir parameter
     → Pass data_dir to extract_local_classes()
     → Pass extracted local classes to template


EXACT CHANGES:
==============

1. REPLACE TEMPLATE
───────────────────
   In agent.py, delete everything from line 69 to ~661
   Paste entire KAGGLE_INFERENCE_TEMPLATE from AGENT_FIXES.py


2. ADD DATA_DIR PARAMETER
────────────────────────
   Line ~855, change:
   
   def propose_and_generate_code(
       llm: LLMClient,
       task_context: str,
       previous_results: Optional[str],
       iteration_dir: Path,
       data_dir: Optional[Path] = None,  # ← ADD THIS
   ) -> Path:


3. EXTRACT & INJECT LOCAL CLASSES
──────────────────────────────────
   Line ~982, change:
   
   OLD:
   inference_script = KAGGLE_INFERENCE_TEMPLATE.format(
       INJECTED_MODEL_CLASSES=model_classes,
       NUM_LOCAL_CLASSES=NUM_SPECIES,
   )
   
   NEW:
   local_classes = extract_local_classes(data_dir)
   logger.info("Extracted %d local species", len(local_classes))
   
   inference_script = KAGGLE_INFERENCE_TEMPLATE.format(
       INJECTED_MODEL_CLASSES=model_classes,
       INJECTED_LOCAL_CLASSES=repr(local_classes),
   )


4. PASS DATA_DIR TO FUNCTION
────────────────────────────
   Line ~993 (in run_agent), change:
   
   OLD:
   script_path = propose_and_generate_code(
       llm, task_context, previous_results_text, iteration_dir
   )
   
   NEW:
   script_path = propose_and_generate_code(
       llm, task_context, previous_results_text, iteration_dir, data_dir
   )


VERIFY:
=======

   python3 -m py_compile agent.py


TEST:
=====

   .venv/bin/python regenerate_inference.py \
       --iteration-dir ./experiments/iterations_0201-0250/iter_0200_20260518_092346
   
   .venv/bin/python generate_local_submission.py \
       --iteration-dir ./experiments/iterations_0201-0250/iter_0200_20260518_092346 \
       --test-audio-dir /mnt/disks/data/birdclef/train_soundscapes \
       --output-dir ./test_submission
   
   # Check: should have thousands of rows (one per chunk)
   wc -l test_submission/submission.csv


EXPECTED RESULT:
================

submission.csv now has:
  ✓ 8400+ rows (vs 700 before)
  ✓ Row IDs like "BC2026_Train_0001_S08_20250606_030007_5"
  ✓ One row per 5-second chunk
  ✓ All values float32 (not object)
  ✓ Species properly mapped by name
  ✓ Untraced species filled with 0.0
  ✓ Kaggle-ready format


REFERENCE:
==========

See INTEGRATION_GUIDE.md for full step-by-step with line numbers
See BEFORE_AFTER.md for detailed before/after comparison
See AGENT_FIXES.py for complete corrected code
