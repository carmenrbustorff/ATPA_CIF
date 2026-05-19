# BirdCLEF 2026 — Experiment Results

Tracking all training runs for benchmarking and report comparison.
Each run gets its own entry. The "Comparison Table" at the bottom
summarizes everything for quick reference.

---

## Run 1 — Manual SimpleCNN Baseline

**Date:** 2026-05-09
**Branch:** feature/train-script-clean
**Author:** Francisca

### Configuration
| Parameter | Value |
|---|---|
| Model | `simple_cnn_torch` (from `models.py`) |
| Pretrained weights | None (from scratch) |
| Epochs | 5 |
| Batch size | 32 |
| Learning rate | 1e-3 |
| Optimizer | Adam |
| Loss | BCEWithLogitsLoss (multi-label) |
| Val split | 20% (random) |
| Augmentation | None |
| Hardware | NVIDIA L4 (cloud) — driver 550.163.01, CUDA 12.4 |
| DataLoader | `num_workers=3, pin_memory=True`, persistent_workers=False |

Per-Epoch Metrics
| Epoch | Val AUC (macro) | Time/epoch |
|---|---|---|
| 1 | 0.5239 | ~30 min |
| 2 | 0.5168 | ~30 min |
| 3 | 0.5080 | ~30 min |
| 4 | 0.5166 | ~30 min |
| 5 | 0.4997 | ~30 min |

*Note: Train/val loss not currently logged in baseline.py — only val AUC. Future runs (with train.py) will log all three.*

### Summary
- **Best val AUC:** 0.5239 (epoch 1)
- **Best epoch:** 1
- **Final epoch AUC:** 0.4997 (below random)
- **Total training time:** ~2.5 h
- **GPU peak memory:** ~1.7 GB (well below L4's 23 GB)
- **Saved checkpoint:** `baseline_model.pt`

### Observations & Interpretation
- AUC fluctuated between 0.4997 and 0.5239 across all 5 epochs — essentially random performance (0.50). No upward learning trend.
- **The best epoch (1) was already the best**. Subsequent epochs showed no improvement, and Epoch 5 actually dropped slightly below random — strong evidence the model learned almost nothing meaningful.
- **Why this is the expected baseline result:**
  - SimpleCNN trained from scratch (no pretrained weights) → no head start from ImageNet features
  - 5 epochs is far too few for a small model to learn 206-way classification
  - No data augmentation → model overfits to specific time crops
  - Severe class imbalance (35,549 samples / 206 classes ≈ 173 per class on average, but heavily skewed)
  - Multi-label loss (BCE) applied to single-label data → mismatched signal
- **This is the central finding of Run 1:** naive from-scratch training fails. This motivates Run 2 (EfficientNet with transfer learning), Run 3 (augmentation), etc.

### Engineering challenges encountered
- TorchCodec library failure on the VM (FFmpeg dependency missing). Patched `data_loader.py` to use librosa instead — committed as part of this branch.
- DataLoader was running single-threaded by default → 2 hr/epoch initially. Fixed by setting `num_workers=3, pin_memory=True` → ~30 min/epoch (~4× speedup, but still data-loading-bottlenecked since librosa is single-threaded per worker).
- L4 GPU utilization stayed near 0% with sporadic bursts to 30%. Confirmed via `nvidia-smi` that the bottleneck was audio decoding on CPU, not GPU compute. For future runs: enable `persistent_workers=True` to skip respawn overhead each epoch (~5 min savings per epoch).
- Cloud capacity issues (us-west1-a unavailability) cost ~3 hours of downtime over the project so far.

- **Loss function bug discovered after Run 1:** baseline.py used `BCEWithLogitsLoss` on outputs that the model had already passed through sigmoid internally — equivalent to applying sigmoid twice, which weakens the gradient signal. Fixed to `BCELoss` for future runs. Run 1 numbers (AUC ~0.52) are reported as-is for transparency, but the bug likely contributed to the flat learning curve. Run 2 onwards uses correct loss math.
---

## Run 2 - Autonomous Agent (EfficientNet)

**Date:** 2026-05-19
**Agent:** constrained autonomous agent (LLM = `qwen2.5-coder:14b`)

### Winning configuration (Iteration 5)
| Parameter | Value |
|---|---|
| Model | `efficientnet_torch` |
| Epochs | 7 |
| Learning rate | 0.0001 |
| Batch size | 32 |
| Augmentation | True (random-crop) |

### Results
- **Best val AUC:** 0.6061 (massive improvement over the 0.52 SimpleCNN baseline)
- **Best epoch:** 7
- **Run ID:** agent_qwen2.5-coder:14b_20260519_132905

### Summary
- The autonomous constrained agent found a strong EfficientNet configuration that achieved **AUC=0.6061**, substantially outperforming Run 1 (SimpleCNN, AUC ~0.52).
- Pre-caching mel-spectrograms as `.npy` files reduced per-epoch overhead — epoch training time dropped to ~46 seconds, which allowed the agent to complete 5 full autonomous experiments in under 30 minutes.


---

## Comparison Table (across all runs)
| Run | Model | Pretrained | Epochs | Best Val AUC | Time | Notes |
|---|---|---|---|---|---|---|
| 1 | SimpleCNN | No | 5 | 0.5239 (so far) | ~2.5 h | Floor measurement |
| 2 | EfficientNet-B0 | ImageNet | 5 | TBD | TBD | Frozen base, then unfreeze top 20 |
| 3 | Agent (Carmen, unconstrained) | varies | 15 iters | TBD | TBD | LLM = qwen3-coder |
| 4 | Agent (constrained, your variant) | varies | 15 iters | TBD | TBD | LLM = qwen3-coder, fixed scaffold |

*Empty rows kept as planning placeholders.*

---

## LLM Benchmark Table (for the agent runs)

Tracks how different LLMs perform as the agent's "brain" — measuring not just final AUC but also the **skip rate** the professor flagged.

| LLM | Iterations | Successful | Skipped (bad code) | Skip Rate | Best AUC |
|---|---|---|---|---|---|
| qwen3-coder | TBD | — | — | — | — |
| nemotron | TBD | — | — | — | — |
| gemma3 | TBD | — | — | — | — |

---

## Notes for the Final Report

Concepts from the course this project applies to deep learning evaluation:

- **Validation split** (preventing test-set leakage)
- **Loss curves** (train vs val to detect overfitting / underfitting)
- **Macro vs micro averaging** for multi-class AUC
- **Class imbalance handling** (only 206 of 234 species in train data)
- **Transfer learning vs from-scratch** (Run 1 vs Run 2 comparison)
- **Hyperparameter sensitivity** (how does LR affect convergence?)

Open questions to investigate:
- Why does Run 1 plateau so quickly? (Likely: model capacity, no augmentation, no class balancing)
- Does the agent (constrained) outperform the agent (unconstrained) at low iteration counts?
- Which LLM produces the most reliably-executable code?