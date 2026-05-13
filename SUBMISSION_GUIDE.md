# BirdCLEF+ 2026 Submission Guide

## Quick Start: Generate & Upload Submission

### Step 1: Generate Submission Notebook (Local)

```bash
cd /home/carme/ATPA_CIF
source .venv/bin/activate
python3 generate_pytorch_submission.py \
    --model experiments/iter_0049_20260513_145457/model.pt \
    --output submission_pytorch.ipynb
```

This creates `submission_pytorch.ipynb` with:
- ✅ PyTorch model inference code
- ✅ Audio preprocessing (librosa)
- ✅ Batch inference on test_soundscapes
- ✅ Submission CSV generation
- ✅ Runtime budget tracking (90 min)

### Step 2: Upload to Kaggle Notebook

1. Go to [BirdCLEF+ 2026 Competition](https://www.kaggle.com/competitions/birdclef-2026)
2. Create a **New Notebook**
3. Upload `submission_pytorch.ipynb`
4. Set data sources:
   - `birdclef-2026` (main dataset)
   - `birdclef-pytorch-weights` (your model folder)
5. Run all cells
6. Submit `submission.csv`

### Step 3: Monitor Execution

The notebook will:
- Print runtime remaining every 10 files
- Show species predictions (234 columns)
- Generate CSV with format: `row_id, species_1, species_2, ..., species_234`

---

## What's in the Submission?

### Input Data
- **Test soundscapes:** ~50-500 individual audio files (variable length)
- **Preprocessing:** 5-second chunks → mel-spectrograms (128×216)
- **Format:** OGG/WAV audio at 22,050 Hz

### Model Architecture
```
Input (1, 128, 216)
    ↓
Conv2d(1, 32) + BatchNorm + ReLU
    ↓
Conv2d(32, 64) + BatchNorm + ReLU
    ↓
AdaptiveAvgPool2d(1,1)
    ↓
Flatten → Linear(64, 128) → ReLU
    ↓
Linear(128, 234) → Sigmoid
    ↓
Output: 234 probability scores [0,1]
```

### Output Format
```csv
row_id,Acadian Flycatcher,Alder Flycatcher,American Avocet,...
soundscape_12345_5,0.001,0.002,0.023,...
soundscape_12345_10,0.015,0.003,0.101,...
soundscape_12346_5,0.023,0.001,0.034,...
```

**One row per 5-second chunk** from each soundscape file.

---

## Model Specifications

| Parameter | Value |
|-----------|-------|
| Architecture | CNN (2 conv layers) |
| Input shape | (1, 128, 216) mel-spectrogram |
| Output | 234 sigmoid probabilities |
| Parameters | ~135,000 |
| Inference device | CPU (Kaggle) |
| Training data | 320 samples (10 batches × 32 samples) |
| Best validation AUC | 0.5292 |

---

## Troubleshooting

### Problem: "ModuleNotFoundError: torch"
**Solution:** Kaggle notebooks have PyTorch pre-installed. If you see this locally, run:
```bash
pip install torch torchvision torchaudio
```

### Problem: "Model weights not found"
**Solution:** 
1. Upload `model.pt` to a Kaggle dataset
2. Update the notebook's `MODEL_WEIGHTS` path
3. Or upload model weights via dataset: `/kaggle/input/birdclef-pytorch-weights/best_model.pt`

### Problem: "CUDA out of memory"
**Solution:** The notebook is CPU-only (BATCH_SIZE=4). If needed, reduce batch size:
```python
BATCH_SIZE = 2  # More conservative
```

### Problem: "Approaching time limit, stopping"
**Explanation:** Kaggle enforces 90-minute time limit. This is expected if:
- Many test files (>1000)
- Large audio files (>1 min each)
- Network latency

The notebook will submit partial results—still valid!

---

## Advanced: Use Your Own Best Model

To use a different iteration's model:

1. **Find best model:**
```bash
cat experiments/agent_state.json | python3 -m json.tool | grep -A 2 "best_iteration"
```

2. **Copy the model:**
```bash
cp experiments/iter_XXXX_TIMESTAMP/model.pt best_model.pt
```

3. **Generate new notebook:**
```bash
python3 generate_pytorch_submission.py --model best_model.pt
```

4. **Upload both files to Kaggle dataset**, then update notebook paths

---

## Expected Results

With the current model (Iteration 49, AUC 0.5292):
- **Random baseline:** AUC 0.5 (non-informative predictions)
- **Expected score:** ~0.53-0.55 (slight improvement over baseline)
- **Full ensemble (10 models):** ~0.60-0.65 (if properly implemented)

---

## Next Steps After Submission

1. **Run more iterations** (20-50) to improve model quality
2. **Implement ensemble** of top 5 models (average predictions)
3. **Retrain on full dataset** instead of 320 samples
4. **Add regularization** (data augmentation, dropout)
5. **Hyperparameter tuning** based on convergence trends

---

**Ready to submit?** Run the notebook on Kaggle! 🚀

