# BirdCLEF+ 2026 Submission Guide

## Recommended Workflow

1. Package your trained checkpoint as a Kaggle dataset. The notebook expects the weights to be available under `/kaggle/input/<dataset-name>/`.
2. Regenerate the submission notebook locally:

```bash
cd /home/carme/ATPA_CIF
source .venv/bin/activate
python3 generate_pytorch_submission.py \
  --weights-dataset birdclef-pytorch-weights \
  --weights-filename model.pt \
  --output submission_related/submission_pytorch.ipynb
```

3. On Kaggle, create a notebook with two data sources attached:
   - The competition dataset that contains `sample_submission.csv` and `test_soundscapes`
   - Your model dataset that contains `model.pt`
4. Run the notebook to generate `submission.csv`, then submit that file.

## What The Generator Produces

The generated notebook is intentionally minimal:

- It discovers the competition input folder at runtime instead of hard-coding a dataset name.
- It loads the saved PyTorch state dict from the attached model dataset.
- It uses the repo’s training-time audio settings: 32 kHz, 128 mel bins, 5-second chunks, and the 2-conv BirdCLEF model.
- It writes a Kaggle-ready `submission.csv` with rows aligned to `sample_submission.csv`.

## Files To Keep

- `generate_pytorch_submission.py` is the source of truth for the notebook template.
- `submission_pytorch.ipynb` is the canonical submission notebook to upload or paste into Kaggle.

## Troubleshooting

- If the notebook cannot find the weights, check the model dataset name and the filename you passed to the generator.
- If the notebook cannot find the competition input, make sure the BirdCLEF competition dataset is attached in Kaggle.
- If inference is slow, reduce `BATCH_SIZE` in the notebook cell that runs prediction.
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

