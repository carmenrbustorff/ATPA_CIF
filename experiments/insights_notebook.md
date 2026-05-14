
After 48 unsucessfull iterations, got AUC to positive territory! iteration 49 AUC: 0.5292.
generated submission notebook with best model weights, and uploaded to Kaggle dataset to have it available for submission. 
troubleshot submission issues, and documented solutions in the notebook for future reference.
after more troubleshooting, submitted successfully with the current model, and documented expected results and next steps for improvement.- score:0.47
increased sample size that was mistakenly hardcoded to 320, to 5000 for training and 1000 for evaluation, and retrained model - iteration 77
The scale-up to 5,000 samples was successful. The system did not crash, the GPU handled the memory load perfectly, and it processed the data in about 6 minutes (366.12 seconds).
    However, the training loop has a critical bug that requires immediate fixing.The Critical Bug: Epoch 2, Loss: 0.0
        fixed batch size error - data was being exhausted in first batch
        The GPU memory didn't crash. The script correctly processed all 6 epochs.It completed everything in about 4 minutes.
        but the LLM was unavailable for anaylysis step still -> likely ollama sleeping time
    