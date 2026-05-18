
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
    The Step 6 analysis actually ran. The 15-second pause and the 120-second timeout fixes worked perfectly. The agent successfully woke up, analyzed the crashed code, and saved its thoughts. The pipeline is fully whole.
Iteration 85: The model achieved a final_train_loss of 0.02 but an AUC of 0.50. It completely memorized the 5,000 training samples (loss near zero) but failed to learn anything real, resulting in random guessing on the validation set.
Iteration 86: The agent read the analysis, likely applied the dropout or augmentation constraints you added to the prompt, and tried again. This time, the training loss stopped at 0.3699 (meaning the model was forced to generalize instead of memorize), and the validation AUC jumped to a new best of 0.5629.

Ready to increase sample size further, but will need to monitor GPU memory and training time closely. Next steps:
    increase iterations per run
    increase sample size further (e.g., 10,000 or 20,000) if GPU can handle it

Iteration 87: Increased training samples to 15,000: took 6 minuntes to run each iteration with 6 epochs, batch size1 6. AUC= .501

To decrease messiness and save VM space, created 50 iteration long buckets and deleted all models except the best one

each sucesfull iteration is taking abou 10 minutes to run
iteration 94:  Timeouts:   The latest experiment timed out after 600 seconds (10 minutes) -> ran 10 more iteratations and if problem persists, will need to investigate further and potentially optimize code or increase timeout limit.

Iteration 95: new max AUC: 0.76!  Script completed in 469.9 s., took 10 minutes for the whole run. Was also the first of the cycle (1/10)

Added data augmentation to data loader and to agent task so it is able to perform it
Model is severly overfitting, increase sample size to 35,000 for training and 15,000 for evaluation, epochs to 20
After including data augmentation to the pipeline the model repeatedly hit issues related to file corruption, unsupported formats, or missing files.
Issues presist after fixing data loader: file handling and auc scoring are the main faults
iteration 152: the agent is now hallucinating some non real issues (related to training time)

DRAMA: From Iteration 96 onward, almost every run shows auc: 0.0 and metrics: {}.
This means the PyTorch script train.py crashed before it even reached the end of the script to save the dictionary.

Root cause of the "DRAMA" identified: A cascade of fatal pipeline exceptions was crashing train.py before the evaluation loop could calculate or save the JSON metrics.
Data Loader Overhaul (The OGG Apocalypse): Discovered native torchaudio throws System errors when reading .ogg files on this Linux OS. Replaced backend with soundfile and implemented a recursive try/except failsafe so the DataLoader silently swaps out corrupted files instead of passing blank tensors or crashing.
Architecture & AMP Stabilization: Fixed an AttributeError caused by PyTorch's Subset wrapper hiding the num_classes variable by enforcing a global NUM_SPECIES constant. Resolved a Mixed Precision (AMP) RuntimeError by stripping the final Sigmoid layer and enforcing BCEWithLogitsLoss to stabilize the FP16 math.
Prompt Engineering (Breaking the Echo Chamber): The LLM got stuck in a confirmation bias loop, feeding its own broken code and path variables (causing an IsADirectoryError) back into subsequent iterations. Injected strict "Anti-Laziness" and "Self-Correction Override" constraints into the agent.py prompt to force the agent to ignore its flawed logic and explicitly use kwargs.
AMP API Deprecation: The pipeline crashed due to TypeError and FutureWarning errors related to Mixed Precision. The agent hallucinated a mix of PyTorch 1.x and 2.x syntax (torch.cuda.amp.autocast("cuda")). Forced a hard migration to the torch.amp V2 API.
Context Truncation: Discovered the LLM was failing to generate the validation loops not because of logic errors, but because it was exhausting its output token limit explaining its own code. Implemented a "Prioritize Code over Analysis" constraint to ensure the full train.py script writes to disk.

Iteration 161+ (Strategic Pivot: Transfer Learning & Neural Scaling)
Architecture Overhaul: Re-aligned the agent's objective with the project's advanced track requirements by deprecating from-scratch CNN generation. Enforced a strict transfer learning constraint, mandating the agent use the timm library to deploy pre-trained vision models (e.g., EfficientNet) optimized for 2D Mel-spectrograms. This offloads feature-extraction learning and significantly raises the baseline AUC.s
Applying Neural Scaling Laws: Addressed the massive CPU/Compute bottlenecks caused by processing the full dataset. Enforced a strict subsetting protocol (5,000 training samples) to allow the autonomous loop to execute rapid, low-cost experimentation. Heavy compute and full datasets will be reserved solely for the exploitation phase once a winning architecture is identified.

Iteration 161+ (YAMNet Integration & Hardware Scaling)
Feature Extraction Pivot: Replaced the custom CNN approach with a frozen audio-specific pre-trained model (YAMNet). This aligns with the course's recommended transfer learning strategy. By freezing the base model, the architecture focuses exclusively on training a lightweight, multi-label classification head for the 206 Pantanal species.
Batch Sizing & Scaling: The shift to a frozen feature extractor drastically reduced GPU VRAM consumption. Scaled the batch_size up to 128 to accelerate epoch times. Maintained a 5,000-sample subset strictly for architectural debugging (ensuring embedding tensors properly connect to the dense layers), with the intent to scale back to the full 35k dataset once the pipeline is proven mathematically stable.

the agent is getting incredibly lazy so pivoting slgihtly, will add a more developed template into the task constraints for it to follow

iteration 174: The agent has finally overcome its "laziness" and learned how to write the complete pipeline but the execution hit the 1000-second (16.6 minute) timeout limit.
    forcing the agent to iterate faster by slashing the number of epochs during this exploratory phase, and give it a slightly larger timeout buffer just in case.
175- Finally tracked metrics again! metrics.json: {'final_train_loss': 0.039597701948881146, 'final_auc': 0.5043372783298752, 'num_params': 4387850, 'epochs_trained': 3, 'batch_size': 128, 'training_samples': 5000, 'eval_samples': 1000}
    UndefinedMetricWarning: Only one class is present in y_true. ROC AUC score is not defined in that case.
        the agent needs to explicitly tell scikit-learn to skip calculating AUC for species that do not appear in that specific 1000-sample validation slice.
            Because the target matrix contains 206 distinct bird species, a single audio clip will have a 1 for the target species and a 0 for the other 205 species. This means 99.5% of the dataset targets are zeros.
            If the model is blind to the spectrogram features (which happens when the pre-trained ImageNet backbone is frozen and cannot interpret audio data), the optimizer takes the easiest path to minimize loss: it learns to output a large negative number for every single class. By predicting a flat 0% probability across the board, it guesses 99.5% of the targets correctly. The Binary Cross-Entropy loss drops to a flawless 0.03, but because the model isn't actually separating classes, validation AUC lands exactly at 0.50 (random chance).
        integrated a clean, working version of the ReduceLROnPlateau scheduler directly into your skeleton template
Fixed double one-hot encoding bug: The dataloader's `BCEWithLogitsLoss` configuration natively yields `[batch_size, 206]` multi-hot arrays. Applying a manual integer enumeration loop over these arrays mangled the validation target matrix, resulting in corrupted metrics. Removed the redundant manual encoding step.
    Silenced static analysis artifacts:   Injected explicit type casting (`int()`) and `# type: ignore` directives directly into the base `TASK_CONTEXT_TEMPLATE` to suppress false-positive Pylance typing warnings regarding `torch.amp.autocast`, `GradScaler`, and PyTorch dataset `__len__` methods.
    Resolved orchestrator timeout limitation:   Scaling up `max_epochs` to 20 to accommodate SpecAugment learning requirements triggered a hardcoded 2000-second subprocess limit in `agent.py`, resulting in a `TimeoutExpired` kill mid-execution. The timeout threshold must be extended (e.g., to 3600 seconds) to safely allow complete multi-epoch iteration cycles.
    Addressed `ValueError: Found array with 0 feature(s)` in AUC calculation:   The validation script crashed because the rigid `y_true_one_hot == 1` boolean mask dropped all columns due to floating-point formatting (`1.0`) from the dataloader outputs.
  Implemented a bulletproof metric mask: Replaced the strict equality check with `(col_sums > 0) & (col_sums < len(y_true_one_hot))`. This guarantees the scikit-learn ROC-AUC function only evaluates columns containing both positive and negative signals, bypassing precision quirks and preventing division-by-zero crashes.
  Integrated a `try/except` failsafe defaulting to 0.5000 to ensure continuous iteration and guarantee that `metrics.json` serialization never blocks the pipeline.

Iteration 188: - The dataset was successfully initialized with 5000 training samples and 1000 validation samples. The EfficientNet-B0 model was loaded correctly, including the conversion of input convolutions from 3 to 1 channel. The training loss decreased steadily over 20 epochs, indicating that the model was learning effectively.

Moved Validation Inline: Shifted the evaluation pass directly into the core epoch training loop. This provides immediate, real-time tracking of validation AUC and convergence trends rather than waiting for the entire multi-epoch run to conclude.
Implemented Best-Model Checkpointing: Added a tracking mechanism that updates and saves model.pt dynamically only when a new peak validation AUC is achieved, protecting the best weights from late-stage overfitting or representation collapse.
Expanded Training Volumetric Capacity: Adjusted the hardcoded limits in data_loader.py (MAX_TRAIN raised from 5,000 to 15,000; MAX_VAL from 1,000 to 3,000). This triples the available signal, scaling average representation to ~72 samples per species to help the model learn features under SpecAugment masking.
Calibrated Loss Function Weights: Relaxed the positive class penalty factor (pos_weight) from 50.0 down to 15.0 within BCEWithLogitsLoss. This maintains a structural penalty against the lazy "all-zeros" prediction shortcut while preventing the optimizer from over-correcting into a flat positive bias.
Excised Metric Array Regression: Removed an incorrect np.eye conversion that treated multi-hot validation matrices as single-label integer sequences. Restored the robust matrix slicing logic to directly evaluate the multi-hot array using the (col_sums > 0) & (col_sums < len(y_true_one_hot)) mask, avoiding silent exceptions and guaranteeing valid macro-averaged AUC scores.
Optimized Hardware Pipeline Throughput: Maintained an optimized execution layout with batch_size = 128 to fully saturate parallel processing pipelines on the Tesla V100 and limit CPU queuing cycles during audio feature decoding.

Validation AUC Debug: 
    Script execution went dark with high CPU usage; no real-time telemetry. Train loss decreased smoothly, but Validation Macro AUC froze at exactly 0.5000.Swapped subprocess.run(capture_output=True) for subprocess.Popen with a line-by-line reader and injected the -u unbuffered flag to force live logging. Set num_workers=4 and pin_memory=True in the training template to parallelize audio CPU loading, dropping epoch times from over an hour to ~2 minutes.
    The Exception Mask: A strict instruction forcing multi_class='ovr' broke scikit-learn on multi-label 2D arrays. A generic try/except block caught the ValueError silently, forcing a fallback score of 0.5000.
    __getitem__ was returning raw scalar class integers (e.g., 150) instead of float arrays.
    Fed raw integers to BCEWithLogitsLoss, which evaluated them as target weights of 150.0 instead of a target probability of 1.0. The model collapsed into predicting uniform probabilities (flat base rate), yielding random-guessing performance (0.5000 AUC).Training: Escaped this because a custom mixup_collate_fn performed a hidden conversion layer that masked the issue during the forward pass.

Iteration 2: The Regularization Penalty
    Objective: Mitigate the severe domain shift observed between the local validation AUC (0.9329) and Kaggle hidden test set AUC (0.653) from Iteration 1.Architecture: efficientnet_b1 (1-channel, scaled from B0) with a 512-dimensional custom classification head.
    Loss Function: Focal Loss ($\gamma=2.0$, $\alpha=0.25$) to penalize errors on minority classes.
    Augmentation Strategy: SpecAugment (Time/Frequency masking) + Gaussian Noise Injection ($\sigma=0.03$) applied directly to the log-mel spectrograms.
    Results:
        Validation AUC: 0.8171 (Epoch 20)
        Leaderboard AUC: 0.600
    Core Insights:
      The Flaw of Spectral Noise: Injecting synthetic white noise directly onto a spectrogram matrix does not simulate acoustic environmental noise (like wind or rain). Instead, it acts as "spectral pixel noise" (like static on a television screen), obliterating the micro-harmonics of the bird calls. We successfully regularized the model, but we blinded it in the process.
      Focal Loss Behavior: As expected, Focal Loss dropped the baseline training loss to microscopic levels (~0.0017) rapidly, proving it successfully downweighted the easily classified negative background targets.
      Engineering & Environment Hardening
        Kaggle Black-Box Quirks Solved:The Internet Lockout: Kaggle's scoring containers disable internet access. Model initialization must explicitly set pretrained=False (e.g., in timm), otherwise, the pipeline crashes trying to fetch Hugging Face weights.
        Hidden Dataset Traps: Kaggle occasionally hides the training metadata CSVs during the scoring run, which kills scripts expecting them. We implemented a robust try/except blast chamber wrapping the entire main() function to prevent the kernel from dying.
        OOM Prevention: Hidden soundscapes can be up to two hours long. To avoid the Linux OOM (Out-of-Memory) killer, the pipeline was refactored to chunk audio on the CPU, only passing a single 5-second tensor to the GPU at a time.
        The Pandas Left-Join: Kaggle's rigid metric throws an unhandled exception if the output CSV rows don't perfectly match the hidden index. A Pandas Left-Join against the dummy sample_submission.csv guarantees strict compliance, padding missing species classes with 0.0.
        Architectural Alignment:A discrepancy between the local agent.py checkpoint and the Kaggle inference notebook will trigger a size mismatch crash. When the local agent scales up the linear head (e.g., from 128 to 512) or the backbone (B0 to B1), the notebook script's class definition must be manually updated to mirror it perfectly before loading the state_dict.

Updated Task context to explore more fruitfull models. 
    By explicitly stating that deviation causes a "fatal script crash" and framing the backbone as a downstream VRAM limitation, the LLM is much less likely to "helpfully" upgrade the model size.
