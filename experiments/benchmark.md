# BirdCLEF+ 2026 — Benchmark Table

_Last updated: 2026-04-30 15:46 UTC_

| Run ID       | Source | Model Arch         | LLM            | Best AUC | Final AUC | Err Rate | Train Time (s) | Epochs | Notes                         |
| ------------- | ------- | ------------------- | --------------- | --------: | ---------: | --------: | --------------: | ------: | ------------------------------ |
| test_run_002 | manual | efficientnet_torch | deepseek-r1:8b |   0.7891 |    0.7891 |     0.0% |          580.0 |      5 | EfficientNet test entry       |
| test_run_001 | manual | simple_cnn_torch   | deepseek-r1:8b |   0.7123 |    0.7123 |    10.0% |          310.0 |      5 | Baseline SimpleCNN test entry |
| test_run_003 | manual | simple_cnn_torch   | qwen2.5-coder  |   0.6540 |    0.6540 |    20.0% |          220.0 |      3 | qwen2.5-coder agent run       |

## Column definitions
| Column | Description |
|--------|-------------|
| Best AUC | Highest macro ROC-AUC achieved across all epochs/iterations |
| Final AUC | Macro ROC-AUC at the last epoch/iteration |
| Err Rate | Fraction of code-generation iterations that exited non-zero |
| Train Time | Wall-clock training time in seconds |

> Metric: macro-averaged ROC-AUC skipping classes with no true-positive labels.
