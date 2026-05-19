# ATPA\_CIF — BirdCLEF+ 2026 Autonomous Research Agent

> **Advanced Topics in Predictive Analytics**
> *Carmen Bustorff Silva, Inês Martins, Francisca Menano*

An AI-powered **autonomous research agent** for the
[BirdCLEF+ 2026](https://www.kaggle.com/competitions/birdclef-2026) audio
classification task (Track B). The agent acts as a *"machine that builds
machines"*, iterating on deep learning models without human intervention by
combining a local LLM (via Ollama) with an automated experiment loop.

---

## Repository Structure

```
ATPA_CIF/
├── agent.py                 # Core 7-step autonomous agent loop
├── llm_client.py            # Ollama API client (model-agnostic)
├── preprocessing.py         # Audio → mel-spectrogram templates
├── models.py                # 2D CNN & transfer-learning scaffolds
├── generate_submission.py   # Kaggle submission notebook generator
├── requirements.txt         # Python dependencies
├── experiments/             # Auto-created: logs, code, metrics per iteration
└── README.md
```

---

## Setup

### 1 — Python environment

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2 — Install and start Ollama

Ollama provides a local OpenAI-compatible API for open-weight LLMs.

```bash
# macOS / Linux
curl -fsSL https://ollama.ai/install.sh | sh

# Pull a supported model (choose one)
ollama pull gemma3          # Google Gemma 3 — recommended
ollama pull qwen2.5-coder   # Qwen 2.5 Coder — strong code generation
ollama pull llama3          # Meta LLaMA 3

# Start the server (keep this terminal open, or run as a daemon)
ollama serve
```

The agent connects to `http://127.0.0.1:11434` by default.

### 3 — (Optional) Download the BirdCLEF+ 2026 dataset

```bash
# Requires the Kaggle CLI and accepted competition rules
pip install kaggle
kaggle competitions download -c birdclef-2026
```

---

## Quick Start — Single Command

```bash
# Run 5 iterations of the autonomous agent (synthetic data if no dataset)
python agent.py --iterations 5 --model gemma3

# With a real dataset
python agent.py --iterations 10 --model qwen2.5-coder --data-dir /path/to/birdclef-2026

# Generate a Kaggle submission notebook from the best trained model
python generate_submission.py --output submission_notebook.ipynb
```

---

## Agent Loop — 7-Step Cycle

| Step | Name | Description |
|------|------|-------------|
| 1 | **Data Exploration** | Scans the audio dataset and computes statistics |
| 2 | **Architecture Proposal** | LLM proposes a new or improved model architecture |
| 3 | **Code Generation** | Extracts runnable Python from the LLM response |
| 4 | **Sandboxed Execution** | Runs the generated script in a subprocess (with timeout) |
| 5 | **Results Capture** | Parses metrics from stdout / `metrics.json` |
| 6 | **LLM Analysis** | LLM analyses results and suggests improvements |
| 7 | **Iteration** | Updates state and repeats from Step 1 |

All artefacts (LLM proposals, generated code, metrics, analysis) are saved
under `experiments/<iteration_id>/`.

---

## Modules

### `llm_client.py`

Model-agnostic Ollama client. Supports any model available locally.

```python
from llm_client import LLMClient
llm = LLMClient(model="gemma3")
print(llm.propose_architecture(context="..."))
```

### `preprocessing.py`

Audio-to-mel-spectrogram conversion using **librosa** or **torchaudio**.

```python
from preprocessing import file_to_melspec
spec = file_to_melspec("bird_call.ogg", backend="librosa", n_mels=64, duration=5.0)
# spec.shape → (64, 216)
```

### `models.py`

Ready-to-train model scaffolds for PyTorch.

```python
from models import build_simple_cnn_torch, build_efficientnet_torch
model = build_simple_cnn_torch()                    # fast baseline
model = build_efficientnet_torch(freeze_base=True) # transfer learning
```

Available models:

| Name | Description |
|------|-------------|
| `simple_cnn_torch` | Lightweight 3-block CNN |
| `efficientnet_torch` | EfficientNetB0 + ImageNet weights |

### `generate_submission.py`

Generates a Jupyter notebook that runs CPU-only inference within 90 minutes.

```bash
python generate_submission.py --output submission_notebook.ipynb
```

---

## Technical Constraints

- **Framework**: PyTorch (GPU-optimized with CUDA support)
- **Small-scale first**: default 5 s clips, 128-mel spectrograms
- **Multi-label**: 206 species, BCEWithLogitsLoss (raw logits output)
- **Kaggle budget**: ≤ 90 minutes CPU-only inference

---

## Experiments Directory

Each agent iteration creates a subdirectory under `experiments/`:

```
experiments/
├── agent_state.json            # Persistent state (best AUC, history)
└── iter_0001_20260416_120000/
    ├── dataset_summary.json    # Step 1 output
    ├── llm_proposal.txt        # Step 2 raw LLM response
    ├── train.py                # Step 3 generated training script
    ├── execution.json          # Step 4+5 stdout/stderr + metrics
    └── llm_analysis.txt        # Step 6 LLM analysis
```

---

## Dependencies

See `requirements.txt`. Key packages:

```
torch, torchaudio, torchvision, librosa, numpy, pandas,
scikit-learn, ollama, kaggle, jupyter, nbformat
```

all done!