# QUICK START - VIDEO PRESENTATION GUIDE
## BirdCLEF+ 2026 Autonomous Research Agent

### 📋 What You're Creating
A **20-25 minute video** showing:
- ✅ The agent running autonomously
- ✅ Design solutions & architecture choices
- ✅ Challenges overcome & solutions
- ✅ Results: 96.08% AUC achieved after 202 iterations

### 🚀 Quick Start (5 steps)

#### Step 1: Prepare Environment (2 min)
```bash
source ~/.venv/bin/activate
bash setup_recording.sh
```

#### Step 2: Pre-record the Demo (10-15 min)
```bash
python agent_demo.py --mode demo 2>&1 | tee demo_run.log
```
This captures a 2-iteration run showing the agent working.

#### Step 3: Generate Results Report (1 min)
```bash
python results_reporter.py --output RESULTS_SUMMARY.md
```
Creates a comprehensive report for Part 5 of the video.

#### Step 4: Set Up Recording (5 min)
- Open OBS Studio / Camtasia / QuickTime
- Set resolution: 1920x1080
- Set frame rate: 30fps
- Terminal font: 16-18pt
- Clear background, good lighting for webcam

#### Step 5: Record (25 min)
Follow the **VIDEO_SCRIPT.md** narration script:
- Play back pre-recorded demo run OR run live
- Narrate using provided script
- Show code snippets when indicated
- Pause for visual emphasis

### 📊 Video Structure (Narration Flow)

```
0:00-0:30  │ TITLE CARD
           │ Team intro: "Carmen Bustorff Silva with Inês Martins & Francisca Menano"
           │
0:30-3:30  │ PROBLEM & CONTEXT
           │ • BirdCLEF+ 2026 challenge
           │ • 206 bird species, imbalanced dataset
           │ • Solution: Autonomous agent for architecture search
           │
3:30-15:00 │ DEMO - AGENT RUNNING (Main section!)
           │ Run: python agent_demo.py --mode demo
           │ Narrate the 7-step cycle:
           │   1. Data Exploration → Dataset summary
           │   2. Architecture Proposal → LLM suggestion
           │   3. Code Generation → Extract Python
           │   4. Sandboxed Execution → Train model
           │   5. Results Capture → Parse metrics
           │   6. LLM Analysis → Improvement suggestions
           │   7. Iteration → Loop continues
           │
15:00-19:00│ DESIGN SOLUTIONS (Show code snippets)
           │ • GPU Memory: AMP + GradScaler + cache clearing
           │ • Focal Loss: Handle class imbalance
           │ • SpecAugment: Frequency/time masking
           │ • Mixup: Label smoothing
           │ • Early Stopping: Prevent overfitting
           │
19:00-21:30│ CHALLENGES & SOLUTIONS
           │ • OOM errors → try/except with batch skipping
           │ • Bad LLM code → Fallback script
           │ • Class imbalance → ROC-AUC over valid classes
           │ • Slow convergence → Combined augmentation
           │
21:30-23:30│ RESULTS & METRICS
           │ Show: RESULTS_SUMMARY.md
           │ Highlight:
           │   • Best AUC: 96.08%
           │   • Iterations: 202
           │   • Success rate: 198/202
           │   • Best architecture: EfficientNet-B1
           │
23:30-25:00│ CONCLUSION
           │ Key takeaways:
           │   - Automation > manual tuning
           │   - Domain-specific augmentation matters
           │   - Transfer learning is powerful
           │   - AI agents can augment human research
           │
```

### 🎥 What People Will See During Demo

The agent will show:
```
════════════════════════════════════════════════════════════════════════════════
 BirdCLEF+ 2026 AUTONOMOUS RESEARCH AGENT
════════════════════════════════════════════════════════════════════════════════

████████████████████████████████████████████████████████████████████████████████
  ► INITIALIZING AGENT
████████████████████████████████████████████████████████████████████████████████

  Dataset:     ~/birdclef-data (206 bird species)
  GPU:         NVIDIA L4 (24GB VRAM)
  Torch:       PyTorch with automatic mixed precision
  Model:       EfficientNet-B1 with custom head
  Resuming from iteration: 202
  Current best AUC: 0.9608

════════════════════════════════════════════════════════════════════════════════
ITERATION 1 / 2  [iter_0203_20260519_161000]
════════════════════════════════════════════════════════════════════════════════

[INFO] Step 1: Exploring data…
[INFO] Found 7000 audio files across 206 species directories.
[INFO] Total duration estimate: 35000 seconds (~9.7 hours)

[INFO] Step 2: Requesting architecture proposal from LLM…
[LLM thinking for ~15 seconds...]

[INFO] Step 3: Extracting code from LLM response…
[INFO] Found 1 code block in LLM response.

[INFO] Step 4: Executing training script…

Epoch 1/40, Train Loss: 0.5231
Epoch 1/40, Validation AUC: 0.5843

[... training continues for ~6 minutes ...]

Epoch 25/40, Train Loss: 0.2341
Epoch 25/40, Validation AUC: 0.9456
--> Saved new best checkpoint with AUC: 0.9456

Early stopping triggered after 5 epochs without improvement.

[INFO] Step 5: Capturing results…
[INFO] Metrics: {'final_auc': 0.9456, 'best_auc': 0.9456}

[INFO] Step 6: Sending results to LLM for analysis…
[INFO] LLM analysis: "Excellent improvement! Consider..."

[INFO] Iteration 1 complete. Best AUC so far: 0.9456

════════════════════════════════════════════════════════════════════════════════
ITERATION 2 / 2  [iter_0204_20260519_161350]
════════════════════════════════════════════════════════════════════════════════
[... similar flow ...]
```

**Total demo runtime: ~12-15 minutes** (perfect for video segment)

### 💡 Narration Key Points (Practice These)

**When showing data exploration:**
> "First, the agent scans the dataset. We have 7,000 audio files across 206 bird species—that's about 34 examples per species on average. Very imbalanced."

**When LLM is thinking:**
> "Now the LLM is proposing an architecture. It has access to the previous results, so it can suggest strategic improvements. This takes about 15-20 seconds."

**When training starts:**
> "The agent generates Python training code directly from the LLM's proposal. This code uses EfficientNet-B1, Focal Loss for class imbalance, and SpecAugment for regularization."

**When metrics appear:**
> "Notice the validation AUC improved from 58% to 94%! This happened in just 25 epochs thanks to our augmentation strategy and Focal Loss."

**When best checkpoint is saved:**
> "Each time we find a better AUC, we save the model checkpoint. Over 202 iterations, the agent kept improving, eventually reaching 96.08%."

### 🎬 Recording Tips

**Before you hit record:**
- [ ] Tested microphone (no echo, 70-80dB audio level)
- [ ] Terminal font at 16-18pt
- [ ] OBS/Camtasia settings: 1920x1080, 30fps, H.264
- [ ] Closed all unnecessary apps
- [ ] Phone on silent
- [ ] Practiced narration 1-2 times

**During recording:**
- Speak clearly, moderate pace
- Pause 2-3 sec when code runs
- Don't apologize for pauses (just wait silently)
- Let audience read code on screen

**If you mess up:**
- Pause, take a breath, restart sentence
- Don't restart entire video
- Easy to fix in post-production

### 📈 Results You'll Show

From `RESULTS_SUMMARY.md`:
- **Top 5 Best Models** with AUC scores
- **AUC Evolution** (bar chart in text)
- **Architecture Distribution** (EfficientNet-B1 dominated)
- **Key Design Insights** (GPU optimization, loss functions, etc.)
- **Best Model Specs** (full code snippet of architecture)

### 🔗 Files You're Using

| File | Purpose |
|------|---------|
| `agent_demo.py` | Demo wrapper with formatted output |
| `results_reporter.py` | Generates results.md report |
| `VIDEO_SCRIPT.md` | **Main narration script** ← Follow this! |
| `setup_recording.sh` | Pre-recording checklist |
| `VIDEO_MATERIALS_README.md` | Comprehensive guide |

### ✅ Minimal Viable Video Path

**If you're in a hurry:**
1. `bash setup_recording.sh` (2 min)
2. `python agent_demo.py --mode demo 2>&1 | tee demo.log` (pre-record, 15 min)
3. Open `VIDEO_SCRIPT.md`
4. Record screen + narration (25 min)
5. Basic trim in iMovie/Premiere (30 min)
6. Upload to YouTube

**Total time: ~90 minutes for complete video**

### 🎯 What Makes This Video Special

- **Live demonstration** of an autonomous AI agent
- **Real metrics** (96.08% AUC is competitive!)
- **Technical depth** without being overwhelming
- **Reproducible** (anyone can run the agent)
- **Shows thought process** (why these design choices)

### 🚀 Advanced Options

- Add real-time metrics graph showing AUC over iterations
- Record voice separately for cleaner audio
- Use motion graphics for the 7-step cycle
- Show LLM prompts/responses as they happen
- Include comparison: manual tuning vs. autonomous agent

---

## Next Steps

1. **Read:** `VIDEO_SCRIPT.md` (full script with timing)
2. **Prepare:** Run `bash setup_recording.sh`
3. **Record:** Follow script while running agent_demo.py
4. **Edit:** Use Premiere/Final Cut/DaVinci
5. **Publish:** YouTube + GitHub

**Questions?** Check `VIDEO_MATERIALS_README.md` for detailed info.

---

**Happy recording! 🎬📊**
