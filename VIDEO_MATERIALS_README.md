# BirdCLEF+ 2026 Video Presentation Materials

This directory contains all materials needed to create the video presentation showing the autonomous research agent and its results.

## Files Overview

### 1. **VIDEO_SCRIPT.md** (Main Resource)
Complete narration script for the video presentation with:
- Full narration for each scene (8 scenes total)
- Exact terminal commands to run
- Expected output screenshots
- Timing guidance for each section
- Production tips and post-processing advice
- Alternative shortened 10-minute version

**Use this as your primary guide.** Follow the scenes sequentially and use the provided narration.

### 2. **agent_demo.py** (Demo Wrapper)
Enhanced version of `agent.py` optimized for video recording with:
- Colored, formatted console output
- Section headers with progress indicators
- Summary statistics displays
- Two modes:
  - `--mode demo`: Run 2-3 iterations with formatted output
  - `--mode summary`: Display experiment statistics only

**Example usage:**
```bash
python agent_demo.py --mode demo
python agent_demo.py --mode summary
```

### 3. **results_reporter.py** (Report Generator)
Generates comprehensive markdown reports from experiment history:
- Top 10 best models with AUCs
- AUC evolution over iterations
- Architecture distribution
- Key design insights
- Challenge solutions
- Best model specifications

**Example usage:**
```bash
python results_reporter.py --output results_report.md
```

### 4. **setup_recording.sh** (Pre-Recording Checklist)
Automated setup script that:
- Verifies Python environment
- Checks GPU availability
- Confirms LLM client is ready
- Generates results report
- Displays experiment statistics
- Provides recording recommendations

**Run before recording:**
```bash
bash setup_recording.sh
```

---

## Video Recording Workflow

### Phase 1: Preparation (10 minutes)
```bash
# 1. Activate environment
source ~/.venv/bin/activate

# 2. Run setup script
bash setup_recording.sh

# 3. Generate results report
python results_reporter.py --output RESULTS_SUMMARY.md

# 4. Test terminal visibility
echo "Test text - make sure font is readable"
```

### Phase 2: Recording (20-25 minutes total)

**Option A: Pre-record + Narrate (Recommended)**
```bash
# 1. Pre-record the demo run to a log file
python agent_demo.py --mode demo 2>&1 | tee pre_recorded_run.log

# 2. In your video editor, play back log with narration overlay
# This gives you perfect timing and editing flexibility
```

**Option B: Live Recording**
```bash
# Record screen while running:
python agent_demo.py --mode demo

# Narrate in real-time (requires practice)
```

### Phase 3: Post-Production (varies)

Recommended editing software and tasks:
- **Final Cut Pro / Premiere Pro / DaVinci Resolve:**
  - Trim long pauses (waiting for GPU)
  - Add title cards between sections
  - Insert graphics/charts from RESULTS_SUMMARY.md
  - Adjust audio levels

- **Add Captions:**
  - Enable accessibility
  - Improves engagement
  - Tools: Rev, Descript, or manual in editor

- **Graphics Overlays:**
  - Lower-third with key metrics
  - Code snippets highlighted
  - Architecture diagrams

---

## Video Structure (Detailed Timeline)

| Time | Section | Duration | Main Activity |
|------|---------|----------|---|
| 0:00-0:30 | Title Card | 0:30 | Team intro, project name |
| 0:30-3:30 | Problem & Context | 3:00 | Explain challenge, show architecture |
| 3:30-15:00 | Demo - Agent Running | 11:30 | Run agent_demo.py with narration |
| 15:00-19:00 | Design Solutions | 4:00 | Show code snippets, explain decisions |
| 19:00-21:30 | Challenges & Solutions | 2:30 | Technical obstacles and fixes |
| 21:30-23:30 | Results & Metrics | 2:00 | Show RESULTS_SUMMARY.md |
| 23:30-25:00 | Conclusion | 1:30 | Key takeaways, closing |

---

## Narration Tips

### Pacing
- Speak clearly at moderate pace (not too fast)
- Pause 2-3 seconds when code executes
- Pause 1 second after each major point
- Total video: 20-25 minutes including pauses

### Emphasis
- Highlight numbers: "96 point zero 8 percent"
- Emphasize decisions: "This was **crucial** for handling GPU memory"
- Question-Answer: "Why Focal Loss? Because..."
- Build suspense: "After 202 iterations, the best model achieved..."

### Energy
- Start with enthusiasm (project intro)
- Maintain engagement through demo
- Build to climax (best results)
- End with inspiration (what's possible with automation)

---

## Example Terminal Output Reference

When you run `agent_demo.py --mode demo`, you'll see:

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

████████████████████████████████████████████████████████████████████████████████
  ► RUNNING AGENT LOOP
████████████████████████████████████████████████████████████████████████████████

============================================================
ITERATION 1 / 3  [iter_0203_20260519_161000]
============================================================
[INFO] Step 1: Exploring data…
[INFO] Found 7000 audio files across 206 species directories.
[INFO] Step 2: Requesting architecture proposal from LLM…
[INFO] Step 3: Extracting code from LLM response…
[INFO] Step 4: Executing training script…

Epoch 1/40, Train Loss: 0.5231
Epoch 1/40, Validation AUC: 0.5843
...
```

This output is already formatted for screen recording—clear, readable, and shows progress.

---

## Key Metrics to Highlight in Video

When showing results, emphasize:

1. **Performance:**
   - Best AUC: 0.9608 (96.08%)
   - Total iterations: 202
   - Success rate: 198/202 (98%)

2. **Efficiency:**
   - Average training time: ~6-8 minutes per iteration
   - Total compute: ~33 GPU hours
   - Model size: 40MB

3. **Robustness:**
   - Classes handled: 206 bird species
   - No manual architecture tuning required
   - Automatic error recovery

4. **Architecture:**
   - Backbone: EfficientNet-B1 (8.7M parameters)
   - Training: Focal Loss + Mixup + SpecAugment
   - Optimization: AdamW + AMP + Early Stopping

---

## Audio Specifications

- **Microphone:** USB condenser mic (reduce echo)
- **Bitrate:** 192 kbps, 48 kHz (CD quality)
- **Noise:** Use noise gate to remove background hum
- **Levels:** Peak at -6dB, average at -12dB to -18dB

---

## Troubleshooting

### Agent runs slowly
- GPU shared by 3 users; wait for availability
- Or pre-record during off-peak hours
- Or show pre-recorded run during narration

### Ollama not responding
- Verify: `ollama list`
- Start: `ollama serve` (separate terminal)
- Wait 30 seconds for model to load

### Terminal text too small
- Zoom terminal: Ctrl/Cmd + "+"
- Or resize window to 1920x1080 with larger font

### Recording stutters
- Close other applications
- Lower screen resolution to 1280x720
- Reduce screen refresh rate to 30Hz

### Audio cuts out
- Check mic isn't muted
- Verify recording software has mic access
- Test mic before full recording

---

## Final Checklist Before Recording

- [ ] Virtual environment activated
- [ ] GPU verified working
- [ ] Ollama running and responding
- [ ] Terminal font set to 16-18pt
- [ ] Terminal background light, text dark
- [ ] Screen resolution 1920x1080 or higher
- [ ] Microphone tested and working
- [ ] Recording software (OBS/Camtasia) configured
- [ ] External monitor closed to avoid distractions
- [ ] Phone on silent
- [ ] Practiced narration timing
- [ ] Results report generated
- [ ] Demo run pre-recorded (optional but recommended)

---

## Files Generated During Recording

After running the demo, you'll have:

```
experiments/
├── agent_state.json          # Updated iteration count
├── iterations_0201-0250/     # New iteration directories
│   ├── iter_0203_*/
│   │   ├── llm_proposal.txt
│   │   ├── train.py
│   │   ├── metrics.json
│   │   ├── model.pt
│   │   └── dataset_summary.json
│   └── ...
└── RESULTS_SUMMARY.md        # Generated report

RESULTS_SUMMARY.md            # Comprehensive report
pre_recorded_run.log          # If using Option A
```

---

## Publishing & Sharing

After editing and finalizing:

1. **Format:** MP4 (H.264 codec), 1920x1080, 30fps
2. **Size:** ~500MB-1GB (typical for 20-25 min video)
3. **Upload Platforms:**
   - YouTube (supports up to 256GB)
   - GitHub (in releases or as link to YouTube)
   - Kaggle Notebooks (embed or link)

4. **Metadata:**
   - Title: "BirdCLEF+ 2026: Autonomous Research Agent"
   - Tags: autonomous-ai, machine-learning, kaggle, birdclef
   - Description: Link to GitHub repo + paper/report links

---

## Additional Resources

- **Agent Code:** `agent.py` (main autonomous loop)
- **Training Code:** `train.py` (per-iteration training script)
- **Model Definitions:** `models.py` (EfficientNet-B1 scaffold)
- **Data Pipeline:** `data_loader.py`, `preprocessing.py`
- **README:** `README.md` (project overview)

---

## Questions? Issues?

Refer back to:
- `VIDEO_SCRIPT.md` for narration and timing
- `agent_demo.py --help` for command options
- `results_reporter.py --help` for report options
- Original `agent.py` for implementation details

Good luck with your presentation! 🎬📊🚀
