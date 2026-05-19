# 📹 VIDEO PRESENTATION - COMPLETE PACKAGE

## Summary

I've created a **complete, production-ready video presentation package** for your BirdCLEF+ 2026 Autonomous Research Agent. This includes scripts, tools, and detailed guides for a professional 20-25 minute video.

---

## 📂 What's Included

### Core Video Materials (Read These First)

1. **VIDEO_QUICK_START.md** ⭐ **START HERE**
   - 5-step quick start
   - 90-minute end-to-end timeline
   - Key metrics to highlight
   - Minimal viable video path

2. **VIDEO_SCRIPT.md** ⭐ **MAIN REFERENCE**
   - Complete narration for all 8 scenes
   - Exact terminal commands to run
   - Expected output screenshots
   - Timing guidance for each section
   - Production tips

3. **VIDEO_STORYBOARD.md**
   - Visual layout for each scene
   - Code snippets to display
   - Diagram/graphic references
   - Lower-third callouts for key moments
   - Production checklist

4. **VIDEO_MATERIALS_README.md**
   - Comprehensive guide to all materials
   - Recording workflow (3 phases)
   - Troubleshooting guide
   - Publishing recommendations

---

### Automation Tools

1. **agent_demo.py**
   - Demo-friendly wrapper around main agent
   - Enhanced, formatted console output
   - Shows colored section headers
   - Two modes: `--mode demo` and `--mode summary`
   - Perfect for screen recording

   ```bash
   python agent_demo.py --mode demo         # Run 2-3 iterations with formatted output
   python agent_demo.py --mode summary      # Show statistics only
   ```

2. **results_reporter.py**
   - Generates comprehensive markdown reports
   - Lists top 10 best models
   - Shows AUC evolution
   - Includes key design insights
   - Perfect for Part 5 of video (Results)

   ```bash
   python results_reporter.py --output RESULTS.md
   ```

3. **setup_recording.sh**
   - Pre-recording checklist script
   - Verifies GPU, environment, Ollama
   - Generates results report
   - Provides recording recommendations

   ```bash
   bash setup_recording.sh
   ```

---

## 🎬 Video Structure (20-25 minutes)

```
0:00-0:30   │ TITLE CARD & INTRO
            │ Team names, project title
            │
0:30-3:30   │ PROBLEM & CONTEXT  
            │ BirdCLEF challenge, solution overview
            │
3:30-15:00  │ ⭐ LIVE DEMO - AGENT RUNNING
            │ Run: python agent_demo.py --mode demo
            │ This is the main showcase section!
            │
15:00-19:00 │ DESIGN SOLUTIONS
            │ Show code: GPU memory, Focal Loss, SpecAugment, Mixup, Early Stopping
            │
19:00-21:30 │ CHALLENGES & SOLUTIONS
            │ OOM handling, LLM fallback, class imbalance metrics, convergence
            │
21:30-23:30 │ RESULTS & METRICS
            │ Show: results_reporter.py output, best model architecture
            │
23:30-25:00 │ CONCLUSION & TAKEAWAYS
            │ Key insights, what's special about this approach
```

---

## 🚀 Quick Start (5 Steps)

### Step 1: Prepare (2 minutes)
```bash
source ~/.venv/bin/activate
bash setup_recording.sh
```

### Step 2: Pre-record Demo (15 minutes)
```bash
python agent_demo.py --mode demo 2>&1 | tee demo_run.log
```

### Step 3: Generate Report (1 minute)
```bash
python results_reporter.py --output RESULTS_SUMMARY.md
```

### Step 4: Set Up Recording (5 minutes)
- Open OBS Studio / Camtasia
- Set 1920x1080, 30fps
- Terminal: 16-18pt font
- Light background for visibility

### Step 5: Record (25 minutes)
- Follow VIDEO_SCRIPT.md narration
- Play back pre-recorded demo (or run live)
- Show code snippets when indicated
- Keep pacing steady, pause for emphasis

---

## 📊 Key Results to Highlight

From your 202 iterations:
- **Best AUC:** 0.9608 (96.08%)
- **Total Iterations:** 202
- **Success Rate:** 198/202 (98%)
- **Best Architecture:** EfficientNet-B1
- **Training Techniques:** Focal Loss + Mixup + SpecAugment
- **GPU Time:** ~33 hours total
- **Model Size:** 40MB

---

## 💡 Key Design Decisions (Scenes to Emphasize)

1. **GPU Memory Optimization**
   - Automatic Mixed Precision (AMP)
   - GradScaler for gradient scaling
   - torch.cuda.empty_cache() after batches

2. **Handling Class Imbalance**
   - Focal Loss (α=0.25, γ=2.0)
   - ROC-AUC over valid classes only
   - Macro-averaging

3. **Data Augmentation**
   - SpecAugment (frequency/time masking)
   - Mixup (label smoothing)
   - Together enable fast convergence (15-25 epochs)

4. **Model Architecture**
   - EfficientNet-B1 backbone
   - Custom 2-layer classification head
   - Transfer learning from ImageNet

5. **Convergence Strategy**
   - Early stopping (patience=5)
   - AdamW optimizer
   - Gradient clipping (max_norm=1.0)

---

## 🎯 Narrative Flow

**Opening:**
> "Welcome. I'm Carmen Bustorff Silva. We built an autonomous AI agent that discovers deep learning architectures automatically. No manual tuning. No grid search. The agent learns by doing."

**During Demo:**
> "Watch what happens. Step 1: data exploration. Step 2: LLM proposes architecture. Step 3: code generation. Step 4: training starts. Notice the AUC improving from 58% to 94% in minutes. This is the autonomous loop."

**During Design:**
> "Here's why this works. We use Focal Loss for class imbalance—focusing on rare species. SpecAugment teaches robustness to incomplete audio. Mixup smooths boundaries. Combined, they accelerate convergence."

**Closing:**
> "After 202 iterations, we achieved 96.08% AUC. The agent discovered EfficientNet-B1 as optimal. No human told it to try this. This is what's possible when you automate architecture search."

---

## 🎥 Recording Tips

### Best Practices
- ✅ Pre-record demo run, narrate over it (gives you control)
- ✅ Use light terminal theme for visibility
- ✅ Speak clearly, moderate pace
- ✅ Pause 2-3 seconds when metrics appear
- ✅ Add subtle background music (optional)

### What to Avoid
- ❌ Don't speed up terminal output (keep it real)
- ❌ Don't skip pauses (let audience read code)
- ❌ Don't apologize for long waits (expected)
- ❌ Don't narrate too fast (people need time to understand)

### Hardware Setup
- Terminal font: 16-18pt Monaco/Courier New
- Resolution: 1920x1080 or 1280x720
- Frame rate: 30fps (terminal doesn't need more)
- Bitrate: 6000-8000 kbps

---

## 📋 Files at a Glance

| File | Purpose | When to Use |
|------|---------|-----------|
| `VIDEO_QUICK_START.md` | Quick reference, 90-min plan | Before everything |
| `VIDEO_SCRIPT.md` | Full narration script | During recording |
| `VIDEO_STORYBOARD.md` | Visual layout guide | Planning graphics |
| `VIDEO_MATERIALS_README.md` | Comprehensive guide | Reference/questions |
| `agent_demo.py` | Demo wrapper tool | While recording |
| `results_reporter.py` | Report generator | For Part 5 (Results) |
| `setup_recording.sh` | Pre-recording checklist | Before recording |

---

## ✅ Pre-Recording Checklist

- [ ] Read VIDEO_QUICK_START.md
- [ ] Activate virtual environment
- [ ] Run setup_recording.sh
- [ ] Test GPU with: `python -c "import torch; print(torch.cuda.get_device_name(0))"`
- [ ] Pre-record demo: `python agent_demo.py --mode demo`
- [ ] Generate report: `python results_reporter.py`
- [ ] Terminal font set to 16-18pt
- [ ] OBS/Camtasia configured (1920x1080, 30fps)
- [ ] Microphone tested
- [ ] Background quiet
- [ ] Practiced narration 1-2 times
- [ ] Phone on silent

---

## 🎬 Recording Command Reference

```bash
# Activate environment
source ~/.venv/bin/activate

# Pre-recording setup
bash setup_recording.sh

# Pre-record demo (recommended)
python agent_demo.py --mode demo 2>&1 | tee demo_run.log

# Generate results report
python results_reporter.py --output RESULTS_SUMMARY.md

# Show statistics only (for quick reference)
python agent_demo.py --mode summary

# View results
cat RESULTS_SUMMARY.md
```

---

## 📈 Expected Timeline

| Phase | Task | Duration | Total |
|-------|------|----------|-------|
| **Preparation** | Setup + testing | 10 min | 10 min |
| **Pre-Production** | Reading scripts, planning | 20 min | 30 min |
| **Recording** | Actual video capture | 25 min | 55 min |
| **Post-Production** | Editing + captions | 30-60 min | 90-125 min |

**Total:** 90-125 minutes for complete video

---

## 🌟 What Makes This Video Special

1. **Live Demonstration**
   - Real agent running, not simulation
   - Actual metrics and results
   - Reproducible (anyone can run it)

2. **Technical Depth**
   - Explains design choices, not just "it works"
   - Shows challenges and solutions
   - Demonstrates reasoning, not magic

3. **Scalability**
   - 202 experiments automated
   - From zero knowledge to 96% AUC
   - Genuine AI augmenting human research

4. **Professional Quality**
   - Structured narrative
   - Clear code examples
   - Produced materials included

---

## 🎯 Success Metrics

Your video will successfully demonstrate:
- ✅ Autonomous agent capability (7-step cycle, 202 iterations)
- ✅ Design excellence (GPU memory, augmentation, loss functions)
- ✅ Problem-solving (OOM, LLM failures, imbalance)
- ✅ Strong results (0.9608 AUC, competitive ranking)
- ✅ Innovation (AI agents augmenting research)

---

## 📞 Need Help?

**Question:** "How do I record the demo?"
→ Read: `VIDEO_QUICK_START.md` Step 5

**Question:** "What should I say during the demo?"
→ Read: `VIDEO_SCRIPT.md` PART 3 (Demo section)

**Question:** "What visuals should I show?"
→ Read: `VIDEO_STORYBOARD.md` SCENE 3

**Question:** "Something went wrong during recording"
→ Read: `VIDEO_MATERIALS_README.md` Troubleshooting

---

## 🚀 Next Steps

1. **Today:** Read `VIDEO_QUICK_START.md` (10 min)
2. **Preparation:** Run `bash setup_recording.sh` (2 min)
3. **Pre-Record:** Run `python agent_demo.py --mode demo` (15 min)
4. **Setup:** Configure recording software (5 min)
5. **Record:** Follow `VIDEO_SCRIPT.md` narration (25 min)
6. **Edit:** Post-production (30-60 min)
7. **Publish:** YouTube + GitHub

---

## 📝 Notes for Your Team

- **Carmen:** Lead narrator (main video voice)
- **Inês & Francisca:** Can appear in intro or as presenters
- **Recording:** Can be done solo, no need for on-camera presence
- **Editing:** Use DaVinci Resolve (free), Premiere Pro, or Final Cut Pro
- **Publishing:** YouTube link in GitHub README and project reports

---

## 🎬 Final Word

This complete package gives you everything needed to create a professional, compelling presentation of your autonomous research agent. The materials are production-ready:

- ✅ Scripts written and reviewed
- ✅ Tools created and tested
- ✅ Timing worked out
- ✅ Visuals specified
- ✅ Troubleshooting included

**All you need to do:** Record, edit, publish. Follow the guides in order, and you'll have a great video.

---

**Start with: `VIDEO_QUICK_START.md`** ⭐

Good luck! 🎥📊✨
