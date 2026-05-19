#!/usr/bin/env python3
"""
Demo-optimized version of agent.py for video presentation.

This version includes enhanced console output formatting, progress indicators,
and timing information suitable for screen recording presentations.

Usage:
    python agent_demo.py --iterations 3 --model deepseek-r1:8b --data-dir ~/birdclef-data
"""

import sys
import os
from pathlib import Path
from datetime import datetime
import json

# Import the main agent components
sys.path.insert(0, str(Path(__file__).parent))
from agent import (
    run_agent, 
    load_state,
    EXPERIMENTS_DIR,
)

def print_banner(text: str, char: str = "=", width: int = 80) -> None:
    """Print a formatted banner."""
    print("\n")
    print(char * width)
    print(f" {text.center(width - 4)} ")
    print(char * width)
    print()

def print_section(text: str) -> None:
    """Print a section header."""
    print(f"\n{'█' * 80}")
    print(f"  ► {text}")
    print(f"{'█' * 80}\n")

def print_result_box(iteration: int, auc: float, is_best: bool = False) -> None:
    """Print formatted iteration result."""
    marker = "🏆 NEW BEST" if is_best else "✓ Result"
    print(f"  {marker:12} │ Iteration {iteration:3d} │ AUC: {auc:.4f}")

def display_summary_statistics() -> None:
    """Display summary of all experiments run so far."""
    state = load_state(EXPERIMENTS_DIR)
    
    print_section("📊 EXPERIMENT SUMMARY")
    
    if state["history"]:
        print(f"  Total Iterations:  {state['iteration']:3d}")
        print(f"  Best AUC Found:    {state['best_auc']:.4f}")
        print(f"  Best Iteration:    {state['best_iteration']}")
        
        # Calculate statistics
        aucs = [h["auc"] for h in state["history"] if h["auc"] > 0]
        if aucs:
            print(f"  Average AUC:       {sum(aucs)/len(aucs):.4f}")
            print(f"  Successful Runs:   {len(aucs)}/{len(state['history'])}")
            print(f"  Improvement:       +{max(aucs) - min(aucs):.4f}")
        
        print(f"\n  Top 5 Best Results:")
        sorted_history = sorted(state["history"], key=lambda x: x["auc"], reverse=True)
        for idx, record in enumerate(sorted_history[:5], 1):
            print(f"    {idx}. {record['iteration']:30s} AUC: {record['auc']:.4f}")
    else:
        print("  No experiments yet.")
    
    print()

def main_demo() -> None:
    """Run demo-optimized agent."""
    print_banner("BirdCLEF+ 2026 AUTONOMOUS RESEARCH AGENT", char="╔")
    
    print("""
    This is a demonstration of the autonomous research agent for BirdCLEF+ 2026.
    
    The agent will:
    1. Explore the dataset and collect statistics
    2. Ask the LLM to propose architecture improvements
    3. Generate and execute training code
    4. Capture and analyze results
    5. Iterate and improve the model
    
    Starting in DEMO MODE (limited iterations for quick showcase)
    """)
    
    print_section("🚀 INITIALIZING AGENT")
    print(f"  Dataset:     ~/birdclef-data (206 bird species)")
    print(f"  GPU:         NVIDIA L4 (24GB VRAM)")
    print(f"  Torch:       PyTorch with automatic mixed precision")
    print(f"  Model:       EfficientNet-B1 with custom head")
    print()
    
    # Load current state before running
    state_before = load_state(EXPERIMENTS_DIR)
    print(f"  Resuming from iteration: {state_before['iteration']}")
    print(f"  Current best AUC: {state_before['best_auc']:.4f}")
    print()
    
    # Run the main agent
    print_section("🔄 RUNNING AGENT LOOP")
    run_agent(
        num_iterations=3,  # Demo with 3 iterations
        model_name="deepseek-r1:8b",
        data_dir=Path(os.path.expanduser("~/birdclef-data")),
        exec_timeout=3600,  # 1 hour timeout for demo
    )
    
    # Display final summary
    state_after = load_state(EXPERIMENTS_DIR)
    print_banner("DEMO COMPLETE", char="═")
    print_section("📈 FINAL STATISTICS")
    
    if state_after["best_auc"] > state_before["best_auc"]:
        print(f"  ✨ NEW BEST AUC FOUND!")
        print(f"     Previous best: {state_before['best_auc']:.4f}")
        print(f"     New best:      {state_after['best_auc']:.4f}")
        print(f"     Improvement:   +{state_after['best_auc'] - state_before['best_auc']:.4f}")
    else:
        print(f"  No improvement in this demo run.")
        print(f"  Current best: {state_after['best_auc']:.4f}")
    
    print(f"\n  Total iterations completed: {state_after['iteration']}")
    print()
    
    display_summary_statistics()
    
    print_section("💾 ARTIFACTS SAVED")
    print(f"  Experiments directory: {EXPERIMENTS_DIR}")
    print(f"  Each iteration contains:")
    print(f"    - llm_proposal.txt  : LLM's architecture suggestion")
    print(f"    - train.py          : Generated training code")
    print(f"    - metrics.json      : Training results")
    print(f"    - model.pt          : Best model checkpoint")
    print()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Demo version of BirdCLEF+ agent")
    parser.add_argument("--mode", choices=["demo", "summary"], default="demo",
                       help="Mode: 'demo' runs agent, 'summary' shows results only")
    args = parser.parse_args()
    
    if args.mode == "demo":
        main_demo()
    else:
        print_banner("BirdCLEF+ 2026 - EXPERIMENT RESULTS SUMMARY")
        display_summary_statistics()
