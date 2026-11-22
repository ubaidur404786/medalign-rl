"""Shared project settings, so every notebook uses the same names and paths."""

from pathlib import Path

# --- Models ---------------------------------------------------------------
BASE_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"          # main model
TINY_MODEL = "HuggingFaceTB/SmolLM2-135M-Instruct"  # fast debugging model

# Hugging Face repo names for the trained models (fill in your HF username)
HF_USER = "ubaidii123"
HF_REPOS = {
    "sft": f"{HF_USER}/medalign-sft",
    "ppo": f"{HF_USER}/medalign-ppo",
    "grpo": f"{HF_USER}/medalign-grpo",
    "golden": f"{HF_USER}/medalign-golden-eval",
}

# --- Paths ----------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results"
OUTPUTS_DIR = ROOT / "outputs"   # local checkpoints (git-ignored)

# --- Reproducibility ------------------------------------------------------
SEED = 42

# The disclaimer that every model card and the app must show
DISCLAIMER = "Educational demo only. Not medical advice."
