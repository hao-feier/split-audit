"""Central path configuration. All stage scripts import from here."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT = ROOT / "data"
RAW_ROOT = DATA_ROOT / "raw"
GMW_ROOT = RAW_ROOT / "gmw"          # GMW v4.1.12 layers (labels)
S2_ROOT = RAW_ROOT / "s2"            # raw S2 L2A tiles (AWS open data)
PROCESSED_ROOT = Path(os.environ.get("MANGROVE_PROCESSED_ROOT", str(DATA_ROOT / "processed")))
SPLITS_ROOT = Path(os.environ.get("MANGROVE_SPLITS_ROOT", str(DATA_ROOT / "splits")))
# split assignments (per region); env override for buffer-sensitivity runs

EXPERIMENTS_ROOT = ROOT / "experiments"
ANALYSIS_ROOT = ROOT / "analysis"
RESULTS_ROOT = Path(os.environ.get("MANGROVE_RESULTS_ROOT", str(ROOT / "results")))
FIGURES_ROOT = ROOT / "figures"
PAPER_ROOT = ROOT / "paper"
LOGS_ROOT = ROOT / "logs"

for p in (GMW_ROOT, S2_ROOT, PROCESSED_ROOT, SPLITS_ROOT, RESULTS_ROOT, FIGURES_ROOT, LOGS_ROOT):
    p.mkdir(parents=True, exist_ok=True)


def ckpt_path(model: str, seed: int) -> Path:
    """Checkpoint for one (model, seed) run; seed is part of the name so
    multi-seed repeats never overwrite each other (2026-09-21)."""
    return RESULTS_ROOT / "checkpoints" / f"{model}_s{seed}.pt"


def metrics_path(model: str, seed: int) -> Path:
    return RESULTS_ROOT / f"metrics_{model}_s{seed}.json"


def log_path(model: str, seed: int) -> Path:
    return RESULTS_ROOT / f"train_log_{model}_s{seed}.jsonl"
