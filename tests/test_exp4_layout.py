"""Local (CPU) smoke test for exp4_visualize figure layout.

Monkeypatches run_inference to return random predictions so the *drawing*
code (subplot grids, panel letters, shared legend, scale bar, mosaic) can be
exercised without a GPU. Run:  python tests/test_exp4_layout.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import experiments.exp4_visualize as E4
from mangrove.datasets import load_region, load_split


def fake_inference(model_name, region, seed=2026):
    data = load_region(region)
    idx = np.asarray(load_split(region, "test"))
    y = data["change"][idx]
    ps = 512
    rng = np.random.RandomState(abs(hash(model_name + region)) % (2**31))
    pred = rng.randint(0, 4, size=(len(idx), ps, ps))
    # force some change so the failure-case selection has candidates
    pred[:, 100:200, 100:200] = 1
    return pred, y


def test_exp4_layout():
    E4.run_inference = fake_inference
    E4.FIGURES_ROOT = Path("/tmp/exp4synth")
    E4.FIGURES_ROOT.mkdir(parents=True, exist_ok=True)
    sys.argv = ["exp4_visualize.py", "--regions", "beilunhe", "zhanjiang"]
    E4.main()
    print("SYNTHETIC LAYOUT TEST OK")