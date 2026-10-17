"""Unit test for cd_metrics (omission/commission conditional rates)."""
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from mangrove.metrics import cd_metrics
from mangrove.config import CHANGE_CLASSES

# 100 px: 40 stable_mangrove, 10 loss, 10 gain, 40 background.
y_true = np.array([0]*40 + [1]*10 + [2]*10 + [3]*40)
# predictions:
#  loss: 7 correct, 3 -> background
#  gain: 5 correct, 5 -> background
#  extra: 4 background px predicted as gain
y_pred = np.array([0]*40
                   + [1]*7 + [3]*3          # loss
                   + [2]*5 + [3]*5          # gain
                   + [2]*4 + [3]*36)        # background (4 mispredicted as gain)

def test_cd_metrics():
    m = cd_metrics(y_true, y_pred, 4, CHANGE_CLASSES)
    exp_omit = (20 - 12) / 20          # 8 of 20 true changes missed -> 0.4
    exp_comm = 4 / 80                  # 4 false alarms of 80 unchanged -> 0.05
    print(f"OA={m['OA']:.3f} mIoU={m['mIoU']:.3f} F1_macro={m['F1_macro']:.3f}")
    print(f"CD_Accuracy_changed={m['CD_Accuracy_changed']:.3f} "
      f"Omission={m['Omission_Error']:.4f} (exp {exp_omit}) "
      f"Commission={m['Commission_Error']:.4f} (exp {exp_comm})")
    assert abs(m["Omission_Error"] - exp_omit) < 1e-9, "omission mismatch"
    assert abs(m["Commission_Error"] - exp_comm) < 1e-9, "commission mismatch"
    # edge case: no changes at all
    y0 = np.array([0]*50 + [3]*50)
    m0 = cd_metrics(y0, y0, 4, CHANGE_CLASSES)
    assert m0["Omission_Error"] == 0.0 and m0["Commission_Error"] == 0.0
    print("PASS")
