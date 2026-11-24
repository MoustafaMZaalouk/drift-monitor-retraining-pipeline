"""Shared configuration. Every number that affects results lives here or in datasets.py."""
from pathlib import Path

SEED = 42
REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data" / "raw"
ARTIFACTS_DIR = REPO_ROOT / "artifacts"

# PSI smoothing floor so empty bins do not explode the statistic.
PSI_EPS = 1e-4
# Number of quantile bins for numeric PSI, fit on the reference window only.
PSI_N_BINS = 10
# Validation-window quantile used to set each signal's alert threshold.
TUNE_QUANTILE = 0.95
# Cooldown candidates (periods); smallest one that keeps validation alerts
# within the budget is chosen, then frozen.
COOLDOWN_GRID = (1, 2, 4, 7, 14)
# Max fraction of validation periods allowed to alert when picking a cooldown.
ALERT_BUDGET_FRAC = 0.05
# Periods after an event ends during which an alert still counts as detecting it.
DETECTION_GRACE_PERIODS = 2

# Trailing window (in periods) over which input/score distributions are
# computed at each stream step. Single-day windows are too small for stable
# distribution statistics (24 hourly rows), so bike monitors a rolling week;
# shoppers months are already large enough.
MONITOR_WINDOW = {"bike": 7, "shoppers": 1}
