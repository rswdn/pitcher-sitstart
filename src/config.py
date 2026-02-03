"""Configuration constants for the sit/start analyzer."""

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

# Rolling windows
RECENT_STARTS_WINDOW = 5
REST_DAYS_MAX = 6  # clip rest days for modeling features

# Scoring thresholds
START_THRESHOLD = 0.65
STREAM_THRESHOLD = 0.45

# Paths
PARK_FACTORS_PATH = BASE_DIR / "park_factors.json"

# Output defaults
DEFAULT_OUTPUT_DIR = BASE_DIR / "data"
CACHE_DIR = DEFAULT_OUTPUT_DIR / "statcast_cache"
FG_LOCAL_DIR = DEFAULT_OUTPUT_DIR / "fg_manual"

# Feature weights for the heuristic stream score.
HEURISTIC_WEIGHTS = {
    "k_bb_pct_recent": 0.40,
    "k_pct_recent": 0.10,
    "bb_pct_recent": -0.2,
    "avg_velo_recent": 0.1,
    "avg_ev_recent": -0.1,
    "park_run_factor": -0.05,
}
