"""
Predict CLI entry point.
"""

import os
import json
from src.config import get_config
from src.pipeline import run_test_and_validate

if __name__ == "__main__":
    cfg = get_config()
    thresh = cfg.selected_threshold
    val_json = os.path.join(cfg.reports_dir, "val_results.json")
    if os.path.exists(val_json):
        with open(val_json, "r", encoding="utf-8") as f:
            saved = json.load(f)
            thresh = saved.get("best_threshold", thresh)
    run_test_and_validate(cfg, threshold=thresh)
