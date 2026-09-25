"""
Evaluate CLI entry point.
"""

import os
import json
from src.config import get_config
from src.pipeline import train_and_validate

if __name__ == "__main__":
    cfg = get_config()
    train_and_validate(cfg)
