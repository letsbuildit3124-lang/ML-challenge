"""
Antigravity Arctic Pipeline Compatibility Wrapper.
Delegates to the model-agnostic Antigravity V5 Dense Pipeline (src.dense_pipeline).
"""

import sys
import warnings
from src.dense_pipeline import main as dense_main

if __name__ == "__main__":
    warnings.warn(
        "src.arctic_pipeline is deprecated and retained for backward compatibility. "
        "Use 'python3 -m src.dense_pipeline --smoke' or '--production' instead.",
        DeprecationWarning,
        stacklevel=2
    )
    dense_main()
