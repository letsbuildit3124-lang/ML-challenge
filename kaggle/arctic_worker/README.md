# Kaggle Arctic GPU Worker Package

This directory contains the standalone worker executed remotely on Kaggle GPUs (Nvidia L4 / T4).

## Files:
- `kaggle_worker.py`: Worker script that reads input Parquet chunks, runs batched FP16 Arctic inference, and outputs normalized `.npy` embeddings, target ID Parquets, and metadata checksums.
- `kernel-metadata.json`: Kernel specification for the official Kaggle CLI (`kaggle kernels push`).

## Competition Compliance:
- **No external lookups**: The worker never queries search engines, web domains, or external business databases.
- **Data protection**: Only the text fields (`business_name`, `business_address`, `country`) from the current chunk are passed.
